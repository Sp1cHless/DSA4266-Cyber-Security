"""Sequence -> unfitted buckets -> existing 6:2:2 split -> train-only fitting."""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np

from CIC_new.preprocessing import bucket_statistics as bf, sequence_index
from .statistics import FIXED, LEARNED, FIELDS, NUMERIC_SCHEMA, raw_bucket, category_indices, project_histogram
from .storage import RawWriter, arrays, shard_files, original_catalog, write_json

LOG = logging.getLogger(__name__)


def aggregate_originals(out, shard_size):
    connection = sequence_index.connect(out)
    try:
        writer = RawWriter(out / "raw_originals", connection, shard_size)
        starts = iter(connection.execute("SELECT sequence_id,window_start_ms FROM original_sequences ORDER BY sequence_id"))
        query = "SELECT sequence_id," + ",".join(bf.EVENT_COLUMNS) + " FROM work.events ORDER BY sequence_id,flow_start_milliseconds,raw_rowid"
        for count, (sid, group) in enumerate(itertools.groupby(connection.execute(query), key=lambda row: row[0]), 1):
            expected, start = next(starts)
            if sid != expected:
                raise AssertionError("Sequence/event order mismatch")
            events = bf.local_events(np.asarray([row[1:] for row in group], np.int64), start)
            writer.append(sid, raw_bucket(events))
            if count % 10000 == 0:
                LOG.info("Buckets before split: %s originals", f"{count:,}")
        writer.flush()
    finally:
        connection.close()


def select_pool(files, catalog, split_code, limit, seed):
    """Uniform priority reservoir over real active buckets, independent of labels."""
    rng = np.random.default_rng(seed)
    priority = np.empty(0)
    file_indices = np.empty(0, np.int64)
    positions = np.empty(0, np.int64)
    for part, path in enumerate(files):
        with np.load(path, allow_pickle=False) as shard:
            eligible = shard["mask"].astype(bool) & (catalog["split"][shard["sample_id"]] == split_code)[:, None]
            flat = np.flatnonzero(eligible)
        priority = np.concatenate((priority, rng.random(len(flat))))
        file_indices = np.concatenate((file_indices, np.full(len(flat), part, np.int64)))
        positions = np.concatenate((positions, flat))
        if len(priority) > limit:
            keep = np.argpartition(priority, limit - 1)[:limit]
            priority, file_indices, positions = priority[keep], file_indices[keep], positions[keep]
    if not len(positions):
        raise ValueError("Embedding fitting requires nonempty real train and validation bucket pools")
    order = np.lexsort((positions, file_indices))
    return file_indices[order], positions[order]


def subset_coo(coo, selected, total_rows):
    mapping = np.full(total_rows, -1, np.int64)
    mapping[selected] = np.arange(len(selected))
    take = mapping[coo[:, 0]] >= 0
    result = coo[take].copy()
    result[:, 0] = mapping[result[:, 0]]
    return result


def write_pool(folder, split, selection, files, schema, scaler):
    file_indices, positions = selection
    fixed_parts, numeric_parts, ids_parts, bucket_parts, labels_parts = [], [], [], [], []
    sparse = {field: [] for field in LEARNED}
    offset = 0
    for part in np.unique(file_indices):
        shard = arrays(files[int(part)])
        selected = positions[file_indices == part]
        rows = len(shard["sample_id"]) * 10
        numeric = bf.transform_log(shard["numeric"].reshape(-1, 82)[selected], NUMERIC_SCHEMA)
        numeric_parts.append(((numeric - scaler["mean"]) / scaler["scale"]).astype(np.float32))
        fixed_parts.append(np.concatenate([
            project_histogram(subset_coo(shard["hist_" + f], selected, rows), len(selected), schema["vocabularies"][f])
            for f in FIXED], axis=1).astype(np.float32))
        ids_parts.append(shard["sample_id"][selected // 10])
        bucket_parts.append(selected % 10)
        labels_parts.append(shard["y_bucket"].max(axis=1)[selected // 10])
        for field in LEARNED:
            coo = subset_coo(shard["hist_" + field], selected, rows)
            coo[:, 1] = category_indices(coo[:, 1], schema["vocabularies"][field])
            coo[:, 0] += offset
            sparse[field].append(coo)
        offset += len(selected)
    payload = dict(numeric=np.concatenate(numeric_parts), fixed=np.concatenate(fixed_parts),
                   sample_id=np.concatenate(ids_parts), bucket=np.concatenate(bucket_parts),
                   label=np.concatenate(labels_parts))
    for field in LEARNED:
        coo = np.concatenate(sparse[field])
        order = np.argsort(coo[:, 0], kind="stable")
        coo = coo[order]
        totals = np.bincount(coo[:, 0], weights=coo[:, 2], minlength=offset)
        lengths = np.bincount(coo[:, 0], minlength=offset)
        payload[field + "_offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        payload[field + "_indices"] = coo[:, 1].astype(np.int64)
        payload[field + "_weights"] = (coo[:, 2] / totals[coo[:, 0]]).astype(np.float32)
    np.savez_compressed(folder / (split + ".npz"), **payload)
    LOG.info("Embedding %s pool: %s real active buckets", split, f"{offset:,}")


def prepare(db, out, seed=4266, shard_size=2048, train_pool=50000, val_pool=2048):
    db, out = Path(db).resolve(), Path(out).resolve()
    if not db.is_file():
        raise FileNotFoundError(db)
    if min(shard_size, train_pool, val_pool) < 1:
        raise ValueError("Shard and pool sizes must be positive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Fresh preparation directory required: {out}")
    out.mkdir(parents=True, exist_ok=True)
    signature = sequence_index.signature(db)
    manifest = dict(version="scheme1-prepared-v1", complete=False, source=signature, seed=seed,
                    train_pool_limit=train_pool, validation_pool_limit=val_pool,
                    order=["sequence_construction", "bucket_aggregation", "split", "train_only_fitting"],
                    target_ratios=[.6, .2, .2], shift_buckets=0)
    write_json(out / "manifest.json", manifest)
    sequence_index.index_source(out, db)
    aggregate_originals(out, shard_size)
    sequence_index.split_originals(out, seed)
    catalog = original_catalog(out)
    files = shard_files(out)
    vocabulary = {field: set() for field in FIELDS}
    coverage_witnesses = set()
    moments = bf.Moments(82)
    for part, path in enumerate(files):
        shard = arrays(path)
        training = catalog["split"][shard["sample_id"]] == 0
        active = shard["mask"].astype(bool) & training[:, None]
        moments.add(bf.transform_log(shard["numeric"][active], NUMERIC_SCHEMA))
        for field in FIELDS:
            coo = shard["hist_" + field]
            retained = coo[training[coo[:, 0] // 10]]
            if field in LEARNED and len(retained):
                codes, first = np.unique(retained[:, 1], return_index=True)
                for code, row in zip(codes, first):
                    if int(code) not in vocabulary[field]:
                        coverage_witnesses.add((part, int(retained[row, 0])))
            vocabulary[field].update(map(int, retained[:, 1]))
    schema = dict(version="scheme1-fields-v1", numeric_names=bf.NUMERIC_NAMES,
                  fixed_fields=list(FIXED), learned_fields=list(LEARNED),
                  vocabularies={field: sorted(values) for field, values in vocabulary.items()},
                  unknown_index=0, port_zero_is_category=True, port_eligibility="protocol in (6,17)",
                  dns_eligibility="query_id != 0 OR query_type != 0 OR answer_ttl != 0",
                  icmp_eligibility="protocol == 1", service_rule="same eight destination-port groups as scheme2; TCP/UDP only",
                  category_pooling="per-field eligible-flow mean; all training-observed port codes retained",
                  training_pool_rule="uniform active-bucket reservoir plus one witness for every learned category",
                  histogram_only_fields=["src_port_all", "dst_port_all", "destination"],
                  log1p_columns=NUMERIC_SCHEMA["log1p_columns"], timesteps=10, bucket_seconds=10)
    scaler = moments.finish()
    scaler["population"] = "all real retained training active buckets; no synthetic/held-out buckets"
    write_json(out / "schema.json", schema)
    write_json(out / "numeric_scaler.json", scaler)
    pools = out / "embedding_pools"
    pools.mkdir()
    for code, name, limit in ((0, "train", train_pool), (1, "validation", val_pool)):
        selection = select_pool(files, catalog, code, limit, seed + 2 + code)
        if code == 0:
            pairs = np.unique(np.concatenate((np.column_stack(selection),
                                              np.asarray(sorted(coverage_witnesses), np.int64).reshape(-1, 2))), axis=0)
            selection = pairs[:, 0], pairs[:, 1]
        write_pool(pools, name, selection, files, schema, scaler)
    if signature != sequence_index.signature(db):
        raise AssertionError("Source changed during preparation")
    counts = {name: int(np.count_nonzero(catalog["split"][1:] == i))
              for i, name in enumerate(("train", "validation", "test", "excluded"))}
    manifest.update(complete=True, counts=counts,
                    original_train_positive=int(catalog["label"][catalog["split"] == 0].sum()))
    write_json(out / "manifest.json", manifest)
    LOG.info("Preparation complete: %s", counts)
    return manifest
