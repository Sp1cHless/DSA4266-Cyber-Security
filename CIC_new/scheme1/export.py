"""Persist a shared donor plan and export fixed, model-independent scheme1 arrays."""
from __future__ import annotations

from collections import Counter, defaultdict
import logging
from pathlib import Path
import shutil

import numpy as np

from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.dataset import Pool
from .statistics import FIELDS, FIXED, LEARNED, NUMERIC_SCHEMA, categorical_features, model_features, merge_buckets
from .storage import (RawStore, arrays, shard_files, original_catalog, state_at,
                      identity, digest, load_tables, readonly, read_json, write_json)

LOG = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")


def make_plan(prepared, catalog, seed):
    bg = defaultdict(lambda: defaultdict(list))
    attacks = defaultdict(lambda: defaultdict(list))
    train = np.flatnonzero(catalog["split"] == 0)
    for sid in train[catalog["label"][train] == 0]:
        active = catalog["active"][sid]
        band = 0 if active == 1 else 1 if active <= 4 else 2
        bg[band][str(catalog["ip"][sid])].append(int(sid))
    connection = readonly(Path(prepared) / "manifest.sqlite")
    try:
        for sid, attack in connection.execute("SELECT sequence_id,attack FROM sequence_attacks ORDER BY attack,sequence_id"):
            if catalog["split"][sid] == 0:
                attacks[attack][str(catalog["ip"][sid])].append(sid)
    finally:
        connection.close()
    positive_count = int(catalog["label"][train].sum())
    if not positive_count or len({host for group in bg.values() for host in group}) < 2:
        raise ValueError("Need real attacks and at least two pure-benign training sources")
    backgrounds, positives = Pool(bg), Pool(attacks)
    rng = np.random.default_rng(seed)
    lefts, rights, targets, bands, kinds = [], [], [], [], []
    for malicious in (True, False):
        for _ in range(positive_count):
            if malicious:
                right, host, target = positives.pick(rng)
                left, _, band = backgrounds.pick(rng, exclude=host)
            else:
                left, host, band = backgrounds.pick(rng)
                right, _, _ = backgrounds.pick(rng, exclude=host)
                target = ""
            lefts.append(left)
            rights.append(right)
            targets.append(target)
            bands.append(band)
            kinds.append("synthetic_malicious" if malicious else "synthetic_benign")
    maximum = len(catalog["split"]) - 1
    return dict(sample_id=np.arange(maximum + 1, maximum + 1 + 2 * positive_count, dtype=np.int64),
                left_id=np.array(lefts, np.int64), right_id=np.array(rights, np.int64),
                right_shift_buckets=np.zeros(2 * positive_count, np.int8), target_attack=np.array(targets, dtype="U64"),
                background_band=np.array(bands, np.uint8), sample_type=np.array(kinds, dtype="U20"))


class Writer:
    def __init__(self, folder, size):
        self.folder = Path(folder)
        self.folder.mkdir()
        self.size, self.part, self.rows, self.counts, self.files = size, 0, [], Counter(), []

    def append(self, features, state, **metadata):
        self.rows.append(dict(X=features, mask=state["mask"], y_bucket=state["y_bucket"],
                              y_sequence=np.uint8(state["y_bucket"].max()), flow_counts=state["flow_counts"],
                              attack_flow_counts=state["attack_flow_counts"], **metadata))
        self.counts[metadata["sample_type"]] += 1
        if len(self.rows) >= self.size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        payload = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        path = self.folder / f"part_{self.part:06d}.npz"
        np.savez_compressed(path, **payload)
        self.files.append(dict(path=str(path.relative_to(self.folder.parent)).replace("\\", "/"),
                               count=len(self.rows), sha256=digest(path)))
        self.part += 1
        self.rows.clear()


def export(prepared, embeddings, out, *, seed=4267, shard_size=2048):
    prepared, embeddings, out = map(lambda path: Path(path).resolve(), (prepared, embeddings, out))
    if shard_size < 1:
        raise ValueError("Positive shard size required")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Fresh dataset directory required: {out}")
    tables, embedding_manifest = load_tables(prepared, embeddings)
    schema = read_json(prepared / "schema.json")
    catalog = original_catalog(prepared)
    names = list(bf.NUMERIC_NAMES)
    for field in FIELDS:
        names.extend(([f"{field}_UNK_ratio"] + [f"{field}_code_{code}_ratio" for code in schema["vocabularies"][field]])
                     if field in FIXED else [f"{field}_embedding_{i}" for i in range(tables[field].shape[1])])
    schema.update(feature_names=names, feature_count=len(names), embedding_dimensions=embedding_manifest["dimensions"],
                  representation="82 exact statistics + fixed category proportions + mean learned field embeddings")
    moments = bf.Moments(len(names))
    files = shard_files(prepared)
    for number, path in enumerate(files, 1):
        shard = arrays(path)
        active = shard["mask"].astype(bool) & (catalog["split"][shard["sample_id"]] == 0)[:, None]
        # Apply the active TRAIN selector before accumulating any fitted moments.
        numeric = bf.transform_log(shard["numeric"][active], NUMERIC_SCHEMA)
        categorical = categorical_features(shard, schema, tables)[active]
        moments.add(np.concatenate((numeric, categorical), axis=1))
        if number % 50 == 0:
            LOG.info("Final train-only scaler: %s raw shards", number)
    scaler = moments.finish()
    scaler["population"] = "all real retained training active buckets only; numeric log1p before scaling, no whole-bucket L2"
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(complete=False, format="model-independent-sequence-shards", representation="scheme1",
                    prepared_path=str(prepared), prepared_identity=identity(prepared),
                    embedding_checkpoint_sha256=embedding_manifest["embedding_sha256"],
                    seed=seed, feature_width=len(names), timesteps=10, shift_mode="none", target_ratios=[.6, .2, .2],
                    training_population="benign and malicious", no_whole_bucket_encoder=True)
    write_json(out / "manifest.json", manifest)
    write_json(out / "schema.json", schema)
    write_json(out / "scaler.json", scaler)
    shutil.copy2(embeddings / "embeddings.npz", out / "embeddings.npz")
    shutil.copy2(embeddings / "manifest.json", out / "embedding_manifest.json")
    plan = make_plan(prepared, catalog, seed)
    np.savez_compressed(out / "donor_plan.npz", **plan)
    manifest["donor_plan_sha256"] = digest(out / "donor_plan.npz")
    writers = {name: Writer(out / name, shard_size) for name in SPLITS}
    for number, file in enumerate(files, 1):
        shard = arrays(file)
        features = model_features(shard, schema, tables, scaler)
        for row, sid in enumerate(shard["sample_id"]):
            sid = int(sid)
            split = int(catalog["split"][sid])
            if split == 3:
                continue
            kind = "real_malicious" if catalog["label"][sid] else "real_benign"
            writers[SPLITS[split]].append(features[row], {key: shard[key][row] for key in
                ("mask", "y_bucket", "flow_counts", "attack_flow_counts")}, sample_id=sid, sample_type=kind,
                left_id=sid, right_id=-1, right_shift_buckets=0, target_attack="", background_band=-1,
                left_ip=str(catalog["ip"][sid]), right_ip="", primary_attack=str(catalog["attack"][sid]))
        if number % 50 == 0:
            LOG.info("Real dataset export: %s/%s raw shards", number, len(files))
    with RawStore(prepared) as store:
        store.prime(np.unique(np.concatenate((plan["left_id"], plan["right_id"]))), out / "donor_cache.sqlite")
        for i, sid in enumerate(plan["sample_id"]):
            left, right = int(plan["left_id"][i]), int(plan["right_id"][i])
            state = merge_buckets(store.get(left), store.get(right))
            features = model_features(state, schema, tables, scaler)
            writers["train"].append(features, state, sample_id=int(sid),
                sample_type=str(plan["sample_type"][i]), left_id=left, right_id=right, right_shift_buckets=0,
                target_attack=str(plan["target_attack"][i]), background_band=int(plan["background_band"][i]),
                left_ip=str(catalog["ip"][left]), right_ip=str(catalog["ip"][right]),
                primary_attack=str(catalog["attack"][right]))
            if (i + 1) % 1000 == 0:
                LOG.info("Zero-shift compositions: %s/%s", i + 1, len(plan["sample_id"]))
    for writer in writers.values():
        writer.flush()
    positive_count = len(plan["sample_id"]) // 2
    manifest.update(complete=True, P=positive_count, augmentation_per_class=positive_count,
                    counts={split: dict(writer.counts) for split, writer in writers.items()},
                    total_samples={split: sum(writer.counts.values()) for split, writer in writers.items()},
                    shards=[record for writer in writers.values() for record in writer.files],
                    schema_sha256=digest(out / "schema.json"), scaler_sha256=digest(out / "scaler.json"))
    write_json(out / "manifest.json", manifest)
    LOG.info("Dataset export complete: %s", manifest["total_samples"])
    return manifest
