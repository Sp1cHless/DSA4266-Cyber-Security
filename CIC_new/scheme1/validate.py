"""Stream all exported data; independently reconstruct sampled examples from flows."""
from __future__ import annotations

from collections import Counter, defaultdict
import logging
from pathlib import Path

import numpy as np

from CIC_new.preprocessing import bucket_statistics as bf, split_policy
from CIC_new.preprocessing.sequence_index import signature
from .statistics import FIELDS, LEARNED, NUMERIC_SCHEMA, raw_bucket, categorical_features, model_features
from .storage import (RawStore, arrays, digest, identity, original_catalog, readonly,
                      read_json, write_json, shard_files)

LOG = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(dataset, prepared=None, *, deep_samples=8, seed=4266):
    dataset = Path(dataset).resolve()
    manifest = read_json(dataset / "manifest.json")
    require(manifest.get("complete") and manifest.get("representation") == "scheme1", "Completed scheme1 dataset required")
    require(deep_samples >= 0, "deep_samples must be nonnegative")
    prepared = Path(prepared or manifest["prepared_path"]).resolve()
    prep = read_json(prepared / "manifest.json")
    require(identity(prepared) == manifest["prepared_identity"], "Preparation fingerprint mismatch")
    require(prep["order"][:3] == ["sequence_construction", "bucket_aggregation", "split"], "Wrong pipeline order")
    require(manifest["target_ratios"] == [.6, .2, .2], "Split target must remain 6:2:2")
    for name in ("schema", "scaler"):
        require(digest(dataset / (name + ".json")) == manifest[name + "_sha256"], name + " checksum mismatch")
    require(digest(dataset / "embeddings.npz") == manifest["embedding_checkpoint_sha256"], "Embedding checksum mismatch")
    require(digest(dataset / "donor_plan.npz") == manifest["donor_plan_sha256"], "Donor plan checksum mismatch")
    schema, scaler = read_json(dataset / "schema.json"), read_json(dataset / "scaler.json")
    tables = arrays(dataset / "embeddings.npz")
    embedding_manifest = read_json(dataset / "embedding_manifest.json")
    require(embedding_manifest["prepared_identity"] == identity(prepared) and embedding_manifest.get("frozen"), "Wrong embedding provenance")
    require(not embedding_manifest["test_used"], "Embedding selection must not use test")
    for field in LEARNED:
        require(tables[field].shape == (len(schema["vocabularies"][field]) + 1, schema["embedding_dimensions"][field]), "Bad table shape: " + field)
        require(np.isfinite(tables[field]).all() and np.all(tables[field][0] == 0), "Bad category table: " + field)
        require(embedding_manifest["training_categories_covered"][field] == len(schema["vocabularies"][field]),
                "Some training category embeddings lack target observations: " + field)
    catalog = original_catalog(prepared)
    maximum = len(catalog["split"]) - 1
    train_ids = np.flatnonzero(catalog["split"] == 0)
    p = int(catalog["label"][train_ids].sum())
    require(manifest["P"] == p, "P differs from real retained training positives")
    plan = arrays(dataset / "donor_plan.npz")
    require(len(plan["sample_id"]) == 2 * p, "Wrong synthesis budget")
    require(np.array_equal(plan["sample_id"], np.arange(maximum + 1, maximum + 1 + 2 * p)), "Invalid synthetic IDs")
    require(np.all(plan["right_shift_buckets"] == 0), "First version must use zero shift")
    require(np.count_nonzero(plan["sample_type"] == "synthetic_malicious") == p and
            np.count_nonzero(plan["sample_type"] == "synthetic_benign") == p, "Synthetic class budget mismatch")
    for side in ("left_id", "right_id"):
        require(np.all((plan[side] > 0) & (plan[side] <= maximum)), "Invalid donor ID")
        require(np.all(catalog["split"][plan[side]] == 0), "Donor outside retained training data")
    require(np.all(catalog["label"][plan["left_id"]] == 0), "Left donor must be benign-only")
    require(np.all(catalog["ip"][plan["left_id"]] != catalog["ip"][plan["right_id"]]), "Same-source donor pair")
    require(np.array_equal(catalog["label"][plan["right_id"]], plan["sample_type"] == "synthetic_malicious"), "Donor/target class mismatch")
    activity = catalog["active"][plan["left_id"]]
    require(np.array_equal(plan["background_band"], np.where(activity == 1, 0, np.where(activity <= 4, 1, 2))), "Wrong background occupancy band")
    connection = readonly(prepared / "manifest.sqlite")
    try:
        memberships = set(connection.execute("SELECT sequence_id,attack FROM sequence_attacks"))
    finally:
        connection.close()
    for right, target, kind in zip(plan["right_id"], plan["target_attack"], plan["sample_type"]):
        require((int(right), str(target)) in memberships if kind == "synthetic_malicious" else target == "", "Target family missing from donor")

    # Audit all retained time groups and conservative cross-fold flow envelopes.
    retained = np.flatnonzero(catalog["split"] < 3)
    order = np.argsort(catalog["start"][retained], kind="stable")
    ordered = retained[order]
    equal_time = np.diff(catalog["start"][ordered]) == 0
    require(not np.any(equal_time & (np.diff(catalog["split"][ordered].astype(int)) != 0)), "Shared global time across splits")
    require(not split_policy.interval_overlap(catalog["first"][1:], catalog["end"][1:],
                                              catalog["start"][1:], catalog["split"][1:]).any(), "Retained flow envelope overlap")

    # Independently refit transforms and recover full train vocabularies from unfitted originals.
    moments = bf.Moments(manifest["feature_width"])
    vocabulary = {f: set() for f in FIELDS}
    raw_seen = np.zeros(maximum + 1, bool)
    raw_flows = raw_attacks = 0
    for number, path in enumerate(shard_files(prepared), 1):
        shard = arrays(path)
        ids = shard["sample_id"]
        require(np.all((ids > 0) & (ids <= maximum)) and not raw_seen[ids].any() and len(np.unique(ids)) == len(ids), "Duplicate/invalid raw original")
        raw_seen[ids] = True
        require(np.array_equal(shard["flow_counts"].sum(axis=1), catalog["flows"][ids]), "Raw flow count mismatch")
        require(np.array_equal(shard["attack_flow_counts"].sum(axis=1), catalog["attacks"][ids]), "Raw attack count mismatch")
        raw_flows += int(shard["flow_counts"].sum())
        raw_attacks += int(shard["attack_flow_counts"].sum())
        training = catalog["split"][ids] == 0
        active = shard["mask"].astype(bool) & training[:, None]
        numeric = bf.transform_log(shard["numeric"][active], NUMERIC_SCHEMA)
        category = categorical_features(shard, schema, tables)[active]
        moments.add(np.concatenate((numeric, category), axis=1))
        for field in FIELDS:
            coo = shard["hist_" + field]
            vocabulary[field].update(map(int, coo[training[coo[:, 0] // 10], 1]))
        if number % 100 == 0:
            LOG.info("Validation: %s raw shards reconciled", number)
    controls = read_json(prepared / "source_controls.json")
    require(raw_seen[1:].all() and raw_flows == controls["flows"] and raw_attacks == controls["attack_flows"], "Raw source reconciliation failed")
    for field in FIELDS:
        require(sorted(vocabulary[field]) == schema["vocabularies"][field], "Non-training or truncated vocabulary: " + field)
    fitted = moments.finish()
    require(fitted["n"] == scaler["n"], "Scaler population count mismatch")
    require(np.allclose(fitted["mean"], scaler["mean"], rtol=1e-10, atol=1e-10) and
            np.allclose(fitted["scale"], scaler["scale"], rtol=1e-10, atol=1e-10), "Scaler does not match real training originals")
    for code, split in ((0, "train"), (1, "validation")):
        pool_path = prepared / "embedding_pools" / (split + ".npz")
        require(digest(pool_path) == embedding_manifest[split + "_pool_sha256"], "Embedding pool checksum mismatch")
        pool = arrays(pool_path)
        require(np.all(catalog["split"][pool["sample_id"]] == code), "Wrong embedding fitting/selection pool")
        if code == 0:
            for field in LEARNED:
                require(np.array_equal(np.unique(pool[field + "_indices"]),
                                       np.arange(1, len(schema["vocabularies"][field]) + 1)),
                        "Training pool does not cover every learned category: " + field)

    seen = np.zeros(maximum + 2 * p + 1, bool)
    counts = {split: Counter() for split in SPLITS}
    reservoirs = defaultdict(list)
    rng = np.random.default_rng(seed)
    listed = {record["path"] for record in manifest["shards"]}
    actual = {str(path.relative_to(dataset)).replace("\\", "/") for split in SPLITS for path in (dataset / split).glob("part_*.npz")}
    require(listed == actual and len(listed) == len(manifest["shards"]), "Missing, extra or duplicate shard")
    for record in manifest["shards"]:
        path = dataset / record["path"]
        require(digest(path) == record["sha256"], "Shard checksum mismatch: " + record["path"])
        shard = arrays(path)
        split = path.parent.name
        n = len(shard["sample_id"])
        require(n == record["count"] and shard["X"].shape == (n, 10, manifest["feature_width"]), "Wrong shard shape")
        require(shard["X"].dtype == np.float32 and np.isfinite(shard["X"]).all(), "Invalid model inputs")
        for key in ("mask", "y_bucket", "flow_counts", "attack_flow_counts"):
            require(shard[key].shape == (n, 10), "Wrong bucket metadata shape: " + key)
        for key, values in shard.items():
            require(len(values) == n, "Metadata row count mismatch: " + key)
        mask, labels, flows, attacks = (shard[k] for k in ("mask", "y_bucket", "flow_counts", "attack_flow_counts"))
        require(np.all(np.isin(mask, (0, 1))) and np.array_equal(mask, flows > 0), "Bad activity mask")
        require(np.all((attacks >= 0) & (attacks <= flows)) and np.array_equal(labels, attacks > 0), "Bad bucket attack labels/counts")
        require(np.array_equal(shard["y_sequence"], labels.max(axis=1)), "Bad sequence labels")
        require(np.all(shard["X"][mask == 0] == 0) and np.all(shard["right_shift_buckets"] == 0), "Nonzero empty input or displacement")
        ids, left, right = (shard[k] for k in ("sample_id", "left_id", "right_id"))
        require(np.all((ids > 0) & (ids < len(seen))) and len(np.unique(ids)) == n and not seen[ids].any(), "Duplicate/invalid exported ID")
        seen[ids] = True
        real = right == -1
        require(np.all(left > 0) and np.all(left <= maximum) and np.all(right >= -1) and np.all(right <= maximum), "Bad provenance ID")
        require(np.array_equal(ids[real], left[real]) and np.all(catalog["split"][ids[real]] == SPLITS.index(split)), "Original assigned to wrong split")
        require(np.all(ids[~real] > maximum) and (split == "train" or real.all()), "Synthetic outside train or conflicting ID")
        expected_counts = catalog["flows"][left] + np.where(real, 0, catalog["flows"][np.maximum(right, 0)])
        expected_attacks = catalog["attacks"][left] + np.where(real, 0, catalog["attacks"][np.maximum(right, 0)])
        require(np.array_equal(flows.sum(axis=1), expected_counts) and np.array_equal(attacks.sum(axis=1), expected_attacks), "Donor flow conservation failed")
        require(np.array_equal(shard["left_ip"], catalog["ip"][left]), "Wrong left source audit metadata")
        require(np.array_equal(shard["right_ip"][~real], catalog["ip"][right[~real]]), "Wrong right source audit metadata")
        expected_kind = np.where(shard["y_sequence"] > 0, "malicious", "benign")
        expected_kind = np.char.add(np.where(real, "real_", "synthetic_"), expected_kind)
        require(np.array_equal(shard["sample_type"], expected_kind), "Wrong sample group")
        indices = ids[~real] - maximum - 1
        require(np.all(indices >= 0) and np.all(indices < 2 * p), "Unknown synthetic plan ID")
        for key in ("left_id", "right_id", "target_attack", "background_band", "sample_type"):
            require(np.array_equal(shard[key][~real], plan[key][indices]), "Export differs from donor plan: " + key)
        counts[split].update(map(str, shard["sample_type"]))
        if deep_samples:
            for kind in np.unique(shard["sample_type"]):
                group = reservoirs[(split, str(kind))]
                rows = np.flatnonzero(shard["sample_type"] == kind)
                priority = rng.random(len(rows))
                for j in np.argsort(priority)[:deep_samples]:
                    i = rows[j]
                    group.append((float(priority[j]), {key: value[i].copy() for key, value in shard.items()}))
                group.sort(key=lambda row: row[0])
                del group[deep_samples:]
    expected_seen = np.zeros_like(seen)
    expected_seen[retained] = True
    expected_seen[maximum + 1:] = True
    require(np.array_equal(seen, expected_seen), "Missing retained original, included excluded original, or missing synthetic")
    for code, split in enumerate(SPLITS):
        original = np.flatnonzero(catalog["split"] == code)
        positive = int(catalog["label"][original].sum())
        expected = Counter(real_benign=len(original) - positive, real_malicious=positive)
        if split == "train":
            expected.update(synthetic_benign=p, synthetic_malicious=p)
        require(counts[split] == expected and dict(counts[split]) == manifest["counts"][split], "Final group count mismatch: " + split)
        require(sum(counts[split].values()) == manifest["total_samples"][split], "Final split total mismatch")

    reconstructed = 0
    with RawStore(prepared) as store:
        for group in reservoirs.values():
            for _, row in group:
                events = store.events(row["left_id"])
                if row["right_id"] >= 0:
                    events = np.concatenate((events, store.events(row["right_id"])))
                reference = raw_bucket(events)  # independent event-level reference, not the merge implementation
                require(np.array_equal(reference["flow_counts"], row["flow_counts"]) and
                        np.array_equal(reference["attack_flow_counts"], row["attack_flow_counts"]), "Sampled bucket conservation mismatch")
                expected = model_features(reference, schema, tables, scaler)
                require(np.allclose(expected, row["X"], rtol=2e-5, atol=2e-5), "Sampled features differ from merged-flow reference")
                reconstructed += 1
    source = Path(prep["source"]["path"])
    require(source.is_file() and signature(source) == prep["source"], "Original source fingerprint changed")
    report = dict(passed=True, complete=True, dataset=str(dataset), target_ratios=[.6, .2, .2],
                  counts={split: dict(c) for split, c in counts.items()}, feature_width=manifest["feature_width"],
                  P=p, all_exported_samples_checked=int(seen.sum()), all_shards_checked=len(manifest["shards"]),
                  raw_originals_checked=int(raw_seen.sum()), train_only_scaler_and_vocabularies_refitted=True,
                  independent_flow_reconstructions=reconstructed, deep_samples_per_group=deep_samples,
                  scope="all shapes/counts/provenance/checksums/ownership; feature arithmetic sampled against original flows",
                  source_fingerprint_unchanged=True, model_accuracy_evaluated=False)
    write_json(dataset / "validation.json", report)
    LOG.info("PASS: %s exported examples; %s independent flow reconstructions", report["all_exported_samples_checked"], reconstructed)
    return report
