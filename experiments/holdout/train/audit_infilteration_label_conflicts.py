"""Read-only reference audit of exact training signatures and imported labels."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from train.common.io import fresh, read_json, sha256, unique_name, write_json


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_float_bytes(row):
    values = row.copy()
    values[values == 0] = 0  # Ignore sign bits of zero; do not round any nonzero value.
    return values.tobytes()


def main():
    prior = PROJECT / "train/runs/20261024_122203_834564_pure_infilteration_raw_flow_audit_3c5af9"
    run = PROJECT / "train/lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc"
    summary = read_json(run / "summary.json")
    with (prior / "raw_flow_profiles.csv").open(encoding="utf-8", newline="") as stream:
        pure_rows = [r for r in csv.DictReader(stream) if r["group"] in ("pure_FN", "pure_TP")]
    target_ids = np.array([int(r["sample_id"]) for r in pure_rows])
    cache = PROJECT / f"train/cache/v1_{summary['dataset_manifest_sha256'][:20]}"
    val_ids = np.load(cache / "validation/ids.npy", mmap_mode="r")
    lookup = {int(sid): i for i, sid in enumerate(val_ids)}
    val_positions = np.array([lookup[int(sid)] for sid in target_ids])
    vx, vm = [np.load(cache / f"validation/{key}.npy", mmap_mode="r") for key in ("x", "mask")]
    target = np.concatenate((vx[val_positions].reshape(len(target_ids), -1), vm[val_positions]), axis=1)
    digests = [hashlib.sha256(canonical_float_bytes(row)).digest() for row in target]
    records = {key: {"groups": Counter(), "positive_primary_attacks": Counter(), "examples": defaultdict(list)} for key in set(digests)}
    train = {key: np.load(cache / f"train/{key}.npy", mmap_mode="r") for key in ("x", "mask", "y", "types", "ids", "attack")}
    ready = read_json(cache / "train/ready.json")
    if ready["cohort_sha256"] != summary["train_cohort_sha256"] or summary["weighting"] != "equal_four_groups_mean_weight_one":
        raise ValueError("Training cohort or objective policy mismatch")
    if sha256(run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Frozen thresholds changed")
    n = len(train["y"])
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        matrix = np.concatenate((train["x"][start:end].reshape(end - start, -1), train["mask"][start:end]), axis=1)
        for offset, row in enumerate(matrix):
            key = hashlib.sha256(canonical_float_bytes(row)).digest()
            if key not in records:
                continue
            i = start + offset
            group, label = str(train["types"][i]), int(train["y"][i])
            if label != int(group.endswith("malicious")):
                raise ValueError("Training group/label mismatch")
            record = records[key]
            record["groups"][group] += 1
            if label:
                record["positive_primary_attacks"][str(train["attack"][i])] += 1
            if len(record["examples"][group]) < 3:
                record["examples"][group].append(int(train["ids"][i]))
        if start % (25 * 4096) == 0:
            print(f"Training reference signatures: {end:,}/{n:,} (no model scoring)", flush=True)
    group_weights = {key: n / (4 * value) for key, value in ready["groups"].items()}
    rows = []
    for source, key in zip(pure_rows, digests):
        record = records[key]
        counts = {group: record["groups"].get(group, 0) for group in ready["groups"]}
        pos = sum(counts[g] for g in counts if g.endswith("malicious"))
        neg = sum(counts[g] for g in counts if g.endswith("benign"))
        weighted_pos = sum(counts[g] * group_weights[g] for g in counts if g.endswith("malicious"))
        weighted_neg = sum(counts[g] * group_weights[g] for g in counts if g.endswith("benign"))
        rows.append({"sample_id": int(source["sample_id"]), "validation_group": source["group"],
                     "validation_saved_score": float(source["score"]), **counts,
                     "training_positive_matches": pos, "training_negative_matches": neg,
                     "training_signature_has_both_labels": bool(pos and neg),
                     "weighted_positive_fraction_of_exact_training_matches": weighted_pos / (weighted_pos + weighted_neg) if pos + neg else None,
                     "positive_primary_attacks": json.dumps(dict(record["positive_primary_attacks"])),
                     "training_example_ids": json.dumps(dict(record["examples"]))})

    # Verify retained sequence labels against the imported original NetFlow table.
    sidecar = read_json(PROJECT / "datasets/dataset/flow_metadata.json")
    prepared = Path(sidecar["prepared"])
    from CIC_new.preprocessing.dataset import Originals
    def flat_original(store, positions):
        return np.concatenate([store.arrays[key][positions].reshape(len(positions), -1) for key in
                               ("numeric", "hist_protocol", "hist_l7", "hist_service", "mask")], axis=1)
    def raw_digest(row, counts):
        digest = hashlib.sha256(canonical_float_bytes(row))
        digest.update(counts.tobytes())
        return digest.digest()
    pre_rows = []
    with Originals(prepared, "validation") as validation, Originals(prepared, "train") as real_train:
        vp = np.searchsorted(validation.meta["id"], target_ids)
        if not np.array_equal(validation.meta["id"][vp], target_ids):
            raise ValueError("Pre-encoder target alignment mismatch")
        matrix = flat_original(validation, vp)
        pre_digests = [raw_digest(row, count) for row, count in zip(matrix, validation.arrays["flow_counts"][vp])]
        pre_records = {key: {"labels": Counter(), "examples": {}, "attacks": Counter()} for key in set(pre_digests)}
        for start in range(0, len(real_train.meta), 2048):
            end = min(start + 2048, len(real_train.meta))
            positions = np.arange(start, end)
            matrix = flat_original(real_train, positions)
            for offset, (row, count) in enumerate(zip(matrix, real_train.arrays["flow_counts"][positions])):
                key = raw_digest(row, count)
                if key not in pre_records:
                    continue
                i = start + offset
                label = int(real_train.meta["label"][i])
                value = pre_records[key]
                value["labels"][label] += 1
                value["examples"].setdefault(label, int(real_train.meta["id"][i]))
                if label:
                    value["attacks"][str(real_train.meta["attack"][i])] += 1
            if start % (100 * 2048) == 0:
                print(f"Real-training pre-encoder signatures: {end:,}/{len(real_train.meta):,}", flush=True)
        train_order = np.argsort(train["ids"])
        sorted_train_ids = train["ids"][train_order]
        for number, (source, key) in enumerate(zip(pure_rows, pre_digests)):
            value = pre_records[key]
            example_deltas = {}
            for label, sid in value["examples"].items():
                i = int(train_order[np.searchsorted(sorted_train_ids, sid)])
                if int(train["ids"][i]) != sid:
                    raise ValueError("Training cache example alignment mismatch")
                example_deltas[f"label_{label}"] = float(np.max(np.abs(train["x"][i] - vx[val_positions[number]])))
            pre_rows.append({"sample_id": int(source["sample_id"]), "validation_group": source["group"],
                             "real_train_benign_n": value["labels"][0], "real_train_malicious_n": value["labels"][1],
                             "both_real_training_labels": bool(value["labels"][0] and value["labels"][1]),
                             "real_training_positive_types": json.dumps(dict(value["attacks"])),
                             "example_sequence_ids": json.dumps(value["examples"]),
                             "max_encoded_input_difference_for_examples": json.dumps(example_deltas)})
    source_info = read_json(prepared / "manifest.json")["source"]
    staged = sqlite3.connect((prepared / "work.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    original_db = sqlite3.connect(Path(source_info["path"]).resolve().as_uri() + "?mode=ro", uri=True)
    provenance = []
    try:
        for record in pure_rows:
            sid = int(record["sample_id"])
            retained = staged.execute("SELECT raw_rowid,label FROM events WHERE sequence_id=? ORDER BY raw_rowid", (sid,)).fetchall()
            imported = []
            for start in range(0, len(retained), 500):
                selected = retained[start:start + 500]
                imported.extend(original_db.execute("SELECT rowid,label,attack FROM network_flows WHERE rowid IN (" +
                                                     ",".join("?" for _ in selected) + ")", [r[0] for r in selected]).fetchall())
            imported.sort()
            if [(r[0], r[1]) for r in imported] != retained:
                raise ValueError("Imported and staged labels differ")
            names = Counter(str(r[2]) for r in imported)
            if set(names) != {"Infilteration"} or any(r[1] != 1 for r in imported):
                raise ValueError("Pure Infilteration cohort contains inconsistent imported labels")
            provenance.append({"sample_id": sid, "group": record["group"], "imported_flows": len(imported),
                               "imported_attack_labels": json.dumps(dict(names)), "staged_labels_match_imported": True})
    finally:
        staged.close()
        original_db.close()
    totals = []
    for group in ("pure_FN", "pure_TP"):
        keep = [r for r in rows if r["validation_group"] == group]
        totals.append({"group": group, "n": len(keep),
                       "with_any_exact_train_match": sum(r["training_positive_matches"] + r["training_negative_matches"] > 0 for r in keep),
                       "with_conflicting_train_labels": sum(r["training_signature_has_both_labels"] for r in keep),
                       "negative_training_matches_only": sum(r["training_negative_matches"] > 0 and r["training_positive_matches"] == 0 for r in keep),
                       "positive_training_matches_only": sum(r["training_positive_matches"] > 0 and r["training_negative_matches"] == 0 for r in keep)})
    output = fresh(PROJECT / "train/runs" / unique_name("infilteration_training_signature_audit"))
    write_csv(output / "exact_training_matches.csv", rows)
    write_csv(output / "pre_encoder_training_matches.csv", pre_rows)
    write_csv(output / "imported_label_provenance.csv", provenance)
    result = {"complete": True, "scope": "Exploratory validation error audit with reused training features as reference only",
              "test_read_or_scored": False, "training_reference_read": True, "training_examples_scored_as_validation": False,
              "model_or_threshold_refitted": False, "training_rows_searched": n, "training_group_weights": group_weights,
              "summary": totals, "imported_labels_consistent_for_all_pure_sequences": True,
              "pre_encoder_real_training_summary": [{"group": group,
                  "n": sum(r["validation_group"] == group for r in pre_rows),
                  "any_real_train_match": sum(r["validation_group"] == group and r["real_train_benign_n"] + r["real_train_malicious_n"] > 0 for r in pre_rows),
                  "both_real_train_labels": sum(r["validation_group"] == group and r["both_real_training_labels"] for r in pre_rows),
                  "negative_only": sum(r["validation_group"] == group and r["real_train_benign_n"] > 0 and r["real_train_malicious_n"] == 0 for r in pre_rows),
                  "positive_only": sum(r["validation_group"] == group and r["real_train_malicious_n"] > 0 and r["real_train_benign_n"] == 0 for r in pre_rows)} for group in ("pure_FN", "pure_TP")],
              "imported_sequences_verified": len(provenance), "imported_flow_rows_verified": sum(r["imported_flows"] for r in provenance),
              "raw_flow_audit_directory": str(prior), "source_run": str(run),
              "limitations": ["Training rows are explicitly reused to diagnose available supervision, not to estimate held-out performance.",
                              "The weighted positive fraction is descriptive supervision at exact input signatures, not a calibrated probability or independently validated target.",
                              "Absence of an exact bitwise training match does not imply novelty or lack of nearby training examples.",
                              "Matching imported and staged labels verifies provenance, not the correctness of original maliciousness annotations."]}
    write_json(output / "training_signature_audit.json", result)
    lines = ["# Infilteration: exact training signatures and label provenance", "",
             "This explicitly reuses training features and labels as a diagnostic reference. Training sequences are not scored as held-out examples. No test data, model retraining or threshold fitting.",
             "", "## Exact input signatures", "", "| Validation group | N | Any exact training match | Both training labels | Only negative matches | Only positive matches |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in totals:
        lines.append(f"| {r['group']} | {r['n']} | {r['with_any_exact_train_match']} | {r['with_conflicting_train_labels']} | {r['negative_training_matches_only']} | {r['positive_training_matches_only']} |")
    lines += ["", "Exact signatures contain all 650 Scheme2 input coordinates and ten mask values. All 781,838 frozen training rows are searched, including real and synthetic groups. Group weights are the recorded four-group training objective, not newly fitted weights.",
              "", "## Pre-encoder real-training signatures", "", json.dumps(result["pre_encoder_real_training_summary"], indent=2),
              "These signatures use the scaled original numeric statistics, category histograms, mask and exact flow counts before neural encoding. The search includes all 765,652 real training originals; synthetic training examples do not have an equivalent physically aggregated raw representation. pre_encoder_training_matches.csv records counts and the encoded-input difference for representative matching rows. Positive and negative zero are canonicalized in both searches; no nonzero value is rounded or compared with a tolerance.",
              "", "## Imported-label provenance", "",
              f"All {len(provenance)} pure validation Infilteration sequences ({result['imported_flow_rows_verified']} flows) retain label=1 and attack=Infilteration in the source NetFlow database. The staged labels match every source row. This rules out a staging-label mismatch for these rows; it does not establish that every original flow has distinguishable malicious behavior.",
              "", "## Limitations", ""]
    lines += ["- " + item for item in result["limitations"]]
    lines += ["", "See exact_training_matches.csv for per-case group counts, weighted label fractions and example IDs. See imported_label_provenance.csv for all retained original-label checks.", ""]
    (output / "TRAINING_SIGNATURE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), **result, "pure_fn_matches": [r for r in rows if r['validation_group'] == 'pure_FN' and r['training_positive_matches'] + r['training_negative_matches'] > 0]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
