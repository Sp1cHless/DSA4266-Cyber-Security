"""Measure past-context availability without retraining or scoring new windows.

Original validation anchors and frozen LSTM outcomes define descriptive slices.
Only retained validation flows are read. Training and test flows are not used.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from train.common.io import fresh, read_json, sha256, unique_name, write_json

CONFIGS = ((100, 10), (100, 20), (200, 10), (200, 20),
           (400, 10), (400, 20), (400, 40))
TYPES = ("Infilteration", "Brute_Force_-Web")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eligible(connection, groups, host, start, end):
    # Require complete coverage by validation-assigned global time blocks.
    for block in range(start // 900000 * 900000, end, 900000):
        if groups.get(block) != "validation":
            return None
    neighbors = connection.execute(
        "SELECT sequence_id,flows,attack_flows,split,window_start_ms "
        "FROM original_sequences WHERE source_ip=? AND window_start_ms>=? "
        "AND window_start_ms<? ORDER BY window_start_ms", (host, start, end)).fetchall()
    if any(row[3] != "validation" for row in neighbors):
        return None
    return neighbors


def main():
    run = PROJECT / "train/lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc"
    summary = read_json(run / "summary.json")
    if sha256(run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Frozen thresholds changed")
    threshold = read_json(run / "thresholds.json")["values"]["validation_fpr_1pct"]
    with np.load(run / "predictions.npz", allow_pickle=False) as predictions:
        ids, labels, types, scores = [predictions[k].copy() for k in
                                      ("sample_id", "y", "primary_attack", "score")]
    projector = PROJECT / "train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/lstm/metadata.tsv"
    with projector.open(encoding="utf-8", newline="") as stream:
        visual_ids = {int(row["sample_id"]) for row in csv.DictReader(stream, delimiter="\t")}
    selected = np.flatnonzero(np.isin(types, TYPES) | ((labels == 0) & np.isin(ids, list(visual_ids))))
    if Counter(types[selected]) != Counter({"Infilteration": 366, "Brute_Force_-Web": 181, "Benign": 4000}):
        raise ValueError("Unexpected anchor cohort")
    prepared = Path(read_json(PROJECT / "datasets/dataset/flow_metadata.json")["prepared"])
    manifest = sqlite3.connect((prepared / "manifest.sqlite").as_uri() + "?mode=ro", uri=True)
    events = sqlite3.connect((prepared / "work.sqlite").as_uri() + "?mode=ro", uri=True)
    output = fresh(PROJECT / "train/runs" / unique_name("window_context_availability"))
    write_json(output / "status.json", {"complete": False})
    groups = dict(manifest.execute("SELECT block_start_ms,assigned_split FROM split_groups"))
    event_cache = {}
    target_counts = {}
    rows, availability = [], []
    try:
        for count, index in enumerate(selected):
            sid, attack = int(ids[index]), str(types[index])
            host, anchor_start, base_flows, base_attack, base_active, split = manifest.execute(
                "SELECT source_ip,window_start_ms,flows,attack_flows,active_buckets,split "
                "FROM original_sequences WHERE sequence_id=?", (sid,)).fetchone()
            if split != "validation":
                raise ValueError("Non-validation anchor")
            end = anchor_start + 100000
            intervals = {window: eligible(manifest, groups, host, end - window * 1000, end)
                         for window in (100, 200, 400)}
            if intervals[100] is None:
                raise ValueError("Original retained anchor is not eligible")
            common = intervals[400] is not None
            availability.append({"sample_id": sid, "attack_type": attack,
                                 "eligible_200s": intervals[200] is not None,
                                 "eligible_400s": common})
            for window, bucket in CONFIGS:
                neighbors = intervals[window]
                if neighbors is None:
                    continue
                starts, malicious = [], []
                target_flows = 0
                for neighbor, n, positive, _, _ in neighbors:
                    if neighbor not in event_cache:
                        event_cache[neighbor] = np.array(events.execute(
                            "SELECT flow_start_milliseconds,label FROM events WHERE sequence_id=?",
                            (neighbor,)).fetchall(), dtype=np.int64).reshape(-1, 2)
                        if len(event_cache[neighbor]) != n or event_cache[neighbor][:, 1].sum() != positive:
                            raise ValueError("Raw-flow reconciliation failed")
                    values = event_cache[neighbor]
                    starts.append(values[:, 0])
                    malicious.append(values[:, 1])
                    key = (neighbor, attack)
                    if key not in target_counts:
                        target_counts[key] = manifest.execute(
                            "SELECT COALESCE(SUM(flows),0) FROM sequence_attacks WHERE sequence_id=? AND attack=?",
                            key).fetchone()[0]
                    target_flows += target_counts[key]
                timestamps, y = np.concatenate(starts), np.concatenate(malicious)
                start = end - window * 1000
                assert np.all((timestamps >= start) & (timestamps < end))
                bins = (timestamps - start) // (bucket * 1000)
                active = len(np.unique(bins))
                total, positive = len(y), int(y.sum())
                if window == 100:
                    assert total == base_flows and positive == base_attack
                    if bucket == 10:
                        assert active == base_active
                    else:
                        assert active <= base_active
                row = {"sample_id": sid, "attack_type": attack,
                       "original_outcome": "benign" if labels[index] == 0 else
                       ("TP" if scores[index] >= threshold else "FN"),
                       "original_sparse_le2_active": base_active <= 2,
                       "common_400s_cohort": common, "window_seconds": window,
                       "bucket_seconds": bucket, "bucket_count": window // bucket,
                       "original_flows": base_flows, "original_attack_flows": base_attack,
                       "original_active_buckets": base_active, "flows": total,
                       "attack_flows": positive, "target_attack_flows": int(target_flows),
                       "active_buckets": active, "active_fraction": active / (window // bucket),
                       "flows_per_active_bucket": total / active,
                       "attack_fraction": positive / total,
                       "added_flows": total - base_flows,
                       "added_attack_flows": positive - base_attack,
                       "added_benign_flows": total - positive - (base_flows - base_attack),
                       "has_additional_flows": total > base_flows,
                       "benign_anchor_gains_attack": base_attack == 0 and positive > 0}
                rows.append(row)
            if (count + 1) % 500 == 0:
                print(f"Audited {count + 1}/{len(selected)} validation anchors", flush=True)
    finally:
        manifest.close()
        events.close()
    summaries = []
    for attack in ("Infilteration", "Brute_Force_-Web", "Benign"):
        for outcome in ("all", "FN", "TP"):
            for sparse in (False, True):
                for window, bucket in CONFIGS:
                    chosen = [r for r in rows if r["attack_type"] == attack
                              and (outcome == "all" or r["original_outcome"] == outcome)
                              and (not sparse or r["original_sparse_le2_active"])
                              and r["common_400s_cohort"]
                              and r["window_seconds"] == window and r["bucket_seconds"] == bucket]
                    if not chosen:
                        continue
                    aggregate = {"attack_type": attack, "original_outcome": outcome,
                                 "sparse_only": sparse, "window_seconds": window,
                                 "bucket_seconds": bucket, "bucket_count": window // bucket,
                                 "anchors": len(chosen)}
                    for key in ("flows", "attack_flows", "target_attack_flows", "active_buckets",
                                "active_fraction", "flows_per_active_bucket", "attack_fraction", "added_flows"):
                        aggregate["median_" + key] = float(np.median([r[key] for r in chosen]))
                    for key in ("has_additional_flows", "benign_anchor_gains_attack"):
                        aggregate[key + "_count"] = sum(r[key] for r in chosen)
                    aggregate["gained_attack_flow_count"] = sum(r["added_attack_flows"] > 0 for r in chosen)
                    aggregate["only_added_benign_count"] = sum(r["added_flows"] > 0 and r["added_attack_flows"] == 0 for r in chosen)
                    summaries.append(aggregate)
    write_csv(output / "anchor_window_profiles.csv", rows)
    write_csv(output / "paired_summary.csv", summaries)
    write_csv(output / "eligibility.csv", availability)
    write_json(output / "summary.json", {"anchors": len(selected), "configs": CONFIGS,
               "cohort_counts": dict(Counter(types[selected])), "paired_summaries": summaries,
               "training_data_used": False, "test_flows_used": False,
               "new_model_scores_computed": False, "threshold": threshold,
               "prepared": str(prepared), "source_run": str(run)})
    lines = ["# Past-context and bucket-resolution diagnostic", "",
             "This is a raw-flow availability diagnostic, not a new detection evaluation. No model was retrained and no new-window recall was measured.", "",
             "Anchors: all 366 Infilteration and 181 Brute_Force_-Web original validation sequences, plus the existing seeded 4,000-benign Projector sample. This is an attack-enriched cohort, not the full validation distribution.", "",
             "All windows end at the original 100-second window endpoint. Expansion adds past context only. Full expanded intervals must remain in validation-assigned global blocks, and every observed neighboring source-host sequence must be retained validation. Training/test flows and excluded neighbors are not read. Summaries compare the same 400-second-eligible anchors across configurations. Eligibility depends on the existing split and can bias the retained subset.", "",
             "Original TP/FN slices use the saved LSTM score and frozen 1%-FPR threshold. Those outcomes are anchor descriptors, not predictions for expanded inputs. Adjacent expanded anchors can overlap and are not independent observations. Benign anchors that gain attack context require a new label definition before any detection evaluation.", "",
             "| Anchor type | Original outcome | Sparse only | Window / bucket / count | Anchors | Median flows | Median active buckets | Median attack fraction | Added flows: anchors | Added attack: anchors | Only benign added: anchors |", 
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["sparse_only"] and r["original_outcome"] in ("all", "FN"):
            lines.append(f"| {r['attack_type']} | {r['original_outcome']} | True | {r['window_seconds']} / {r['bucket_seconds']} / {r['bucket_count']} | {r['anchors']} | {r['median_flows']:.1f} | {r['median_active_buckets']:.1f} | {r['median_attack_fraction']:.3f} | {r['has_additional_flows_count']} | {r['gained_attack_flow_count']} | {r['only_added_benign_count']} |")
    lines.extend(["", "## Prospective model experiment", "",
                  "First compare 100s/10s/10, 200s/10s/20 and 200s/20s/10 using an unchanged anchor endpoint and an explicit anchor-vs-window target definition. The 100s/20s/5 control isolates bucket coarsening. Consider 400s only after checking benefit and dilution at 200s.", "",
                  "A recall comparison requires newly prepared features and separately trained models. Refit changed aggregation scalers and encoders on training data only; a frozen-encoder variant can be reported separately as a transfer diagnostic. Keep the global split assignment fixed, recheck expanded support and flow-end exclusions, and compare on common eligible anchors. Select each model threshold under the same benign-FPR rule, and report actual false alarms, type recall, attack fraction and host-level slices. Preserve all original runs. Validation-informed development is not an independent test result.", "",
                  "Increasing bucket count at fixed 100-second duration makes buckets shorter and cannot create additional flows. Coarsening buckets at fixed duration cannot increase the number of nonempty buckets, although it can increase flows per occupied bucket and occupancy fraction. Neither guarantees better recall."])
    (output / "CONTEXT_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "status.json", {"complete": True})
    print(json.dumps({"output": str(output), "anchors": len(selected),
                      "common_400s": sum(a["eligible_400s"] for a in availability)}, indent=2))


if __name__ == "__main__":
    main()
