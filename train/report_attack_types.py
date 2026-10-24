"""Report every primary attack type and export matching Projector subsets."""
import argparse
import csv
from pathlib import Path
import re
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.io import ROOT, fresh, read_json, sha256, unique_name, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True, help="Model subdirectory containing vectors.tsv and metadata.tsv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    summary = read_json(args.run / "summary.json")
    if not summary.get("complete") or summary.get("split") != "validation":
        raise ValueError("Expected a completed real validation run")
    if sha256(args.run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Frozen thresholds changed")
    thresholds = read_json(args.run / "thresholds.json")["values"]
    with np.load(args.run / "predictions.npz", allow_pickle=False) as saved:
        ids, labels, scores, attacks, hosts = [saved[key].copy() for key in
                                              ("sample_id", "y", "score", "primary_attack", "source_host")]
    if len(ids) != summary["metrics"]["n"] or int(labels.sum()) != summary["metrics"]["positives"]:
        raise ValueError("Prediction population mismatch")
    names = sorted(np.unique(attacks[labels == 1]).tolist())
    if set(names) != set(summary["metrics"]["per_primary_attack"]):
        raise ValueError("Attack labels differ from the original report")
    output = fresh(args.out or ROOT / "runs" / unique_name(f"{summary['model']}_all_attack_types_validation"))
    rows = []
    policies = [("1pct", "validation_fpr_1pct"), ("0_1pct", "validation_fpr_0.1pct"), ("default_0_5", "default_0.5")]
    benign = labels == 0
    operating = {}
    for alias, policy in policies:
        predicted = scores >= thresholds[policy]
        counts = {"tp": int(np.count_nonzero(predicted & (labels == 1))),
                  "fn": int(np.count_nonzero(~predicted & (labels == 1))),
                  "fp": int(np.count_nonzero(predicted & benign)),
                  "tn": int(np.count_nonzero(~predicted & benign))}
        if any(count != summary["metrics"]["operating_points"][policy][key] for key, count in counts.items()):
            raise ValueError(f"Saved aggregate metrics do not match predictions: {policy}")
        operating[policy] = {"threshold": thresholds[policy], **counts,
                             "actual_fpr": counts["fp"] / int(benign.sum())}
    for name in names:
        keep = (labels == 1) & (attacks == name)
        row = {"primary_attack": name, "n": int(keep.sum()), "source_hosts": len(np.unique(hosts[keep]))}
        for alias, policy in policies:
            tp = int(np.count_nonzero(scores[keep] >= thresholds[policy]))
            row.update({f"tp_{alias}": tp, f"fn_{alias}": row["n"] - tp, f"recall_{alias}": tp / row["n"]})
        original = summary["metrics"]["per_primary_attack"][name]
        if row["n"] != original["n"] or row["source_hosts"] != original["hosts"] or row["recall_1pct"] != original["recall_at_validation_fpr_1pct"]:
            raise ValueError(f"Per-type result differs from the original report: {name}")
        rows.append(row)
    for alias, policy in policies:
        if sum(row[f"tp_{alias}"] for row in rows) != operating[policy]["tp"]:
            raise ValueError("Per-type totals do not reconcile")
    with (output / "attack_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    export_manifest = read_json(args.projector.parent / "manifest.json")
    model_info = export_manifest["models"][summary["model"]]
    if not export_manifest.get("complete") or model_info["model_sha256"] != summary["model_sha256"]:
        raise ValueError("Projector vectors do not belong to this frozen model")
    for filename, key in (("vectors.tsv", "vectors_sha256"), ("metadata.tsv", "metadata_sha256")):
        if sha256(args.projector / filename) != model_info[key]:
            raise ValueError(f"Projector file changed: {filename}")
    vectors = np.loadtxt(args.projector / "vectors.tsv", delimiter="\t", dtype=np.float32, ndmin=2)
    with (args.projector / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header, metadata = reader.fieldnames, list(reader)
    if len(metadata) != len(vectors):
        raise ValueError("Projector row mismatch")
    source_positions = {int(value): index for index, value in enumerate(ids)}
    positions = np.array([source_positions[int(row["sample_id"])] for row in metadata])
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Duplicate visualization rows")
    for row, position in zip(metadata, positions):
        if row["primary_attack"] != attacks[position] or row["label"] != ("Attack" if labels[position] else "Benign"):
            raise ValueError("Projector metadata does not match saved validation predictions")
    if set(ids[positions[labels[positions] == 1]]) != set(ids[labels == 1]):
        raise ValueError("Visualization must contain every validation attack")
    subtypes = fresh(output / "projector_by_attack")
    subsets = []
    for number, row in enumerate(rows, 1):
        name = row["primary_attack"]
        folder = fresh(subtypes / f"{number:02d}_{re.sub(r'[^A-Za-z0-9_-]', '_', name)}")
        keep = (labels[positions] == 0) | (attacks[positions] == name)
        selected = [metadata[index] for index in np.flatnonzero(keep)]
        np.savetxt(folder / "vectors.tsv", vectors[keep], delimiter="\t", fmt="%.9g", encoding="utf-8")
        with (folder / "metadata.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)
        restored = np.loadtxt(folder / "vectors.tsv", delimiter="\t", dtype=np.float32, ndmin=2)
        with (folder / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
            restored_rows = list(csv.DictReader(stream, delimiter="\t"))
        if not np.array_equal(restored, vectors[keep]) or restored_rows != selected:
            raise ValueError("Subset TSV round-trip failed")
        if sum(item["label"] == "Attack" for item in selected) != row["n"]:
            raise ValueError("Subset attack population mismatch")
        subsets.append({"primary_attack": name, "path": str(folder.resolve()), "rows": len(selected),
                        "attacks": row["n"], "benign": len(selected) - row["n"],
                        "vectors_sha256": sha256(folder / "vectors.tsv"),
                        "metadata_sha256": sha256(folder / "metadata.tsv")})
    manifest = {"complete": True, "model": summary["model"], "run": str(args.run.resolve()),
                "split": "validation", "population_n": len(labels), "positives": int(labels.sum()),
                "attack_type_count": len(rows), "operating_points": operating, "per_primary_attack": rows,
                "predictions_sha256": sha256(args.run / "predictions.npz"),
                "thresholds_sha256": summary["thresholds_sha256"], "test_opened": False,
                "training_data_reused": False, "validation_used_for_checkpoint_and_threshold_selection": True,
                "projector_subsets": subsets, "all_subset_tsv_round_trips_verified": True}
    write_json(output / "manifest.json", manifest)
    lines = [f"# {summary['model'].upper()}: all primary attack types", "",
             f"Full real validation population: {len(labels):,} sequences, including {int(labels.sum()):,} attacks and {int(benign.sum()):,} benign sequences. Counts and recalls below come from original saved scores, not a sampled plot or reloaded inference.",
             "This is binary attack detection broken down by the sequence primary_attack label. The model does not predict attack categories. No per-type threshold is fitted.", "",
             "| Primary attack | N | Source hosts | TP at 1% | FN at 1% | Recall at 1% | TP at 0.1% | FN at 0.1% | Recall at 0.1% |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['primary_attack']} | {row['n']} | {row['source_hosts']} | {row['tp_1pct']} | {row['fn_1pct']} | {row['recall_1pct']:.2%} | {row['tp_0_1pct']} | {row['fn_0_1pct']} | {row['recall_0_1pct']:.2%} |")
    lines += ["", "## Global thresholds and false alarms", ""]
    for alias, policy in policies:
        point = operating[policy]
        lines.append(f"- {policy}: threshold {point['threshold']:.10f}; TP {point['tp']}, FN {point['fn']}, FP {point['fp']}, TN {point['tn']}; actual benign FPR {point['actual_fpr']:.6%}.")
    lines += ["", "The same benign false-alarm population is shared by all attack slices. False alarms cannot be uniquely assigned to an attack type by this binary model.",
              "All 13 primary attack labels are retained, including types with fewer than 20 sequences. Small counts and host/campaign dependence limit conclusions; primary labels do not represent all attack memberships in mixed sequences.",
              "Validation selected checkpoints and thresholds, so these are exploratory validation results. No training examples or test examples enter this report. Historical project work already inspected the Scheme2 test split.",
              "", "## Projector subsets", "",
              "Each directory contains vectors.tsv and metadata.tsv, with every validation attack of that type and the same 4,000 benign visualization examples. These subsets are attack-enriched, and their class ratios are not natural prevalence. No vectors, models or thresholds were refitted.",
              "Load the two files at https://projector.tensorflow.org/. Use Color by = label or outcome_1pct (TP/FN for attacks, FP/TN for benign examples).",
              "Each separately loaded subset causes the website to recompute normalization and projection. For direct geometric comparisons, retain the original all-type view and inspect primary_attack there; do not compare coordinates from independently projected subsets.", ""]
    for subset in subsets:
        folder = Path(subset["path"]).relative_to(output.resolve()).as_posix()
        lines.append(f"- {subset['primary_attack']}: {subset['attacks']} attacks + {subset['benign']} benign; [vectors]({folder}/vectors.tsv), [metadata]({folder}/metadata.tsv).")
    lines += ["", "attack_metrics.csv also includes TP, FN and recall at the default 0.5 threshold. manifest.json contains all counts, hashes and export verification.", ""]
    report = "\n".join(lines)
    (output / "ATTACK_RESULTS.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Saved to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
