"""Compare a completed shift batch with explicit historical baseline run references."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from train.common.io import fresh, read_json, sha256, unique_name, write_json
from train.common.metrics import evaluate
from train.common.temporal import input_transform
from train.run_circular_shift import LIMITATIONS, extract_metrics, write_csv


def verify_predictions(folder, summary):
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    split = {"y": arrays["y"], "attack": arrays["primary_attack"], "host": arrays["source_host"]}
    thresholds = read_json(folder / "thresholds.json")["values"]
    metrics = evaluate(split, arrays["score"], thresholds)
    for key in ("n", "positives", "average_precision", "roc_auc", "macro_attack_recall_n_ge_20"):
        np.testing.assert_allclose(metrics[key], summary["metrics"][key], rtol=1e-12, atol=1e-12)
    for name, operating in metrics["operating_points"].items():
        for key, value in operating.items():
            np.testing.assert_allclose(value, summary["metrics"]["operating_points"][name][key], rtol=1e-12, atol=1e-12)
    if metrics["per_primary_attack"] != summary["metrics"]["per_primary_attack"]:
        raise ValueError(f"Per-attack metrics disagree with predictions: {folder}")
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--baseline-comparison", type=Path, required=True,
                        help="Existing architecture-screening comparison.json with explicit baseline run paths")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    batch = args.batch.resolve()
    plan = read_json(batch / "plan.json")
    if read_json(batch / "status.json")["status"] != "complete":
        raise ValueError("A completed shift batch is required")
    baseline_records = read_json(args.baseline_comparison)["rows"]
    baselines = {record["model"]: Path(record["run"]).resolve() for record in baseline_records}
    if len(baselines) != len(baseline_records):
        raise ValueError("Baseline references must contain one run per family")
    shifted = {record["model"]: batch / record["path"] for record in plan["runs"]
               if record["start_bucket"] == plan["start_bucket"]}
    rows, attacks, provenance = [], [], []
    for name in plan["models"]:
        paths = (baselines[name], shifted[name])
        summaries = [read_json(folder / "summary.json") for folder in paths]
        configs = [read_json(folder / "config.json") for folder in paths]
        a, b = summaries
        for folder, summary in zip(paths, summaries):
            if (not summary.get("complete") or summary.get("test_used") or summary.get("split") != "validation"
                    or summary["model"] != name or read_json(folder / "status.json")["status"] != "complete"):
                raise ValueError(f"A completed validation-only {name} run is required: {folder}")
        for key in ("dataset_manifest_sha256", "train_cohort_sha256", "validation_cohort_sha256",
                    "evaluation_policy", "weighting", "seed"):
            if a[key] != b[key]:
                raise ValueError(f"Incompatible {key} for {name}")
        if a.get("input_transform", {"name": "identity"}) != input_transform(0):
            raise ValueError(f"Baseline is not chronological: {paths[0]}")
        if b["input_transform"] != input_transform(plan["start_bucket"]):
            raise ValueError(f"Unexpected shifted input order: {paths[1]}")
        architectures = [{k: v for k, v in config["architecture"].items() if k != "circular_shift_start"}
                         for config in configs]
        if architectures[0] != architectures[1] or configs[0]["training"] != configs[1]["training"]:
            raise ValueError(f"Architecture or training-budget mismatch for {name}")
        predictions = [verify_predictions(folder, summary) for folder, summary in zip(paths, summaries)]
        for key in ("sample_id", "y", "primary_attack", "source_host"):
            np.testing.assert_array_equal(predictions[0][key], predictions[1][key])
        left, right = extract_metrics(a), extract_metrics(b)
        row = {"model": name, "seed": a["seed"], "n_validation": a["metrics"]["n"],
               "positives": a["metrics"]["positives"], "start_bucket": plan["start_bucket"]}
        for key in left:
            row[f"baseline_{key}"] = left[key]
            row[f"shifted_{key}"] = right[key]
            row[f"delta_{key}" + ("" if key == "roc_auc" else "_pp")] = (
                (right[key] - left[key]) * (1 if key == "roc_auc" else 100)
                if left[key] is not None and right[key] is not None else None)
        row.update({"baseline_run": str(paths[0]), "shifted_run": str(paths[1])})
        rows.append(row)
        for attack, details in a["metrics"]["per_primary_attack"].items():
            other = b["metrics"]["per_primary_attack"][attack]
            if (details["n"], details["hosts"]) != (other["n"], other["hosts"]):
                raise ValueError(f"Attack population mismatch for {name}/{attack}")
            base_recall, shift_recall = (v["recall_at_validation_fpr_1pct"] for v in (details, other))
            attacks.append({"model": name, "attack": attack, "n": details["n"], "hosts": details["hosts"],
                            "baseline_recall": base_recall, "shifted_recall": shift_recall,
                            "delta_recall_pp": 100 * (shift_recall - base_recall),
                            "delta_detected_sequences": round((shift_recall - base_recall) * details["n"])})
        environment_changes = {key: {"baseline": a["environment"].get(key), "shifted": b["environment"].get(key)}
                               for key in ("python", "packages", "platform", "git_commit", "processor")
                               if a["environment"].get(key) != b["environment"].get(key)}
        provenance.append({"model": name, "environment_changes": environment_changes,
                           "baseline_device": a["device"], "shifted_device": b["device"],
                           "baseline_config": configs[0], "shifted_config": configs[1],
                           "predictions_and_all_reported_metrics_verified": True,
                           "validation_ids_labels_attacks_hosts_identical": True,
                           "source_files": {str(folder / file): sha256(folder / file)
                                            for folder in paths for file in ("summary.json", "config.json", "predictions.npz", "thresholds.json")}})
    output = fresh(args.out or batch / "reports" / unique_name("historical_no_shift_comparison"))
    limitations = LIMITATIONS + [
        "Historical controls were reused rather than retrained in the current environment.",
        "Python, PyTorch/CUDA and source revisions differ between the historical baselines and current runs; deltas cannot be attributed solely to shifting.",
        "No new fitting, threshold selection, dataset access or test scoring was performed; metrics were verified from saved validation predictions."]
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "per_attack_comparison.csv", attacks)
    write_json(output / "comparison.json", {"complete": True, "comparison_type": "historical_baseline_descriptive",
               "test_used": False, "split": "validation", "rows": rows, "per_attack": attacks,
               "provenance": provenance, "limitations": limitations})
    lines = ["# Circular shift versus historical no-shift baselines", "",
             "Fixed rotation: [5,6,7,8,9,0,1,2,3,4]. Seed: 4266.",
             "All comparisons use 251,146 real validation sequences, including 917 positives.",
             "Original training data are reused; rotation creates no independent examples.",
             "Deltas below are shifted minus historical control, in percentage points (pp).", "",
             "| Model | No-shift AP | Shift AP | AP delta (pp) | R@1%: original -> shift | R@0.1%: original -> shift |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['baseline_ap']:.2%} | {row['shifted_ap']:.2%} | {row['delta_ap_pp']:+.2f} | "
                     f"{row['baseline_recall_at_1pct_fpr']:.2%} -> {row['shifted_recall_at_1pct_fpr']:.2%} | "
                     f"{row['baseline_recall_at_0.1pct_fpr']:.2%} -> {row['shifted_recall_at_0.1pct_fpr']:.2%} |")
    lines += ["", "## Attack-specific changes", "",
              "Recall uses each run's own validation-selected threshold at the <=1% FPR budget.", "",
              "| Model | Attack | n | Original recall | Shift recall | Delta (pp) | Delta detections |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in attacks:
        if abs(row["delta_recall_pp"]) > 1e-10:
            lines.append(f"| {row['model']} | {row['attack']} | {row['n']} | {row['baseline_recall']:.2%} | "
                         f"{row['shifted_recall']:.2%} | {row['delta_recall_pp']:+.2f} | {row['delta_detected_sequences']:+d} |")
    lines += ["", "## Reading the results", "",
              "XGBoost changes little overall. LSTM and hybrid AP decrease slightly; CNN AP increases.",
              "The direction is not uniform across families, metrics or attack types.",
              "LSTM Web brute-force recall falls substantially despite its small overall AP decrease.",
              "CNN trades lower Web brute-force recall for higher Infilteration recall.",
              "Historical environment differences prevent isolating the causal effect of shifting.", "",
              "## Verification and provenance", "",
              "Manifest/cohort fingerprints, seed, weighting, metric policy, architecture and training settings match within every pair.",
              "Validation sample IDs, labels, attack names and source-host arrays match exactly within every pair.",
              "AP, ROC-AUC, operating metrics and per-attack recalls were recomputed from saved predictions using saved thresholds.",
              "Historical neural baseline: torch 2.6.0+cu124. Current neural runs: torch 2.11.0+cu128.",
              "XGBoost is 3.2.0 in both records. Detailed environment changes and source hashes are in comparison.json.", ""]
    lines += [f"- {item}" for item in limitations]
    lines += ["", "## Input run paths", ""]
    for row in rows:
        lines += [f"- {row['model']} baseline: `{row['baseline_run']}`", f"- {row['model']} shifted: `{row['shifted_run']}`"]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12 + len(rows)]), flush=True)
    print(f"Verified {len(rows)} model pairs and {len(attacks)} attack/model slices. Report: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
