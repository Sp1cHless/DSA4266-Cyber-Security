"""Retrain model families with matched chronological and fixed circular-shift inputs."""
import argparse
import csv
from pathlib import Path
import sys
import traceback

if __package__ in (None, ""):
    # Replace the script directory: train/xgboost must not shadow the installed library.
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from train.common.experiment import MODELS, add_common_arguments, validate_arguments, main as train_model
from train.common.io import ROOT, read_json, write_json, fresh, unique_name
from train.common.temporal import input_transform

LIMITATIONS = [
    "Temporal perturbation diagnostics on the existing real validation cohort; not final test results.",
    "All runs reuse the same original training cohort; rotation creates no independent observations.",
    "Training and validation use the same fixed rotation; retraining permits adaptation to the input order.",
    "Circular shifts retain most adjacent pairs and change the sequence boundary; they do not remove all temporal information.",
    "Checkpoints and thresholds are selected separately per run using validation only.",
    "One seed does not establish statistical significance. Historical project work already inspected the test split.",
    "For flattened XGBoost inputs a fixed shift permutes columns; differences can also reflect column sampling and split ties.",
]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_metrics(summary):
    metrics = summary["metrics"]
    one = metrics["operating_points"]["validation_fpr_1pct"]
    strict = metrics["operating_points"]["validation_fpr_0.1pct"]
    return {"ap": metrics["average_precision"], "roc_auc": metrics["roc_auc"],
            "recall_at_1pct_fpr": one["recall"], "actual_fpr_at_1pct": one["fpr"],
            "recall_at_0.1pct_fpr": strict["recall"], "actual_fpr_at_0.1pct": strict["fpr"],
            "macro_attack_recall_n_ge_20": metrics["macro_attack_recall_n_ge_20"]}


def summarize(batch, plan, output):
    """Read completed run metadata only; never load shards or refit thresholds."""
    loaded = {}
    reference = None
    for record in plan["runs"]:
        folder = batch / record["path"]
        summary = read_json(folder / "summary.json")
        config = read_json(folder / "config.json")
        if (not summary.get("complete") or summary.get("test_used") or summary.get("split") != "validation"
                or read_json(folder / "status.json")["status"] != "complete"):
            raise ValueError(f"A completed validation-only run is required: {folder}")
        if (summary["model"] != record["model"] or summary["seed"] != plan["seed"]
                or summary.get("input_transform", {"name": "identity"}) != input_transform(record["start_bucket"])):
            raise ValueError(f"Run differs from the saved circular-shift plan: {folder}")
        contract = {key: summary[key] for key in ("dataset_manifest_sha256", "train_cohort_sha256",
                    "validation_cohort_sha256", "evaluation_policy", "weighting", "seed")}
        contract["packages"] = summary["environment"]["packages"]
        if reference is not None and contract != reference:
            raise ValueError("Incompatible data, evaluation policy, seed or software versions in shift comparison")
        reference = contract
        loaded[(record["model"], record["start_bucket"])] = (summary, config, record["path"])
    rows, attacks = [], []
    for name in plan["models"]:
        shifted, config, shifted_path = loaded[(name, plan["start_bucket"])]
        control = loaded.get((name, 0))
        baseline = control[0] if control else None
        if control:
            left = {k: v for k, v in control[1]["architecture"].items() if k != "circular_shift_start"}
            right = {k: v for k, v in config["architecture"].items() if k != "circular_shift_start"}
            if left != right or control[1]["training"] != config["training"] or baseline["device"] != shifted["device"]:
                raise ValueError(f"Unmatched architecture, training settings or device for {name}")
        row = {"model": name, "seed": plan["seed"], "start_bucket": plan["start_bucket"]}
        a = extract_metrics(baseline) if baseline else {}
        b = extract_metrics(shifted)
        for metric, value in b.items():
            original = a.get(metric)
            row[f"baseline_{metric}"] = original
            row[f"shifted_{metric}"] = value
            suffix = "" if metric == "roc_auc" else "_pp"
            row[f"delta_{metric}{suffix}"] = ((value - original) * (1 if metric == "roc_auc" else 100)
                                                if original is not None and value is not None else None)
        row.update({"baseline_fit_seconds": baseline["fit_seconds"] if baseline else None,
                    "shifted_fit_seconds": shifted["fit_seconds"],
                    "baseline_run": control[2] if control else None, "shifted_run": shifted_path})
        rows.append(row)
        original_attacks = baseline["metrics"]["per_primary_attack"] if baseline else {}
        for attack, details in shifted["metrics"]["per_primary_attack"].items():
            original = original_attacks.get(attack, {}).get("recall_at_validation_fpr_1pct")
            value = details["recall_at_validation_fpr_1pct"]
            attacks.append({"model": name, "start_bucket": plan["start_bucket"], "attack": attack,
                            "n": details["n"], "hosts": details["hosts"], "baseline_recall": original,
                            "shifted_recall": value, "delta_recall_pp": (value - original) * 100 if original is not None else None})
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "per_attack_comparison.csv", attacks)
    write_json(output / "comparison.json", {"complete": True, "split": "validation", "test_used": False,
               "paired_controls": not plan["shift_only"], "rows": rows, "per_attack": attacks,
               "comparison_contract": reference, "limitations": LIMITATIONS})
    lines = ["# Circular-shift retraining comparison", "",
             f"Fixed start bucket: {plan['start_bucket']} (zero-based). Seed: {plan['seed']}.", "",
             "Real validation; each run selects its own checkpoint and operating thresholds.",
             "Delta = shifted minus chronological control. AP/recall deltas are percentage points (pp).", "",
             "| Model | Original AP | Shift AP | Delta AP (pp) | Original R@1% | Shift R@1% | Original R@0.1% | Shift R@0.1% |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = []
        for field in ("baseline_ap", "shifted_ap", "delta_ap_pp", "baseline_recall_at_1pct_fpr",
                      "shifted_recall_at_1pct_fpr", "baseline_recall_at_0.1pct_fpr", "shifted_recall_at_0.1pct_fpr"):
            value = row[field]
            values.append("N/A" if value is None else (f"{value:+.3f}" if field == "delta_ap_pp" else f"{value:.3%}"))
        lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
    lines += ["", "See comparison.csv for ROC-AUC, realized FPR, macro recall and runtime;",
              "see per_attack_comparison.csv for attack-specific changes.", ""]
    if plan["shift_only"]:
        lines += ["Chronological controls were not run; baseline values and deltas are unavailable.", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--circular-shift-start", type=int, choices=range(1, 10), default=5)
    parser.add_argument("--shift-only", action="store_true", help="Run shifted models only; no paired deltas are produced")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, help="Override the epoch budget for neural models")
    parser.add_argument("--rounds", type=int, help="Override the XGBoost round budget")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--summarize", type=Path, help="Read a completed batch and write a fresh report without training")
    args = parser.parse_args(argv)
    validate_arguments(args)
    if args.summarize:
        batch = args.summarize.resolve()
        output = fresh(args.out or batch / "reports" / unique_name("circular_shift_comparison"))
        summarize(batch, read_json(batch / "plan.json"), output)
        return output
    if len(set(args.models)) != len(args.models):
        raise ValueError("Each model may appear only once")
    for key in ("epochs", "rounds", "batch_size", "patience"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    starts = [args.circular_shift_start] if args.shift_only else [0, args.circular_shift_start]
    output = fresh(args.out or ROOT / "runs" / unique_name(f"four_models_circular_start{args.circular_shift_start}_seed{args.seed}"))
    records = [{"model": name, "start_bucket": start, "path": f"{name}/start{start}"}
               for name in args.models for start in starts]
    plan = {"models": args.models, "seed": args.seed, "start_bucket": args.circular_shift_start,
            "shift_only": args.shift_only, "runs": records, "test_used": False,
            "dataset": str(args.dataset.resolve()), "requested_device": args.device,
            "limitations": LIMITATIONS}
    write_json(output / "plan.json", plan)
    completed = []
    write_json(output / "status.json", {"status": "running", "completed_runs": completed, "test_used": False})
    try:
        for index, record in enumerate(records, 1):
            name, start = record["model"], record["start_bucket"]
            print(f"[{index}/{len(records)}] {name}: circular_shift_start={start}; batch={output}", flush=True)
            device = "cpu" if name == "xgboost" and args.device == "mps" else args.device
            command = ["--dataset", str(args.dataset), "--cache", str(args.cache), "--device", device,
                       "--threads", str(args.threads), "--seed", str(args.seed),
                       "--out", str(output / record["path"]), "--circular-shift-start", str(start)]
            keys = ("rounds", "patience") if name == "xgboost" else ("epochs", "batch_size", "patience")
            for key in keys:
                value = getattr(args, key)
                if value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
            train_model(name, command)
            completed.append(record["path"])
            write_json(output / "status.json", {"status": "running", "completed_runs": completed, "test_used": False})
        summarize(output, plan, output)
        write_json(output / "status.json", {"status": "complete", "completed_runs": completed, "test_used": False})
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "completed_runs": completed, "error": repr(error),
                   "traceback": traceback.format_exc(), "test_used": False})
        raise
    print(f"Circular-shift batch saved to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
