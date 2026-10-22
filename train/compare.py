"""Compare completed validation runs with identical data and metric contracts."""
import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.experiment import MODELS
from train.common.io import read_json, write_json, fresh, unique_name


def compare(paths, output=None, allow_partial=False):
    paths = [Path(p).resolve() for p in paths]
    summaries = [read_json(p / "summary.json") for p in paths]
    if not summaries or any(not s.get("complete") or s.get("split") != "validation" or s.get("test_used") for s in summaries):
        raise ValueError("Only completed validation screening runs can be compared")
    names = [s["model"] for s in summaries]
    if len(names) != len(set(names)):
        raise ValueError("Provide one run per model; compare seeds in separate screens")
    if not allow_partial and set(names) != set(MODELS):
        raise ValueError("All four models are required; use --allow-partial for a progress table")
    for key in ("dataset_manifest_sha256", "train_cohort_sha256", "validation_cohort_sha256",
                "evaluation_policy", "weighting", "seed"):
        if len({s[key] for s in summaries}) != 1:
            raise ValueError(f"Incompatible comparison: {key}")
    rows = []
    for path, summary in zip(paths, summaries):
        metrics = summary["metrics"]
        one = metrics["operating_points"]["validation_fpr_1pct"]
        strict = metrics["operating_points"]["validation_fpr_0.1pct"]
        rows.append({"model": summary["model"], "average_precision": metrics["average_precision"],
                     "roc_auc": metrics["roc_auc"], "recall_at_1pct_fpr": one["recall"],
                     "actual_fpr_at_1pct": one["fpr"], "precision_at_1pct": one["precision"],
                     "f1_at_1pct": one["f1"], "recall_at_0.1pct_fpr": strict["recall"],
                     "actual_fpr_at_0.1pct": strict["fpr"],
                     "macro_attack_recall_n_ge_20": metrics["macro_attack_recall_n_ge_20"],
                     "fit_seconds": summary["fit_seconds"],
                     "validation_sequences_per_second": summary["validation_sequences_per_second"],
                     "model_mib": summary["model_bytes"] / 1024 ** 2, "device": summary["device"],
                     "platform": summary.get("environment", {}).get("platform", "not recorded"), "run": str(path)})
    rows.sort(key=lambda r: (-r["average_precision"], -r["recall_at_0.1pct_fpr"], -r["roc_auc"]))
    output = fresh(output or paths[0].parent / unique_name("comparison"))
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    complete = set(names) == set(MODELS)
    write_json(output / "comparison.json", {"complete_screen": complete, "split": "validation",
               "selection_rule": "AP descending; exact ties: recall at 0.1% FPR, then ROC-AUC",
               "recommended_model": rows[0]["model"] if complete else None, "rows": rows,
               "limitations": ["One seed and validation-selected checkpoints; not final test performance.",
                   "Runs may come from different members and machines; speed is descriptive and is not used for ranking.",
                   "Ranking is an empirical screening recommendation, not a significance or robustness claim."]})
    lines = ["# Scheme2 model screening", "", "Real validation population; primary metric: average precision (AP).", "",
             "| Model | AP | ROC-AUC | Recall at <=1% FPR | Recall at <=0.1% FPR | Fit seconds | Device |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['average_precision']:.4%} | {row['roc_auc']:.5f} | "
                     f"{row['recall_at_1pct_fpr']:.2%} | {row['recall_at_0.1pct_fpr']:.2%} | "
                     f"{row['fit_seconds']:.1f} | {row['device']} |")
    lines += ["", f"AP-first screening candidate: **{rows[0]['model']}**." if complete else "Partial screen; no final candidate.",
              "", "Inspect per-attack recall and operating-point false alarms before adopting the candidate.",
              "Results use one seed and reuse validation for selection. No test scores enter this ranking.",
              "Runtime includes validation-based early stopping. Different machines/devices prevent a controlled speed comparison.",
              "Historical project experiments already inspected this test split; reserve a fresh holdout for a new final claim."]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Comparison saved to {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", type=Path, nargs="+")
    source.add_argument("--root", type=Path, help="A collection directory with xgboost/lstm/cnn/hybrid run subdirectories")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    paths = args.runs or [args.root / model for model in MODELS if (args.root / model / "summary.json").exists()]
    compare(paths, args.out, args.allow_partial)


if __name__ == "__main__":
    main()
