"""Train one bucket-supervised original LSTM and reuse the completed baseline."""
import argparse
import copy
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from train.common.experiment import main as train_model
from train.common.io import ROOT, PROJECT, read_json, write_json, fresh, unique_name, sha256, environment
from train.common.metrics import POLICY
from train.common.temporal_experiment import attack_details
from train.lstm_bucket.runner import REDUCTION
from train.run_circular_shift import write_csv

REFERENCE = ROOT / "runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/seed4266/lstm/start0"
FILES = ("config.json", "summary.json", "model.pt", "thresholds.json", "predictions.npz", "data_audit.json")
CODE = ("train/run_lstm_bucket_ablation.py", "train/lstm_bucket/model.py", "train/lstm_bucket/runner.py",
        "train/common/experiment.py", "train/common/torch_runner.py", "train/common/data.py",
        "train/lstm/model.py", "train/common/networks.py", "train/common/metrics.py")


def compare(out, reference, trained, plan):
    baseline, added = [read_json(p / "summary.json") for p in (reference, trained)]
    configs = [read_json(p / "config.json") for p in (reference, trained)]
    for key in ("dataset_manifest_sha256", "train_cohort_sha256", "validation_cohort_sha256", "evaluation_policy", "weighting", "seed", "input_transform"):
        if baseline[key] != added[key]:
            raise ValueError("Comparison contract mismatch: " + key)
    for key in ("architecture", "training", "device", "threads", "environment"):
        if configs[0][key] != configs[1][key]:
            raise ValueError("Detector configuration mismatch: " + key)
    if added["test_used"] or read_json(trained / "data_audit.json")["test_opened"]:
        raise ValueError("Unexpected test use")
    if added["fit"]["parameters"] - baseline["fit"]["parameters"] != plan["extra_parameters"]:
        raise ValueError("Unexpected model parameter change")
    with np.load(reference / "predictions.npz", allow_pickle=False) as a, np.load(trained / "predictions.npz", allow_pickle=False) as b:
        for key in ("sample_id", "y", "primary_attack", "source_host"):
            np.testing.assert_array_equal(a[key], b[key])
        predictions = [{key: p[key].copy() for key in p.files} for p in (a, b)]
    previous = {(r["attack"], r["operating_point"]): r for r in attack_details(reference, baseline["metrics"])}
    rows = []
    for row in attack_details(trained, added["metrics"]):
        old = previous[row["attack"], row["operating_point"]]
        target = (predictions[0]["y"] == 1) & (predictions[0]["primary_attack"] == row["attack"])
        a = predictions[0]["score"] >= old["threshold"]
        b = predictions[1]["score"] >= row["threshold"]
        rows.append({**row, "baseline_tp": old["tp"], "baseline_fn": old["fn"], "baseline_recall": old["recall"],
            "baseline_global_fpr": old["global_fpr"], "recovered": int((target & ~a & b).sum()),
            "lost": int((target & a & ~b).sum()),
            "delta_recall_pp": 100 * (row["recall"] - old["recall"]) if row["n"] else None})
    write_csv(out / "attack_comparison.csv", rows)
    result = {"baseline": str(reference), "bucket_supervised": str(trained), "baseline_reused": True,
        "baseline_metrics": baseline["metrics"], "bucket_supervised_metrics": added["metrics"],
        "baseline_parameters": baseline["fit"]["parameters"], "bucket_supervised_parameters": added["fit"]["parameters"],
        "rows": rows, "same_cohorts_and_inputs_verified": True, "bucket_supervision": plan["config"]["bucket_supervision"],
        "bucket_metrics_diagnostic_only": read_json(trained / "bucket_metrics.json"),
        "test_used": False, "limitations": plan["limitations"]}
    write_json(out / "comparison.json", result)
    lines = ["# LSTM bucket-supervision comparison", "",
        "One new auxiliary-supervision run; the completed original LSTM baseline was reused read-only.", "",
        f"Baseline AP: {baseline['metrics']['average_precision']:.6f}; auxiliary AP: {added['metrics']['average_precision']:.6f}.", "",
        "| Attack | Global FPR budget | N | Baseline TP/FN | Auxiliary TP/FN | Baseline recall | Auxiliary recall | Delta (pp) | Recovered | Lost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["n"]:
            lines.append(f"| {r['attack']} | {r['operating_point']} | {r['n']} | {r['baseline_tp']}/{r['baseline_fn']} | {r['tp']}/{r['fn']} | {r['baseline_recall']:.2%} | {r['recall']:.2%} | {r['delta_recall_pp']:+.2f} | {r['recovered']} | {r['lost']} |")
    lines += ["", "## Limits", ""] + ["- " + line for line in plan["limitations"]]
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    reference = args.reference_run.resolve()
    config, summary = [read_json(reference / name) for name in ("config.json", "summary.json")]
    if (config["model"] != "lstm" or not summary.get("complete") or summary.get("test_used")
            or summary.get("split") != "validation" or config["evaluation_policy"] != POLICY
            or config["input_transform"] != {"name": "identity"}
            or config["architecture"].get("circular_shift_start", 0)
            or config["weighting"] != "equal_four_groups_mean_weight_one"
            or config["selection_metric"] != "validation_average_precision_sklearn"
            or read_json(reference / "data_audit.json")["test_opened"]):
        raise ValueError("Use a completed original chronological Scheme2 LSTM reference")
    hashes = {name: sha256(reference / name) for name in FILES}
    if hashes["model.pt"] != summary["model_sha256"] or hashes["thresholds.json"] != summary["thresholds_sha256"]:
        raise ValueError("Reference weights or thresholds changed")
    dataset = Path(config["dataset"])
    if sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Reference dataset changed")
    current = environment()
    for key in ("python", "packages", "machine"):
        if current[key] != config["environment"][key]:
            raise ValueError("Use the same installed runtime as the selected reference: " + key)
    variant = {"architecture": copy.deepcopy(config["architecture"]), "training": copy.deepcopy(config["training"]),
               "bucket_supervision": {"weight": .5, "reduction": REDUCTION}}
    plan = {"reference": str(reference), "reference_sha256": hashes, "baseline_reused": True, "new_runs": 1,
        "dataset": str(dataset.resolve()), "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "config": variant, "seed": config["seed"], "device": config["device"], "threads": config["threads"],
        "cache": str(args.cache.resolve()), "environment": current,
        "code_sha256": {name: sha256(PROJECT / name) for name in CODE},
        "extra_parameters": config["architecture"]["hidden_size"] + 1,
        "training_data_reused": True, "encoder_retrained": False, "test_used": False,
        "limitations": ["Only one new run with fixed auxiliary weight 0.5; no baseline refit or hyperparameter search.",
            "Both completed fits reuse original training data and the previously inspected validation cohort; this is exploratory validation, not independent test evidence.",
            "Original 100s/10s features, frozen encoder, inputs, splits, donor pairs and synthetic augmentation remain unchanged.",
            "Existing training bucket targets add finer supervision; labels are never model inputs.",
            "A bucket is positive if it contains any attack-labeled flow, including mixed buckets; empty buckets are excluded from the auxiliary loss.",
            "Auxiliary BCE is averaged over active buckets per sequence, then combined with sequence BCE and existing four-group sample weights; no extra bucket-class weighting is applied.",
            "The original LSTM and sequence head initialize from scratch with the reference seed; the zero-initialized bucket head adds hidden_size+1 parameters and preserves the baseline initialization RNG stream.",
            "AdamW, batching, gradient clipping, maximum epochs and patience follow the original baseline. Checkpoints still use overall sequence validation AP.",
            "Sequence scores come from the original sequence head; global FPR thresholds follow the unchanged validation policy, without attack-specific thresholds.",
            "Validation bucket labels are read only after checkpoint selection for descriptive auxiliary metrics; they do not select checkpoints or sequence thresholds.",
            "No test feature or label shards are read. Original Scheme2 donor metadata may already describe all splits."]}
    if args.dry_run:
        print({key: plan[key] for key in ("reference", "new_runs", "baseline_reused", "config", "seed", "device", "extra_parameters", "encoder_retrained", "test_used")})
        return None
    out = fresh(args.out or ROOT / "runs" / unique_name("lstm_bucket_supervision_seed" + str(config["seed"])))
    write_json(out / "plan.json", plan)
    write_json(out / "config.json", variant)
    write_json(out / "status.json", {"status": "running", "test_used": False})
    try:
        trained = train_model("lstm_bucket", ["--dataset", str(dataset), "--config", str(out / "config.json"),
            "--out", str(out / "bucket_supervised"), "--cache", str(args.cache), "--device", config["device"],
            "--threads", str(config["threads"]), "--seed", str(config["seed"]), "--circular-shift-start", "0"])
        if hashes != {name: sha256(reference / name) for name in FILES}:
            raise ValueError("Completed reference artifacts changed during this experiment")
        if sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
            raise ValueError("Source dataset manifest changed during this experiment")
        compare(out, reference, trained, plan)
        write_json(out / "status.json", {"status": "complete", "test_used": False})
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error), "test_used": False})
        raise
    print(out, flush=True)
    return out


if __name__ == "__main__":
    main()
