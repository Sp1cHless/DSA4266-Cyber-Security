"""Train one attention-pooling LSTM with the existing matched Scheme2 baseline protocol."""
import argparse
import copy
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from train.common.experiment import main as train_model
from train.common.io import ROOT, fresh, read_json, write_json, unique_name, sha256, environment
from train.common.temporal_experiment import attack_details
from train.run_circular_shift import write_csv

REFERENCE = ROOT / "runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/seed4266/lstm/start0"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    reference = args.reference_run.resolve()
    config, summary = read_json(reference / "config.json"), read_json(reference / "summary.json")
    if (summary["model"] != "lstm" or not summary.get("complete") or summary.get("test_used")
            or summary.get("split") != "validation" or config.get("input_transform", {}).get("start_bucket", 0)):
        raise ValueError("Use a completed chronological validation-only LSTM reference")
    for file, field in (("model.pt", "model_sha256"), ("thresholds.json", "thresholds_sha256")):
        if sha256(reference / file) != summary[field]:
            raise ValueError("Reference artifact changed: " + file)
    dataset = Path(config["dataset"])
    if not (dataset / "manifest.json").is_file():
        dataset = ROOT.parent / "datasets/dataset"
    if sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Reference dataset differs from the local dataset")
    current = environment()
    for key in ("python", "packages", "machine"):
        if current[key] != config["environment"][key]:
            raise ValueError("The matched reference uses a different runtime; select a same-runtime reference with --reference-run")
    variant = {"architecture": copy.deepcopy(config["architecture"]), "training": copy.deepcopy(config["training"])}
    plan = {"reference": str(reference), "dataset": str(dataset), "seed": config["seed"],
        "device": config["device"], "threads": config["threads"], "config": variant,
        "change": "Replace masked uniform mean of recurrent states with learned masked attention; preserve max and final state.",
        "initialization": "Attention logits start at zero, giving the original uniform mean.",
        "extra_parameters": config["architecture"]["hidden_size"] + 1,
        "test_used": False, "training_data_reused": True,
        "limitations": ["Exploratory validation comparison on the same previously inspected cohort.",
            "Both fits reuse the existing training data; no independent data are created.",
            "The new LSTM and classification head train from scratch, using the same seed and original budget.",
            "No bucket labels or oracle masks are supplied to the model; no bucket encoder is retrained.",
            "The model adds 65 parameters at hidden_size=64, so parameter counts are disclosed rather than claimed identical.",
            "Checkpoint selection remains overall validation AP; thresholds remain global validation FPR thresholds.",
            "One seed is a mechanism screen, not evidence of significance or a final test result."]}
    if args.dry_run:
        print(plan)
        return None
    out = fresh(args.out or ROOT / "runs" / unique_name("lstm_attention_pooling_seed" + str(config["seed"])))
    write_json(out / "plan.json", plan)
    write_json(out / "config.json", variant)
    write_json(out / "status.json", {"status": "running", "test_used": False})
    try:
        trained = train_model("lstm_context", ["--config", str(out / "config.json"), "--dataset", str(dataset),
            "--cache", str(ROOT / "cache"), "--device", config["device"], "--threads", str(config["threads"]),
            "--seed", str(config["seed"]), "--out", str(out / "attention")])
        new_summary = read_json(trained / "summary.json")
        for key in ("dataset_manifest_sha256", "train_cohort_sha256", "validation_cohort_sha256", "evaluation_policy", "weighting", "seed"):
            if new_summary[key] != summary[key]:
                raise ValueError("Comparison contract mismatch: " + key)
        with np.load(reference / "predictions.npz", allow_pickle=False) as left, np.load(trained / "predictions.npz", allow_pickle=False) as right:
            for key in ("sample_id", "y", "primary_attack", "source_host"):
                np.testing.assert_array_equal(left[key], right[key])
        previous = {(r["attack"], r["operating_point"]): r for r in attack_details(reference, summary["metrics"])}
        rows = []
        for row in attack_details(trained, new_summary["metrics"]):
            old = previous[(row["attack"], row["operating_point"])]
            rows.append({**row, "baseline_tp": old["tp"], "baseline_fn": old["fn"],
                "baseline_recall": old["recall"], "delta_recall_pp": 100 * (row["recall"] - old["recall"]) if row["n"] else None,
                "baseline_global_fpr": old["global_fpr"]})
        write_csv(out / "attack_comparison.csv", rows)
        write_json(out / "comparison.json", {"baseline": str(reference), "attention": str(trained),
            "baseline_ap": summary["metrics"]["average_precision"], "attention_ap": new_summary["metrics"]["average_precision"],
            "baseline_parameters": summary["fit"]["parameters"], "attention_parameters": new_summary["fit"]["parameters"],
            "rows": rows, "limitations": plan["limitations"], "test_used": False})
        write_json(out / "status.json", {"status": "complete", "test_used": False})
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error), "test_used": False})
        raise
    print(str(out), flush=True)
    return out


if __name__ == "__main__":
    main()
