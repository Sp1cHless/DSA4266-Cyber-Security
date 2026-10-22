"""Matched Scheme1 bucket-L2 ablation: train, frozen evaluation, comparisons.

Reuse final Scheme1 samples and apply L2 after scaling/composition at read time.
No original data, caches, embeddings or existing model weights are rewritten.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.evaluate_autoencoder import read, write, digest, evaluate
from CIC_new.modelling.train_autoencoder import train_temporal
from CIC_new.preprocessing.dataset import ModelInputs
from CIC_new.run_scheme2_experiment import (
    CONTROLS, latest_run, latest_evaluation, check_dataset_pair, compare_reports,
)
from CIC_new.train_autoencoder import new_run_directory


def verify_reference(run, report_path, representation):
    run, report_path = Path(run).resolve(), Path(report_path).resolve()
    config, summary, report = read(run/"config.json"), read(run/"summary.json"), read(report_path)
    if not summary.get("complete") or not report.get("complete") or report.get("test_used"):
        raise ValueError("Completed validation-only reference results required")
    if config.get("input_transform", "none") != "none":
        raise ValueError("The reference must be an unmodified baseline")
    if Path(report["training_run"]).resolve() != run:
        raise ValueError("Reference evaluation belongs to another run")
    for name, expected in report["source_sha256"].items():
        if digest(run/name) != expected:
            raise ValueError(f"Reference artifact changed: {name}")
    if digest(run/"sample_draws.jsonl") != report["sampling"]["sample_audit_sha256"]:
        raise ValueError("Reference sampling audit changed")
    data = Path(config["dataset"]).resolve()
    manifest = read(data/"manifest.json")
    if not manifest.get("complete") or manifest.get("representation") != representation:
        raise ValueError("Wrong reference representation")
    if digest(run/"dataset_manifest.json") != digest(data/"manifest.json"):
        raise ValueError("Reference dataset manifest changed")
    for name in ("schema.json", "scaler.json", "embeddings.npz", "frozen_bucket_encoder.pt"):
        if (run/name).exists() and digest(run/name) != digest(data/name):
            raise ValueError(f"Reference input artifact changed: {name}")
    return config


def check_l2_inputs(dataset):
    """Sample every group, compare with an independent norm, check sampler IDs."""
    result = {}
    for split in ("train", "validation"):
        with ModelInputs(dataset, split) as raw, ModelInputs(dataset, split, "bucket_l2") as transformed:
            rng = np.random.default_rng(4266)
            ids = np.unique(np.concatenate([
                rng.choice(pool, min(512, len(pool)), replace=False)
                for kind in np.unique(raw.arrays["sample_type"])
                for pool in [np.flatnonzero(raw.arrays["sample_type"] == kind)]
            ]))
            a, b = raw.take(ids), transformed.take(ids)
            expected = a["X"].astype(np.float64)
            expected /= np.maximum(np.sqrt(np.sum(expected**2, axis=-1, keepdims=True)), 1e-8)
            expected *= a["mask"][..., None]
            if not np.allclose(b["X"], expected, rtol=2e-6, atol=1e-7):
                raise ValueError("L2 transform differs from independent calculation")
            for key in a:
                if key != "X" and not np.array_equal(a[key], b[key]):
                    raise ValueError(f"L2 modified sample metadata: {key}")
            active = b["mask"] > 0
            norms = np.linalg.norm(b["X"], axis=-1)
            if not np.isfinite(b["X"]).all() or np.any(b["X"][~active] != 0):
                raise ValueError("Invalid transformed padding or nonfinite input")
            if not np.allclose(norms[active], 1., rtol=2e-6, atol=1e-7):
                raise ValueError("Expected nonzero unit vectors for all sampled active buckets")
            if split == "train":
                for left, right in zip(raw.batches(128, 3, 4266), transformed.batches(128, 3, 4266)):
                    for key in ("sample_id", "sample_type"):
                        if not np.array_equal(left[key], right[key]):
                            raise ValueError("L2 changed the sampler stream")
            result[split] = dict(sampled_sequences=len(ids), feature_width=b["X"].shape[-1],
                minimum_active_norm=float(norms[active].min()), maximum_active_norm=float(norms[active].max()),
                maximum_arithmetic_difference=float(np.max(np.abs(b["X"]-expected))))
    return dict(input_transform="bucket_l2", all_final_features_including_flow_count=True,
                source_cache_reused=True, stateless=True, test_opened=False, splits=result)


def latest_l2_run():
    candidates = [p.parent for p in (ROOT/"CIC_new/outputs/ae_runs").glob("*/summary.json")
                  if read(p).get("complete") and read(p.parent/"config.json").get("input_transform") == "bucket_l2"]
    if not candidates:
        raise FileNotFoundError("No completed Scheme1 L2 run; provide --run")
    return max(candidates, key=lambda p: p.name)


def write_three_way(out, first, second):
    models = {**first["models"], **second["models"]}
    write(out/"three_way_comparison.json", dict(complete=True, test_used=False,
        controls=first["controls"], models=models, limitations=first["limitations"]))
    lines = ["# Scheme1 / Scheme1 L2 / Scheme2", "",
        "Frozen-AE real-train-only linear probe; paired validation, one seed, no test.",
        "Scheme1 L2 changes only the final per-bucket normalization of Scheme1 inputs.", "",
        "| Representation | Best epoch | ROC-AUC | AP | Recall @ FPR <=1% | Recall @ FPR <=0.1% |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ("scheme1", "scheme1_l2", "scheme2"):
        m = models[name]
        points = m["exploratory_validation_roc_points"]
        lines.append(f"| {name} | {m['best_epoch']} | {m['roc_auc']:.5f} | {m['average_precision']:.2%} | "
                     f"{points['0.01']['recall']:.2%} | {points['0.001']['recall']:.2%} |")
    lines += ["", "Fixed-FPR points are selected on validation curves, not deployment thresholds.",
              "Raw reconstruction MSE has different targets/scales and must not rank the representations.",
              "See comparison.json and vs_scheme2/comparison.json for train-calibrated thresholds and mixed-flow recall."]
    (out/"THREE_WAY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "check", "train", "evaluate"), default="all")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--baseline-evaluation", type=Path, help="Existing Scheme1 report.json")
    parser.add_argument("--scheme2-run", type=Path)
    parser.add_argument("--scheme2-evaluation", type=Path, help="Existing Scheme2 report.json")
    parser.add_argument("--run", type=Path, help="Completed L2 run, for evaluation only")
    parser.add_argument("--out", type=Path, help="Fresh training directory; default unique scheme1_l2 folder")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.evaluation_batch_size <= 0:
        parser.error("Positive evaluation batch size required")
    if args.run is not None and args.stage != "evaluate":
        parser.error("--run is only for --stage evaluate")
    if args.out is not None and args.stage not in ("all", "train"):
        parser.error("--out is only for all/train")
    baseline = (args.baseline_run or latest_run("scheme1")).resolve()
    scheme2 = (args.scheme2_run or latest_run("scheme2")).resolve()
    baseline_report = args.baseline_evaluation or latest_evaluation(baseline)
    scheme2_report = args.scheme2_evaluation or latest_evaluation(scheme2)
    config = verify_reference(baseline, baseline_report, "scheme1")
    other = verify_reference(scheme2, scheme2_report, "scheme2")
    if any(config[k] != other[k] for k in CONTROLS):
        raise ValueError("Scheme1/Scheme2 reference training controls differ")
    dataset = Path(config["dataset"]).resolve()
    paired = check_dataset_pair(dataset, Path(other["dataset"]))
    checked = check_l2_inputs(dataset)
    controls = {k: config[k] for k in CONTROLS}
    print(f"Scheme1 reference: {baseline}\nScheme2 reference: {scheme2}\nDataset (reused): {dataset}", flush=True)
    print(f"Only new transform: per-bucket L2 over all final features, including flow count\nMatched settings: {controls}", flush=True)
    print(f"[L2 input check] {checked['splits']}", flush=True)
    if args.stage == "check":
        print("CHECK PASSED: no training or probe fitting; test not opened", flush=True)
        return
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.stage in ("all", "train"):
        training = SimpleNamespace(**controls, dataset=dataset, out=args.out, device=device,
            cpu_threads=config["cpu_threads"], log_every=config["log_every"],
            representation="scheme1", input_transform="bucket_l2")
        if training.out is None:
            training.out = new_run_directory(training)
        run = training.out.resolve()
        train_temporal(training)
        write(run/"l2_preflight.json", dict(paired=paired, normalization=checked,
            baseline_run=str(baseline), scheme2_run=str(scheme2)))
        if args.stage == "train":
            print(f'TRAIN COMPLETE: {run}; next: python CIC_new/run_scheme1_l2_experiment.py --stage evaluate --run "{run}"', flush=True)
            return
    else:
        run = (args.run or latest_l2_run()).resolve()
        candidate = read(run/"config.json")
        if candidate.get("input_transform") != "bucket_l2" or Path(candidate["dataset"]).resolve() != dataset:
            raise ValueError("Expected an L2 run on the original Scheme1 reference dataset")
        if any(candidate[k] != config[k] for k in CONTROLS):
            raise ValueError("L2 run training controls differ from reference")
    parent = run/"evaluations"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=stamp+"_scheme1_l2_comparison_", dir=parent))
    write(out/"evaluation_status.json", dict(status="running", complete=False))
    try:
        evaluate(run, out, device, args.evaluation_batch_size)
        write(out/"input_checks.json", dict(paired=paired, normalization=checked))
        first = compare_reports(baseline, baseline_report, run, out/"report.json", out,
                                model_names=("scheme1", "scheme1_l2"))
        second_out = out/"vs_scheme2"
        second_out.mkdir()
        second = compare_reports(scheme2, scheme2_report, run, out/"report.json", second_out,
                                 model_names=("scheme2", "scheme1_l2"))
        write_three_way(out, first, second)
    except BaseException as error:
        write(out/"evaluation_status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
              complete=False, error=str(error)))
        raise
    write(out/"evaluation_status.json", dict(status="complete", complete=True, comparison_complete=True,
          finished_at=datetime.now().astimezone().isoformat()))
    print(f"COMPLETE: {out/'THREE_WAY.md'}", flush=True)


if __name__ == "__main__":
    main()
