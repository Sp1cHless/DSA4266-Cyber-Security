"""Shared train/validation entry point; test shards are never opened here."""
import argparse
import importlib
from pathlib import Path
import time
import traceback
import numpy as np

from .data import Scheme2, training_weights, check_disjoint
from .io import ROOT, DEFAULT_DATASET, read_json, write_json, fresh, unique_name, environment, sha256
from .metrics import POLICY, choose_thresholds, evaluate, save_curves

MODELS = ("xgboost", "lstm", "cnn", "hybrid")


def add_common_arguments(parser):
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="auto: CPU XGBoost; CUDA neural models if available, otherwise CPU (including Mac)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4266)


def validate_arguments(args):
    if args.threads < 1 or args.seed < 0:
        raise ValueError("threads must be positive and seed must be nonnegative")


def adapter(name):
    return importlib.import_module("train.xgboost.model" if name == "xgboost" else "train.common.torch_runner")


def resolve_device(name, requested):
    # CPU is the portable Mac path. MPS is opt-in until verified on the target Mac.
    if name == "xgboost" and requested == "mps":
        raise ValueError("XGBoost does not use MPS; select --device cpu or auto")
    if requested == "cpu" or (name == "xgboost" and requested == "auto"):
        return "cpu"
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable; use --device cpu or auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested


def score(name, model, split, device, threads):
    engine = adapter(name)
    if name == "xgboost":
        return engine.predict(model, split, device, threads)
    return engine.predict(model, split, device)


def load_model(name, config, folder, device, threads):
    engine = adapter(name)
    if name == "xgboost":
        return engine.load(folder / "model.ubj", device, threads)
    engine.setup(config["seed"], threads, device)
    return engine.load(name, config, folder / "model.pt", device)


def export_evaluation(folder, split, scores, thresholds):
    report = evaluate(split, scores, thresholds)
    write_json(folder / "metrics.json", report)
    np.savez_compressed(folder / "predictions.npz", sample_id=split["ids"], y=split["y"],
                        score=scores, primary_attack=split["attack"], source_host=split["host"])
    save_curves(folder / "curves.npz", split["y"], scores)
    return report


def main(name, argv=None):
    parser = argparse.ArgumentParser(description=f"Train the {name} baseline on frozen Scheme2 inputs")
    add_common_arguments(parser)
    parser.add_argument("--config", type=Path, default=ROOT / name / "config.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patience", type=int)
    if name == "xgboost":
        parser.add_argument("--rounds", type=int)
    else:
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    validate_arguments(args)
    config = read_json(args.config)
    for key in ("patience", "epochs", "rounds", "batch_size"):
        value = getattr(args, key, None)
        if value is not None:
            config["training"][key] = value
    if any(v <= 0 for v in config["training"].values()):
        raise ValueError("All training settings must be positive")
    device = resolve_device(name, args.device)
    data = Scheme2(args.dataset, args.cache)
    out = fresh(args.out or ROOT / name / "runs" / unique_name(f"{name}_seed{args.seed}"))
    config.update({"model": name, "seed": args.seed, "device": device, "threads": args.threads,
                   "dataset": str(data.path), "dataset_manifest_sha256": data.identity,
                   "evaluation_policy": POLICY, "weighting": "equal_four_groups_mean_weight_one",
                   "selection_metric": "validation_average_precision_sklearn",
                   "test_used": False, "environment": environment()})
    source_paths = sorted([p for p in ROOT.rglob("*.py") if not any(
        v in p.relative_to(ROOT).parts for v in ("runs", "cache", "__pycache__", "tests"))])
    config["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "running", "test_used": False})
    opened = []
    started = time.perf_counter()
    try:
        training = data.load("train"); opened.append(training)
        validation = data.load("validation"); opened.append(validation)
        check_disjoint(training, validation)
        weights = training_weights(training)
        write_json(out / "data_audit.json", {"dataset_manifest_sha256": data.identity,
                   "train": training.info, "validation": validation.info, "test_opened": False,
                   "train_group_weight_totals": {g: float(weights[training["types"] == g].sum(dtype=np.float64))
                                                 for g in training.info["groups"]}})
        engine = adapter(name)
        training_started = time.perf_counter()
        if name == "xgboost":
            model, fit_info = engine.fit(training, validation, weights, config, device, args.threads, args.seed, out)
        else:
            model, fit_info = engine.fit(name, training, validation, weights, config, device, args.threads, args.seed, out)
        fit_seconds = time.perf_counter() - training_started
        # A full validation pass includes input preparation, inference and transfer to host scores.
        prediction_started = time.perf_counter()
        scores = score(name, model, validation, device, args.threads)
        prediction_seconds = time.perf_counter() - prediction_started
        thresholds = choose_thresholds(validation["y"], scores)
        write_json(out / "thresholds.json", {"source": "real_validation_only", "decision": "score >= threshold",
                                             "values": thresholds})
        metrics = export_evaluation(out, validation, scores, thresholds)
        model_path = out / ("model.ubj" if name == "xgboost" else "model.pt")
        summary = {"complete": True, "split": "validation", "model": name, "seed": args.seed,
                   "dataset_manifest_sha256": data.identity, "train_cohort_sha256": training.info["cohort_sha256"],
                   "validation_cohort_sha256": validation.info["cohort_sha256"],
                   "evaluation_policy": POLICY, "weighting": config["weighting"], "device": device,
                   "environment": config["environment"],
                   "fit": fit_info, "fit_seconds": fit_seconds, "validation_prediction_seconds": prediction_seconds,
                   "validation_sequences_per_second": len(validation) / max(prediction_seconds, 1e-9),
                   "model_bytes": model_path.stat().st_size, "model_sha256": sha256(model_path),
                   "thresholds_sha256": sha256(out / "thresholds.json"),
                   "wall_seconds": time.perf_counter() - started, "metrics": metrics, "test_used": False,
                   "limitations": ["Validation selects checkpoints and thresholds; results are exploratory.",
                       "This is one supervised training seed, not the earlier AE plus linear readout.",
                       "Historical Scheme2 work already inspected this dataset's test split.",
                       "The original split does not guarantee unseen hosts or campaigns."]}
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {"status": "complete", "test_used": False})
        print(f"Completed {name}: validation AP={metrics['average_precision']:.6f}; {out}", flush=True)
        return out
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                                         "test_used": False})
        raise
    finally:
        for split in opened:
            split.close()
