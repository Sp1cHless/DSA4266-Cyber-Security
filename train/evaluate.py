"""Evaluate a frozen run using its saved validation thresholds, without refitting."""
import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    # Running this file directly puts train/ first on sys.path, where it shadows the
    # installed xgboost package with the local train/xgboost/ baseline. Import the
    # project from the repository root and keep train/ off the search path.
    script_dir = Path(__file__).resolve().parent
    sys.path[0] = str(script_dir.parent)
    sys.path[1:] = [entry for entry in sys.path[1:] if not entry or Path(entry).resolve() != script_dir]

from train.common.data import Scheme2, check_disjoint
from train.common.experiment import load_model, score, resolve_device, export_evaluation
from train.common.io import ROOT, read_json, write_json, sha256, fresh, unique_name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Override a relocated dataset with an identical manifest")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    run = args.run.resolve()
    config = read_json(run / "config.json")
    summary = read_json(run / "summary.json")
    if not summary.get("complete") or read_json(run / "status.json")["status"] != "complete":
        raise ValueError("Only a completed training run can be evaluated")
    data = Scheme2(args.dataset or config["dataset"], args.cache)
    if data.identity != summary["dataset_manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from training dataset")
    name = summary["model"]
    model_path = run / ("model.ubj" if name == "xgboost" else "model.pt")
    if sha256(model_path) != summary["model_sha256"] or sha256(run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Model or frozen thresholds changed")
    thresholds = read_json(run / "thresholds.json")["values"]
    device = resolve_device(name, args.device)
    model = load_model(name, config, run, device, args.threads)
    out = fresh(args.out or run / "evaluations" / unique_name(args.split))
    # Record the evaluation plan before any selected split arrays are opened.
    write_json(out / "plan.json", {"run": str(run), "split": args.split, "thresholds": thresholds,
                                   "input_transform": config.get("input_transform", {"name": "identity"}),
                                   "model_sha256": summary["model_sha256"], "refitting": False,
                                   "historical_test_previously_inspected": True})
    opened = []
    try:
        training = data.load("train"); opened.append(training)
        split = data.load(args.split); opened.append(split)
        check_disjoint(training, split)
        started = time.perf_counter()
        scores = score(name, model, split, device, args.threads)
        seconds = time.perf_counter() - started
        report = export_evaluation(out, split, scores, thresholds)
        scope = "Frozen evaluation on the existing split; not a fresh untouched final holdout."
        if config.get("input_transform", {}).get("name") == "circular_shift":
            scope = ("Temporal perturbation diagnostic under the saved circular shift on the existing split; "
                     "not chronological-input performance or a fresh untouched final holdout.")
        write_json(out / "summary.json", {"complete": True, "split": args.split, "model": name,
                   "metrics": report, "prediction_seconds": seconds, "device": device, "refitting": False,
                   "input_transform": config.get("input_transform", {"name": "identity"}),
                   "threshold_source": "saved_real_validation_thresholds", "test_used": args.split == "test",
                   "scope": scope})
        write_json(out / "status.json", {"status": "complete"})
        print(f"Frozen {args.split}: AP={report['average_precision']:.6f}; {out}", flush=True)
        return out
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        for split in opened:
            split.close()


if __name__ == "__main__":
    main()
