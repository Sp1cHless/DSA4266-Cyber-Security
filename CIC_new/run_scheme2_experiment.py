"""Scheme2 AE training and frozen validation; optional historical comparison.

Full runs are launched only when this entry point is invoked by the user. The
check stage reads train/validation metadata and checksums without fitting models.
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

from CIC_new.evaluate_autoencoder import evaluate, read, write, digest, binary_metrics
from CIC_new.modelling.train_autoencoder import train_temporal
from CIC_new.train_autoencoder import new_run_directory

CONTROLS = ("seed", "batch_size", "epochs", "steps_per_epoch", "width", "latent", "lr",
            "real_only", "val_limit", "patience", "min_delta")
META = ("sample_id", "sample_type", "y_sequence", "left_id", "right_id", "right_shift_buckets",
        "left_ip", "right_ip", "primary_attack", "mask", "y_bucket", "flow_counts")


def latest_run(representation):
    candidates = []
    for path in (ROOT/"CIC_new/outputs/ae_runs").glob("*/summary.json"):
        config = read(path.parent/"config.json")
        status = path.parent/"run_status.json"
        if read(path).get("complete") and (not status.exists() or read(status)["status"] == "complete"):
            if read(Path(config["dataset"])/"manifest.json").get("representation") == representation and config.get("input_transform","none")=="none":
                candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(f"No completed {representation} AE run; specify the corresponding --baseline-run or --run")
    return max(candidates, key=lambda p: p.name)


def latest_dataset():
    candidates = [p.parent/"dataset" for p in (ROOT/"CIC_new/outputs/scheme2_runs").glob("*/workflow.json")
                  if read(p).get("complete") and read(p)["status"] == "complete"]
    if not candidates:
        raise FileNotFoundError("No completed scheme2 dataset; specify --dataset")
    return max(candidates, key=lambda p: p.parent.name)


def latest_evaluation(run):
    candidates = [p for p in Path(run).glob("evaluations/*/report.json") if read(p).get("complete")]
    if not candidates:
        raise FileNotFoundError("Baseline has no completed evaluation; run evaluate_autoencoder.py --run <baseline> first")
    return max(candidates, key=lambda p: p.parent.name)


def metadata_chunks(folder, split):
    manifest = read(folder/"manifest.json")
    records = {a["path"]: a for a in manifest["shards"]}
    for path in sorted((folder/split).glob("part_*.npz")):
        record = records[path.relative_to(folder).as_posix()]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Shard changed after validation: {path}")
        with np.load(path, allow_pickle=False) as archive:
            a = {k: archive[k] for k in META}
        if len(a["sample_id"]) != record["count"]:
            raise ValueError("Shard sample count differs from manifest")
        yield a


def check_dataset_pair(baseline, scheme2):
    baseline, scheme2 = Path(baseline).resolve(), Path(scheme2).resolve()
    a, b = read(baseline/"manifest.json"), read(scheme2/"manifest.json")
    if not a.get("complete") or not b.get("complete") or a.get("representation") != "scheme1" or b.get("representation") != "scheme2":
        raise ValueError("Completed scheme1 and scheme2 datasets are required")
    if not read(scheme2/"validation.json").get("passed") or not read(baseline/"validation.json").get("passed"):
        raise ValueError("Both data validations must pass")
    if Path(b["paired_reference_dataset"]).resolve() != baseline or a["counts"] != b["counts"]:
        raise ValueError("Scheme2 is not paired with the baseline dataset")
    if a["shift_mode"] != "none" or b["shift_mode"] != "none":
        raise ValueError("This comparison expects zero displacement")
    plan_hash = digest(baseline/"donor_plan.npz")
    if plan_hash != digest(scheme2/"donor_plan.npz") or a["donor_plan_sha256"] != plan_hash or b["donor_plan_sha256"] != plan_hash:
        raise ValueError("Donor plans differ or were changed")
    checked = {}
    for split in ("train", "validation"):
        streams = [iter(metadata_chunks(folder, split)) for folder in (baseline, scheme2)]
        blocks = [next(s, None) for s in streams]
        offsets = [0, 0]
        count = 0
        while all(block is not None for block in blocks):
            n = min(len(blocks[i]["sample_id"])-offsets[i] for i in (0, 1))
            for key in META:
                if not np.array_equal(blocks[0][key][offsets[0]:offsets[0]+n], blocks[1][key][offsets[1]:offsets[1]+n]):
                    raise ValueError(f"Paired ordered metadata differ: {split}/{key}")
            count += n
            for i in (0, 1):
                offsets[i] += n
                if offsets[i] == len(blocks[i]["sample_id"]):
                    blocks[i] = next(streams[i], None)
                    offsets[i] = 0
        if any(block is not None for block in blocks) or count != a["total_samples"][split] or count != b["total_samples"][split]:
            raise ValueError("Paired split coverage differs")
        checked[split] = count
    print(f"[paired data checked] train={checked['train']:,}, validation={checked['validation']:,}; test not opened", flush=True)
    return dict(ordered_metadata_equal=True, train_validation_shard_hashes_checked=True,
                donor_plan_sha256=plan_hash, checked_samples=checked, test_opened=False)


def curve_point(labels, logits, fpr_limit):
    """An exploratory validation ROC point; NOT a deployment threshold fit."""
    labels, logits = np.asarray(labels, bool), np.asarray(logits, np.float64)
    if not 0<=fpr_limit<=1 or not labels.any() or labels.all() or not np.isfinite(logits).all():
        raise ValueError("Both classes and finite logits needed")
    order = np.argsort(-logits, kind="stable")
    scores = logits[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores)-1]
    tp = np.cumsum(labels[order], dtype=np.int64)[ends]
    fp = ends+1-tp
    allowed = np.flatnonzero(fp <= fpr_limit*np.count_nonzero(~labels)+1e-10)
    threshold = scores[ends[allowed[-1]]] if len(allowed) else np.nextafter(scores[0], np.inf)
    metrics = binary_metrics(labels, logits, threshold)
    metrics.update(fpr_budget=fpr_limit, role="exploratory point selected on the validation ROC curve; not a deployment threshold")
    return metrics


def draw_prefix_equal(left, right):
    remaining = min(Path(left).stat().st_size, Path(right).stat().st_size)
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while remaining:
            n = min(1048576, remaining)
            if a.read(n) != b.read(n):
                return False
            remaining -= n
    return True


def compare_reports(baseline_run, baseline_report, run, report_path, out, model_names=("scheme1","scheme2")):
    baseline_run, run, out = map(Path, (baseline_run, run, out))
    reports = [read(baseline_report), read(report_path)]
    configs = [read(p/"config.json") for p in (baseline_run, run)]
    summaries = [read(p/"summary.json") for p in (baseline_run, run)]
    if any(not r.get("complete") or r.get("test_used") for r in reports):
        raise ValueError("Complete validation-only reports required")
    for directory, report in zip((baseline_run, run), reports):
        if Path(report["training_run"]).resolve() != directory.resolve():
            raise ValueError("Evaluation belongs to a different training run")
        if any(digest(directory/name) != expected for name, expected in report["source_sha256"].items()):
            raise ValueError("Baseline or candidate training artifact changed after evaluation")
        if digest(directory/"sample_draws.jsonl") != report["sampling"]["sample_audit_sha256"]:
            raise ValueError("Sampling audit changed after evaluation")
    if any(configs[0][key] != configs[1][key] for key in CONTROLS):
        raise ValueError("Training budgets, stopping rules or model settings differ from baseline")
    if not draw_prefix_equal(baseline_run/"sample_draws.jsonl", run/"sample_draws.jsonl"):
        raise ValueError("Sample draw streams differ before the shorter run stops")
    scores = []
    for path in (Path(baseline_report).parent, Path(report_path).parent):
        with np.load(path/"validation_scores.npz", allow_pickle=False) as a:
            scores.append({k: a[k] for k in a.files})
    for key in ("sample_id", "labels", "attack_flow_fraction"):
        if not np.array_equal(scores[0][key], scores[1][key]):
            raise ValueError(f"Paired evaluation population differs: {key}")
    models = {}
    if len(model_names)!=2 or len(set(model_names))!=2:raise ValueError("Two distinct comparison names required")
    for name, directory, summary, report, values in zip(model_names, (baseline_run, run), summaries, reports, scores):
        saved = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
        points = {}
        mixed = (values["attack_flow_fraction"] > 0) & (values["attack_flow_fraction"] < 1)
        for budget in (.01, .001):
            point = curve_point(values["labels"], values["logits"], budget)
            point["mixed_attack_recall"] = float((values["logits"][mixed] >= point["threshold_logit"]).mean()) if mixed.any() else None
            points[str(budget)] = point
        op = report["linear_probe"]["operating_points"]["balanced_prior_0.5"]
        models[name] = dict(training_run=str(directory.resolve()), input_features=saved["model_args"]["features"],
            model_parameter_count=sum(t.numel() for t in saved["model_state"].values()),
            best_epoch=summary["best_epoch"], epochs_completed=summary["epochs_completed"],
            updates=summary["steps"], stopping_reason=summary["stopping_reason"],
            reconstruction_mse_internal_only=summary["best_validation_mse"],
            input_transform=read(directory/"config.json").get("input_transform","none"),
            roc_auc=op["roc_auc"], average_precision=op["average_precision"],
            train_calibrated_operating_points=report["linear_probe"]["operating_points"],
            exploratory_validation_roc_points=points)
    result = dict(complete=True, test_used=False, controls={k: configs[0][k] for k in CONTROLS},
        paired_validation_examples=len(scores[0]["labels"]), paired_positive_examples=int(scores[0]["labels"].sum()),
        identical_draw_stream_prefix=True, models=models,
        baseline_report=str(Path(baseline_report).resolve()), baseline_report_sha256=digest(baseline_report),
        limitations=["reconstruction losses have different targets and scales; do not rank representations by raw MSE",
                     "same hidden widths and latent size; input widths and parameter counts are reported separately",
                     "same maximum budget and stopping rule, not necessarily the same number of completed updates",
                     "fixed-FPR ROC points are exploratory validation statistics, not train-calibrated deployment thresholds",
                     "comparison covers the two complete representation pipelines, not just a causal effect of the addition formula",
                     "one seed and previously selected validation checkpoints; test remains reserved"])
    write(out/"comparison.json", result)
    lines = [f"# {' / '.join(model_names)} AE validation comparison", "",
        "Same splits, donor plan, ordered metadata, sampler stream prefix, maximum budget and stopping rule.",
        "Input targets differ: raw reconstruction MSE must not be used to rank these representations.", "",
        "| Model | Input width | Parameters | Best epoch | Stop epoch | ROC-AUC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in models.items():
        lines.append(f"| {name} | {model['input_features']} | {model['model_parameter_count']} | {model['best_epoch']} | {model['epochs_completed']} | {model['roc_auc']:.5f} | {model['average_precision']:.2%} |")
    lines += ["", "Exploratory validation ROC points below use observed validation FPR budgets.",
        "They are not deployment thresholds; the JSON separately retains all train-calibrated operating points.", "",
        "| Model | FPR budget | Actual FPR | Precision | Recall | Mixed-flow attack recall |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, model in models.items():
        for point in model["exploratory_validation_roc_points"].values():
            mixed_text = f"{point['mixed_attack_recall']:.2%}" if point["mixed_attack_recall"] is not None else "n/a"
            lines.append(f"| {name} | {point['fpr_budget']:.2%} | {point['fpr']:.4%} | {point['precision']:.2%} | {point['recall']:.2%} | {mixed_text} |")
    lines += ["", "No test data used. These are exploratory validation results from one seed.",
        "See comparison.json for controls, operating points, model size and interpretation limits."]
    (out/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def standalone(args):
    """Run the selected Scheme2 pipeline without retired Scheme1 artifacts."""
    run = (args.run or latest_run("scheme2")).resolve() if args.stage == "evaluate" else None
    dataset = Path(read(run/"config.json")["dataset"]) if run else (args.dataset or latest_dataset()).resolve()
    if args.dataset is not None and args.dataset.resolve() != dataset.resolve():
        raise ValueError("--dataset differs from the completed run")
    manifest = read(dataset/"manifest.json")
    if not manifest.get("complete") or manifest.get("representation") != "scheme2" or not read(dataset/"validation.json").get("passed"):
        raise ValueError("Completed validated Scheme2 data required")
    if digest(dataset/"donor_plan.npz") != manifest["donor_plan_sha256"]:
        raise ValueError("Donor plan changed")
    if manifest["shift_mode"] != "none":
        raise ValueError("This workflow expects zero displacement")
    controls = dict(seed=4266, batch_size=128, epochs=500, steps_per_epoch=500, width=32,
        latent=32, lr=.001, real_only=False, val_limit=manifest["total_samples"]["validation"],
        patience=20, min_delta=.001)
    print(f"Scheme2 data: {dataset}\nSettings: {controls}", flush=True)
    if args.stage == "check":
        for record in manifest["shards"]:
            if digest(dataset/record["path"]) != record["sha256"]:
                raise ValueError(f"Shard changed: {record['path']}")
        print("CHECK PASSED: Scheme2 manifests, plan and every shard hash; no fitting performed", flush=True)
        return
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.stage in ("all", "train"):
        training = SimpleNamespace(**controls, dataset=dataset, out=args.out, device=device,
            cpu_threads=2, log_every=50, representation="scheme2", input_transform="none")
        if training.out is None:
            training.out = new_run_directory(training)
        run = training.out.resolve()
        train_temporal(training)
        if args.stage == "train":
            print(f"TRAIN COMPLETE: {run}", flush=True)
            return
    parent = run/"evaluations"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_scheme2_validation_"), dir=parent))
    write(out/"evaluation_status.json", dict(status="running", complete=False))
    try:
        evaluate(run, out, device, args.evaluation_batch_size)
    except BaseException as error:
        write(out/"evaluation_status.json", dict(status="failed", complete=False, error=str(error)))
        raise
    write(out/"evaluation_status.json", dict(status="complete", complete=True))
    print(f"COMPLETE: validation report = {out/'report.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "check", "train", "evaluate"), default="all")
    p.add_argument("--dataset", type=Path, help="Default: latest completed scheme2 dataset")
    p.add_argument("--baseline-run", type=Path, help="Optional rebuilt Scheme1 run for historical paired comparison; default runs Scheme2 independently")
    p.add_argument("--baseline-evaluation", type=Path, help="Baseline report.json; default: latest completed evaluation of the baseline run")
    p.add_argument("--run", type=Path, help="Existing completed scheme2 AE, only for --stage evaluate")
    p.add_argument("--out", type=Path, help="Exact fresh AE output; default: unique scheme2-labelled folder under ae_runs")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--evaluation-batch-size", type=int, default=1024)
    args = p.parse_args()
    if args.evaluation_batch_size <= 0:
        p.error("Positive evaluation batch size required")
    if args.run is not None and args.stage != "evaluate":
        p.error("--run is only valid for --stage evaluate; it does not resume training")
    if args.out is not None and args.stage not in ("all", "train"):
        p.error("--out names a fresh AE run and is only valid for all/train")
    if args.baseline_run is None:
        if args.baseline_evaluation is not None:
            p.error("--baseline-evaluation requires --baseline-run")
        standalone(args)
        return
    baseline = (args.baseline_run or latest_run("scheme1")).resolve()
    baseline_config = read(baseline/"config.json")
    baseline_report = args.baseline_evaluation or latest_evaluation(baseline)
    baseline_evaluation = read(baseline_report)
    if not read(baseline/"summary.json").get("complete") or not baseline_evaluation.get("complete") or Path(baseline_evaluation["training_run"]).resolve() != baseline:
        raise ValueError("A completed baseline and its matching completed evaluation are required")
    if any(digest(baseline/name) != value for name, value in baseline_evaluation["source_sha256"].items()):
        raise ValueError("Baseline artifacts changed after its evaluation")
    if digest(baseline/"sample_draws.jsonl") != baseline_evaluation["sampling"]["sample_audit_sha256"]:
        raise ValueError("Baseline sampling audit changed after evaluation")
    if args.stage == "evaluate":
        run = (args.run or latest_run("scheme2")).resolve()
        dataset = Path(read(run/"config.json")["dataset"]).resolve()
        if any(read(run/"config.json")[key] != baseline_config[key] for key in CONTROLS):
            raise ValueError("Candidate training controls differ from baseline")
        if args.dataset is not None and args.dataset.resolve() != dataset:
            raise ValueError("--dataset differs from the completed run")
    else:
        dataset = (args.dataset or latest_dataset()).resolve()
    reference = Path(baseline_config["dataset"]).resolve()
    if digest(baseline/"dataset_manifest.json") != digest(reference/"manifest.json"):
        raise ValueError("Baseline training dataset manifest changed")
    paired = check_dataset_pair(reference, dataset)
    controls = {k: baseline_config[k] for k in CONTROLS}
    print(f"Baseline: {baseline}\nScheme2 data: {dataset}\nMatched settings: {controls}", flush=True)
    if args.stage == "check":
        print("CHECK PASSED: no AE or probe fitting performed", flush=True)
        return
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.stage in ("all", "train"):
        training = SimpleNamespace(**controls, dataset=dataset, out=args.out, device=device,
            cpu_threads=baseline_config["cpu_threads"], log_every=baseline_config["log_every"], representation="scheme2")
        if training.out is None:
            training.out = new_run_directory(training)
        run = training.out.resolve()
        train_temporal(training)
        if args.stage == "train":
            print(f"TRAIN COMPLETE: {run}; next: python CIC_new/run_scheme2_experiment.py --stage evaluate --run \"{run}\"", flush=True)
            return
    parent = run/"evaluations"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=stamp+"_scheme2_comparison_", dir=parent))
    write(out/"evaluation_status.json", dict(status="running", complete=False, stage="frozen_evaluation"))
    try:
        report = evaluate(run, out, device, args.evaluation_batch_size)
        write(out/"paired_data_check.json", paired)
        compare_reports(baseline, baseline_report, run, out/"report.json", out)
    except BaseException as error:
        write(out/"evaluation_status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", complete=False, error=str(error)))
        raise
    write(out/"evaluation_status.json", dict(status="complete", complete=True, comparison_complete=True,
          finished_at=datetime.now().astimezone().isoformat()))
    print(f"COMPLETE: comparison = {out/'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
