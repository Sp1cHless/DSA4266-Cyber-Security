"""Run paired circular-shift trials with fixed data, checked reuse and Web sparsity slices."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import traceback

if __package__ in (None, ""):
    # Do not leave train/ on the path where it would shadow installed xgboost.
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from train.common.data import Scheme2
from train.common.experiment import MODELS, main as train_model, resolve_device
from train.common.io import ROOT, DEFAULT_DATASET, environment, fresh, read_json, sha256, unique_name, write_json
from train.common.metrics import POLICY, validate_scores
from train.common.temporal import input_transform
from train.run_circular_shift import extract_metrics, write_csv

WEB = "Brute_Force_-Web"
ENV_KEYS = ("python", "platform", "machine", "processor", "packages")
CORE = ("common/data.py", "common/experiment.py", "common/io.py", "common/metrics.py",
        "common/networks.py", "common/temporal.py", "common/torch_runner.py")
# Reviewed compatibility for the original T=10 dataset only. The loader changed
# hardcoded shapes to manifest.timesteps; the added guards reject non-T=10 shifts.
# No objective, optimizer, model, weighting, threshold or split code changed.
TEN_BUCKET_SOURCE_MIGRATIONS = {
    "common/data.py": (
        "c635ec2f6cbc4f1556b7a2b81e63827c4ab38d35dfa8a134fc63b39c105e03f1",
        "c3dabc040aa76e8d708c6edca6f8a916d35eddb1138e03828a865164752f04cb"),
    "common/experiment.py": (
        "bf58307ef8dfbeb3cbb2ac980044315f9c96bc10c4b4c73365b2b97d4deaa770",
        "549d51cda3c855fdcab5eed6298ccd3f518fb58d0908ee2654fef4b2e549c1f9"),
}
LIMITATIONS = [
    "Temporal perturbation diagnostics on the existing real validation split, not independent test results.",
    "Every run reuses the same training cohort and fixed validation cohort; seeds do not create new observations.",
    "The fixed rotation is applied to both training and validation, allowing adaptation during retraining.",
    "Each run selects its checkpoint and thresholds separately using validation under the same policy.",
    "Reported standard deviations describe training-seed variation conditional on this fixed data split.",
    "They do not measure uncertainty over new hosts, campaigns or data splits and are not significance tests.",
    "Web groups use the original number of active buckets; each group's recall uses the global validation threshold.",
    "No Web-specific threshold is fitted. Empty groups have null recall, not zero recall.",
    "Circular shifts retain most adjacent pairs; similar results do not prove that temporal order is irrelevant.",
    "Historical project work already inspected the existing test split. This runner never opens test shards.",
]


def source_hashes(name):
    files = CORE + (("xgboost/model.py",) if name == "xgboost" else (f"{name}/model.py",))
    return {file: sha256(ROOT / file) for file in files}


def check_runtime(plan):
    current = environment()
    if any(current.get(key) != plan["environment"].get(key) for key in ENV_KEYS):
        raise ValueError("Runtime changed: use the original Python/software environment or start a fresh study")
    for name in plan["models"]:
        if source_hashes(name) != plan["source_hashes"][name]:
            raise ValueError(f"Training source changed for {name}; start a fresh study")


def check_run(folder, record, plan):
    """Validate a saved fit without loading weights, source shards or test arrays."""
    summary, config = read_json(folder / "summary.json"), read_json(folder / "config.json")
    if (read_json(folder / "status.json")["status"] != "complete" or not summary.get("complete")
            or summary.get("test_used") or summary.get("split") != "validation"):
        raise ValueError(f"A completed validation-only run is required: {folder}")
    name, start = record["model"], record["start_bucket"]
    expected = {"model": name, "seed": record["seed"], "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
                "device": plan["devices"][name], "evaluation_policy": POLICY,
                "weighting": "equal_four_groups_mean_weight_one", "input_transform": input_transform(start)}
    for key, value in expected.items():
        if summary.get(key, {"name": "identity"} if key == "input_transform" else None) != value:
            raise ValueError(f"Incompatible {key}: {folder}")
        if config.get(key, {"name": "identity"} if key == "input_transform" else None) != value:
            raise ValueError(f"Incompatible config {key}: {folder}")
    architecture = dict(config["architecture"])
    saved_start = architecture.pop("circular_shift_start", 0)
    if name != "xgboost" and saved_start != start:
        raise ValueError(f"Saved architecture has a different rotation: {folder}")
    if (architecture != plan["configs"][name]["architecture"] or config["training"] != plan["configs"][name]["training"]
            or config["threads"] != plan["threads"] or config.get("input_transform", {"name": "identity"}) != input_transform(start)):
        raise ValueError(f"Architecture, training settings, threads or input-transform mismatch: {folder}")
    if any(config["environment"].get(key) != plan["environment"].get(key) for key in ENV_KEYS):
        raise ValueError(f"Saved run uses a different runtime environment: {folder}")
    saved_sources = {key.replace("\\", "/"): value for key, value in config["source_sha256"].items()}
    migrations = []
    for key, value in plan["source_hashes"][name].items():
        if saved_sources.get(key) != value:
            if plan.get("timesteps") != 10 or TEN_BUCKET_SOURCE_MIGRATIONS.get(key) != (saved_sources.get(key), value):
                raise ValueError(f"Saved run uses different training source ({key}): {folder}")
            migrations.append({"file": key, "saved_sha256": saved_sources[key], "current_sha256": value,
                               "reason": "Reviewed T=10-equivalent shape generalization and non-T=10 shift guards only"})
    if migrations:
        record["source_compatibility"] = migrations
    model = "model.ubj" if name == "xgboost" else "model.pt"
    if sha256(folder / model) != summary["model_sha256"] or sha256(folder / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError(f"Model or threshold hash mismatch: {folder}")
    files = ("summary.json", "config.json", "thresholds.json", "predictions.npz", model)
    hashes = {file: sha256(folder / file) for file in files}
    if record.get("hashes") is not None and record["hashes"] != hashes:
        raise ValueError(f"Previously recorded artifacts changed: {folder}")
    return summary, hashes


def validation_index(root, plan):
    target = root / plan.get("validation_index_file", "validation_index.npz")
    if not plan.get("validation_index_sha256"):
        if target.exists():
            # Preserve an interrupted index and write a fresh snapshot.
            target = root / (unique_name("validation_index") + ".npz")
        split = Scheme2(plan["dataset"], plan["cache"]).load("validation")
        try:
            np.savez_compressed(target, sample_id=split["ids"], y=split["y"], attack=split["attack"],
                                host=split["host"], active_buckets=split["mask"].sum(axis=1).astype(np.int64))
            plan["validation_cohort_sha256"] = split.info["cohort_sha256"]
        finally:
            split.close()
        plan["validation_index_sha256"] = sha256(target)
        plan["validation_index_file"] = target.name
        write_json(root / "plan.json", plan)
    if sha256(target) != plan["validation_index_sha256"]:
        raise ValueError("Validation index changed")
    with np.load(target, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def statistics(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, members in grouped.items():
        item = dict(zip(group_keys, group))
        valid = [row for row in members if row["delta"] is not None]
        item.update({"n_pairs": len(members), "n_nonempty_pairs": len(valid), "unit": members[0]["unit"]})
        for field in ("baseline", "shifted", "delta"):
            values = np.array([row[field] for row in valid], dtype=float)
            item[field + "_mean"] = float(values.mean()) if len(values) else None
            item[field + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        deltas = [row["delta"] for row in valid]
        item.update({"positive_seeds": sum(v > 0 for v in deltas), "negative_seeds": sum(v < 0 for v in deltas),
                     "zero_seeds": sum(v == 0 for v in deltas), "delta_min": min(deltas) if deltas else None,
                     "delta_max": max(deltas) if deltas else None})
        output.append(item)
    return output


def paired_row(name, seed, metric, baseline, shifted, unit="pp", **extra):
    scale = 1 if unit == "absolute" else 100
    return {"model": name, "seed": seed, "metric": metric, **extra, "unit": unit,
            "baseline": baseline * scale if baseline is not None else None,
            "shifted": shifted * scale if shifted is not None else None,
            "delta": (shifted - baseline) * scale if baseline is not None and shifted is not None else None}


def make_report(root, plan):
    """Aggregate saved fits and global-threshold Web slices; never fit or score models."""
    index = validation_index(root, plan)
    pairs = defaultdict(dict)
    reference = None
    for record in plan["runs"]:
        folder = Path(record["path"])
        summary, _ = check_run(folder, record, plan)
        contract = {key: summary[key] for key in ("dataset_manifest_sha256", "train_cohort_sha256",
                    "validation_cohort_sha256", "evaluation_policy", "weighting")}
        if reference is not None and contract != reference:
            raise ValueError("Runs do not share the same data and metric contract")
        reference = contract
        if summary["validation_cohort_sha256"] != plan["validation_cohort_sha256"]:
            raise ValueError("Validation index differs from the fitted validation cohort")
        pairs[(record["model"], record["seed"])][record["start_bucket"]] = (summary, folder)
    overall, attacks, web = [], [], []
    for name in plan["models"]:
        for seed in plan["seeds"]:
            baseline, original_path = pairs[(name, seed)][0]
            shifted, shifted_path = pairs[(name, seed)][plan["start_bucket"]]
            a, b = extract_metrics(baseline), extract_metrics(shifted)
            for metric in a:
                overall.append(paired_row(name, seed, metric, a[metric], b[metric],
                                           "absolute" if metric == "roc_auc" else "pp"))
            for attack, details in baseline["metrics"]["per_primary_attack"].items():
                other = shifted["metrics"]["per_primary_attack"][attack]
                if (details["n"], details["hosts"]) != (other["n"], other["hosts"]):
                    raise ValueError("Per-attack populations differ")
                attacks.append(paired_row(name, seed, "recall_at_1pct_fpr",
                    details["recall_at_validation_fpr_1pct"], other["recall_at_validation_fpr_1pct"],
                    attack=attack, n=details["n"], hosts=details["hosts"]))
            predictions = []
            for folder in (original_path, shifted_path):
                with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                    for left, right in (("sample_id", "sample_id"), ("y", "y"), ("primary_attack", "attack"), ("source_host", "host")):
                        np.testing.assert_array_equal(saved[left], index[right])
                    validate_scores(index["y"], saved["score"])
                    predictions.append(saved["score"])
            for group, condition in (("all", index["active_buckets"] >= 1), ("one", index["active_buckets"] == 1),
                                     ("two", index["active_buckets"] == 2), ("three_or_more", index["active_buckets"] >= 3)):
                keep = (index["attack"] == WEB) & (index["y"] == 1) & condition
                n = int(keep.sum())
                for budget in ("validation_fpr_1pct", "validation_fpr_0.1pct"):
                    detections = [scores[keep] >= summary["metrics"]["operating_points"][budget]["threshold"]
                                  for scores, summary in zip(predictions, (baseline, shifted))]
                    recalls = [float(values.mean()) if n else None for values in detections]
                    web.append(paired_row(name, seed, budget, *recalls, group=group, n=n,
                        hosts=int(len(np.unique(index["host"][keep]))), baseline_tp=int(detections[0].sum()),
                        shifted_tp=int(detections[1].sum()), gained=int((~detections[0] & detections[1]).sum()),
                        lost=int((detections[0] & ~detections[1]).sum())))
    aggregate = statistics(overall, ("model", "metric"))
    attack_aggregate = statistics(attacks, ("model", "attack", "metric", "n", "hosts"))
    web_aggregate = statistics(web, ("model", "group", "metric", "n", "hosts"))
    report = fresh(root / "reports" / unique_name("paired_multiseed_comparison"))
    for file, rows in (("per_seed.csv", overall), ("aggregate.csv", aggregate), ("per_attack_per_seed.csv", attacks),
                       ("per_attack_aggregate.csv", attack_aggregate), ("web_per_seed.csv", web), ("web_aggregate.csv", web_aggregate)):
        write_csv(report / file, rows)
    write_json(report / "summary.json", {"complete": True, "test_used": False, "seeds": plan["seeds"],
        "start_bucket": plan["start_bucket"], "aggregate": aggregate, "web_aggregate": web_aggregate,
        "data_contract": reference, "limitations": LIMITATIONS})
    def formatted(value):
        return "N/A" if value is None else f"{value:.3f}"
    lines = ["# Paired multi-seed circular-shift diagnostic", "", f"Seeds: {plan['seeds']}; fixed start: {plan['start_bucket']}.",
             "Rates are percentages; deltas are percentage points, shifted minus no-shift for the same seed.",
             "SD is the sample standard deviation across seed pairs (ddof=1); it is N/A for one seed.", "",
             "| Model | Metric | No-shift mean | Shift mean | Delta mean | Delta SD | Positive / negative seeds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        if row["metric"] in ("ap", "recall_at_1pct_fpr", "recall_at_0.1pct_fpr"):
            lines.append(f"| {row['model']} | {row['metric']} | {formatted(row['baseline_mean'])} | {formatted(row['shifted_mean'])} | "
                         f"{formatted(row['delta_mean'])} | {formatted(row['delta_sd'])} | {row['positive_seeds']} / {row['negative_seeds']} |")
    lines += ["", "## Web recall by original active-bucket count", "",
              "| Model | Group | n | Global FPR budget | No-shift mean | Shift mean | Delta mean | Delta SD |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in web_aggregate:
        lines.append(f"| {row['model']} | {row['group']} | {row['n']} | {row['metric']} | {formatted(row['baseline_mean'])} | "
                     f"{formatted(row['shifted_mean'])} | {formatted(row['delta_mean'])} | {formatted(row['delta_sd'])} |")
    lines += ["", "CSV files retain every seed's difference, all attack recalls, realized FPR, and Web gained/lost detections.", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    (report / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Paired multi-seed report: {report}", flush=True)
    return report


def create_plan(args):
    if args.threads < 1 or len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Require positive threads, distinct models and distinct nonnegative seeds")
    for key in ("epochs", "rounds", "batch_size", "patience"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    data = Scheme2(args.dataset, args.cache)
    if data.manifest["timesteps"] != 10:
        raise ValueError("This study requires the original ten-bucket dataset")
    configs = {name: read_json(ROOT / name / "config.json") for name in args.models}
    for name, config in configs.items():
        config["architecture"].pop("circular_shift_start", None)
        keys = ("rounds", "patience") if name == "xgboost" else ("epochs", "batch_size", "patience")
        for key in keys:
            if getattr(args, key) is not None:
                config["training"][key] = getattr(args, key)
    devices = {name: resolve_device(name, "cpu" if name == "xgboost" and args.device == "mps" else args.device)
               for name in args.models}
    plan = {"version": 1, "timesteps": 10, "models": args.models, "seeds": args.seeds, "start_bucket": args.circular_shift_start,
            "dataset": str(data.path), "dataset_manifest_sha256": data.identity, "cache": str(args.cache.resolve()),
            "threads": args.threads, "devices": devices, "environment": environment(), "configs": configs,
            "source_hashes": {name: source_hashes(name) for name in args.models}, "test_used": False,
            "limitations": LIMITATIONS, "runs": []}
    reusable = {}
    for batch in args.reuse_batch or []:
        batch = batch.resolve()
        old = read_json(batch / "plan.json")
        if read_json(batch / "status.json")["status"] != "complete":
            raise ValueError(f"Reuse requires a completed batch: {batch}")
        for item in old["runs"]:
            seed = item.get("seed", old.get("seed"))
            if item["model"] not in args.models or seed not in args.seeds or item["start_bucket"] not in (0, args.circular_shift_start):
                continue
            key = (item["model"], seed, item["start_bucket"])
            if key in reusable:
                raise ValueError(f"Duplicate reuse candidate: {key}")
            record = {"model": key[0], "seed": key[1], "start_bucket": key[2]}
            folder = Path(item["path"])
            folder = folder if folder.is_absolute() else batch / folder
            _, hashes = check_run(folder, record, plan)
            reusable[key] = {"path": str(folder), "reused": True, "hashes": hashes,
                             "source_compatibility": record.get("source_compatibility", [])}
    root = fresh(args.out or ROOT / "runs" / unique_name(f"circular_multiseed_start{args.circular_shift_start}"))
    (root / "configs").mkdir()
    for name, config in configs.items():
        write_json(root / "configs" / f"{name}.json", config)
    for seed in args.seeds:
        for name in args.models:
            for start in (0, args.circular_shift_start):
                record = {"model": name, "seed": seed, "start_bucket": start, "attempts": []}
                record.update(reusable.get((name, seed, start), {"path": str(root / f"seed{seed}" / name / f"start{start}"), "reused": False}))
                plan["runs"].append(record)
    write_json(root / "plan.json", plan)
    return root, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(4266, 4271)))
    parser.add_argument("--circular-shift-start", type=int, choices=range(1, 10), default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--reuse-batch", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true", help="Validate reuse and save a plan without training")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="Continue the saved plan, preserving every existing fit/attempt")
    mode.add_argument("--summarize", type=Path, help="Write a fresh report from a completed study without fitting")
    for flag in ("epochs", "rounds", "batch-size", "patience"):
        parser.add_argument("--" + flag, type=int)
    args = parser.parse_args(argv)
    if args.resume or args.summarize:
        provided = {arg.split("=", 1)[0] for arg in (sys.argv[1:] if argv is None else argv) if arg.startswith("--")}
        if provided - {"--resume", "--summarize", "--plan-only"}:
            parser.error("Resume/summarize use the saved plan; pass only the mode and optional --plan-only")
        root = (args.resume or args.summarize).resolve()
        plan = read_json(root / "plan.json")
    else:
        root, plan = create_plan(args)
    check_runtime(plan)
    if Scheme2(plan["dataset"], plan["cache"]).identity != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    reused = sum(record["reused"] for record in plan["runs"])
    print(f"Study: {root}\nPlanned fits: {len(plan['runs'])}; reused: {reused}; new fits: {len(plan['runs']) - reused}", flush=True)
    if args.plan_only:
        if not args.resume and not args.summarize:
            write_json(root / "status.json", {"status": "planned", "test_used": False})
        return root
    if args.summarize:
        return make_report(root, plan)
    completed = []
    try:
        validation_index(root, plan)
        for count, record in enumerate(plan["runs"], 1):
            folder = Path(record["path"])
            complete = (folder / "status.json").exists() and read_json(folder / "status.json").get("status") == "complete"
            if record["reused"] or complete:
                _, record["hashes"] = check_run(folder, record, plan)
                print(f"[{count}/{len(plan['runs'])}] Keep completed {record['model']} seed={record['seed']} start={record['start_bucket']}", flush=True)
            else:
                if folder.exists():
                    record["attempts"].append(str(folder))
                    folder = folder.parent / unique_name(f"start{record['start_bucket']}_retry")
                    record["path"] = str(folder)
                write_json(root / "plan.json", plan)
                name = record["model"]
                print(f"[{count}/{len(plan['runs'])}] Fit {name} seed={record['seed']} start={record['start_bucket']}", flush=True)
                write_json(root / "status.json", {"status": "running", "completed_runs": completed,
                           "current_run": record["path"], "test_used": False})
                train_model(name, ["--dataset", plan["dataset"], "--cache", plan["cache"], "--device", plan["devices"][name],
                    "--seed", str(record["seed"]), "--threads", str(plan["threads"]),
                    "--config", str(root / "configs" / f"{name}.json"), "--out", str(folder),
                    "--circular-shift-start", str(record["start_bucket"])])
                _, record["hashes"] = check_run(folder, record, plan)
            completed.append(record["path"])
            write_json(root / "plan.json", plan)
        report = make_report(root, plan)
        write_json(root / "status.json", {"status": "complete", "completed_runs": completed,
                   "latest_report": str(report), "test_used": False})
    except BaseException as error:
        write_json(root / "plan.json", plan)
        write_json(root / "status.json", {"status": "failed", "completed_runs": completed,
                   "error": repr(error), "traceback": traceback.format_exc(), "test_used": False})
        raise
    return root


if __name__ == "__main__":
    main()
