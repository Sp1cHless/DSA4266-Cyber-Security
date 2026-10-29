"""Train four temporal ablations using frozen historical detector settings.

Reuse completed buckets/splits. Build and score train/validation only; no test
feature shard is opened. The historical 100s/10s model is referenced, not refitted.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from CIC_new.run_temporal_ablation import execute, stage_record, now
from train.common.io import PROJECT, ROOT, environment, fresh, read_json, sha256, unique_name, write_json
from train.common.metrics import POLICY
from train.common.temporal_storage import cleanup_case, compatible_storage_upgrade, require_space

BASELINES = {
    "lstm": ROOT / "lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc",
    "cnn": ROOT / "cnn/runs/20261023_024348_386209_cnn_seed4266_84d958",
}
EMBEDDING = PROJECT / "CIC_new/outputs/scheme2_runs/20261022_115847_160650_scheme2_bucket64_shift0_seed4266_0fhh3gjt/bucket_embeddings/config.json"
GEOMETRIES = [(200, 10), (200, 20), (400, 10), (400, 40)]
PACKAGES = ("numpy", "scikit-learn", "torch")
FOCUS_ATTACKS = (("Infilteration", "infilteration"), ("Brute_Force_-Web", "web"))
OPERATING_POINTS = (("validation_fpr_1pct", "1pct"), ("validation_fpr_0.1pct", "0.1pct"))


def reference_contract(baseline, embedding, model):
    config, summary = [read_json(baseline / name) for name in ("config.json", "summary.json")]
    encoder = read_json(embedding)
    if (config["model"] != model or not summary.get("complete") or summary.get("test_used")
            or summary.get("split") != "validation" or config["evaluation_policy"] != POLICY
            or config["weighting"] != "equal_four_groups_mean_weight_one"
            or config["selection_metric"] != "validation_average_precision_sklearn"
            or config.get("input_transform", {"name": "identity"}) != {"name": "identity"}
            or config["architecture"].get("circular_shift_start", 0) != 0):
        raise ValueError("Reference must be a completed chronological baseline for the selected model and original policy")
    for filename, key in (("model.pt", "model_sha256"), ("thresholds.json", "thresholds_sha256")):
        if sha256(baseline / filename) != summary[key]:
            raise ValueError(f"Historical reference changed: {filename}")
    if encoder["seed"] != config["seed"] or encoder["bucket_dim"] != 64 or encoder["lr"] != .001:
        raise ValueError("Encoder reference differs from the existing Scheme2 contract")
    encoder_keys = ("steps", "validate_every", "batch_size", "cpu_threads", "category_dim", "bucket_dim", "val_limit", "lr")
    return {"model": model, "detector": {key: config[key] for key in ("architecture", "training", "seed", "device", "threads")},
            "encoder": {key: encoder[key] for key in encoder_keys},
            "packages": {key: config["environment"]["packages"][key] for key in PACKAGES},
            "baseline_config_sha256": sha256(baseline / "config.json"),
            "baseline_summary_sha256": sha256(baseline / "summary.json"),
            "encoder_config_sha256": sha256(embedding)}


def prepared_cases(folder):
    batch = read_json(folder / "batch.json")
    cases = batch["settings"]["cases"]
    if [(c["sequence_seconds"], c["bucket_seconds"]) for c in cases] != GEOMETRIES:
        raise ValueError("Expected the four completed 200/10, 200/20, 400/10, 400/40 preparations")
    result = []
    for case in cases:
        record = batch["cases"][case["name"]].get("preprocess")
        if not record or stage_record(Path(record["path"]), "preprocess", case) != record:
            raise ValueError(f"Missing, incomplete or changed preparation: {case['name']}")
        result.append({**case, "prepared": record})
    return result


def find_preparations():
    parent = PROJECT / "CIC_new/outputs/temporal_ablation_runs"
    for folder in sorted(parent.glob("*"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if folder.is_dir() and (folder / "batch.json").is_file():
            try:
                prepared_cases(folder)
                return folder.resolve()
            except (ValueError, KeyError, FileNotFoundError):
                continue
    raise ValueError("No matching completed preparations found; supply --prepared-batch")


def representation_command(args, contract, case, target, device):
    prepared = Path(case["prepared"]["path"])
    manifest = read_json(prepared / "manifest.json")
    encoder = contract["encoder"]
    command = [sys.executable, "-u", str(PROJECT / "CIC_new/prepare_scheme2_dataset.py"),
               "--db", manifest["source"]["path"], "--prepared", str(prepared), "--out", str(target),
               "--splits", "train", "validation", "--sequence-seconds", str(case["sequence_seconds"]),
               "--bucket-seconds", str(case["bucket_seconds"]), "--seed", str(contract["detector"]["seed"]),
               "--device", device, "--shard-size", str(args.shard_size)]
    for flag, key in (("embedding-steps", "steps"), ("validate-every", "validate_every"),
                      ("batch-size", "batch_size"), ("cpu-threads", "cpu_threads"),
                      ("category-dim", "category_dim"), ("bucket-dim", "bucket_dim"), ("val-limit", "val_limit")):
        command += ["--" + flag, str(encoder[key])]
    return command


def training_command(contract, dataset, target, config, device):
    return [sys.executable, "-u", "-m", f"train.{contract['model']}.train", "--dataset", str(dataset),
            "--out", str(target), "--cache", str(target.parent / (target.name + "_cache")),
            "--config", str(config), "--seed", str(contract["detector"]["seed"]),
            "--threads", str(contract["detector"]["threads"]), "--device", device,
            "--circular-shift-start", "0"]


def training_record(folder, contract, dataset, device):
    config, summary, status = [read_json(folder / name) for name in ("config.json", "summary.json", "status.json")]
    expected = contract["detector"]
    for key in ("architecture", "training", "seed", "threads"):
        if config[key] != expected[key]:
            raise ValueError(f"Detector settings differ from the historical baseline: {key}")
    if (not summary.get("complete") or summary.get("test_used") or summary.get("split") != "validation"
            or status.get("status") != "complete" or summary["device"] != device
            or summary["evaluation_policy"] != POLICY
            or config["input_transform"] != {"name": "identity"}
            or summary["dataset_manifest_sha256"] != sha256(dataset / "manifest.json")
            or read_json(folder / "data_audit.json")["test_opened"]):
        raise ValueError("Training result does not match the validation-only contract")
    for filename, key in (("model.pt", "model_sha256"), ("thresholds.json", "thresholds_sha256")):
        if sha256(folder / filename) != summary[key]:
            raise ValueError(f"Completed training artifact changed: {filename}")
    names = ("summary.json", "config.json", "model.pt", "thresholds.json", "data_audit.json", "status.json")
    return {"path": str(folder.resolve()), "hashes": {name: sha256(folder / name) for name in names}}


def attack_details(run, metrics):
    """Slice saved real-validation predictions; do not refit per-attack thresholds."""
    thresholds = read_json(run / "thresholds.json")["values"]
    with np.load(run / "predictions.npz", allow_pickle=False) as saved:
        y, scores, attacks, hosts = [saved[key].copy() for key in
                                    ("y", "score", "primary_attack", "source_host")]
    if (len(y) != metrics["n"] or scores.shape != y.shape or attacks.shape != y.shape
            or hosts.shape != y.shape or not np.isfinite(scores).all()):
        raise ValueError("Validation predictions do not match their summary")
    result = []
    for attack, _ in FOCUS_ATTACKS:
        keep = (y == 1) & (attacks == attack)
        n = int(keep.sum())
        if n != metrics["per_primary_attack"].get(attack, {}).get("n", 0):
            raise ValueError("Saved attack population differs from the summary")
        for point, _ in OPERATING_POINTS:
            tp = int(np.count_nonzero(scores[keep] >= thresholds[point]))
            result.append({"attack": attack, "operating_point": point, "n": n,
                           "hosts": int(len(np.unique(hosts[keep]))), "tp": tp, "fn": n - tp,
                           "recall": tp / n if n else None, "threshold": thresholds[point],
                           "global_fpr": metrics["operating_points"][point]["fpr"]})
    return result


def summarize(output, state, baseline):
    model = state["settings"]["model"]
    rows, focused = [], []
    baseline_focus = {}
    references = [("existing_baseline", 100, 10, baseline)]
    references += [(c["name"], c["sequence_seconds"], c["bucket_seconds"],
                    Path(state["cases"][c["name"]][model]["path"]))
                   for c in state["settings"]["cases"] if model in state["cases"].get(c["name"], {})]
    for name, window, bucket, run in references:
        summary = read_json(run / "summary.json")
        metrics = summary["metrics"]
        one, strict = [metrics["operating_points"][k] for k in ("validation_fpr_1pct", "validation_fpr_0.1pct")]
        row = {"case": name, "sequence_seconds": window, "bucket_seconds": bucket,
               "timesteps": window // bucket, "validation_n": metrics["n"], "positives": metrics["positives"],
               "prevalence": metrics["prevalence"], "ap": metrics["average_precision"], "roc_auc": metrics["roc_auc"],
               "recall_at_1pct": one["recall"], "actual_fpr_1pct": one["fpr"],
               "recall_at_0.1pct": strict["recall"], "actual_fpr_0.1pct": strict["fpr"],
               "best_epoch": summary["fit"]["best_epoch"], "run": str(run), "test_used": False,
               "runtime_matches_baseline": name == "existing_baseline" or not state["settings"]["runtime_differences"]}
        details = attack_details(run, metrics)
        if name == "existing_baseline":
            baseline_focus = {(entry["attack"], entry["operating_point"]): entry for entry in details}
        for entry in details:
            reference = baseline_focus[(entry["attack"], entry["operating_point"])]
            change = (100 * (entry["recall"] - reference["recall"])
                      if entry["recall"] is not None and reference["recall"] is not None else None)
            focused.append({"case": name, "sequence_seconds": window, "bucket_seconds": bucket,
                            **entry, "baseline_n": reference["n"], "baseline_recall": reference["recall"],
                            "delta_recall_pp_descriptive": change, "paired_cohorts": False,
                            "runtime_matches_baseline": row["runtime_matches_baseline"]})
        lookup = {(entry["attack"], entry["operating_point"]): entry for entry in details}
        for attack, prefix in FOCUS_ATTACKS:
            row[prefix + "_n"] = lookup[(attack, OPERATING_POINTS[0][0])]["n"]
            for point, suffix in OPERATING_POINTS:
                entry = lookup[(attack, point)]
                for key in ("recall", "tp", "fn"):
                    row[f"{prefix}_{key}_at_{suffix}"] = entry[key]
        rows.append(row)
    with (output / "validation_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "validation_comparison.json", {"rows": rows, "test_used": False,
               "model": model, "focused_attacks": focused,
               "paired_cohorts": False, "runtime_differences": state["settings"]["runtime_differences"],
               "limitations": ["Historical baseline is referenced without retraining.",
                  "Each preparation has a different cohort/prevalence; these are not paired recall deltas.",
                  "Validation selects checkpoints and thresholds. Missing primary types have null recall, not zero.",
                  "Training data are reused for fitting, never reported as held-out evidence."]})
    with (output / "attack_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(focused[0]))
        writer.writeheader()
        writer.writerows(focused)
    lines = [f"# {model.upper()} window comparison: Infilteration and Brute Force Web", "",
             "Existing 100s/10s baseline plus completed new-window runs. Real validation only; test is not used.",
             "Global validation thresholds are selected by the unchanged FPR rule; no attack-specific threshold is fitted.",
             "Recall deltas are descriptive percentage points. Cohorts, prevalence and coverage differ across sequence durations; these are not paired or causal estimates.", "",
             "| Case | Attack | N | R@1% FPR | Delta (pp) | R@0.1% FPR | Delta (pp) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    pct = lambda value: "N/A" if value is None else f"{100 * value:.2f}%"
    delta = lambda value: "N/A" if value is None else f"{value:+.2f}"
    for row in rows:
        for attack, _ in FOCUS_ATTACKS:
            entries = [entry for entry in focused if entry["case"] == row["case"] and entry["attack"] == attack]
            one, strict = entries
            lines.append(f"| {row['case']} | {attack} | {one['n']} | {pct(one['recall'])} | "
                         f"{delta(one['delta_recall_pp_descriptive'])} | {pct(strict['recall'])} | "
                         f"{delta(strict['delta_recall_pp_descriptive'])} |")
    lines += ["", "See attack_comparison.csv for TP/FN, host counts, thresholds and realized global FPR.",
              "See validation_comparison.csv for overall AP, ROC-AUC and prevalence.", ""]
    (output / "ATTACK_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    source = (args.prepared_batch or find_preparations()).resolve()
    cases = prepared_cases(source)
    baseline, embedding = args.baseline_run.resolve(), args.embedding_config.resolve()
    contract = reference_contract(baseline, embedding, args.model)
    if any(read_json(Path(case["prepared"]["path"]) / "manifest.json")["seed"] != contract["detector"]["seed"]
           for case in cases):
        raise ValueError("Prepared split seed differs from the historical baseline seed")
    runtime = environment()
    device = args.device or contract["detector"]["device"]
    differences = {key: {"baseline": expected, "current": runtime["packages"].get(key)}
                   for key, expected in contract["packages"].items() if runtime["packages"].get(key) != expected}
    if device != contract["detector"]["device"]:
        differences["device"] = {"baseline": contract["detector"]["device"], "current": device}
    if args.dry_run:
        print(f"Prepared batch: {source}")
        print(f"Historical reference: {baseline}; not retrained")
        print(f"Runtime differences: {differences or 'none'}")
        parent = args.out or ROOT / "runs" / f"<new_temporal_{args.model}_batch>"
        for case in cases:
            folder = parent / case["name"]
            print(" ".join(representation_command(args, contract, case, folder / "representation_attempt_0001", device)))
            print(" ".join(training_command(contract, folder / "representation_attempt_0001/dataset",
                  folder / f"{args.model}_attempt_0001", parent / f"{args.model}_config.json", device)))
        return
    if differences:
        print(f"Using current environment; historical runtime differences recorded: {differences}", flush=True)
    from train.common.experiment import resolve_device
    resolve_device(args.model, device)
    if args.out is None:
        args.out = ROOT / "runs" / unique_name(args.model + "_temporal_seed" + str(contract["detector"]["seed"]))
    output = args.out.resolve()
    settings = {"model": args.model, "prepared_batch": str(source), "baseline_run": str(baseline), "cases": cases,
                "contract": contract, "python": sys.executable, "packages": runtime["packages"],
                "device": device, "runtime_differences": differences, "shard_size": args.shard_size,
                "pipeline_sha256": {str(p.relative_to(PROJECT)): sha256(p) for base in
                  (ROOT / "common", ROOT / args.model, PROJECT / "CIC_new/preprocessing", PROJECT / "CIC_new/representation_learning")
                  for p in sorted(base.glob("*.py"))}}
    settings["pipeline_sha256"].update({str(Path(__file__).relative_to(PROJECT)): sha256(Path(__file__)),
                                      f"train/run_temporal_{args.model}.py": sha256(ROOT / f"run_temporal_{args.model}.py"),
                                      "train/common/remove_temporal_cache.ps1": sha256(Path(__file__).with_name("remove_temporal_cache.ps1")),
                                      "CIC_new/prepare_scheme2_dataset.py": sha256(PROJECT / "CIC_new/prepare_scheme2_dataset.py")})
    storage_upgrade = False
    if args.resume:
        state = read_json(output / "batch.json")
        if state["settings"] != settings:
            if not compatible_storage_upgrade(state["settings"], settings):
                raise ValueError("Resume data, baseline, runtime, scientific code or settings changed")
            storage_upgrade = True
    else:
        fresh(output)
        state = {"settings": settings, "cases": {}, "started_at": now(), "test_used": False}
        write_json(output / f"{args.model}_config.json", {key: contract["detector"][key] for key in ("architecture", "training")})
    if read_json(output / f"{args.model}_config.json") != {key: contract["detector"][key] for key in ("architecture", "training")}:
        raise ValueError("Saved detector baseline configuration changed")
    lock = output / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        import os
        stream.write(str(os.getpid()))
    try:
        if storage_upgrade:
            backup = output / (unique_name("batch_before_storage_upgrade") + ".json")
            write_json(backup, state)
            state.setdefault("storage_upgrades", []).append({"at": now(), "original_batch": str(backup),
                "previous_pipeline_sha256": state["settings"]["pipeline_sha256"],
                "current_pipeline_sha256": settings["pipeline_sha256"],
                "scope": "cache lifetime and disk preflight only; scientific settings unchanged"})
            state["settings"] = settings
            write_json(output / "batch.json", state)
        # Reclaim previous completed cases before allocating another case's cache.
        for case in cases:
            item = state["cases"].get(case["name"], {})
            representation_ok = "representation" in item
            detector_ok = args.model in item
            if representation_ok and stage_record(Path(item["representation"]["path"]), "dataset", case) != item["representation"]:
                raise ValueError("Completed representation changed: " + case["name"])
            if detector_ok and training_record(Path(item[args.model]["path"]), contract,
                    Path(item["representation"]["path"]) / "dataset", device) != item[args.model]:
                raise ValueError("Completed detector changed: " + case["name"])
            cleanup_case(case, item, args.model, output, representation_ok, detector_ok)
        if args.cleanup_only:
            print(f"CLEANUP COMPLETE: {output}; no training started", flush=True)
            return output
        state.update(complete=False, status="running")
        state.pop("error", None)
        write_json(output / "batch.json", state)
        print(f"Training batch: {output}", flush=True)
        for case in cases:
            item = state["cases"].setdefault(case["name"], {"attempts": []})
            folder = output / case["name"]
            folder.mkdir(exist_ok=True)
            for stage in ("representation", args.model):
                if stage in item:
                    old = item[stage]
                    actual = (stage_record(Path(old["path"]), "dataset", case) if stage == "representation" else
                              training_record(Path(old["path"]), contract, Path(item["representation"]["path"]) / "dataset", device))
                    if actual != old:
                        raise ValueError("Completed stage changed: " + old["path"])
                    print(f"[reuse {case['name']} / {stage}] {old['path']}", flush=True)
                    continue
                number = 1
                while (folder / f"{stage}_attempt_{number:04d}").exists():
                    number += 1
                target = folder / f"{stage}_attempt_{number:04d}"
                command = (representation_command(args, contract, case, target, device) if stage == "representation" else
                           training_command(contract, Path(item["representation"]["path"]) / "dataset", target,
                                            output / f"{args.model}_config.json", device))
                require_space(case, stage, output,
                              Path(item["representation"]["path"]) / "dataset" if stage != "representation" else None)
                item["attempts"].append({"stage": stage, "path": str(target), "started_at": now()})
                write_json(output / "batch.json", state)
                execute(command, folder / f"{stage}_attempt_{number:04d}.log")
                item[stage] = (stage_record(target, "dataset", case) if stage == "representation" else
                               training_record(target, contract, Path(item["representation"]["path"]) / "dataset", device))
                item["attempts"][-1].update(complete=True, finished_at=now())
                write_json(output / "batch.json", state)
                cleanup_case(case, item, args.model, output, "representation" in item, args.model in item)
            summarize(output, state, baseline)
        state.update(complete=True, status="complete", finished_at=now())
        write_json(output / "batch.json", state)
        print(f"COMPLETE: {output / 'validation_comparison.csv'}", flush=True)
        return output
    except BaseException as error:
        state.update(complete=False, status="failed", error=str(error))
        write_json(output / "batch.json", state)
        raise
    finally:
        lock.unlink()


def main(argv=None, model="lstm"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-batch", type=Path, help="Default: latest matching completed four-case batch")
    parser.add_argument("--baseline-run", type=Path, default=BASELINES[model])
    parser.add_argument("--embedding-config", type=Path, default=EMBEDDING)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true",
                        help="With --resume --out, verify completed stages and remove only disposable caches")
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: historical baseline device")
    parser.add_argument("--allow-runtime-drift", action="store_true",
                        help="Compatibility flag; the current environment is already allowed by default")
    args = parser.parse_args(argv)
    args.model = model
    if args.shard_size <= 0 or (args.resume and args.out is None) or (args.cleanup_only and not args.resume):
        parser.error("Positive shard size required; --resume requires --out; --cleanup-only requires --resume")
    try:
        return run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
