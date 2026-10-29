"""Train only 82+15 Scheme2 features; compare with a completed original LSTM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from CIC_new.prepare_distribution_features import prepare
from CIC_new.preprocessing.distribution_features import NAMES, BUNDLE
from train.common.io import ROOT, PROJECT, fresh, read_json, write_json, unique_name, sha256, environment
from train.common.temporal_experiment import attack_details
from train.run_circular_shift import write_csv
from train.common.temporal_storage import remove_cache

PREPARED = PROJECT / "CIC_new/outputs/scheme2_runs/20261022_115847_160650_scheme2_bucket64_shift0_seed4266_0fhh3gjt/preprocessed"
REFERENCE = ROOT / "runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/seed4266/lstm/start0"
ENCODER = PREPARED.parent / "bucket_embeddings/config.json"
CODE = ("CIC_new/prepare_distribution_features.py", "CIC_new/preprocessing/distribution_features.py",
        "CIC_new/preprocessing/dataset.py", "CIC_new/representation_learning/learn_bucket_embeddings.py",
        "CIC_new/representation_learning/bucket_autoencoder.py", "CIC_new/representation_learning/encode_sequences.py",
        "CIC_new/prepare_scheme2_dataset.py", "CIC_new/preprocessing/donor_plan.py", "CIC_new/preprocessing/augmentation.py",
        "CIC_new/preprocessing/export_augmented_dataset.py", "train/run_distribution_feature_ablation.py",
        "train/common/experiment.py", "train/common/data.py", "train/common/torch_runner.py", "train/lstm/model.py", "train/common/networks.py",
        "train/common/temporal_storage.py", "train/common/remove_temporal_cache.ps1")


def artifact(folder, names):
    return {"path": str(folder.resolve()), "hashes": {name: sha256(folder / name) for name in names}}


def verified(record):
    folder = Path(record["path"])
    if any(sha256(folder / name) != digest for name, digest in record["hashes"].items()):
        raise ValueError("Recorded completed artifacts changed: " + str(folder))
    return folder


def reference_record(folder):
    config = read_json(folder / "config.json")
    summary = read_json(folder / "summary.json")
    if not summary.get("complete") or summary.get("split") != "validation" or summary.get("test_used"):
        raise ValueError("Reference must be a completed validation LSTM run")
    if read_json(folder / "data_audit.json")["test_opened"]:
        raise ValueError("Reference unexpectedly opened test data")
    dataset = Path(config["dataset"])
    if sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Reference dataset manifest changed")
    donor = dataset / "donor_plan.npz"
    if sha256(donor) != read_json(dataset / "manifest.json")["donor_plan_sha256"]:
        raise ValueError("Reference donor plan changed")
    record = artifact(folder, ("model.pt", "summary.json", "config.json", "thresholds.json", "predictions.npz", "data_audit.json"))
    if record["hashes"]["model.pt"] != summary["model_sha256"] or record["hashes"]["thresholds.json"] != summary["thresholds_sha256"]:
        raise ValueError("Reference model or thresholds changed")
    return {**record, "dataset": str(dataset.resolve()), "donor_plan_sha256": sha256(donor)}


def next_attempt(folder, stage):
    number = 1
    while (folder / f"{stage}_attempt_{number:04d}").exists():
        number += 1
    return folder / f"{stage}_attempt_{number:04d}"


def command_run(command, log):
    with log.open("x", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
            code = process.wait()
    if code:
        raise RuntimeError(f"Stage failed (exit {code}); inspect {log}")


def check_representation_space(data, owner):
    schema = read_json(data / "schema.json")
    connection = sqlite3.connect((data / "manifest.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        counts = connection.execute("SELECT COUNT(*),SUM(attack_flows>0 AND split='train') FROM original_sequences WHERE split IN ('train','validation')").fetchone()
    finally:
        connection.close()
    real, positive_train = map(int, counts)
    numeric = len(schema["numeric_names"])
    categories = sum(schema["category_slots"].values())
    prepared_bytes = real * 10 * (4 * (numeric + categories) + 16)
    encoded_bytes = real * 10 * (64 * 4 + 20)
    export_bytes = (real + 2 * positive_train) * (10 * (65 * 4 + 4 + 8 + 8) + 1024)
    required = prepared_bytes + encoded_bytes + export_bytes + 2 * 1024**3
    free = shutil.disk_usage(owner).free
    print(f"Representation disk preflight: estimated additional {required/1024**3:.1f} GiB including reserve; free {free/1024**3:.1f} GiB", flush=True)
    if free < required:
        raise RuntimeError("Insufficient free space for the representation stage; preserve this batch, free space, then --resume")


def release_representation_caches(data, representation, out):
    # Only disposable caches created inside this new batch are eligible.
    audit = out / "cache_cleanup"
    for split in ("train", "validation"):
        remove_cache(data / "mmap" / split, data, audit, "prepared")
    remove_cache(representation / "encoded_sequences", representation, audit, "encoded")


def compare(out, state):
    baseline = state["settings"]["reference"]
    folders = [verified(baseline), verified(state["cases"]["distribution97"]["lstm"])]
    summaries = [read_json(path / "summary.json") for path in folders]
    configs = [read_json(path / "config.json") for path in folders]
    for key in ("architecture", "training", "seed", "device", "threads", "environment", "weighting", "input_transform"):
        if configs[0][key] != configs[1][key]:
            raise ValueError("Unmatched detector settings: " + key)
    for key in ("train_cohort_sha256", "validation_cohort_sha256", "evaluation_policy"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError("Unmatched cohorts or evaluation: " + key)
    for folder in folders:
        if read_json(folder / "data_audit.json")["test_opened"]:
            raise ValueError("Unexpected test use")
    with np.load(folders[0] / "predictions.npz", allow_pickle=False) as left, np.load(folders[1] / "predictions.npz", allow_pickle=False) as right:
        for key in ("sample_id", "y", "primary_attack", "source_host"):
            np.testing.assert_array_equal(left[key], right[key])
    donor = Path(baseline["dataset"]) / "donor_plan.npz"
    if sha256(donor) != baseline["donor_plan_sha256"]:
        raise ValueError("Reference donor plan changed during the run")
    representation = Path(state["cases"]["distribution97"]["representation"]["path"])
    with np.load(donor, allow_pickle=False) as a, np.load(representation / "donor_plan.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Donor plan keys differ")
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    previous = {(r["attack"], r["operating_point"]): r for r in attack_details(folders[0], summaries[0]["metrics"])}
    rows = []
    for row in attack_details(folders[1], summaries[1]["metrics"]):
        old = previous[(row["attack"], row["operating_point"])]
        rows.append({**row, "control_tp": old["tp"], "control_fn": old["fn"], "control_recall": old["recall"],
                     "control_global_fpr": old["global_fpr"],
                     "delta_recall_pp": 100 * (row["recall"] - old["recall"]) if row["n"] else None})
    write_csv(out / "attack_comparison.csv", rows)
    result = {"control": str(folders[0]), "distribution": str(folders[1]), "rows": rows,
        "baseline_reused": True,
        "control_metrics": summaries[0]["metrics"], "distribution_metrics": summaries[1]["metrics"],
        "same_donor_pairs_verified": True, "same_prediction_cohort_verified": True,
        "test_used": False, "limitations": state["settings"]["limitations"]}
    write_json(out / "comparison.json", result)
    lines = ["# Distribution-feature LSTM comparison", "", "Only the extended-feature encoder and LSTM were trained. The completed original-feature baseline was reused read-only.", "",
        f"Control AP: {summaries[0]['metrics']['average_precision']:.6f}; extended AP: {summaries[1]['metrics']['average_precision']:.6f}.", "",
        "| Attack | FPR budget | n | Control TP/FN | Extended TP/FN | Control recall | Extended recall | Delta (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["n"]:
            lines.append(f"| {row['attack']} | {row['operating_point']} | {row['n']} | {row['control_tp']}/{row['control_fn']} | {row['tp']}/{row['fn']} | {row['control_recall']:.2%} | {row['recall']:.2%} | {row['delta_recall_pp']:+.2f} |")
    lines += ["", "## Limits", ""] + ["- " + s for s in result["limitations"]]
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE)
    parser.add_argument("--encoder-config", type=Path, default=ENCODER)
    parser.add_argument("--db", type=Path, default=PROJECT / "datasets/cic_2018_traffic.db")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.out and args.resume:
        raise ValueError("Use either --out or --resume")
    reference = read_json(args.reference_run / "config.json")
    baseline = reference_record(args.reference_run)
    encoder = read_json(args.encoder_config)
    if reference["model"] != "lstm" or reference.get("input_transform", {}).get("start_bucket", 0):
        raise ValueError("Use an original chronological LSTM reference")
    if encoder["bucket_dim"] != 64 or encoder["lr"] != .001 or reference["seed"] != encoder["seed"]:
        raise ValueError("Expected the established 64D, lr=.001, matched-seed encoder protocol")
    shard_stats = [(str(p.relative_to(args.prepared)), p.stat().st_size, p.stat().st_mtime_ns)
                   for split in ("train", "validation") for p in sorted((args.prepared / "splits" / split).glob("part_*.npz"))]
    settings = {"prepared": str(args.prepared.resolve()), "prepared_manifest_sha256": sha256(args.prepared / "manifest.json"),
        "prepared_metadata_sha256": sha256(args.prepared / "manifest.sqlite"),
        "prepared_shard_stat_signature": hashlib.sha256(json.dumps(shard_stats).encode()).hexdigest(),
        "staged_source_signature": {"bytes": (args.prepared / "work.sqlite").stat().st_size,
                                    "mtime_ns": (args.prepared / "work.sqlite").stat().st_mtime_ns},
        "prepared_schema_sha256": sha256(args.prepared / "schema.json"), "prepared_scaler_sha256": sha256(args.prepared / "scaler.json"),
        "db": str(args.db.resolve()), "source_signature": {"bytes": args.db.stat().st_size, "mtime_ns": args.db.stat().st_mtime_ns},
        "reference": baseline, "baseline_reused": True, "cases_to_train": ["distribution97"],
        "reference_config_sha256": sha256(args.reference_run / "config.json"), "encoder_config_sha256": sha256(args.encoder_config),
        "architecture": reference["architecture"], "training": reference["training"], "seed": reference["seed"],
        "device": reference["device"], "threads": reference["threads"], "encoder": {key: encoder[key] for key in
            ("seed", "cpu_threads", "batch_size", "steps", "validate_every", "category_dim", "bucket_dim", "lr", "val_limit")},
        "bundle": BUNDLE, "added_features": NAMES, "environment": environment(),
        "code_sha256": {name: sha256(PROJECT / name) for name in CODE},
        "limitations": ["Only the extended-feature encoder and original LSTM are newly trained; the completed seed-matched baseline is reused read-only.",
            "The new representation requires a fresh encoder, while the baseline retains its existing encoder; this comparison does not isolate feature changes from encoder refitting.",
            "The extended encoder adds 15 numeric inputs/outputs; its bottlenecks and 64D output are retained, while detector parameter counts stay identical.",
            "Old normalized features and category vocabularies are retained exactly; only new statistics use newly fitted real-training scalers.",
            "Original training/validation IDs, labels, donor pairs, activity masks and flow counts are preserved; split assignment is not recomputed.",
            "Completed-flow statistics retain the original offline information-availability assumption.",
            "Synthetic augmentation remains frozen-vector addition and L2 normalization, never arithmetic addition of raw quantiles or standard deviations.",
            "The new fit reuses original training data and the already inspected validation cohort; this is one-seed exploratory validation, not independent test evidence.",
            "Checkpoint selection remains overall validation AP. Global validation FPR thresholds are selected independently under the same policy.",
            "No test features are prepared, encoded or scored; original all-split metadata is retained only for donor ownership and synthetic IDs."]}
    if args.dry_run:
        print({k: settings[k] for k in ("cases_to_train", "baseline_reused", "bundle", "added_features", "training", "encoder", "device")})
        print("Read-only baseline: " + baseline["path"])
        return None
    if args.resume:
        out = args.resume.resolve()
        state = read_json(out / "batch.json")
        if state["settings"] != settings:
            raise ValueError("Resume settings, source or code changed; retain this run and use a new output")
        if state.get("status") == "complete":
            for case in state["cases"].values():
                for record in case.values():
                    verified(record)
            print("Completed batch verified: " + str(out))
            return out
    else:
        out = fresh(args.out or ROOT / "runs" / unique_name("lstm_distribution_features_seed" + str(reference["seed"])))
        state = {"settings": settings, "cases": {}, "status": "running", "test_used": False}
    write_json(out / "batch.json", state)
    write_json(out / "detector_config.json", {"architecture": reference["architecture"], "training": reference["training"]})
    try:
        for name, extended in (("distribution97", True),):
            folder = out / name
            folder.mkdir(exist_ok=True)
            case = state["cases"].setdefault(name, {})
            if "prepared" in case:
                data = verified(case["prepared"])
                audit = read_json(data / "extension_audit.json")
                for record in audit["shards"]:
                    if sha256(data / record["path"]) != record["sha256"]:
                        raise ValueError("A completed prepared shard changed")
            else:
                data = prepare(args.prepared, next_attempt(folder, "prepared"), extended)
                case["prepared"] = artifact(data, ("manifest.json", "schema.json", "scaler.json", "manifest.sqlite", "extension_audit.json"))
                write_json(out / "batch.json", state)
            if "representation" in case:
                representation = verified(case["representation"])
            else:
                representation = next_attempt(folder, "representation")
                check_representation_space(data, folder)
                command = [sys.executable, "-u", "-m", "CIC_new.prepare_scheme2_dataset", "--db", str(args.db),
                    "--prepared", str(data), "--out", str(representation), "--splits", "train", "validation",
                    "--device", reference["device"], "--seed", str(reference["seed"])]
                for flag, key in (("embedding-steps", "steps"), ("validate-every", "validate_every"), ("val-limit", "val_limit"),
                    ("batch-size", "batch_size"), ("cpu-threads", "cpu_threads"), ("category-dim", "category_dim"), ("bucket-dim", "bucket_dim")):
                    command += ["--" + flag, str(encoder[key])]
                command_run(command, representation.with_suffix(".log"))
                case["representation"] = artifact(representation, ("workflow.json", "donor_plan.npz", "bucket_embeddings/best.pt", "dataset/manifest.json", "dataset/validation.json"))
                write_json(out / "batch.json", state)
            release_representation_caches(data, representation, out)
            if "lstm" in case:
                verified(case["lstm"])
            else:
                target = next_attempt(folder, "lstm")
                command_run([sys.executable, "-u", "-m", "train.lstm.train", "--dataset", str(representation / "dataset"),
                    "--config", str(out / "detector_config.json"), "--out", str(target), "--cache", str(target.parent / (target.name + "_cache")),
                    "--device", reference["device"], "--threads", str(reference["threads"]), "--seed", str(reference["seed"]),
                    "--circular-shift-start", "0"], target.with_suffix(".log"))
                case["lstm"] = artifact(target, ("model.pt", "summary.json", "config.json", "thresholds.json", "predictions.npz", "data_audit.json"))
                write_json(out / "batch.json", state)
            target = Path(case["lstm"]["path"])
            remove_cache(target.parent / (target.name + "_cache"), target.parent, out / "cache_cleanup", "detector")
        compare(out, state)
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(out / "batch.json", state)
    print(str(out), flush=True)
    return out


if __name__ == "__main__":
    main()
