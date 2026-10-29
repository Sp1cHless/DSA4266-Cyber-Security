"""Sequential Scheme2 temporal ablations, stopping at preprocessing by default.

Only temporal geometry varies between cases. No downstream detector is trained.
Completed stages are reused on explicit resume; partial stages remain untouched.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.preprocessing.sequence_index import read_json, signature, write_json
from CIC_new.preprocessing.temporal import TemporalGrid

DEFAULT_CONFIG = ROOT / "CIC_new/configs/temporal_ablation.json"
SOURCE_FILES = [ROOT / "CIC_new/prepare_scheme2_dataset.py", Path(__file__).resolve()]
SOURCE_FILES += sorted((ROOT / "CIC_new/preprocessing").glob("*.py"))
SOURCE_FILES += sorted((ROOT / "CIC_new/representation_learning").glob("*.py"))


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path):
    raw = read_json(path)
    if not isinstance(raw, dict) or set(raw) != {"cases"} or not raw["cases"]:
        raise ValueError("Config must contain one nonempty 'cases' list")
    result = []
    for item in raw["cases"]:
        if not isinstance(item, dict) or set(item) != {"sequence_seconds", "bucket_seconds"}:
            raise ValueError("Each case must specify only sequence_seconds and bucket_seconds")
        grid = TemporalGrid(**item)
        result.append({"name": f"sequence{grid.sequence_seconds}s_bucket{grid.bucket_seconds}s_T{grid.timesteps}",
                       **grid.settings()})
    if len({case["name"] for case in result}) != len(result):
        raise ValueError("Duplicate temporal configurations")
    return result


def command_for(args, case, folder, stage, prepared=None):
    temporal = ["--sequence-seconds", str(case["sequence_seconds"]),
                "--bucket-seconds", str(case["bucket_seconds"])]
    common = ["--db", str(args.db.resolve()), "--out", str(folder),
              "--seed", str(args.seed), "--shard-size", str(args.shard_size), *temporal]
    if stage == "preprocess":
        return [sys.executable, "-u", "-m", "CIC_new.preprocessing.run_preprocessing", *common]
    return [sys.executable, "-u", str(ROOT / "CIC_new/prepare_scheme2_dataset.py"),
            *common, "--prepared", str(prepared), "--device", args.device,
            "--cpu-threads", str(args.cpu_threads), "--batch-size", str(args.batch_size),
            "--embedding-steps", str(args.embedding_steps), "--validate-every", str(args.validate_every),
            "--val-limit", str(args.val_limit), "--category-dim", str(args.category_dim),
            "--bucket-dim", str(args.bucket_dim)]


def execute(command, log):
    print(subprocess.list2cmdline(command), flush=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", env=environment)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
            if code:
                raise RuntimeError(f"Stage exited with code {code}; see {log}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
        finally:
            process.stdout.close()


def stage_record(folder, stage, case):
    """Check completion and bind reused metadata to exact content hashes."""
    if stage == "preprocess":
        files = ("manifest.json", "schema.json", "scaler.json", "split_notes.json", "v3_counts.json")
        manifest = read_json(folder / "manifest.json")
        schema = read_json(folder / "schema.json")
        if not manifest.get("complete") or manifest.get("version") != 3:
            raise ValueError("Incomplete preprocessing")
        for key in ("sequence_seconds", "bucket_seconds", "timesteps"):
            if schema.get(key) != case[key]:
                raise ValueError("Preprocessing geometry does not match its case")
    else:
        files = ("workflow.json", "dataset/manifest.json", "dataset/validation.json")
        if not read_json(folder / "workflow.json").get("complete"):
            raise ValueError("Incomplete Scheme2 workflow")
        report = read_json(folder / "dataset/validation.json")
        if not report.get("complete") or not report.get("passed"):
            raise ValueError("Incomplete Scheme2 export validation")
    return {"path": str(folder.resolve()), "hashes": {name: digest(folder / name) for name in files}}


def write_index(out, state):
    with (out / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ("case", "sequence_seconds", "bucket_seconds", "timesteps", "preprocess",
                   "dataset", "train_originals", "validation_originals", "test_originals", "error")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in state["settings"]["cases"]:
            item = state["cases"].get(case["name"], {})
            prepared = item.get("preprocess", {}).get("path", "")
            counts = read_json(Path(prepared) / "v3_counts.json") if prepared else {}
            dataset = item.get("dataset", {}).get("path", "")
            writer.writerow(dict(case=case["name"], **{k: case[k] for k in
                                 ("sequence_seconds", "bucket_seconds", "timesteps")},
                                 preprocess=prepared, dataset=str(Path(dataset) / "dataset") if dataset else "",
                                 **{name + "_originals": counts.get(name, "") for name in
                                    ("train", "validation", "test")}, error=item.get("error", "")))


def run(args):
    cases = load_cases(args.config)
    if not args.db.is_file():
        raise FileNotFoundError(args.db)
    if args.dry_run:
        parent = args.out.resolve() if args.out else ROOT / "CIC_new/outputs/temporal_ablation_runs/<unique-run>"
        for case in cases:
            folder = parent / case["name"]
            print(subprocess.list2cmdline(command_for(args, case, folder / "preprocess_attempt_0001", "preprocess")))
            if args.stage == "dataset":
                print(subprocess.list2cmdline(command_for(args, case, folder / "dataset_attempt_0001",
                      "dataset", folder / "preprocess_attempt_0001")))
        return
    if args.out is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        args.out = ROOT / "CIC_new/outputs/temporal_ablation_runs" / f"{stamp}_{uuid.uuid4().hex[:6]}"
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Fresh batch directory required; use --resume for {out}")
    out.mkdir(parents=True, exist_ok=True)
    # An exclusive lock prevents two builders from touching one batch.
    lock = out / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_at": now()}))
    state = None
    try:
        settings = {"source": signature(args.db.resolve()), "cases": cases,
                    "python": str(Path(sys.executable).resolve()),
                    "pipeline_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in SOURCE_FILES},
                    **{key: getattr(args, key) for key in ("seed", "shard_size", "device", "cpu_threads",
                       "batch_size", "embedding_steps", "validate_every", "val_limit", "category_dim", "bucket_dim")}}
        path = out / "batch.json"
        if args.resume:
            state = read_json(path)
            if state["settings"] != settings:
                state = None
                raise ValueError("Resume source, geometry, code or non-temporal settings changed")
        else:
            state = {"settings": settings, "started_at": now(), "cases": {},
                     "prediction_model_training_started": False}
        state.update(status="running", complete=False, requested_stage=args.stage)
        state.pop("error", None)
        write_json(path, state)
        print(f"Batch output: {out}", flush=True)
        failures = []
        for case in cases:
            item = state["cases"].setdefault(case["name"], {"attempts": []})
            item.pop("error", None)
            try:
                for stage in (("preprocess", "dataset") if args.stage == "dataset" else ("preprocess",)):
                    if stage in item:
                        record = item[stage]
                        if stage_record(Path(record["path"]), stage, case) != record:
                            raise ValueError(f"Completed stage metadata changed: {record['path']}")
                        print(f"[reuse {case['name']} / {stage}] {record['path']}", flush=True)
                        continue
                    folder = out / case["name"]
                    folder.mkdir(exist_ok=True)
                    number = 1
                    while (folder / f"{stage}_attempt_{number:04d}").exists():
                        number += 1
                    attempt = folder / f"{stage}_attempt_{number:04d}"
                    attempt.mkdir()
                    item["attempts"].append({"stage": stage, "path": str(attempt), "started_at": now()})
                    write_json(path, state)
                    prepared = item.get("preprocess", {}).get("path")
                    # Logs are siblings because Scheme2 requires an empty stage directory.
                    execute(command_for(args, case, attempt, stage, prepared),
                            folder / f"{stage}_attempt_{number:04d}.log")
                    record = stage_record(attempt, stage, case)
                    item[stage] = record
                    item["attempts"][-1].update(complete=True, finished_at=now())
                    write_json(path, state)
                    write_index(out, state)
            except Exception as error:
                item["error"] = str(error)
                failures.append(case["name"])
                write_json(path, state)
                write_index(out, state)
                if not args.keep_going:
                    raise
        if signature(args.db.resolve()) != settings["source"]:
            raise ValueError("Raw source changed during the batch")
        if failures:
            raise RuntimeError("Failed cases: " + ", ".join(failures))
        state.update(status="complete", complete=True, finished_at=now())
        write_json(path, state)
        write_index(out, state)
        print(f"COMPLETE: {out / 'index.csv'}", flush=True)
    except BaseException as error:
        if state is not None:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         complete=False, error=str(error))
            write_json(out / "batch.json", state)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "datasets/cic_2018_traffic.db")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, help="Fresh batch directory; default creates a unique run")
    parser.add_argument("--stage", choices=("preprocess", "dataset"), default="preprocess")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or scanning flows")
    parser.add_argument("--keep-going", action="store_true", help="Continue later cases after a failed case")
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-steps", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=4096)
    parser.add_argument("--category-dim", type=int, default=16)
    parser.add_argument("--bucket-dim", type=int, default=64)
    args = parser.parse_args()
    if args.seed < 0 or any(getattr(args, key) <= 0 for key in ("shard_size", "cpu_threads", "batch_size",
              "embedding_steps", "validate_every", "val_limit", "category_dim", "bucket_dim")):
        parser.error("Seed must be nonnegative and sizes/training budgets positive")
    if args.resume and args.out is None:
        parser.error("--resume requires the previous --out")
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
