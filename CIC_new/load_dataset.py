"""Read the prepared local CIC dataset without downloads or rebuilding buckets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "local_dataset.json"
SPLITS = ("train", "validation", "test")
ARRAYS = ("X", "mask", "y_bucket", "y_sequence", "sample_id", "flow_counts", "attack_flow_counts")


def metadata(config_path=DEFAULT_CONFIG):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run = (ROOT / config["dataset_directory"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Prepared local dataset missing: {run}. No download or rebuild was attempted.")
    state = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if not state.get("complete") or state["config"].get("pipeline_version") != 2:
        raise ValueError("The selected dataset is not a completed pipeline-v2 run")
    schema = json.loads((run / "feature_schema.json").read_text(encoding="utf-8"))
    checks = json.loads((run / "export_checks.json").read_text(encoding="utf-8"))
    if not all(checks.get(name) is True for name in ("finite", "empty_buckets_zero", "labels_reconciled", "manifest_reconciled")):
        raise ValueError("Dataset export checks did not pass")
    if schema["feature_count"] != config["feature_count"]:
        raise ValueError("Local dataset pointer and feature schema disagree")
    for split in SPLITS:
        if checks["counts"][split] != config["expected_sequences"][split]:
            raise ValueError(f"Local dataset pointer and exported {split} count disagree")
    return config, run, schema


def _validate(values, config, expected_n):
    n = len(values["sample_id"])
    if n != expected_n:
        raise ValueError("Unexpected shard sample count")
    shape = (n, config["buckets_per_sequence"])
    if values["X"].shape != (*shape, config["feature_count"]) or values["X"].dtype != np.float32:
        raise ValueError("Unexpected X shape or dtype")
    if values["y_sequence"].shape != (n,):
        raise ValueError("Unexpected sequence labels")
    for name in ("mask", "y_bucket", "flow_counts", "attack_flow_counts"):
        if values[name].shape != shape:
            raise ValueError(f"Unexpected {name} shape")
    if not np.isfinite(values["X"]).all():
        raise ValueError("Nonfinite model inputs")


def _shard_files(run, split, config):
    state = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    shard_size = state["config"]["shard_sequences"]
    n = config["expected_sequences"][split]
    files = sorted((run / "splits" / split).glob("part_*.npz"))
    expected_files = (n + shard_size - 1)//shard_size
    if [f.name for f in files] != [f"part_{i:06d}.npz" for i in range(expected_files)]:
        raise ValueError(f"Missing or unexpected shard filenames in {split}")
    return files, shard_size, n


def iter_shards(split, config_path=DEFAULT_CONFIG):
    """Yield dictionaries of NumPy arrays, one shard at a time; mask stays separate."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    config, run, _ = metadata(config_path)
    files, shard_size, n = _shard_files(run, split, config)
    for index, file in enumerate(files):
        with np.load(file, allow_pickle=False) as archive:
            values = {name: archive[name] for name in ARRAYS}
        _validate(values, config, min(shard_size, n-index*shard_size))
        yield values


def inspect_local(config_path=DEFAULT_CONFIG):
    """Check manifest/counts and first/last shard of each split, not a full rescan."""
    config, run, schema = metadata(config_path)
    rows = []
    for split in SPLITS:
        files, shard_size, n = _shard_files(run, split, config)
        for index in sorted({0, len(files)-1}):
            with np.load(files[index], allow_pickle=False) as archive:
                values = {name: archive[name] for name in ARRAYS}
            _validate(values, config, min(shard_size, n-index*shard_size))
        rows.append({"split": split, "sequences": n, "shards": len(files)})
    return {"dataset": str(run), "feature_count": schema["feature_count"],
            "buckets_per_sequence": config["buckets_per_sequence"], "splits": rows,
            "verification": "completed build metadata, all shard filenames, and first/last shard arrays per split"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(inspect_local(args.config), indent=2))


if __name__ == "__main__":
    main()
