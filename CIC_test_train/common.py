"""Shared data geometry and utilities for the independent CIC experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BIN_MS = 10_000
SEQ_LEN = 10
DAY_MS = 86_400_000
SPLITS = ("train", "validation", "calibration", "test")
WINDOW_DTYPE = np.dtype([
    ("host", "<i4"), ("start_ms", "<i8"), ("label", "u1"),
    ("active", "u1"), ("attack_bits", "<u8"), ("attack_flows", "<i8"),
    ("rows", "<i4", (SEQ_LEN,)),
])


def dump_json(path: Path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows, fields=None):
    rows = iter(rows)
    first = next(rows, None)
    if fields is None:
        fields = list(first) if first is not None else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(rows)


def empty_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output is nonempty; choose a new directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def feature_groups(names):
    log = [i for i, n in enumerate(names) if n == "flow_count" or n.endswith("_sum")
           or n.startswith("duration_") or n.endswith("_nunique")]
    bounded = [i for i, n in enumerate(names) if n.endswith(("_ratio", "_entropy"))]
    if sorted(log + bounded) != list(range(len(names))):
        raise ValueError("Feature schema contains unhandled feature types")
    return log, bounded


def occupancy(active):
    return np.where(active == 1, "1", np.where(active <= 4, "2-4", "5-10"))


class WindowData:
    """Disk-backed bucket features; each window stores only 10 bucket indices."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.manifest = read_json(folder / "manifest.json")
        if not self.manifest.get("complete"):
            raise ValueError("Prepared dataset is incomplete")
        self.names = self.manifest["feature_names"]
        self.buckets = np.load(folder / "bucket_features.npy", mmap_mode="r")
        self.windows = {s: np.load(folder / f"{s}_windows.npy", mmap_mode="r")
                        for s in SPLITS}

    def close(self):
        for array in (self.buckets, *self.windows.values()):
            array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def batch(self, split, indices):
        windows = self.windows[split][indices]
        rows = windows["rows"]
        mask = rows >= 0
        x = np.array(self.buckets[np.maximum(rows, 0)], dtype=np.float32, copy=True)
        x[~mask] = 0
        return x, mask.astype(np.float32)
