"""Read portable Scheme2 shards; build disposable mmap caches outside source data."""
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np

from .io import read_json, write_json, sha256, ROOT

GROUPS = ("real_benign", "real_malicious", "synthetic_benign", "synthetic_malicious")
CACHE_VERSION = 1


class Scheme2:
    def __init__(self, path, cache=None):
        self.path = Path(path).resolve()
        self.manifest = read_json(self.path / "manifest.json")
        m = self.manifest
        if not m.get("complete") or m.get("representation") != "scheme2":
            raise ValueError("A completed Scheme2 dataset is required")
        if (m.get("timesteps"), m.get("feature_width")) != (10, 65):
            raise ValueError("Expected Scheme2 inputs [N,10,65]")
        if not read_json(self.path / "validation.json").get("passed"):
            raise ValueError("Dataset construction validation did not pass")
        self.identity = sha256(self.path / "manifest.json")
        self.cache = Path(cache or ROOT / "cache").resolve() / f"v{CACHE_VERSION}_{self.identity[:20]}"

    def load(self, split):
        if split not in ("train", "validation", "test"):
            raise ValueError(split)
        records = [r for r in self.manifest["shards"] if Path(r["path"]).parts[0] == split]
        actual = {p.name for p in (self.path / split).glob("part_*.npz")}
        if actual != {Path(r["path"]).name for r in records} or not records:
            raise ValueError(f"Missing or unexpected {split} shards")
        files = [self.path / r["path"] for r in records]
        stat_signature = hashlib.sha256(repr([(str(p.relative_to(self.path)), p.stat().st_size,
                                               p.stat().st_mtime_ns) for p in files]).encode()).hexdigest()
        folder = self.cache / split
        status = folder / "ready.json"
        if status.exists():
            saved = read_json(status)
            if saved["manifest_sha256"] != self.identity or saved["source_stats"] != stat_signature:
                raise ValueError("Source changed after caching; choose a fresh --cache directory")
        else:
            # mkdir is also a concurrency lock. Interrupted caches remain inspectable.
            folder.mkdir(parents=True, exist_ok=False)
            print(f"Preparing {split} cache from {len(files)} verified shards...", flush=True)
            n = self.manifest["total_samples"][split]
            specs = {"x": (np.float32, (n, 10, 65)), "mask": (np.float32, (n, 10)),
                     "y": (np.int64, (n,)), "ids": (np.int64, (n,)), "types": ("U24", (n,)),
                     "attack": ("U80", (n,)), "host": ("U64", (n,)), "flows": (np.int64, (n,))}
            arrays = {k: np.lib.format.open_memmap(folder / f"{k}.npy", mode="w+", dtype=d, shape=s)
                      for k, (d, s) in specs.items()}
            offset = 0
            groups = Counter()
            for record, file in zip(records, files):
                if sha256(file) != record["sha256"]:
                    raise ValueError(f"Shard checksum mismatch: {file}")
                with np.load(file, allow_pickle=False) as a:
                    size = len(a["y_sequence"])
                    if size != record["count"] or offset + size > n:
                        raise ValueError("Shard row count mismatch")
                    x, mask, y = a["X"], a["mask"], a["y_sequence"]
                    if x.shape != (size, 10, 65) or mask.shape != (size, 10):
                        raise ValueError("Invalid input shape")
                    if not np.isfinite(x).all() or not np.isin(mask, (0, 1)).all():
                        raise ValueError("Nonfinite input or invalid mask")
                    if np.any(x[mask == 0] != 0) or np.any(mask.sum(axis=1) == 0):
                        raise ValueError("Invalid padding or completely empty sequence")
                    if not np.isin(y, (0, 1)).all() or not np.array_equal(y, a["y_bucket"].max(axis=1)):
                        raise ValueError("Invalid labels")
                    kinds = a["sample_type"]
                    if not np.isin(kinds, GROUPS).all():
                        raise ValueError("Unknown training group")
                    if not np.array_equal(y, np.char.endswith(kinds, "malicious").astype(y.dtype)):
                        raise ValueError("Group/label mismatch")
                    if split != "train" and np.any(np.char.startswith(kinds, "synthetic")):
                        raise ValueError("Synthetic held-out sequence")
                    if np.any(a["flow_counts"] < 0) or not np.array_equal(a["flow_counts"] > 0, mask > 0):
                        raise ValueError("Invalid bucket flow counts")
                    values = {"x": x, "mask": mask, "y": y, "ids": a["sample_id"], "types": kinds,
                              "attack": a["primary_attack"], "host": a["left_ip"],
                              "flows": a["flow_counts"].sum(axis=1)}
                    for key, value in values.items():
                        arrays[key][offset:offset + size] = value
                    groups.update(kinds.tolist())
                    offset += size
            if offset != n or dict(groups) != self.manifest["counts"][split]:
                raise ValueError("Split population mismatch")
            if len(np.unique(arrays["ids"])) != n or len(np.unique(arrays["y"])) != 2:
                raise ValueError("Duplicate sample IDs or missing class")
            cohort = hashlib.sha256()
            for key in ("ids", "y", "types"):
                cohort.update(arrays[key].tobytes())
            for a in arrays.values():
                a.flush()
                a._mmap.close()
            write_json(status, {"manifest_sha256": self.identity, "source_stats": stat_signature,
                                "n": n, "groups": dict(groups), "cohort_sha256": cohort.hexdigest(),
                                "all_source_shards_sha256_verified": True})
        return Split(folder)


class Split:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = read_json(self.folder / "ready.json")
        self.arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in self.folder.glob("*.npy")}

    def __getitem__(self, key):
        return self.arrays[key]

    def __len__(self):
        return self.info["n"]

    def close(self):
        for a in self.arrays.values():
            a._mmap.close()
        self.arrays.clear()


def training_weights(split):
    counts = split.info["groups"]
    if any(counts.get(g, 0) == 0 for g in GROUPS):
        raise ValueError("Four nonempty training groups are required for equal-group weighting")
    # Mean weight = 1; each of the four groups contributes 25% of the objective.
    weights = np.empty(len(split), dtype=np.float32)
    for group in GROUPS:
        weights[split["types"] == group] = len(split) / (4 * counts[group])
    return weights


def flat_inputs(split):
    # Fixed temporal positions are preserved; mask is explicit, not a learned feature.
    return np.concatenate((split["x"].reshape(len(split), -1), split["mask"]), axis=1)


def check_disjoint(a, b):
    if np.intersect1d(a["ids"], b["ids"]).size:
        raise ValueError("Training and held-out sample IDs overlap")
