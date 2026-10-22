"""Small artificial Scheme2-shaped dataset for exercising all training paths."""
from collections import Counter
from pathlib import Path
import numpy as np
from train.common.io import write_json, sha256
from train.common.data import GROUPS


def make_fixture(path):
    path = Path(path)
    path.mkdir(parents=True)
    rng = np.random.default_rng(17)
    records, counts, totals = [], {}, {}
    offset = 0
    for split in ("train", "validation", "test"):
        kinds = (np.repeat(GROUPS, (128, 32, 32, 32)) if split == "train"
                 else np.repeat(GROUPS[:2], (80, 24)))
        n = len(kinds)
        y = np.char.endswith(kinds, "malicious").astype(np.uint8)
        mask = (rng.random((n, 10)) > 0.25).astype(np.uint8)
        mask[:, 0] = 1
        vectors = rng.normal(0, 0.1, (n, 10, 64))
        vectors[:, :, 0] += y[:, None] * 0.6 - 0.3
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        x = np.concatenate((vectors, rng.normal(size=(n, 10, 1))), axis=-1).astype(np.float32)
        x *= mask[..., None]
        bucket_labels = y[:, None] * mask
        ids = np.arange(offset, offset + n, dtype=np.int64)
        payload = dict(X=x, mask=mask, y_sequence=y, y_bucket=bucket_labels, sample_id=ids,
                       sample_type=kinds, primary_attack=np.where(y, "FixtureAttack", "Benign"),
                       left_ip=np.array([f"10.0.0.{i % 20}" for i in range(n)]),
                       flow_counts=mask.astype(np.int64) * 2)
        folder = path / split
        folder.mkdir()
        file = folder / "part_000000.npz"
        np.savez_compressed(file, **payload)
        records.append({"path": file.relative_to(path).as_posix(), "count": n, "sha256": sha256(file)})
        counts[split] = dict(Counter(kinds.tolist()))
        totals[split] = n
        offset += n
    write_json(path / "manifest.json", {"complete": True, "representation": "scheme2", "timesteps": 10,
               "feature_width": 65, "shards": records, "counts": counts, "total_samples": totals,
               "fixture_only": True})
    write_json(path / "validation.json", {"passed": True, "fixture_only": True})
    return path
