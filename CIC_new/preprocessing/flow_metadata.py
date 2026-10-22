"""Compact real-sequence flow counts for evaluation without another dataset."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_flow_metadata(prepared, dataset):
    """Add sidecars without modifying existing shards, manifests or checkpoints."""
    prepared, dataset = Path(prepared), Path(dataset)
    connection = sqlite3.connect((prepared/"manifest.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    records = {}
    try:
        for split in ("validation", "test"):
            rows = connection.execute(
                "SELECT sequence_id,flows,attack_flows FROM original_sequences WHERE split=? ORDER BY sequence_id",
                (split,)).fetchall()
            values = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
            seen = np.zeros(len(values), bool)
            for path in sorted((dataset/split).glob("part_*.npz")):
                with np.load(path, allow_pickle=False) as a:
                    ids = a["sample_id"]
                    positions = np.searchsorted(values[:, 0], ids)
                    if (positions >= len(values)).any() or not np.array_equal(values[positions, 0], ids):
                        raise ValueError(f"Unknown {split} sequence ID")
                    if seen[positions].any() or len(np.unique(positions)) != len(positions):
                        raise ValueError(f"Duplicate {split} sequence ID")
                    if not np.array_equal(values[positions, 1], a["flow_counts"].sum(axis=1)) or not np.array_equal(
                            values[positions, 2] > 0, a["y_sequence"] > 0) or (a["right_id"] >= 0).any():
                        raise ValueError(f"{split} flow counts or labels differ from original metadata")
                    seen[positions] = True
            if not seen.all():
                raise ValueError(f"Incomplete {split} original coverage")
            path = dataset/f"flow_metadata_{split}.npz"
            payload = dict(sample_id=values[:, 0], total_flows=values[:, 1], attack_flows=values[:, 2],
                           labels=(values[:, 2] > 0).astype(np.uint8))
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if set(saved.files) != set(payload) or any(not np.array_equal(saved[k], v) for k, v in payload.items()):
                        raise ValueError(f"Existing metadata sidecar differs: {path}")
            else:
                np.savez_compressed(path, **payload)
            records[split] = dict(path=path.name, count=len(values), sha256=sha256(path))
    finally:
        connection.close()
    result = dict(complete=True, source="original_sequences metadata; no model predictions or fitting",
                  prepared=str(prepared.resolve()), splits=records)
    target = dataset/"flow_metadata.json"
    if target.exists() and json.loads(target.read_text()) != result:
        raise ValueError("Existing flow metadata index differs")
    if not target.exists():
        target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result
