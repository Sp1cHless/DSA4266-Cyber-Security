"""Reuse original split shards and append train-scaled distribution statistics."""
from collections import defaultdict
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.distribution_features import NAMES, LOG_COLUMNS, BUNDLE, aggregate, logged
from train.common.io import fresh, read_json, write_json, sha256


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def matrix_for_ids(work, ids, starts):
    groups = defaultdict(list)
    fields = ",".join(bf.EVENT_COLUMNS)
    for offset in range(0, len(ids), 800):
        chosen = list(map(int, ids[offset:offset + 800]))
        query = f"SELECT sequence_id,{fields} FROM events WHERE sequence_id IN (" + ",".join("?" for _ in chosen) + ") ORDER BY sequence_id,flow_start_milliseconds,raw_rowid"
        for row in work.execute(query, chosen):
            groups[int(row[0])].append(row[1:])
    output = []
    for sid in ids:
        events = np.asarray(groups[int(sid)], np.int64)
        if not len(events):
            raise ValueError("Missing staged flows for " + str(sid))
        output.append(aggregate(bf.local_events(events, starts[int(sid)])))
    return np.stack(output)


def prepare(source, out, extended):
    source, out = Path(source).resolve(), Path(out).resolve()
    original = read_json(source / "manifest.json")
    schema = read_json(source / "schema.json")
    if (not original.get("complete") or original.get("version") != 3 or schema.get("timesteps") != 10
            or schema.get("sequence_seconds") != 100 or schema.get("bucket_seconds") != 10
            or schema["numeric_names"] != bf.NUMERIC_NAMES):
        raise ValueError("Use the completed original 100s/10s, 82-numeric preparation")
    fresh(out)
    write_json(out / "stage_status.json", {"status": "running"})
    source_db = readonly(source / "manifest.sqlite")
    target_db = sqlite3.connect(out / "manifest.sqlite")
    work = readonly(source / "work.sqlite") if extended else None
    try:
        source_db.backup(target_db)
        target_db.execute("DROP TABLE IF EXISTS v3_locations")
        target_db.execute("DROP TABLE IF EXISTS v3_raw_locations")
        target_db.execute("CREATE TABLE v3_locations(sample_id INTEGER PRIMARY KEY,split TEXT,shard TEXT,row_index INTEGER)")
        starts = dict(source_db.execute("SELECT sequence_id,window_start_ms FROM original_sequences WHERE split IN ('train','validation')"))
        moments = bf.Moments(len(NAMES))
        raw_records = []
        for split in ("train", "validation"):
            (out / "splits" / split).mkdir(parents=True)
            files = sorted((source / "splits" / split).glob("part_*.npz"))
            if not files:
                raise ValueError("Missing source split: " + split)
            if extended:
                (out / "extension_raw" / split).mkdir(parents=True)
                for i, file in enumerate(files):
                    with np.load(file, allow_pickle=False) as a:
                        values = matrix_for_ids(work, a["sample_id"], starts)
                        active = a["mask"].astype(bool)
                        if np.any(values[~active] != 0):
                            raise AssertionError("Feature/mask geometry differs from the original")
                        if split == "train":
                            moments.add(logged(values[active]))
                        path = out / "extension_raw" / split / file.name
                        np.savez_compressed(path, numeric=values, sample_id=a["sample_id"])
                        raw_records.append({"path": str(path.relative_to(out)), "sha256": sha256(path)})
                    if (i + 1) % 10 == 0 or i + 1 == len(files):
                        print(f"[distribution {split}] {i+1}/{len(files)} original shards", flush=True)
        scaler = read_json(source / "scaler.json")
        added_scaler = moments.finish() if extended else None
        if extended:
            if int(scaler["n"]) != int(added_scaler["n"]):
                raise AssertionError("New statistics used a different training active-bucket population")
            scaler = {**scaler, **{k: scaler[k] + added_scaler[k] for k in ("mean", "scale", "constant")}}
            schema = {**schema, "numeric_names": schema["numeric_names"] + NAMES,
                      "log1p_columns": schema["log1p_columns"] + [82 + i for i in LOG_COLUMNS],
                      "feature_bundle": BUNDLE}
        counts, records = {}, []
        for split in ("train", "validation"):
            counts[split] = 0
            for file in sorted((source / "splits" / split).glob("part_*.npz")):
                with np.load(file, allow_pickle=False) as a:
                    payload = {k: a[k] for k in a.files}
                if extended:
                    with np.load(out / "extension_raw" / split / file.name, allow_pickle=False) as added:
                        np.testing.assert_array_equal(added["sample_id"], payload["sample_id"])
                        normalized = (logged(added["numeric"]) - added_scaler["mean"]) / added_scaler["scale"]
                        normalized[payload["mask"] == 0] = 0
                        payload["numeric"] = np.concatenate((payload["numeric"], normalized.astype(np.float32)), axis=-1)
                    np.savez_compressed(out / "splits" / split / file.name, **payload)
                else:
                    shutil.copy2(file, out / "splits" / split / file.name)
                path = out / "splits" / split / file.name
                records.append({"path": str(path.relative_to(out)), "sha256": sha256(path)})
                ids = payload["sample_id"]
                target_db.executemany("INSERT INTO v3_locations VALUES(?,?,?,?)",
                    [(int(sid), split, file.name, i) for i, sid in enumerate(ids)])
                counts[split] += len(ids)
            expected = source_db.execute("SELECT COUNT(*) FROM original_sequences WHERE split=?", (split,)).fetchone()[0]
            if counts[split] != expected:
                raise AssertionError("The original cohort was not fully preserved")
        target_db.commit()
        write_json(out / "schema.json", schema)
        write_json(out / "scaler.json", scaler)
        write_json(out / "v3_counts.json", counts)
        write_json(out / "manifest.json", {**original, "counts": counts, "splits": ["train", "validation"],
            "feature_experiment": {"bundle": BUNDLE if extended else "original_82_control", "base_preparation": str(source),
                "base_manifest_sha256": sha256(source / "manifest.json"), "old_features_preserved_exactly": True,
                "split_recomputed": False, "test_features_read": False}, "complete": True})
        write_json(out / "extension_audit.json", {"complete": True, "extended": extended, "numeric_width": len(schema["numeric_names"]),
            "shards": records, "raw_extension_shards": raw_records, "metadata_sha256": sha256(out / "manifest.sqlite"),
            "schema_sha256": sha256(out / "schema.json"), "scaler_sha256": sha256(out / "scaler.json"),
            "new_scaler_population": "all real retained training active buckets only", "test_features_read": False,
            "metadata_note": "Original all-split IDs/labels retained only for original donor ownership and synthetic ID allocation."})
        write_json(out / "stage_status.json", {"status": "complete"})
    except Exception as error:
        write_json(out / "stage_status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        source_db.close()
        target_db.close()
        if work is not None:
            work.close()
    return out
