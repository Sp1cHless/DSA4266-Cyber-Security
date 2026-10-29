"""Temporal geometry, sequential batches and frozen Scheme2 export on fixtures."""
import hashlib
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from CIC_new import run_temporal_ablation as batch
from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing import split_policy as sp
from CIC_new.preprocessing.temporal import TemporalGrid
from CIC_new.preprocessing.run_preprocessing import prepare
from CIC_new.tests.fixtures import event, fixture_database

ROOT = Path(__file__).resolve().parents[2]


class TemporalTests(unittest.TestCase):
    def test_geometry_boundaries_and_window_purge(self):
        grid = TemporalGrid(200, 20)
        schema = bf.make_schema([], grid)
        values, mask, labels, counts, attacks = bf.aggregate(
            np.stack([event(0), event(19999), event(20000, label=1), event(199999)]), schema)
        self.assertEqual(values.shape, (10, 83))
        np.testing.assert_array_equal(np.flatnonzero(mask), [0, 1, 9])
        self.assertEqual((counts.sum(), attacks.sum(), labels.sum()), (4, 1, 1))
        with self.assertRaises(ValueError):
            bf.aggregate(np.stack([event(200000)]), schema)
        for durations in ((0, 10), (100, 30), (128, 1), (100.0, 10), (True, 1)):
            with self.assertRaises(ValueError):
                TemporalGrid(*durations)
        first, end = np.array([1, 200001]), np.array([199999, 210000])
        start, codes = np.array([0, 200000]), np.array([0, 1], np.uint8)
        self.assertEqual(sp.interval_overlap(first, end, start, codes, window_ms=200000).sum(), 0)
        end[0] = 200000
        self.assertEqual(sp.interval_overlap(first, end, start, codes, window_ms=200000).tolist(), [True, False])

    def test_full_matrix_preprocess_resume_then_complete_scheme2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db, out = root / "flows.sqlite", root / "batch"
            fixture_database(db)
            source_hash = hashlib.sha256(db.read_bytes()).hexdigest()
            command = [sys.executable, str(ROOT / "CIC_new/run_temporal_ablation.py"),
                       "--db", str(db), "--out", str(out), "--device", "cpu",
                       "--cpu-threads", "1", "--shard-size", "17", "--embedding-steps", "1",
                       "--validate-every", "1", "--batch-size", "8", "--val-limit", "16"]

            def run(extra=(), success=True):
                result = subprocess.run(command + list(extra), cwd=root, text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                return result

            # Reuse the existing baseline; only four new configurations are scheduled.
            cases = batch.load_cases(batch.DEFAULT_CONFIG)
            self.assertEqual([(c["sequence_seconds"], c["bucket_seconds"]) for c in cases],
                             [(200, 10), (200, 20), (400, 10), (400, 40)])
            self.assertEqual(run(["--dry-run"]).stdout.count("run_preprocessing"), 4)
            self.assertFalse(out.exists())
            run()
            state = batch.read_json(out / "batch.json")
            self.assertTrue(state["complete"])
            self.assertFalse(state["prediction_model_training_started"])
            self.assertFalse(list(out.rglob("*.pt")))
            self.assertEqual(len(state["cases"]), 4)
            original_splits = {}
            for case in state["settings"]["cases"]:
                item = state["cases"][case["name"]]
                prepared = Path(item["preprocess"]["path"])
                schema = batch.read_json(prepared / "schema.json")
                for key in ("sequence_seconds", "bucket_seconds", "timesteps"):
                    self.assertEqual(schema[key], case[key])
                notes = batch.read_json(prepared / "split_notes.json")
                self.assertEqual(notes["block_seconds"], 900)
                self.assertEqual(notes["embargo_seconds"], 100)
                self.assertEqual(notes["retained_interval_overlap_sequences"], 0)
                self.assertEqual(notes["adjacent_source_pairs_cross_split"], 0)
                connection = sqlite3.connect(prepared / "manifest.sqlite")
                try:
                    membership = connection.execute(
                        "SELECT source_ip,window_start_ms,flows,attack_flows,primary_attack,split "
                        "FROM original_sequences ORDER BY sequence_id").fetchall()
                    old = original_splits.setdefault(case["sequence_seconds"], membership)
                    self.assertEqual(membership, old)  # Bucket-only changes cannot alter the split.
                finally:
                    connection.close()
                for split in ("train", "validation", "test"):
                    for shard in (prepared / "splits" / split).glob("*.npz"):
                        with np.load(shard) as arrays:
                            self.assertEqual(arrays["numeric"].shape[1:], (case["timesteps"], 82))
                            np.testing.assert_array_equal(arrays["flow_counts"] > 0, arrays["mask"] > 0)
                            for field in ("protocol", "l7", "service"):
                                np.testing.assert_allclose(arrays["hist_" + field].sum(-1), arrays["mask"], atol=1e-6)
            files_before = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in out.rglob("*") if p.is_file() and p.name not in ("batch.json", "index.csv")}
            self.assertIn("reuse", run(["--resume"]).stdout)
            files_after = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in out.rglob("*") if p.is_file() and p.name not in ("batch.json", "index.csv")}
            self.assertEqual(files_before, files_after)
            run(success=False)
            run(["--resume", "--seed", "7"], success=False)
            self.assertTrue(batch.read_json(out / "batch.json")["complete"])
            # A later dataset stage reuses every raw preparation and trains only input encoders.
            run(["--resume", "--stage", "dataset"])
            state = batch.read_json(out / "batch.json")
            for case in state["settings"]["cases"]:
                item = state["cases"][case["name"]]
                self.assertEqual(sum(a["stage"] == "preprocess" for a in item["attempts"]), 1)
                exported = Path(item["dataset"]["path"]) / "dataset"
                manifest = batch.read_json(exported / "manifest.json")
                self.assertEqual(manifest["timesteps"], case["timesteps"])
                self.assertEqual(manifest["feature_width"], 65)
                self.assertEqual(manifest["shift_mode"], "none")
                self.assertTrue(batch.read_json(exported / "validation.json")["passed"])
                for split in ("validation", "test"):
                    self.assertFalse(any(k.startswith("synthetic") for k in manifest["counts"][split]))
                with np.load(next((exported / "train").glob("*.npz"))) as a:
                    self.assertEqual(a["X"].shape[1:], (case["timesteps"], 65))
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), source_hash)
            self.assertFalse((out / "RUNNING.lock").exists())

    def test_failed_stage_kept_and_retried_in_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db, config, out = root / "flows.sqlite", root / "cases.json", root / "batch"
            fixture_database(db)
            config.write_text(json.dumps({"cases": [{"sequence_seconds": 200, "bucket_seconds": 10}]}))
            args = SimpleNamespace(db=db, config=config, out=out, stage="preprocess", resume=False,
                       dry_run=False, keep_going=False, seed=4266, shard_size=17, device="cpu",
                       cpu_threads=1, batch_size=8, embedding_steps=1, validate_every=1,
                       val_limit=16, category_dim=16, bucket_dim=64)

            def interrupt(command, log):
                partial = Path(command[command.index("--out") + 1])
                (partial / "partial.txt").write_text("preserve this incomplete stage")
                raise RuntimeError("Injected interruption")

            with patch.object(batch, "execute", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "Injected interruption"):
                    batch.run(args)
            args.resume = True
            batch.run(args)
            item = next(iter(batch.read_json(out / "batch.json")["cases"].values()))
            self.assertEqual(len(item["attempts"]), 2)
            self.assertEqual((Path(item["attempts"][0]["path"]) / "partial.txt").read_text(),
                             "preserve this incomplete stage")
            self.assertTrue(item["preprocess"]["path"].endswith("preprocess_attempt_0002"))

    def test_csv_and_sqlite_share_parameterized_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db, source = root / "flows.sqlite", root / "flows.csv"
            fixture_database(db)
            connection = sqlite3.connect(db)
            try:
                cursor = connection.execute("SELECT * FROM network_flows ORDER BY rowid")
                with source.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([column[0] for column in cursor.description])
                    writer.writerows(cursor)
            finally:
                connection.close()
            prepare(root / "sql", db=db, sequence_seconds=200, bucket_seconds=20, shard_size=17)
            prepare(root / "csv", csv=source, sequence_seconds=200, bucket_seconds=20, shard_size=17)
            for path in (root / "sql/splits").rglob("*.npz"):
                other = root / "csv" / path.relative_to(root / "sql")
                with np.load(path) as left, np.load(other) as right:
                    self.assertEqual(set(left.files), set(right.files))
                    for key in left.files:
                        np.testing.assert_array_equal(left[key], right[key])


if __name__ == "__main__":
    unittest.main()
