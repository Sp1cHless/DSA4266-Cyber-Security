"""Artificial database-to-dataset checks; no real-data detector results."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from CIC_new.prepare_scheme2_dataset import validate_export
from CIC_new.preprocessing.donor_plan import make_plan, original_catalog
from CIC_new.preprocessing.run_preprocessing import prepare
from CIC_new.tests.fixtures import fixture_database

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_map(folder):
    import sqlite3
    connection = sqlite3.connect(Path(folder) / "manifest.sqlite")
    try:
        return dict(connection.execute("SELECT sequence_id,split FROM original_sequences"))
    finally:
        connection.close()


class SchemeTwoPreparationTests(unittest.TestCase):
    def test_database_export_resume_reuse_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db, out = root / "flows.sqlite", root / "scheme2"
            fixture_database(db)
            source_hash = hashlib.sha256(db.read_bytes()).hexdigest()
            command = [sys.executable, str(ROOT / "CIC_new/prepare_scheme2_dataset.py"),
                       "--db", str(db), "--out", str(out), "--embedding-steps", "2",
                       "--validate-every", "1", "--batch-size", "8", "--category-dim", "4",
                       "--bucket-dim", "64", "--shard-size", "17", "--val-limit", "16",
                       "--device", "cpu", "--cpu-threads", "1"]

            def run(arguments, success=True):
                result = subprocess.run(arguments, cwd=root, text=True, capture_output=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            # Default data path: no historical reference and no existing encoder.
            run(command)
            workflow = read_json(out / "workflow.json")
            manifest = read_json(out / "dataset/manifest.json")
            self.assertTrue(workflow["complete"])
            self.assertFalse(workflow["prediction_model_training_started"])
            self.assertEqual(manifest["representation"], "scheme2")
            self.assertEqual(manifest["feature_width"], 65)
            self.assertNotIn("paired_reference_dataset", manifest)
            report = read_json(out / "dataset/validation.json")
            self.assertTrue(report["passed"])
            self.assertEqual(report["all_exported_samples_checked"], sum(manifest["total_samples"].values()))
            self.assertTrue((out / "dataset/flow_metadata.json").is_file())
            plan = make_plan(out / "preprocessed", original_catalog(out / "preprocessed"), 4267)
            with np.load(out / "dataset/donor_plan.npz", allow_pickle=False) as saved:
                for key, expected in plan.items():
                    np.testing.assert_array_equal(saved[key], expected)
            for split in ("validation", "test"):
                self.assertFalse(any(key.startswith("synthetic") for key in manifest["counts"][split]))

            hashes = {p.relative_to(out / "dataset").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (out / "dataset").rglob("*") if p.is_file()}
            resumed = run(command + ["--resume"])
            self.assertIn("reuse completed stage", resumed.stdout)
            self.assertEqual(hashes, {p.relative_to(out / "dataset").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (out / "dataset").rglob("*") if p.is_file()})
            run(command, success=False)
            run(command + ["--resume", "--seed", "7"], success=False)

            # Reuse an exact completed preparation/encoder in a fresh export.
            reuse_command = command.copy()
            reused = root / "reuse"
            reuse_command[reuse_command.index("--out") + 1] = str(reused)
            run(reuse_command + ["--prepared", str(out / "preprocessed"),
                                 "--bucket-checkpoint", str(out / "bucket_embeddings/best.pt")])
            self.assertFalse((reused / "preprocessed").exists())
            self.assertFalse((reused / "bucket_embeddings").exists())
            self.assertEqual(read_json(reused / "dataset/manifest.json")["counts"], manifest["counts"])
            self.assertEqual((reused / "dataset/donor_plan.npz").read_bytes(), (out / "dataset/donor_plan.npz").read_bytes())
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), source_hash)

            # The unchanged train loader consumes the newly exported contract.
            from train.common.data import Scheme2, check_disjoint
            stores = []
            try:
                dataset = Scheme2(out / "dataset", cache=root / "cache")
                for split in ("train", "validation", "test"):
                    stores.append(dataset.load(split))
                for left, right in ((0, 1), (0, 2), (1, 2)):
                    check_disjoint(stores[left], stores[right])
            finally:
                for store in stores:
                    store.close()

            # Corrupt a disposable fixture shard and require arithmetic validation to fail.
            path = next((out / "dataset/train").glob("part_*.npz"))
            with np.load(path, allow_pickle=False) as saved:
                payload = {key: saved[key] for key in saved.files}
            payload["X"][0, 0, 0] += 1
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "vector addition/L2"):
                validate_export(out / "dataset", out / "encoded_sequences", donor_plan=out / "donor_plan.npz")
            run(command + ["--resume"], success=False)
            self.assertFalse(read_json(out / "workflow.json")["complete"])
            self.assertFalse(read_json(out / "dataset/validation.json")["passed"])

    def test_training_holdout_keeps_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = root / "flows.sqlite"
            fixture_database(db)
            control, holdout = root / "control", root / "holdout"
            prepare(control, db=db, seed=4266, shard_size=17)
            prepare(holdout, db=db, seed=4266, shard_size=17, holdout=("Beta",))
            before, after = split_map(control), split_map(holdout)
            self.assertEqual(set(before), set(after))

            # Fold assignment is untouched; only training membership of the declared type changes.
            changed = {sid for sid, fold in before.items() if after[sid] != fold}
            self.assertTrue(changed)
            self.assertTrue(all(before[sid] == "train" and after[sid] == "excluded" for sid in changed))
            for sid, fold in before.items():
                if fold in ("validation", "test"):
                    self.assertEqual(after[sid], fold)

            notes = read_json(holdout / "split_notes.json")
            self.assertEqual(notes["train_holdout_types"], ["Beta"])
            entry = notes["train_holdout"][0]
            self.assertEqual(entry["attack_type"], "Beta")
            self.assertGreater(entry["train_sequences_removed"], 0)
            self.assertGreater(entry["validation_sequences_primary"] + entry["test_sequences_primary"], 0)
            self.assertGreaterEqual(entry["validation_sequences_containing_type"], entry["validation_sequences_primary"])
            self.assertGreaterEqual(entry["test_sequences_containing_type"], entry["test_sequences_primary"])
            coverage = {row["attack_type"]: row for row in notes["coverage"]}
            self.assertEqual(coverage["Beta"]["train_sequences"], 0)
            self.assertTrue(coverage["Beta"]["held_out_from_training"])
            self.assertGreater(coverage["Alpha"]["train_sequences"], 0)
            self.assertIn("train_holdout_residual", notes)

            control_manifest = read_json(control / "manifest.json")
            holdout_manifest = read_json(holdout / "manifest.json")
            self.assertEqual(holdout_manifest["train_holdout_attacks"], ["Beta"])
            self.assertLess(holdout_manifest["counts"]["train"], control_manifest["counts"]["train"])

            with self.assertRaisesRegex(ValueError, "absent from the source"):
                prepare(root / "missing", db=db, seed=4266, shard_size=17, holdout=("Absent",))

            # A holdout cannot be retrofitted to an existing preparation.
            command = [sys.executable, str(ROOT / "CIC_new/prepare_scheme2_dataset.py"),
                       "--db", str(db), "--out", str(root / "reuse"), "--prepared", str(holdout),
                       "--holdout-train-attacks", "Beta", "--embedding-steps", "2", "--validate-every", "1",
                       "--batch-size", "8", "--category-dim", "4", "--bucket-dim", "64", "--shard-size", "17",
                       "--val-limit", "16", "--device", "cpu", "--cpu-threads", "1"]
            result = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("cannot be added to an existing preparation", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
