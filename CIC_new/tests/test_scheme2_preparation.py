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
from CIC_new.tests.fixtures import fixture_database

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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
            # Historical workflow settings did not record explicit temporal flags.
            historical = read_json(out / "workflow.json")
            historical["settings"].pop("sequence_seconds")
            historical["settings"].pop("bucket_seconds")
            historical["settings"].pop("splits")
            (out / "workflow.json").write_text(json.dumps(historical), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
