"""Scheme2 remains usable after deleting retired Scheme1 outputs."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from CIC_new.evaluate_autoencoder import flow_composition
from CIC_new.preprocessing.flow_metadata import export_flow_metadata
from CIC_new.tests.fixtures import fixture_database

ROOT = Path(__file__).resolve().parents[2]


class SchemeTwoIndependenceTests(unittest.TestCase):
    def test_compact_metadata_without_reference_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared, data = root/"prepared", root/"dataset"
            prepared.mkdir()
            with sqlite3.connect(prepared/"manifest.sqlite") as db:
                db.execute("CREATE TABLE original_sequences(sequence_id INTEGER,flows INTEGER,attack_flows INTEGER,split TEXT)")
                db.executemany("INSERT INTO original_sequences VALUES(?,?,?,?)", [(3, 3, 0, "validation"), (8, 4, 1, "validation"), (9, 2, 1, "test")])
            db.close()
            for split, ids, counts, labels in (("validation", [8, 3], [[4, 0], [2, 1]], [1, 0]), ("test", [9], [[2, 0]], [1])):
                (data/split).mkdir(parents=True)
                np.savez(data/split/"part_000000.npz", sample_id=ids, flow_counts=counts, y_sequence=labels, right_id=np.full(len(ids), -1))
            export_flow_metadata(prepared, data)
            export_flow_metadata(prepared, data)
            class Store:
                folder = data
                split = "validation"
                arrays = dict(sample_id=np.array([8, 3]), y_sequence=np.array([1, 0]))
                manifest = dict(representation="scheme2")
                def __len__(self): return 2
            totals, attacks = flow_composition(Store())
            np.testing.assert_array_equal(totals, [4, 3])
            np.testing.assert_array_equal(attacks, [1, 0])
            with (data/"flow_metadata_validation.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                flow_composition(Store())

    def test_database_to_dataset_without_scheme1_and_stage_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db, out = root/"flows.sqlite", root/"scheme2"
            fixture_database(db)
            command = [sys.executable, str(ROOT/"CIC_new/prepare_scheme2_dataset.py"),
                "--db", str(db), "--out", str(out), "--embedding-steps", "2", "--validate-every", "1",
                "--batch-size", "8", "--category-dim", "4", "--bucket-dim", "8",
                "--shard-size", "17", "--val-limit", "16", "--device", "cpu", "--cpu-threads", "1"]
            for extra in ([], ["--resume"]):
                result = subprocess.run(command+extra, cwd=root, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            manifest = json.loads((out/"dataset/manifest.json").read_text())
            self.assertEqual(manifest["representation"], "scheme2")
            self.assertNotIn("paired_reference_dataset", manifest)
            self.assertTrue((out/"dataset/flow_metadata.json").is_file())
            command = [sys.executable, str(ROOT/"CIC_new/run_scheme2_experiment.py"),
                "--stage", "check", "--dataset", str(out/"dataset"), "--device", "cpu"]
            result = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("CHECK PASSED", result.stdout)
            # Execute new default train/all/evaluate paths with a tiny CPU budget.
            import CIC_new.run_scheme2_experiment as experiment
            original_train = experiment.train_temporal
            def short_train(args):
                self.assertEqual((args.epochs, args.steps_per_epoch, args.batch_size), (500, 500, 128))
                args.epochs, args.steps_per_epoch, args.batch_size = 1, 2, 8
                args.width, args.latent = 4, 3
                return original_train(args)
            args = SimpleNamespace(stage="all", run=None, dataset=out/"dataset", out=root/"ae",
                device="cpu", evaluation_batch_size=32)
            with patch.object(experiment, "train_temporal", side_effect=short_train):
                experiment.standalone(args)
                args.stage, args.out = "train", root/"ae_train_only"
                experiment.standalone(args)
            args.stage, args.run, args.out = "evaluate", root/"ae", None
            with patch.object(experiment, "train_temporal", side_effect=AssertionError("Evaluation must not train")):
                experiment.standalone(args)
            self.assertEqual(len(list((root/"ae/evaluations").glob("*/report.json"))), 2)
            # Generic trainer defaults to Scheme2 even with no --dataset flag.
            import CIC_new.train_autoencoder as generic
            with patch.object(sys, "argv", ["train_autoencoder.py", "--out", str(root/"generic"),
                    "--epochs", "1", "--steps-per-epoch", "2", "--batch-size", "8", "--device", "cpu"]), \
                    patch.object(experiment, "latest_dataset", return_value=out/"dataset"):
                generic.main()
            self.assertEqual(json.loads((root/"generic/config.json").read_text())["dataset_representation"], "scheme2")


if __name__ == "__main__":
    unittest.main()
