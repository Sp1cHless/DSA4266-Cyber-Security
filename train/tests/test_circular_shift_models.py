"""Artificial-data coverage for four-model shift training, reports and frozen replay."""
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train.common.data import flat_inputs
from train.common.experiment import load_model
from train.common.io import read_json
from train.evaluate import main as evaluate_model
from train.run_circular_shift import main as run_batch
from train.tests.fixture import make_fixture


class ArraySplit(dict):
    def __len__(self):
        return len(self["x"])


class AllModelShiftTests(unittest.TestCase):
    def test_actual_script_and_module_entry_points(self):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = make_fixture(root / "dataset")
            common = ["--dataset", str(dataset), "--cache", str(root / "cache"),
                      "--device", "cpu", "--threads", "1", "--epochs", "1", "--rounds", "2"]
            variants = (
                ([str(project / "train" / "run_circular_shift.py")], root, ["--shift-only"]),
                (["-m", "train.run_circular_shift"], project, []),
            )
            for index, (entry, cwd, flags) in enumerate(variants):
                with self.subTest(entry=entry):
                    batch = root / f"batch{index}"
                    result = subprocess.run([sys.executable, *entry, *common, *flags, "--out", str(batch)],
                                            cwd=cwd, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(read_json(batch / "status.json")["status"], "complete")
                    self.assertEqual(len(read_json(batch / "status.json")["completed_runs"]), 4 if flags else 8)
                    # Direct frozen evaluation must also import the installed XGBoost library.
                    replay = subprocess.run([sys.executable, str(project / "train" / "evaluate.py"),
                                             "--run", str(batch / "xgboost" / "start5"),
                                             "--cache", str(root / "cache"), "--device", "cpu", "--threads", "1"],
                                            cwd=root, capture_output=True, text=True, timeout=120)
                    self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)

    def test_flattened_columns_and_neural_inputs_follow_the_same_bucket_order(self):
        torch.set_num_threads(1)
        x = np.arange(3 * 10 * 65, dtype=np.float32).reshape(3, 10, 65) / 1000
        mask = np.tile(np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32), (3, 1))
        x *= mask[..., None]
        split = ArraySplit(x=x, mask=mask)
        originals = x.copy(), mask.copy()
        for start in range(10):
            order = list(range(start, 10)) + list(range(start))
            expected = np.concatenate((x[:, order].reshape(3, -1), mask[:, order]), axis=1)
            np.testing.assert_array_equal(flat_inputs(split, start), expected)
        for name in ("cnn", "hybrid"):
            model_type = importlib.import_module(f"train.{name}.model").Model
            baseline = model_type(dropout=0).eval()
            for start in range(10):
                with self.subTest(model=name, start=start):
                    model = model_type(dropout=0, circular_shift_start=start).eval()
                    model.load_state_dict(baseline.state_dict())
                    order = list(range(start, 10)) + list(range(start))
                    with torch.no_grad():
                        actual = model(torch.from_numpy(x), torch.from_numpy(mask))
                        expected = baseline(torch.from_numpy(x[:, order]), torch.from_numpy(mask[:, order]))
                    torch.testing.assert_close(actual, expected)
            for start in (-1, 10, 0.5, True):
                with self.assertRaisesRegex(ValueError, "integer from 0 to 9"):
                    model_type(circular_shift_start=start)
        np.testing.assert_array_equal(x, originals[0])
        np.testing.assert_array_equal(mask, originals[1])
        with self.assertRaises(ValueError):
            flat_inputs(split, 10)

    def test_paired_cpu_batch_reports_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = make_fixture(root / "dataset")
            test_shard = dataset / "test" / "part_000000.npz"
            hidden = test_shard.with_suffix(".hidden")
            test_shard.rename(hidden)
            # Exercise default family list, default offset and auto CPU fallback.
            command = ["--dataset", str(dataset), "--cache", str(root / "cache"), "--threads", "1",
                       "--epochs", "1", "--rounds", "3", "--batch-size", "32", "--out", str(root / "batch")]
            with patch("torch.cuda.is_available", return_value=False):
                batch = run_batch(command)
            hidden.rename(test_shard)
            plan = read_json(batch / "plan.json")
            self.assertEqual(len(plan["runs"]), 8)
            self.assertEqual(read_json(batch / "status.json")["status"], "complete")
            report = read_json(batch / "comparison.json")
            self.assertTrue(report["paired_controls"])
            self.assertFalse(report["test_used"])
            self.assertEqual(len(report["rows"]), 4)
            self.assertEqual(len(report["per_attack"]), 4)
            for row in report["rows"]:
                self.assertAlmostEqual(row["delta_ap_pp"], (row["shifted_ap"] - row["baseline_ap"]) * 100)
                name = row["model"]
                run = batch / row["shifted_run"]
                config = read_json(run / "config.json")
                audit = read_json(run / "data_audit.json")
                self.assertFalse(audit["test_opened"])
                self.assertEqual(audit["input_transform"]["bucket_order"], [5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
                with np.load(batch / row["baseline_run"] / "predictions.npz") as a, np.load(run / "predictions.npz") as b:
                    np.testing.assert_array_equal(a["sample_id"], b["sample_id"])
                    np.testing.assert_array_equal(a["y"], b["y"])
                model = load_model(name, config, run, "cpu", 1)
                if name == "xgboost":
                    self.assertEqual(model.attr("circular_shift_start"), "5")
                    bad = {**config, "input_transform": {"name": "identity"}}
                    with self.assertRaisesRegex(ValueError, "circular-shift mismatch"):
                        load_model(name, bad, run, "cpu", 1)
                    # Simulate a historical chronological model with no transform attribute.
                    old_run = batch / name / "start0"
                    old_config = read_json(old_run / "config.json")
                    old_model = load_model(name, old_config, old_run, "cpu", 1)
                    old_model.set_attr(circular_shift_start=None)
                    legacy = root / "legacy.ubj"
                    old_model.save_model(legacy)
                    importlib.import_module("train.xgboost.model").load(legacy, "cpu", 1, 0)
                else:
                    self.assertEqual(model.circular_shift_start, 5)
                for split in ("validation", "test"):
                    output = evaluate_model(["--run", str(run), "--cache", str(root / "cache"),
                                             "--device", "cpu", "--threads", "1", "--split", split])
                    self.assertEqual(read_json(output / "summary.json")["input_transform"], config["input_transform"])
                    if split == "validation":
                        with np.load(run / "predictions.npz") as a, np.load(output / "predictions.npz") as b:
                            np.testing.assert_allclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)
            with patch("train.run_circular_shift.train_model", side_effect=AssertionError("Must not train")):
                output = run_batch(["--summarize", str(batch)])
            self.assertEqual(read_json(output / "comparison.json")["rows"], report["rows"])
            with self.assertRaises(FileExistsError):
                run_batch(command)
            # Budget mismatch must not be silently presented as a matched shift effect.
            from train.common.io import write_json
            config_path = batch / "cnn" / "start5" / "config.json"
            changed = read_json(config_path)
            changed["training"]["epochs"] += 1
            write_json(config_path, changed)
            with self.assertRaisesRegex(ValueError, "Unmatched architecture, training settings or device"):
                run_batch(["--summarize", str(batch)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_shift_only_cuda_batch_and_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = make_fixture(root / "dataset")
            batch = run_batch(["--dataset", str(dataset), "--cache", str(root / "cache"),
                               "--device", "cuda", "--threads", "1", "--epochs", "1", "--rounds", "2",
                               "--circular-shift-start", "3", "--shift-only", "--out", str(root / "batch")])
            report = read_json(batch / "comparison.json")
            self.assertFalse(report["paired_controls"])
            self.assertEqual(len(read_json(batch / "plan.json")["runs"]), 4)
            for row in report["rows"]:
                self.assertIsNone(row["baseline_ap"])
                self.assertIsNone(row["delta_ap_pp"])
                run = batch / row["shifted_run"]
                output = evaluate_model(["--run", str(run), "--cache", str(root / "cache"),
                                         "--device", "cuda", "--threads", "1"])
                with np.load(run / "predictions.npz") as a, np.load(output / "predictions.npz") as b:
                    np.testing.assert_allclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
