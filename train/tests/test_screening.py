"""Integration checks cover fitting, checkpoint reload, comparisons and leakage guards."""
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import torch

from train.common.data import Scheme2, training_weights
from train.common.experiment import main as train_model, load_model, score, resolve_device
from train.common.io import read_json, environment
from train.common.metrics import choose_thresholds, at_threshold
from train.compare import compare, main as compare_cli
from train.evaluate import main as evaluate_model
from train.tests.fixture import make_fixture


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = make_fixture(self.root / "dataset")
        self.cache = self.root / "cache"

    def test_tied_scores_respect_false_alarm_budget(self):
        y = np.array([0] * 1000 + [1] * 3)
        scores = np.array([0.8] * 20 + [0.1] * 980 + [0.95, 0.85, 0.8])
        thresholds = choose_thresholds(y, scores)
        for name in ("validation_fpr_1pct", "validation_fpr_0.1pct"):
            result = at_threshold(y, scores, thresholds[name])
            self.assertLessEqual(result["fpr"], 0.001)
            self.assertEqual(result["tp"], 2)
        # A constant scorer must reject all to satisfy these budgets, with a finite threshold.
        thresholds = choose_thresholds(y, np.ones_like(scores))
        self.assertTrue(np.isfinite(thresholds["validation_fpr_1pct"]))
        self.assertEqual(at_threshold(y, np.ones_like(scores), thresholds["validation_fpr_1pct"])["fp"], 0)

    def test_loader_weights_and_corruption_rejection(self):
        source = Scheme2(self.data, self.cache)
        train = source.load("train")
        try:
            weights = training_weights(train)
            totals = [weights[train["types"] == g].sum() for g in train.info["groups"]]
            np.testing.assert_allclose(totals, np.repeat(len(train) / 4, 4))
        finally:
            train.close()
        # Read the ready cache a second time, then reject a changed source shard.
        source.load("train").close()
        shard = self.data / "train" / "part_000000.npz"
        with shard.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            source.load("train")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Scheme2(self.data, self.root / "fresh_cache").load("train")

    def test_four_cpu_models_frozen_evaluation_and_comparison(self):
        screen = self.root / "screen"
        # Missing test shards must not prevent any training or validation comparison.
        test_shard = self.data / "test" / "part_000000.npz"
        hidden = test_shard.with_suffix(".hidden")
        test_shard.rename(hidden)
        # Each member runs one entry point. Simulate default auto on a non-CUDA machine.
        with patch("torch.cuda.is_available", return_value=False):
            for name in ("xgboost", "lstm", "cnn", "hybrid"):
                train_model(name, ["--dataset", str(self.data), "--cache", str(self.cache),
                                  "--out", str(screen / name), "--threads", "1"] +
                            (["--rounds", "3"] if name == "xgboost" else ["--epochs", "1"]))
        hidden.rename(test_shard)
        # Compare copied summary files only, independent of all original dataset/model paths.
        collected = self.root / "collected"
        for name in ("xgboost", "lstm", "cnn", "hybrid"):
            (collected / name).mkdir(parents=True)
            shutil.copy2(screen / name / "summary.json", collected / name / "summary.json")
        compare_cli(["--root", str(collected), "--out", str(screen / "comparison")])
        ranking = read_json(screen / "comparison" / "comparison.json")
        self.assertEqual(len(ranking["rows"]), 4)
        self.assertTrue(ranking["complete_screen"])
        for name in ("xgboost", "lstm", "cnn", "hybrid"):
            run = screen / name
            summary = read_json(run / "summary.json")
            self.assertFalse(summary["test_used"])
            validation_eval = evaluate_model(["--run", str(run), "--cache", str(self.cache), "--device", "cpu"])
            with np.load(run / "predictions.npz") as a, np.load(validation_eval / "predictions.npz") as b:
                np.testing.assert_allclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)
            test_eval = evaluate_model(["--run", str(run), "--cache", str(self.cache), "--split", "test", "--device", "cpu"])
            self.assertTrue(read_json(test_eval / "summary.json")["test_used"])
        with self.assertRaises(FileExistsError):
            train_model("cnn", ["--dataset", str(self.data), "--out", str(screen / "cnn"), "--device", "cpu"])
        # Corrupt compatibility metadata, not the saved model, and ensure ranking refuses it.
        with patch("train.compare.read_json", side_effect=[
            {**read_json(screen / name / "summary.json"), "seed": i}
            for i, name in enumerate(("xgboost", "lstm", "cnn", "hybrid"))]):
            with self.assertRaisesRegex(ValueError, "seed"):
                compare([screen / n for n in ("xgboost", "lstm", "cnn", "hybrid")], self.root / "bad_comparison")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_models_and_default_device_selection(self):
        self.assertEqual(resolve_device("cnn", "auto"), "cuda")
        self.assertEqual(resolve_device("xgboost", "auto"), "cpu")
        for name in ("xgboost", "lstm", "cnn", "hybrid"):
            run = train_model(name, ["--dataset", str(self.data), "--cache", str(self.cache),
                                    "--device", "cuda", "--threads", "1", "--out", str(self.root / name)] +
                              (["--rounds", "2"] if name == "xgboost" else ["--epochs", "1"]))
            config = read_json(run / "config.json")
            data = Scheme2(self.data, self.cache).load("validation")
            try:
                model = load_model(name, config, run, "cuda", 1)
                predictions = score(name, model, data, "cuda", 1)
                with np.load(run / "predictions.npz") as saved:
                    np.testing.assert_allclose(predictions, saved["score"], rtol=1e-6, atol=1e-7)
            finally:
                data.close()

    def test_cpu_mps_selection_and_no_git_environment(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            for name in ("xgboost", "cnn", "lstm", "hybrid"):
                self.assertEqual(resolve_device(name, "auto"), "cpu")
                self.assertEqual(resolve_device(name, "cpu"), "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA"):
                resolve_device("cnn", "cuda")
            with self.assertRaisesRegex(ValueError, "MPS"):
                resolve_device("lstm", "mps")
        with patch("torch.backends.mps.is_available", return_value=True):
            for name in ("cnn", "lstm", "hybrid"):
                self.assertEqual(resolve_device(name, "mps"), "mps")
            with self.assertRaisesRegex(ValueError, "XGBoost"):
                resolve_device("xgboost", "mps")
        with patch("train.common.io.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(environment()["git_commit"])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable; requires a Mac")
    def test_actual_mps_training_and_reload(self):
        for name in ("cnn", "lstm", "hybrid"):
            run = train_model(name, ["--dataset", str(self.data), "--cache", str(self.cache),
                                    "--device", "mps", "--epochs", "1", "--out", str(self.root / name)])
            evaluation = evaluate_model(["--run", str(run), "--cache", str(self.cache), "--device", "mps"])
            with np.load(run / "predictions.npz") as a, np.load(evaluation / "predictions.npz") as b:
                np.testing.assert_allclose(a["score"], b["score"], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
