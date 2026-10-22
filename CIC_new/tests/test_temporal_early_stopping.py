"""Check stopping, cumulative improvements and actual best-weight restoration."""
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from CIC_new.modelling.train_autoencoder import EarlyStopping, train_temporal
from CIC_new.train_autoencoder import main as terminal_main, new_run_directory


class TemporalStoppingTests(unittest.TestCase):
    def test_output_directories_are_unique_even_with_identical_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(real_only=False, epochs=500, batch_size=128, seed=4266)
            with patch("CIC_new.train_autoencoder.ROOT", root), patch("CIC_new.train_autoencoder.datetime") as clock:
                clock.now.return_value = datetime(2026, 10, 22, tzinfo=timezone.utc)
                first = new_run_directory(args)
                (first / "existing.txt").write_text("first run")
                second = new_run_directory(args)
                self.assertNotEqual(first, second)
                self.assertIn("scheme1_ae_aug_e500_b128_seed4266_", first.name)
                self.assertEqual(list(second.iterdir()), [])
                self.assertEqual((first / "existing.txt").read_text(), "first run")
                args.real_only = True
                self.assertIn("scheme1_ae_real_", new_run_directory(args).name)

    def test_cumulative_relative_improvement_and_invalid_settings(self):
        stop = EarlyStopping(patience=2, min_delta=.01)
        for value, wait, expected in ((1., 0, False), (.996, 1, False),
                                      (.989, 0, False), (1., 1, False), (1., 2, True)):
            self.assertEqual(stop.update(value), expected)
            self.assertEqual(stop.wait, wait)
        for patience, delta in ((-1, .001), (2, -1.), (2, 1.), (2, float("nan"))):
            with self.assertRaises(ValueError):
                EarlyStopping(patience, delta)
        with self.assertRaises(ValueError):
            stop.update(float("nan"))

    def test_stopping_budget_and_best_checkpoint_restoration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "dataset"
            data.mkdir()
            rng = np.random.default_rng(2)
            kinds = np.array(["real_benign", "synthetic_benign", "real_malicious", "synthetic_malicious"] * 2)
            for split, n in (("train", 8), ("validation", 4)):
                folder = data / split
                folder.mkdir()
                labels = np.array([0, 0, 1, 1] * 2)[:n]
                np.savez_compressed(folder / "part_000000.npz",
                    X=rng.normal(size=(n, 10, 3)).astype(np.float32), mask=np.ones((n, 10), np.float32),
                    y_sequence=labels, sample_id=np.arange(n), sample_type=kinds[:n],
                    left_id=np.arange(n), right_id=np.full(n, -1), right_shift_buckets=np.zeros(n),
                    left_ip=np.full(n, "10.0.0.1"), right_ip=np.full(n, ""), primary_attack=np.full(n, "fixture"))
            (data / "manifest.json").write_text(json.dumps(dict(
                complete=True, format="model-independent-sequence-shards", feature_width=3, timesteps=10,
                embedding_checkpoint_sha256="fixture", shift_mode="none",
                total_samples=dict(train=8, validation=4, test=4))))
            # No test directory: an attempt to use test during fitting would fail.
            for name, patience, scores, best_epoch, epochs, reason in (
                ("plateau", 2, [2., 1., .9995, 1.01, .9995], 2, 3, "early_stopping"),
                ("disabled", 0, [2., 1., 1.1, 1.2, 1.], 1, 3, "max_epochs"),
                ("improving", 2, [2., 1., .9, .8, .8], 3, 3, "max_epochs")):
                with self.subTest(name=name):
                    out = root / name
                    snapshots = []
                    values = iter(scores)
                    def evaluate(model, store, ids, batch_size, device, return_vectors=False):
                        model.eval()
                        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                        return np.full(len(ids), next(values)), np.zeros((len(ids), 3)) if return_vectors else None
                    args = SimpleNamespace(dataset=data, out=out, device="cpu", seed=4266, cpu_threads=1,
                        batch_size=8, epochs=8 if name=="plateau" else 3, steps_per_epoch=1,
                        width=4, latent=3, log_every=1, val_limit=4, lr=.001, real_only=False,
                        patience=patience, min_delta=.001)
                    with patch("CIC_new.modelling.train_autoencoder.evaluate_temporal", side_effect=evaluate) as evaluator:
                        if name == "improving":
                            # Exercise the actual no --out entry path without touching real data or outputs.
                            command = ["train_autoencoder.py", "--dataset", str(data), "--device", "cpu",
                                       "--epochs", "3", "--steps-per-epoch", "1", "--batch-size", "8",
                                       "--width", "4", "--latent", "3", "--log-every", "1", "--cpu-threads", "1",
                                       "--patience", "2"]
                            with patch("CIC_new.train_autoencoder.ROOT", root), patch("sys.argv", command):
                                terminal_main()
                            out = next((root / "CIC_new/outputs/ae_runs").iterdir())
                        else:
                            train_temporal(args)
                    summary = json.loads((out / "summary.json").read_text())
                    self.assertEqual(json.loads((out / "run_status.json").read_text())["status"], "complete")
                    self.assertEqual(summary["epochs_completed"], epochs)
                    self.assertEqual(summary["steps"], epochs)
                    self.assertEqual(summary["best_epoch"], best_epoch)
                    self.assertEqual(summary["stopping_reason"], reason)
                    self.assertEqual(summary["plateau_detected"], reason=="early_stopping")
                    self.assertAlmostEqual(summary["restored_validation_mse"], summary["best_validation_mse"])
                    self.assertEqual(evaluator.call_count, epochs+2)
                    for key in snapshots[-1]:
                        torch.testing.assert_close(snapshots[-1][key], snapshots[best_epoch][key], rtol=0, atol=0)
                    self.assertFalse((data / "model_cache/test").exists())
            for error, expected_status in ((KeyboardInterrupt(), "interrupted"), (RuntimeError("fixture failure"), "failed")):
                with self.subTest(status=expected_status):
                    args.out = root / expected_status
                    with patch("CIC_new.modelling.train_autoencoder._train_temporal", side_effect=error):
                        with self.assertRaises(type(error)):
                            train_temporal(args)
                    status = json.loads((args.out / "run_status.json").read_text())
                    self.assertEqual(status["status"], expected_status)
                    self.assertFalse(status["complete"])
                    self.assertFalse((args.out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
