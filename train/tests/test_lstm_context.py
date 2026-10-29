"""Focused checks for the frozen LSTM context interventions."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from analysis.diagnose_lstm_context import intervene
from train.lstm.model import Model
from train.lstm_context.model import Model as AttentionModel
from train.common import torch_runner
from train.tests.fixture import make_fixture
from train.common.io import read_json, write_json, sha256
from train.common.experiment import main as train_model
from train.run_lstm_context_ablation import main as run_ablation


class ContextTest(unittest.TestCase):
    def test_mechanism_identities_and_shift(self):
        torch.manual_seed(4266)
        for start in (0, 5):
            model = Model(hidden_size=8, dropout=0, circular_shift_start=start).eval()
            x = torch.randn(3, 10, 65)
            mask = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            x *= mask[..., None]
            attacks = mask.clone()
            attacks[1, 1] = 0
            attacks[2] = 0
            result = intervene(model, x, mask, attacks)
            with torch.no_grad():
                np.testing.assert_allclose(result["original"], model(x, mask).numpy(), atol=1e-7)
            for name in ("oracle_mean_attack", "oracle_pool_attack", "oracle_remove_background"):
                np.testing.assert_allclose(result[name][[0, 2]], result["original"][[0, 2]], atol=1e-7)
            rotated = torch.roll(mask, -start, 1)
            final_is_active = rotated[:, -1].bool().numpy()
            np.testing.assert_allclose(result["last_active_final"][final_is_active], result["original"][final_is_active], atol=1e-7)
            self.assertTrue(all(np.isfinite(value).all() for value in result.values()))

    def test_attention_initial_identity_and_fit_reload(self):
        for steps in (10, 20, 40):
            torch.manual_seed(4266)
            baseline = Model(hidden_size=8, dropout=0).eval()
            torch.manual_seed(4266)
            attention = AttentionModel(hidden_size=8, dropout=0).eval()
            mask = (torch.arange(steps)[None, :] % 3 == 0).float().repeat(4, 1)
            x = torch.randn(4, steps, 65) * mask[..., None]
            with torch.no_grad():
                torch.testing.assert_close(attention(x, mask), baseline(x, mask), atol=1e-7, rtol=1e-6)
            self.assertEqual(sum(p.numel() for p in attention.parameters()) - sum(p.numel() for p in baseline.parameters()), 9)

        rng = np.random.default_rng(4266)

        class ArtificialSplit(dict):
            def __len__(self):
                return len(self["y"])

        split = ArtificialSplit(x=rng.normal(size=(24, 10, 65)).astype(np.float32),
                                mask=np.ones((24, 10), np.float32), y=np.arange(24) % 2)
        config = {"architecture": {"hidden_size": 8, "dropout": 0},
                  "training": {"epochs": 1, "patience": 1, "batch_size": 8,
                               "learning_rate": .001, "weight_decay": .0001}}
        with tempfile.TemporaryDirectory() as folder:
            fitted, result = torch_runner.fit("lstm_context", split, split, np.ones(24, np.float32),
                                              config, "cpu", 1, 4266, Path(folder))
            loaded = torch_runner.load("lstm_context", config, Path(folder) / "model.pt", "cpu")
            self.assertEqual(result["epochs_completed"], 1)
            self.assertGreater(float(fitted.attention.weight.detach().abs().sum()), 0)
            np.testing.assert_array_equal(torch_runner.predict(fitted, split, "cpu"),
                                          torch_runner.predict(loaded, split, "cpu"))

    def test_runner_train_validation_only_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_fixture(root / "dataset")
            (dataset / "test").rename(dataset / "test_unavailable")
            manifest = read_json(dataset / "manifest.json")
            for split in ("train", "validation"):
                file = dataset / split / "part_000000.npz"
                with np.load(file, allow_pickle=False) as saved:
                    payload = {key: saved[key] for key in saved.files}
                positive = np.flatnonzero(payload["y_sequence"])
                payload["primary_attack"] = payload["primary_attack"].astype("U80")
                payload["primary_attack"][positive[:len(positive)//2]] = "Infilteration"
                payload["primary_attack"][positive[len(positive)//2:]] = "Brute_Force_-Web"
                np.savez_compressed(file, **payload)
                next(r for r in manifest["shards"] if r["path"] == f"{split}/part_000000.npz")["sha256"] = sha256(file)
            write_json(dataset / "manifest.json", manifest)
            config = {"architecture": {"hidden_size": 8, "dropout": 0},
                      "training": {"epochs": 1, "patience": 1, "batch_size": 32,
                                   "learning_rate": .001, "weight_decay": .0001}}
            write_json(root / "config.json", config)
            control = train_model("lstm", ["--dataset", str(dataset), "--config", str(root / "config.json"),
                "--out", str(root / "control"), "--device", "cpu", "--threads", "1", "--cache", str(root / "cache")])
            before = sha256(control / "model.pt")
            with patch("train.run_lstm_context_ablation.ROOT", root):
                output = run_ablation(["--reference-run", str(control), "--out", str(root / "attention_batch")])
            result = read_json(output / "comparison.json")
            self.assertFalse(result["test_used"])
            self.assertEqual(result["attention_parameters"] - result["baseline_parameters"], 9)
            self.assertEqual(len(result["rows"]), 4)
            self.assertEqual(sha256(control / "model.pt"), before)
            self.assertFalse(read_json(output / "attention/data_audit.json")["test_opened"])
            self.assertEqual(read_json(output / "attention/config.json")["training"], config["training"])


if __name__ == "__main__":
    unittest.main()
