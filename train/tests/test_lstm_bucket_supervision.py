"""Focused checks for mask-safe supervision and unchanged baseline behavior."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from train.common.data import Scheme2
from train.common.experiment import main as train_model, load_model
from train.common.io import read_json, write_json, sha256
from train.common import torch_runner
from train.lstm.model import Model as Baseline
from train.lstm_bucket.model import Model
from train.lstm_bucket.runner import REDUCTION, bucket_bce_per_sequence, load_bucket_targets
from train.run_lstm_bucket_ablation import main as run_ablation
from train.tests.fixture import make_fixture


class BucketSupervisionTest(unittest.TestCase):
    def test_initialization_masked_loss_and_shared_gradient(self):
        torch.manual_seed(4266)
        baseline = Baseline(hidden_size=8, dropout=.2)
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(4266)
        auxiliary = Model(hidden_size=8, dropout=.2)
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, auxiliary.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in auxiliary.parameters()) - sum(p.numel() for p in baseline.parameters()), 9)
        baseline.eval()
        auxiliary.eval()
        x = torch.randn(4, 10, 65)
        mask = (torch.arange(10) % 3 != 1).float().repeat(4, 1)
        x *= mask[..., None]
        with torch.no_grad():
            torch.testing.assert_close(baseline(x, mask), auxiliary(x, mask), atol=0, rtol=0)
        logits = torch.tensor([[0., 9., -8.], [0., 0., 0.]], requires_grad=True)
        labels = torch.tensor([[1., 0., 0.], [1., 0., 1.]])
        active = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
        loss = bucket_bce_per_sequence(logits, labels, active)
        torch.testing.assert_close(loss, torch.full((2,), float(np.log(2))))
        loss.mean().backward()
        torch.testing.assert_close(logits.grad[0, 1:], torch.zeros(2), atol=0, rtol=0)
        torch.testing.assert_close(logits.grad.abs().sum(1), torch.full((2,), .25))
        with torch.no_grad():
            auxiliary.bucket_head.weight.fill_(.1)
        _, buckets = auxiliary.forward_with_buckets(x, mask)
        bucket_bce_per_sequence(buckets, torch.zeros_like(buckets), mask).mean().backward()
        self.assertGreater(float(auxiliary.lstm.weight_ih_l0.grad.abs().sum()), 0)

    def test_artificial_end_to_end_and_zero_weight_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_fixture(root / "dataset")
            (dataset / "test").rename(dataset / "test_unavailable")
            manifest = read_json(dataset / "manifest.json")
            for split in ("train", "validation"):
                file = dataset / split / "part_000000.npz"
                with np.load(file, allow_pickle=False) as saved:
                    payload = {key: saved[key] for key in saved.files}
                payload["y_bucket"][:] = 0
                payload["y_bucket"][:, 0] = payload["y_sequence"]
                positive = np.flatnonzero(payload["y_sequence"])
                payload["primary_attack"] = payload["primary_attack"].astype("U80")
                payload["primary_attack"][positive[:len(positive)//2]] = "Infilteration"
                payload["primary_attack"][positive[len(positive)//2:]] = "Brute_Force_-Web"
                np.savez_compressed(file, **payload)
                next(r for r in manifest["shards"] if r["path"] == f"{split}/part_000000.npz")["sha256"] = sha256(file)
            write_json(dataset / "manifest.json", manifest)
            config = {"architecture": {"hidden_size": 8, "dropout": .2},
                      "training": {"epochs": 1, "patience": 1, "batch_size": 32,
                                   "learning_rate": .001, "weight_decay": .0001}}
            write_json(root / "config.json", config)
            common = ["--dataset", str(dataset), "--device", "cpu", "--threads", "1", "--cache", str(root / "cache")]
            baseline = train_model("lstm", common + ["--config", str(root / "config.json"), "--out", str(root / "baseline")])
            baseline_hashes = {p.name: sha256(p) for p in baseline.iterdir() if p.is_file()}
            dataset_hash = sha256(dataset / "manifest.json")
            zero = {**config, "bucket_supervision": {"weight": 0., "reduction": REDUCTION}}
            write_json(root / "zero.json", zero)
            zero_run = train_model("lstm_bucket", common + ["--config", str(root / "zero.json"), "--out", str(root / "zero")])
            with np.load(baseline / "predictions.npz") as a, np.load(zero_run / "predictions.npz") as b:
                np.testing.assert_array_equal(a["score"], b["score"])
            original_state = torch.load(baseline / "model.pt", weights_only=True)["state_dict"]
            zero_state = torch.load(zero_run / "model.pt", weights_only=True)["state_dict"]
            for key, value in original_state.items():
                torch.testing.assert_close(value, zero_state[key], atol=0, rtol=0)
            arguments = ["--reference-run", str(baseline), "--cache", str(root / "cache"), "--out", str(root / "batch")]
            run_ablation(arguments + ["--dry-run"])
            self.assertFalse((root / "batch").exists())
            output = run_ablation(arguments)
            self.assertEqual(read_json(output / "status.json")["status"], "complete")
            plan = read_json(output / "plan.json")
            self.assertEqual(plan["new_runs"], 1)
            self.assertFalse(plan["encoder_retrained"])
            fitted = output / "bucket_supervised"
            fitted_config = read_json(fitted / "config.json")
            self.assertEqual(fitted_config["training"], config["training"])
            self.assertEqual(fitted_config["bucket_supervision"]["weight"], .5)
            result = read_json(output / "comparison.json")
            self.assertEqual(len(result["rows"]), 4)
            self.assertEqual(result["bucket_supervised_parameters"] - result["baseline_parameters"], 9)
            self.assertTrue(result["baseline_reused"])
            self.assertFalse(result["test_used"])
            audit = read_json(fitted / "bucket_target_audit.json")
            self.assertFalse(audit["validation_bucket_labels_used_during_fit"])
            self.assertEqual(audit["train"]["positive_buckets"], manifest["counts"]["train"]["real_malicious"] + manifest["counts"]["train"]["synthetic_malicious"])
            metrics = read_json(fitted / "bucket_metrics.json")
            self.assertFalse(metrics["used_for_checkpoint_selection"])
            self.assertFalse(metrics["used_for_sequence_scores_or_thresholds"])
            model = load_model("lstm_bucket", fitted_config, fitted, "cpu", 1)
            self.assertGreater(float(model.bucket_head.weight.detach().abs().sum()), 0)
            self.assertFalse(torch.equal(original_state["lstm.weight_ih_l0"], model.state_dict()["lstm.weight_ih_l0"]))
            store = Scheme2(dataset, root / "cache")
            validation = store.load("validation")
            try:
                with np.load(fitted / "predictions.npz") as saved:
                    np.testing.assert_array_equal(torch_runner.predict(model, validation, "cpu"), saved["score"])
                labels, _ = load_bucket_targets(dataset, "validation", validation)
                self.assertEqual(int(labels.sum()), int(validation["y"].sum()))
                with self.assertRaises(ValueError):
                    load_bucket_targets(dataset, "test", validation)
                class Reordered(dict):
                    def __len__(self):
                        return len(self["y"])
                bad = Reordered(mask=validation["mask"], ids=validation["ids"][::-1], y=validation["y"])
                with self.assertRaises(AssertionError):
                    load_bucket_targets(dataset, "validation", bad)
                self.assertNotIn("y_bucket", validation.arrays)
            finally:
                validation.close()
            self.assertEqual(sha256(dataset / "manifest.json"), dataset_hash)
            self.assertEqual(baseline_hashes, {p.name: sha256(p) for p in baseline.iterdir() if p.is_file()})


if __name__ == "__main__":
    unittest.main()
