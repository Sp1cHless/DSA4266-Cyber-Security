"""Artificial-fixture checks for fixed rotations and frozen LSTM replay."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from train.common.data import Scheme2
from train.common.experiment import main as train_model, load_model, score
from train.common.io import read_json, write_json
from train.compare import compare
from train.evaluate import main as evaluate_model
from train.lstm.model import Model
from train.tests.fixture import make_fixture


class CircularShiftTests(unittest.TestCase):
    def test_all_offsets_rotate_every_channel_and_mask_without_mutating_inputs(self):
        torch.set_num_threads(1)
        x = torch.arange(2 * 10 * 65, dtype=torch.float32).reshape(2, 10, 65) / 1000
        mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
                             [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
        x *= mask.unsqueeze(-1)
        original_x, original_mask = x.clone(), mask.clone()
        baseline = Model(dropout=0).eval()
        for start in range(10):
            with self.subTest(start=start):
                model = Model(dropout=0, circular_shift_start=start).eval()
                model.load_state_dict(baseline.state_dict())
                captured = []
                hook = model.lstm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
                order = list(range(start, 10)) + list(range(start))
                with torch.no_grad():
                    actual = model(x, mask)
                    expected = baseline(x[:, order], mask[:, order])
                hook.remove()
                torch.testing.assert_close(captured[0][..., :65], x[:, order])
                torch.testing.assert_close(captured[0][..., 65], mask[:, order])
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(x, original_x)
                torch.testing.assert_close(mask, original_mask)
        for invalid in (-1, 10, 2.5, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "integer from 0 to 9"):
                Model(circular_shift_start=invalid)

    def test_cli_training_config_override_and_frozen_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = make_fixture(root / "dataset")
            common = ["--dataset", str(dataset), "--cache", str(root / "cache"),
                      "--device", "cpu", "--threads", "1", "--epochs", "1"]
            custom = read_json(Path(__file__).resolve().parents[1] / "lstm" / "config.json")
            custom["architecture"]["circular_shift_start"] = 5
            write_json(root / "custom.json", custom)
            variants = {
                "baseline": [],
                "shifted": ["--circular-shift-start", "5"],
                "configured": ["--config", str(root / "custom.json")],
                "overridden": ["--config", str(root / "custom.json"), "--circular-shift-start", "0"],
            }
            # Missing test shards must not prevent this experiment's training.
            test_shard = dataset / "test" / "part_000000.npz"
            hidden = test_shard.with_suffix(".hidden")
            test_shard.rename(hidden)
            for label, extra in variants.items():
                train_model("lstm", common + ["--out", str(root / label)] + extra)
                summary = read_json(root / label / "summary.json")
                self.assertFalse(summary["test_used"])
                self.assertFalse(read_json(root / label / "data_audit.json")["test_opened"])
                expected_name = "identity" if label in ("baseline", "overridden") else "circular_shift"
                self.assertEqual(summary["input_transform"]["name"], expected_name)
            for left, right in (("baseline", "overridden"), ("shifted", "configured")):
                with np.load(root / left / "predictions.npz") as a, np.load(root / right / "predictions.npz") as b:
                    np.testing.assert_array_equal(a["score"], b["score"])
            test_shard_path = hidden.rename(test_shard)
            self.assertTrue(test_shard_path.exists())
            run = root / "shifted"
            config = read_json(run / "config.json")
            self.assertEqual(config["input_transform"]["bucket_order"], [5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
            model = load_model("lstm", config, run, "cpu", 1)
            self.assertEqual(model.circular_shift_start, 5)
            split = Scheme2(dataset, root / "cache").load("validation")
            try:
                x, mask = torch.from_numpy(split["x"].copy()), torch.from_numpy(split["mask"].copy())
                chronological = Model(**{**config["architecture"], "circular_shift_start": 0}).eval()
                chronological.load_state_dict(model.state_dict())
                with torch.no_grad():
                    expected = torch.sigmoid(chronological(torch.roll(x, -5, 1), torch.roll(mask, -5, 1))).numpy()
                np.testing.assert_allclose(score("lstm", model, split, "cpu", 1), expected, rtol=1e-6, atol=1e-7)
            finally:
                split.close()
            for split_name in ("validation", "test"):
                output = evaluate_model(["--run", str(run), "--cache", str(root / "cache"),
                                         "--device", "cpu", "--threads", "1", "--split", split_name])
                self.assertEqual(read_json(output / "summary.json")["input_transform"], config["input_transform"])
                self.assertIn("Temporal perturbation diagnostic", read_json(output / "summary.json")["scope"])
                if split_name == "validation":
                    with np.load(run / "predictions.npz") as a, np.load(output / "predictions.npz") as b:
                        np.testing.assert_array_equal(a["score"], b["score"])
            with self.assertRaises(FileExistsError):
                train_model("lstm", common + ["--out", str(run), "--circular-shift-start", "5"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_model("lstm", common + ["--circular-shift-start", "10"])
            custom["architecture"]["circular_shift_start"] = -1
            write_json(root / "invalid.json", custom)
            with self.assertRaisesRegex(ValueError, "integer from 0 to 9"):
                train_model("lstm", common + ["--config", str(root / "invalid.json")])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_shifted_training_and_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = make_fixture(root / "dataset")
            run = train_model("lstm", ["--dataset", str(dataset), "--cache", str(root / "cache"),
                                       "--device", "cuda", "--threads", "1", "--epochs", "1",
                                       "--out", str(root / "shifted"), "--circular-shift-start", "5"])
            output = evaluate_model(["--run", str(run), "--cache", str(root / "cache"),
                                     "--device", "cuda", "--threads", "1"])
            self.assertEqual(read_json(run / "config.json")["architecture"]["circular_shift_start"], 5)
            with np.load(run / "predictions.npz") as a, np.load(output / "predictions.npz") as b:
                np.testing.assert_allclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)

    def test_screening_rejects_mixed_input_orders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = {"complete": True, "split": "validation", "test_used": False,
                    "dataset_manifest_sha256": "same", "train_cohort_sha256": "same",
                    "validation_cohort_sha256": "same", "evaluation_policy": "same",
                    "weighting": "same", "seed": 4266}
            for name in ("lstm", "cnn"):
                (root / name).mkdir()
                summary = {**base, "model": name}
                if name == "lstm":
                    summary["input_transform"] = {"name": "circular_shift", "start_bucket": 5}
                write_json(root / name / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "input_transform"):
                compare([root / "lstm", root / "cnn"], root / "comparison", allow_partial=True)


if __name__ == "__main__":
    unittest.main()
