"""Small artificial checks for normalization, unchanged donors and frozen eval."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from CIC_new.evaluate_autoencoder import digest, evaluate, read, write
from CIC_new.modelling.train_autoencoder import train_temporal
from CIC_new.preprocessing.dataset import ModelInputs, transform_model_inputs
from CIC_new.run_scheme1_l2_experiment import main, check_l2_inputs
from CIC_new.run_scheme2_experiment import latest_run
from CIC_new.train_autoencoder import main as train_main


def fixture_dataset(folder, reference=None):
    folder.mkdir()
    rng = np.random.default_rng(8)
    groups = np.array(["real_benign", "synthetic_benign", "real_malicious", "synthetic_malicious"]*4)
    shards = []
    for split, n in (("train", 16), ("validation", 8)):
        root = folder/split
        root.mkdir()
        labels = np.array([0, 0, 1, 1]*4, np.int64)[:n]
        kinds = groups[:n] if split == "train" else np.where(labels > 0, "real_malicious", "real_benign")
        ids = np.arange(n)+(1 if split == "train" else 100)
        mask = np.ones((n, 10), np.float32)
        mask[::2, 1:] = 0
        x = rng.normal(size=(n, 10, 3)).astype(np.float32)*mask[..., None]
        if reference is not None:
            x = transform_model_inputs(x, mask, "bucket_l2")
        payload = dict(X=x, mask=mask, y_sequence=labels, sample_id=ids, sample_type=kinds,
            left_id=ids, right_id=np.where(np.char.startswith(kinds, "synthetic"), ids+1000, -1),
            right_shift_buckets=np.zeros(n, np.int64), left_ip=np.full(n, "10.0.0.1"),
            right_ip=np.where(np.char.startswith(kinds, "synthetic"), "10.0.0.2", ""),
            primary_attack=np.where(labels > 0, "attack", "Benign"),
            y_bucket=(labels[:, None]*mask).astype(np.int64),
            flow_counts=(mask*4).astype(np.int64), attack_flow_counts=(mask*labels[:, None]).astype(np.int64))
        path = root/"part_000000.npz"
        np.savez_compressed(path, **payload)
        shards.append(dict(path=path.relative_to(folder).as_posix(), count=n, sha256=digest(path)))
    if reference is None:
        np.savez(folder/"donor_plan.npz", sample_id=np.array([2, 4]))
    else:
        (folder/"donor_plan.npz").write_bytes((reference/"donor_plan.npz").read_bytes())
    manifest = dict(complete=True, format="model-independent-sequence-shards", representation="scheme2" if reference else "scheme1",
        feature_width=3, timesteps=10, total_samples=dict(train=16, validation=8, test=0),
        counts=dict(train=dict(real_benign=4, synthetic_benign=4, real_malicious=4, synthetic_malicious=4),
                    validation=dict(real_benign=4, real_malicious=4), test={}),
        embedding_checkpoint_sha256="fixture", shift_mode="none", shards=shards,
        donor_plan_sha256=digest(folder/"donor_plan.npz"))
    if reference is not None:
        manifest["paired_reference_dataset"] = str(reference.resolve())
    write(folder/"manifest.json", manifest)
    write(folder/"schema.json", dict(feature_names=["a", "b", "c"]))
    write(folder/"scaler.json", dict(mean=[0, 0, 0], scale=[1, 1, 1]))
    write(folder/"validation.json", dict(passed=True))


class SchemeOneL2Tests(unittest.TestCase):
    def test_formula_padding_zero_and_dominance(self):
        x = np.array([[[3., 4., 0.], [1., 2., 3.], [0., 0., 0.], [1399., 1., 1.]]], np.float32)
        mask = np.array([[1., 0., 1., 1.]], np.float32)
        before = x.copy()
        normalized = transform_model_inputs(x, mask, "bucket_l2")
        np.testing.assert_allclose(normalized[0, 0], [.6, .8, 0.], rtol=1e-6)
        np.testing.assert_array_equal(normalized[0, 1:3], 0.)
        self.assertAlmostEqual(float(normalized[0, 3, 0]/normalized[0, 3, 1]), 1399., places=3)
        np.testing.assert_array_equal(x, before)
        self.assertIs(transform_model_inputs(x, mask), x)
        with self.assertRaises(ValueError):
            transform_model_inputs(x*np.nan, mask, "bucket_l2")

    def test_complete_cli_paths_and_unchanged_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, scheme2 = root/"data1", root/"data2"
            fixture_dataset(reference)
            fixture_dataset(scheme2, reference)
            runs = root/"CIC_new/outputs/ae_runs"
            runs.mkdir(parents=True)
            configs = []
            for data, name in ((reference, "a_scheme1"), (scheme2, "b_scheme2")):
                run = runs/name
                args = SimpleNamespace(dataset=data, out=run, device="cpu", seed=4266, cpu_threads=1,
                    batch_size=8, epochs=2, steps_per_epoch=2, width=4, latent=3,
                    log_every=2, val_limit=8, lr=.001, real_only=False, patience=2, min_delta=.001)
                train_temporal(args)
                out = run/"evaluations/original"
                out.mkdir(parents=True)
                evaluate(run, out, "cpu", 8)
                configs.append(run)
            baseline, other = configs
            protected = {p: digest(p) for folder in (reference, scheme2, baseline, other)
                         for p in folder.rglob("*") if p.is_file()}
            base = ["run_scheme1_l2_experiment.py", "--baseline-run", str(baseline),
                    "--scheme2-run", str(other), "--device", "cpu", "--evaluation-batch-size", "8"]
            def invoke(extra):
                with patch("sys.argv", [*base, *extra]):
                    main()
            with patch("CIC_new.run_scheme1_l2_experiment.train_temporal", side_effect=AssertionError("unexpected fitting")):
                invoke(["--stage", "check"])
            with patch("CIC_new.train_autoencoder.ROOT", root):
                invoke([])  # Real default all-stage and automatically reserved output.
            candidate = next(p for p in runs.iterdir() if "scheme1_l2" in p.name)
            self.assertEqual(read(candidate/"config.json")["input_transform"], "bucket_l2")
            self.assertEqual(torch.load(candidate/"best.pt", weights_only=True)["input_transform"], "bucket_l2")
            comparison = next((candidate/"evaluations").glob("*/three_way_comparison.json"))
            self.assertEqual(set(read(comparison)["models"]), {"scheme1", "scheme1_l2", "scheme2"})
            self.assertTrue(read(comparison.parent/"comparison.json")["identical_draw_stream_prefix"])
            self.assertTrue(read(comparison.parent/"report.json")["training_artifacts_unchanged"])
            self.assertEqual(read(comparison.parent/"report.json")["input_transform"], "bucket_l2")
            models = read(comparison)["models"]
            self.assertEqual(models["scheme1"]["model_parameter_count"], models["scheme1_l2"]["model_parameter_count"])
            with patch("CIC_new.run_scheme2_experiment.ROOT", root):
                self.assertEqual(latest_run("scheme1"), baseline)
            frozen = digest(candidate/"best.pt")
            invoke(["--stage", "evaluate", "--run", str(candidate)])
            self.assertEqual(digest(candidate/"best.pt"), frozen)
            split_run = runs/"c_scheme1_l2_split"
            invoke(["--stage", "train", "--out", str(split_run)])
            self.assertFalse((split_run/"evaluations").exists())
            invoke(["--stage", "evaluate", "--run", str(split_run)])
            # The standalone trainer also records the transform and labels a fresh output.
            command = ["train_autoencoder.py", "--dataset", str(reference), "--device", "cpu",
                       "--input-transform", "bucket_l2", "--epochs", "1", "--steps-per-epoch", "1",
                       "--batch-size", "8", "--width", "4", "--latent", "3", "--cpu-threads", "1"]
            with patch("CIC_new.train_autoencoder.ROOT", root), patch("sys.argv", command):
                train_main()
            self.assertEqual(len(list(runs.glob("*scheme1_l2_ae_aug*"))), 2)
            for path, expected in protected.items():
                self.assertEqual(digest(path), expected, str(path))
            self.assertFalse((reference/"model_cache/test").exists())
            self.assertFalse((scheme2/"model_cache/test").exists())
            config = read(candidate/"config.json")
            config["input_transform"] = "none"
            write(candidate/"config.json", config)
            with self.assertRaisesRegex(ValueError, "input transforms disagree"):
                evaluate(candidate, root, "cpu", 8)


if __name__ == "__main__":
    unittest.main()
