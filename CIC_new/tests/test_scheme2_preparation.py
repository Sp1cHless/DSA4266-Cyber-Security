"""Only artificial small data: paired scheme1/scheme2 exports and stage reuse."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from CIC_new.scheme1.pipeline import build
from CIC_new.scheme1.storage import digest, read_json, arrays
from CIC_new.prepare_scheme2_dataset import validate_export, check_same_originals
from CIC_new.tests.fixtures import fixture_database
from CIC_new.modelling.train_autoencoder import train_temporal
from CIC_new.evaluate_autoencoder import evaluate

ROOT = Path(__file__).resolve().parents[2]


class SchemeTwoPreparationTests(unittest.TestCase):
    def test_direct_workflow_shared_plan_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root/"flows.sqlite"
            fixture_database(db)
            before = digest(db)
            baseline = root/"baseline"
            build(db=db, out=baseline, shard_size=17, train_pool=16, val_pool=16,
                  embedding_steps=3, batch_size=8, validate_every=3, device="cpu", cpu_threads=1)
            reference = baseline/"dataset"
            reference_hash = digest(reference/"manifest.json")
            plan_hash = digest(reference/"donor_plan.npz")
            out = root/"scheme2"
            command = [sys.executable, str(ROOT/"CIC_new/prepare_scheme2_dataset.py"),
                "--db", str(db), "--reference-dataset", str(reference), "--out", str(out),
                "--embedding-steps", "3", "--batch-size", "8", "--validate-every", "1",
                "--category-dim", "4", "--bucket-dim", "8", "--val-limit", "16",
                "--shard-size", "17", "--device", "cpu", "--cpu-threads", "1"]
            def terminal(extra=()):
                return subprocess.run([*command, *extra], cwd=root, capture_output=True, text=True)
            result = terminal()
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            workflow = read_json(out/"workflow.json")
            self.assertTrue(workflow["complete"])
            self.assertFalse(workflow["prediction_model_training_started"])
            manifest = read_json(out/"dataset/manifest.json")
            self.assertEqual(manifest["feature_width"], 9)
            self.assertEqual(manifest["representation"], "scheme2")
            self.assertEqual(manifest["counts"], read_json(reference/"manifest.json")["counts"])
            self.assertEqual(digest(out/"dataset/donor_plan.npz"), plan_hash)
            report = read_json(out/"dataset/validation.json")
            self.assertTrue(report["passed"])
            self.assertTrue(report["identical_reference_originals_and_splits"])
            self.assertTrue(report["identical_reference_donor_plan"])
            self.assertEqual(report["all_exported_samples_checked"], sum(manifest["total_samples"].values()))
            plan = arrays(reference/"donor_plan.npz")
            produced = {k: [] for k in ("sample_id", "left_id", "right_id", "sample_type")}
            for path in sorted((out/"dataset/train").glob("part_*.npz")):
                a = arrays(path)
                synthetic = a["right_id"] >= 0
                self.assertTrue(np.all(a["right_shift_buckets"] == 0))
                for key in produced:
                    produced[key].extend(a[key][synthetic])
            for key in produced:
                np.testing.assert_array_equal(produced[key], plan[key])
            result = terminal(("--resume",))
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("reuse completed stage", result.stdout)
            self.assertNotEqual(terminal().returncode, 0)
            self.assertEqual(digest(db), before)
            self.assertEqual(digest(reference/"manifest.json"), reference_hash)
            self.assertEqual(digest(reference/"donor_plan.npz"), plan_hash)
            self.assertEqual(check_same_originals(out/"preprocessed", baseline/"prepared"), 300)
            # Explicitly supplying existing V3 preparation/checkpoint skips both fitting stages.
            reuse = root/"reuse"
            reuse_command = command.copy()
            reuse_command[reuse_command.index("--out")+1] = str(reuse)
            result = subprocess.run([*reuse_command, "--prepared", str(out/"preprocessed"),
                "--bucket-checkpoint", str(out/"bucket_embeddings/best.pt")], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertFalse((reuse/"preprocessed").exists())
            self.assertFalse((reuse/"bucket_embeddings").exists())
            self.assertEqual(digest(reuse/"dataset/donor_plan.npz"), plan_hash)
            self.assertTrue(read_json(reuse/"dataset/validation.json")["passed"])
            # Fit a tiny baseline, then exercise matched scheme2 train -> frozen eval -> comparison.
            baseline_ae = root/"baseline_ae"
            val_size = read_json(reference/"manifest.json")["total_samples"]["validation"]
            train_temporal(SimpleNamespace(dataset=reference, out=baseline_ae, device="cpu", seed=4266,
                cpu_threads=1, batch_size=8, epochs=1, steps_per_epoch=2, width=4, latent=3,
                log_every=1, val_limit=val_size, lr=.001, real_only=False, patience=2, min_delta=.001))
            baseline_eval = baseline_ae/"evaluations/first"
            baseline_eval.mkdir(parents=True)
            evaluate(baseline_ae, baseline_eval, "cpu", 32)
            candidate = root/"scheme2_ae"
            experiment = [sys.executable, str(ROOT/"CIC_new/run_scheme2_experiment.py"),
                "--dataset", str(out/"dataset"), "--baseline-run", str(baseline_ae),
                "--out", str(candidate), "--device", "cpu", "--evaluation-batch-size", "32"]
            result = subprocess.run(experiment, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            comparison = read_json(next((candidate/"evaluations").glob("*/comparison.json")))
            self.assertTrue(comparison["identical_draw_stream_prefix"])
            self.assertFalse(comparison["test_used"])
            self.assertEqual(comparison["models"]["scheme2"]["input_features"], 9)
            self.assertEqual(comparison["paired_positive_examples"], read_json(baseline_eval/"report.json")["linear_probe"]["operating_points"]["balanced_prior_0.5"]["positives"])
            candidate_hash = digest(candidate/"best.pt")
            second = [sys.executable, str(ROOT/"CIC_new/run_scheme2_experiment.py"), "--stage", "evaluate",
                "--run", str(candidate), "--baseline-run", str(baseline_ae), "--device", "cpu", "--evaluation-batch-size", "32"]
            result = subprocess.run(second, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(digest(candidate/"best.pt"), candidate_hash)
            self.assertEqual(len(list((candidate/"evaluations").glob("*/comparison.json"))), 2)
            self.assertFalse((out/"dataset/model_cache/test").exists())
            # Altering a prepared vector must fail arithmetic validation and stage reuse.
            path = next((out/"dataset/train").glob("part_*.npz"))
            payload = arrays(path)
            payload["X"][0, 0, 0] += 1
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "vector addition/L2"):
                validate_export(out/"dataset", out/"encoded_sequences", reference)
            result = terminal(("--resume",))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(read_json(out/"workflow.json")["complete"])
            self.assertFalse(read_json(out/"dataset/validation.json")["passed"])


if __name__ == "__main__":
    unittest.main()
