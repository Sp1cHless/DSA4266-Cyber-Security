"""Synthetic held-out evaluation: paired shards, fixed probes, no fitting."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from CIC_new.evaluate_autoencoder import read, write, digest, binary_metrics
from CIC_new.evaluate_test_set import evaluate, paired_test_chunks, NAMES, TRAIN_POINTS
from CIC_new.modelling.autoencoder import TemporalAE
from CIC_new.tests.test_scheme1_l2_experiment import fixture_dataset


def fixture(root):
    data1, data2 = root/"data1", root/"data2"
    fixture_dataset(data1)
    fixture_dataset(data2, data1)
    models = {}
    for name, data, transform in zip(NAMES, (data1, data2), ("bucket_l2", "none")):
        with np.load(data/"validation/part_000000.npz") as a:
            payload = {key: a[key] for key in a.files}
        payload["sample_id"] += 1000
        payload["left_id"] = payload["sample_id"].copy()
        if name == "scheme2":
            del payload["attack_flow_counts"]
        folder = data/"test"
        folder.mkdir()
        # Different shard boundaries exercise streaming pairing, including legacy
        # Scheme2 shards without attack-flow counts.
        blocks = ((0, 3), (3, 8)) if name == "scheme2" else ((0, 8),)
        manifest = read(data/"manifest.json")
        manifest["total_samples"]["test"] = 8
        manifest["counts"]["test"] = dict(real_benign=4, real_malicious=4)
        for number, (lo, hi) in enumerate(blocks):
            path = folder/f"part_{number:06d}.npz"
            np.savez_compressed(path, **{k:v[lo:hi] for k,v in payload.items()})
            manifest["shards"].append(dict(path=path.relative_to(data).as_posix(), count=hi-lo, sha256=digest(path)))
        write(data/"manifest.json", manifest)
        run = root/name
        run.mkdir()
        config = dict(dataset=str(data), input_transform=transform)
        write(run/"config.json", config)
        write(run/"summary.json", dict(complete=True, best_epoch=1))
        write(run/"dataset_manifest.json", manifest)
        torch.manual_seed(4)
        args = dict(features=3, width=4, latent=3, timesteps=10)
        model = TemporalAE(**args)
        torch.save(dict(model_args=args, model_state=model.state_dict(), epoch=1, input_transform=transform), run/"best.pt")
        params = dict(mean=[0., 0., 0.], scale=[1., 1., 1.], weights=[1., -1., .1], bias=0.,
                      validation_used_for_fitting=False)
        write(run/"linear_probe.json", params)
        y, scores = np.array([0, 0, 1, 1]), np.array([-1., .1, 1., 2.])
        points = {key:binary_metrics(y, scores, threshold) for key, threshold in zip(TRAIN_POINTS, (0., .5, 1.5))}
        report = dict(linear_probe=dict(parameters=params, operating_points=points),
            source_sha256={f:digest(run/f) for f in ("best.pt", "config.json", "summary.json")})
        write(run/"report.json", report)
        models[name] = dict(config=config, report=report, run=run, report_path=run/"report.json",
                            values=dict(labels=y, logits=scores))
    return models, data1, data2


class FrozenTestEvaluationTests(unittest.TestCase):
    def test_complete_frozen_evaluation_and_no_test_threshold_choice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, one, two = fixture(root)
            protected = {p:digest(p) for p in root.rglob("*") if p.is_file()}
            out = root/"evaluation"
            out.mkdir()
            with patch("CIC_new.evaluate_autoencoder.fit_probe", side_effect=AssertionError("No fitting allowed")):
                result = evaluate(models, out, "cpu", 4)
            self.assertTrue(result["complete"])
            self.assertTrue(result["test_used"])
            self.assertFalse(result["parameter_fitting"])
            self.assertFalse(result["test_threshold_selection"])
            self.assertEqual(result["test_examples"], 8)
            self.assertEqual(result["test_attacks"], 4)
            self.assertEqual(result["mixed_attack_sequences"], 4)
            plan = read(out/"frozen_evaluation_plan.json")
            for name in NAMES:
                self.assertEqual(plan["references"][name]["thresholds"]["train_benign_99pct"], .5)
                self.assertEqual(result["metrics"][name]["train_benign_99.9pct"]["threshold_logit"], 1.5)
            self.assertTrue((out/"TEST_COMPARISON.md").exists())
            for path, expected in protected.items():
                self.assertEqual(digest(path), expected, str(path))
            self.assertFalse((one/"model_cache/test").exists())
            self.assertFalse((two/"model_cache/test").exists())

    def test_reject_changed_or_unpaired_held_out_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, one, two = fixture(root)
            path = two/"test/part_000000.npz"
            with np.load(path) as a:
                payload = {key:a[key] for key in a.files}
            payload["sample_id"][0] += 1
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                list(paired_test_chunks(one, two))
            manifest = read(two/"manifest.json")
            for row in manifest["shards"]:
                if row["path"] == "test/part_000000.npz":
                    row["sha256"] = digest(path)
            write(two/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "pairing differs"):
                list(paired_test_chunks(one, two))


if __name__ == "__main__":
    unittest.main()
