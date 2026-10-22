"""Checks for paired uncertainty, fair order perturbations and frozen inference."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from CIC_new.diagnose_superimposition import paired_effect, order_indices, operating_points, temporal_probe, strata
from CIC_new.evaluate_autoencoder import evaluate, read, digest
from CIC_new.modelling.train_autoencoder import train_temporal
from CIC_new.tests.test_scheme1_l2_experiment import fixture_dataset


class SuperimpositionDiagnosticsTests(unittest.TestCase):
    def test_host_bootstrap_and_boundaries(self):
        a = np.ones(5, bool)
        b = np.zeros(5, bool)
        r = paired_effect(a, b, np.ones(5, bool), np.array(["a", "a", "a", "b", "b"]), repetitions=50)
        self.assertEqual(r["source_hosts"], 2)
        self.assertEqual(r["only_scheme1_l2"], 5)
        self.assertEqual(r["host_bootstrap_low"], -1.)
        self.assertEqual(r["host_bootstrap_high"], -1.)
        context = dict(primary_attack=np.array(["Benign", "attack", "attack", "attack"]),
            active_buckets=np.array([1, 2, 2, 10]), total_flows=np.array([1, 100, 100, 100]),
            attack_buckets=np.array([0, 1, 1, 10]), protocol=np.full(4, "TCP"), dns_fraction=np.zeros(4))
        groups = {(d, n): s for d, n, s in strata(context, dict(labels=np.array([0, 1, 1, 1]),
            attack_flow_fraction=np.array([0., .01, .1, 1.]))) }
        np.testing.assert_array_equal(groups["composition", "mixed"], [False, True, True, False])
        np.testing.assert_array_equal(groups["composition", "pure_attack"], [False, False, False, True])
        self.assertEqual(groups["attack_fraction", "(0,1%]"].sum(), 1)

    def test_order_preserves_padding_content_and_singletons(self):
        mask = np.array([[1, 0, 0, 0], [1, 0, 1, 1]], np.float32)
        x = np.arange(8).reshape(2, 4)
        for mode in ("shuffle", "reverse"):
            indices = order_indices(mask, mode)
            changed = np.take_along_axis(x, indices, axis=1)
            np.testing.assert_array_equal(changed[mask == 0], x[mask == 0])
            np.testing.assert_array_equal(changed[0], x[0])
            np.testing.assert_array_equal(np.sort(changed[1, mask[1] > 0]), np.sort(x[1, mask[1] > 0]))
        np.testing.assert_array_equal(order_indices(mask, "reverse")[1], [3, 1, 2, 0])

    def test_frozen_probe_reproduces_scores_and_does_not_refit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = root/"data1", root/"data2"
            fixture_dataset(one)
            fixture_dataset(two, one)
            models = {}
            for name, data, transform in (("scheme1", one, "none"), ("scheme1_l2", one, "bucket_l2"), ("scheme2", two, "none")):
                run = root/name
                train_temporal(SimpleNamespace(dataset=data, out=run, device="cpu", seed=4266, cpu_threads=1,
                    batch_size=8, epochs=1, steps_per_epoch=1, width=4, latent=3, input_transform=transform,
                    log_every=1, val_limit=8, lr=.001, real_only=False, patience=2, min_delta=.001))
                out = run/"evaluations/first"
                out.mkdir(parents=True)
                evaluate(run, out, "cpu", 8)
                with np.load(out/"validation_scores.npz") as a:
                    values = {k: a[k] for k in a.files}
                models[name] = dict(run=run, config=read(run/"config.json"), report=read(out/"report.json"),
                    report_path=out/"report.json", values=values)
            with np.load(one/"validation/part_000000.npz") as a:
                context = {k: a[k] for k in ("sample_id", "mask")}
            protected = {p: digest(p) for p in root.rglob("*") if p.is_file()}
            out = root/"diagnostic"
            out.mkdir()
            with patch("CIC_new.diagnose_superimposition.torch.cuda.is_available", return_value=False):
                report = temporal_probe(models, context, operating_points(models), out, benign_limit=2)
            self.assertFalse(report["threshold_refit"])
            self.assertEqual(report["sampled_sequences"], 6)
            self.assertEqual(len(report["rows"]), 72)
            for path, expected in protected.items():
                self.assertEqual(digest(path), expected)
            self.assertFalse((one/"model_cache/test").exists())


if __name__ == "__main__":
    unittest.main()
