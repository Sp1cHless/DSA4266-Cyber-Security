"""Reference checks for tied ranking metrics, probe fitting and the draw audit."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from CIC_new.evaluate_autoencoder import KINDS, audit_sampling, binary_metrics, fit_probe, flow_composition, input_feature_names
from CIC_new.run_scheme2_experiment import curve_point


class EvaluationTests(unittest.TestCase):
    def test_scheme2_names_and_tied_fixed_fpr_points(self):
        names = input_feature_names(dict(representation="scheme2", feature_width=9), {})
        self.assertEqual(names, [f"bucket_embedding_{i}" for i in range(8)]+["log_flow_count_scaled"])
        y = np.array([0, 1, 0, 1])
        point = curve_point(y, np.zeros(4), .5)
        self.assertEqual(point["fp"], 0)
        self.assertEqual(point["recall"], 0)
        point = curve_point(y, np.array([-2., 2., -1., 1.]), 0.)
        self.assertEqual(point["fpr"], 0)
        self.assertEqual(point["recall"], 1.)

    def test_scheme2_flow_counts_align_reference_ids_and_reject_wrong_totals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, data = root/"reference", root/"scheme2"
            for folder in (reference, data):
                (folder/"validation").mkdir(parents=True)
            np.savez(reference/"validation/part_000000.npz", sample_id=[3, 8],
                flow_counts=[[2, 1], [4, 0]], attack_flow_counts=[[0, 0], [1, 0]], y_sequence=[0, 1])
            payload = dict(sample_id=[8, 3], flow_counts=[[4, 0], [2, 1]])
            target = data/"validation/part_000000.npz"
            np.savez(target, **payload)
            class Store:
                folder = data
                split = "validation"
                arrays = dict(sample_id=np.array([8, 3]), y_sequence=np.array([1, 0]))
                manifest = dict(representation="scheme2", paired_reference_dataset=str(reference))
                def __len__(self): return 2
            totals, attacks = flow_composition(Store())
            np.testing.assert_array_equal(totals, [4, 3])
            np.testing.assert_array_equal(attacks, [1, 0])
            payload["flow_counts"][0][0] = 999
            np.savez(target, **payload)
            with self.assertRaisesRegex(ValueError, "total flows or labels differ"):
                flow_composition(Store())

    def test_ranking_metrics_include_ties_and_imbalance(self):
        y = np.array([0, 1, 1, 0])
        perfect = binary_metrics(y, [1., 3., 2., 0.], 1.5)
        for key in ("roc_auc", "average_precision", "precision", "recall", "f1"):
            self.assertAlmostEqual(perfect[key], 1.)
        tied = binary_metrics(y, np.zeros(4))
        self.assertAlmostEqual(tied["roc_auc"], .5)
        self.assertAlmostEqual(tied["average_precision"], .5)
        reversed_scores = binary_metrics(y, [-1., -3., -2., 0.])
        self.assertAlmostEqual(reversed_scores["roc_auc"], 0.)
        self.assertAlmostEqual(reversed_scores["average_precision"], (1/3+1/2)/2)
        single = binary_metrics([1, 1], [0., 1.])
        self.assertIsNone(single["roc_auc"])

    def test_probe_uses_training_transform_and_handles_constant_dimension(self):
        z = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]], np.float32)
        y = np.array([0, 0, 1, 1])
        parameters, train_scores, scores = fit_probe(z, y, np.array([[-100., 0.], [100., 0.]], np.float32), "cpu")
        self.assertEqual(parameters["mean"], [0., 0.])
        self.assertEqual(parameters["scale"][1], 1.)
        self.assertFalse(parameters["validation_used_for_fitting"])
        self.assertLess(parameters["maximum_absolute_gradient"], 1e-4)
        self.assertTrue(np.all(train_scores[y == 1] > train_scores[y == 0]))
        self.assertLess(scores[0], 0)
        self.assertGreater(scores[1], 0)

    def test_audit_checks_every_update_and_training_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = SimpleNamespace(arrays=dict(sample_id=np.arange(4), sample_type=np.array(KINDS)))
            class Store:
                arrays = store.arrays
                def __len__(self):
                    return 4
            lines = [dict(sample_id=i, step=step, sample_type=name)
                     for step in (1, 2) for i, name in enumerate(KINDS)]
            audit = root/"sample_draws.jsonl"
            audit.write_text("\n".join(json.dumps(row) for row in lines)+"\n")
            report = audit_sampling(root, Store(), dict(steps=2, sample_draws=8), dict(batch_size=4, real_only=False), root)
            self.assertTrue(report["all_updates_balanced"])
            self.assertTrue(all(v["coverage"] == 1 for v in report["groups"].values()))
            lines[0]["sample_id"] = 100
            audit.write_text("\n".join(json.dumps(row) for row in lines)+"\n")
            with self.assertRaisesRegex(ValueError, "Unknown sample"):
                audit_sampling(root, Store(), dict(steps=2, sample_draws=8), dict(batch_size=4, real_only=False), root)


if __name__ == "__main__":
    unittest.main()
