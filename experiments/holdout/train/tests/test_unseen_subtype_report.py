"""Artificial checks for the control-versus-holdout unseen-subtype report."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.common.io import sha256, write_json
from train.common.metrics import choose_thresholds, evaluate
from train.unseen_subtype_report import main as report


def make_folder(folder, model, labels, scores, attacks, hosts, split="validation"):
    folder = Path(folder)
    folder.mkdir()
    thresholds = choose_thresholds(labels, scores)
    write_json(folder / "thresholds.json", {"values": thresholds})
    metrics = evaluate({"y": labels, "attack": attacks, "host": hosts}, scores, thresholds)
    write_json(folder / "summary.json", {"complete": True, "split": split, "model": model,
                                         "metrics": metrics, "thresholds_sha256": sha256(folder / "thresholds.json")})
    np.savez_compressed(folder / "predictions.npz", sample_id=np.arange(len(labels)), y=labels, score=scores,
                        primary_attack=attacks, source_host=hosts)
    return folder


def populations(seed=5):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * 120 + [1] * 160)
    attacks = np.array(["Benign"] * 120 + ["SeenA"] * 70 + ["SeenB"] * 50 + ["HeldA"] * 25 + ["HeldB"] * 15)
    hosts = np.array([f"10.0.0.{i % 9}" for i in range(len(labels))])
    signal = labels * 0.45
    control = np.clip(signal + rng.normal(0, 0.25, len(labels)), 0, 1)
    holdout = np.clip(control * 0.55 + rng.normal(0, 0.05, len(labels)), 0, 1)
    return labels, attacks, hosts, control, holdout


class UnseenSubtypeReportTests(unittest.TestCase):
    def test_report_compares_frozen_pairs_and_rejects_moved_populations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels, attacks, hosts, control, holdout = populations()
            left = make_folder(root / "control", "cnn", labels, control, attacks, hosts)
            right = make_folder(root / "holdout", "cnn", labels, holdout, attacks, hosts)
            out = root / "report"
            report(["--control", str(left), "--holdout", str(right),
                    "--held-out-attacks", "HeldB,HeldA", "--out", str(out)])
            for name in ("UNSEEN_SUBTYPE.md", "unseen_subtype.csv", "manifest.json"):
                self.assertTrue((out / name).is_file())
            import csv
            with (out / "unseen_subtype.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            pooled = rows[0]
            self.assertEqual(pooled["primary_attack"], "ALL HELD OUT")
            self.assertEqual(int(pooled["n"]), 40)
            self.assertEqual(pooled["held_out_from_training"], "True")
            seen = {row["primary_attack"]: row for row in rows[1:]}
            self.assertEqual(seen["HeldA"]["held_out_from_training"], "True")
            self.assertEqual(seen["SeenA"]["held_out_from_training"], "False")
            self.assertLessEqual(float(seen["HeldA"]["recall_holdout_1pct"]),
                                 float(seen["HeldA"]["recall_control_1pct"]) + 1e-12)

            shorter = root / "short"
            make_folder(shorter, "cnn", labels[:-1], holdout[:-1], attacks[:-1], hosts[:-1])
            with self.assertRaisesRegex(ValueError, "population"):
                report(["--control", str(left), "--holdout", str(shorter),
                        "--held-out-attacks", "HeldA,HeldB", "--out", str(root / "bad")])

            with self.assertRaisesRegex(ValueError, "same model families"):
                other = make_folder(root / "other", "lstm", labels, holdout, attacks, hosts)
                report(["--control", str(left), "--holdout", str(other),
                        "--held-out-attacks", "HeldA", "--out", str(root / "bad2")])

            manifest = __import__("json").loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["held_out_attacks"], ["HeldA", "HeldB"])
            self.assertFalse(manifest["refitting"])
            self.assertTrue(manifest["control_and_holdout_share_prediction_population"])


if __name__ == "__main__":
    unittest.main()
