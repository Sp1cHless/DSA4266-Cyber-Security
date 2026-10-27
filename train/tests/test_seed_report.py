"""Artificial checks for the multi-seed control-versus-holdout aggregation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.common.io import write_json
from train.common.metrics import choose_thresholds, evaluate
from train.seed_report import main as report


def make_run(root, model, seed, labels, scores, attacks, hosts, dataset_sha):
    run = Path(root) / f"{model}_seed{seed}"
    run.mkdir(parents=True)
    thresholds = choose_thresholds(labels, scores)
    write_json(run / "thresholds.json", {"values": thresholds})
    write_json(run / "summary.json", {
        "complete": True, "split": "validation", "model": model, "seed": seed,
        "dataset_manifest_sha256": dataset_sha,
        "metrics": evaluate({"y": labels, "attack": attacks, "host": hosts}, scores, thresholds)})
    evaluation = run / "test_eval"
    evaluation.mkdir()
    write_json(evaluation / "plan.json", {"thresholds": thresholds})
    write_json(evaluation / "summary.json", {
        "complete": True, "split": "test", "model": model,
        "metrics": evaluate({"y": labels, "attack": attacks, "host": hosts}, scores, thresholds)})
    np.savez_compressed(evaluation / "predictions.npz", sample_id=np.arange(len(labels)), y=labels,
                        score=scores, primary_attack=attacks, source_host=hosts)
    return run


def populations(seed):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * 120 + [1] * 160)
    attacks = np.array(["Benign"] * 120 + ["SeenA"] * 70 + ["SeenB"] * 50 + ["HeldA"] * 25 + ["HeldB"] * 15)
    hosts = np.array([f"10.0.0.{i % 9}" for i in range(len(labels))])
    control = np.clip(labels * 0.45 + rng.normal(0, 0.25, len(labels)), 0, 1)
    holdout = np.clip(control * 0.55 - 0.12 * labels + rng.normal(0, 0.05, len(labels)), 0, 1)
    return labels, attacks, hosts, control, holdout


def build(root, seeds=(11, 12, 13)):
    control, holdout = [], []
    for seed in seeds:
        labels, attacks, hosts, left, right = populations(seed)
        control.append(make_run(root / "control", "cnn", seed, labels, left, attacks, hosts, "a" * 64))
        holdout.append(make_run(root / "holdout", "cnn", seed, labels, right, attacks, hosts, "b" * 64))
    (root / "holdout_dataset").mkdir(parents=True, exist_ok=True)
    write_json(root / "holdout_dataset" / "manifest.json", {"train_holdout_attacks": ["HeldA", "HeldB"]})
    return control, holdout


class SeedSweepReportTests(unittest.TestCase):
    def test_report_pairs_seeds_and_summarises_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control, holdout = build(root)
            out = root / "report"
            report(["--arm", "control=" + ",".join(map(str, control)),
                    "--arm", "holdout=" + ",".join(map(str, holdout)),
                    "--arm-dataset", f"holdout={root / 'holdout_dataset'}",
                    "--out", str(out)])
            for name in ("SEED_SWEEP.md", "seed_sweep_arms.csv", "seed_sweep_deltas.csv",
                         "seed_sweep_subtypes.csv", "manifest.json"):
                self.assertTrue((out / name).is_file(), name)
            with (out / "seed_sweep_subtypes.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            pooled = rows[0]
            self.assertEqual(pooled["primary_attack"], "ALL HELD OUT")
            self.assertEqual(int(pooled["n"]), 40)
            self.assertEqual(pooled["seeds"], "11;12;13")
            self.assertEqual(len(pooled["per_seed_delta_1pct"].split(";")), 3)
            self.assertLess(float(pooled["delta_1pct_mean"]), 0.0)
            self.assertEqual(int(pooled["seeds_lower_1pct"]), 3)
            seen = {row["primary_attack"]: row for row in rows[1:]}
            self.assertEqual(seen["HeldA"]["held_out"], "True")
            self.assertEqual(seen["SeenA"]["held_out"], "False")
            with (out / "seed_sweep_deltas.csv").open(encoding="utf-8", newline="") as stream:
                delta = next(csv.DictReader(stream))
            self.assertEqual(delta["seeds_lower_1pct"], "3")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seeds"], [11, 12, 13])
            self.assertTrue(manifest["repeated_test_use"])
            self.assertFalse(manifest["refitting"])

    def test_report_rejects_moved_population_and_bad_seed_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control, holdout = build(root)
            labels, attacks, hosts, _, right = populations(11)
            moved = make_run(root / "moved", "cnn", 11, labels[:-1], right[:-1], attacks[:-1], hosts[:-1], "b" * 64)
            with self.assertRaisesRegex(ValueError, "population"):
                report(["--arm", "control=" + ",".join(map(str, control)),
                        "--arm", f"holdout={moved},{holdout[1]},{holdout[2]}",
                        "--held-out-attacks", "holdout=HeldA,HeldB", "--out", str(root / "bad")])
            with self.assertRaisesRegex(ValueError, "duplicate seeds"):
                report(["--arm", "control=" + ",".join(map(str, control)),
                        "--arm", f"holdout={holdout[0]},{holdout[0]},{holdout[2]}",
                        "--held-out-attacks", "holdout=HeldA,HeldB", "--out", str(root / "bad2")])
            with self.assertRaisesRegex(ValueError, "expected 3 seeds"):
                report(["--arm", "control=" + ",".join(map(str, control)),
                        "--arm", f"holdout={holdout[0]},{holdout[1]}",
                        "--held-out-attacks", "holdout=HeldA,HeldB", "--out", str(root / "bad3")])
            with self.assertRaisesRegex(ValueError, "named control"):
                report(["--arm", f"holdout={holdout[0]},{holdout[1]},{holdout[2]}",
                        "--held-out-attacks", "holdout=HeldA,HeldB", "--out", str(root / "bad4")])


if __name__ == "__main__":
    unittest.main()
