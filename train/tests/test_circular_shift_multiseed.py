"""Small artificial-data checks for paired seeds, reuse, resume and Web slices."""
from contextlib import redirect_stderr
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from train.common.experiment import main as train_model
from train.common.io import read_json, sha256, write_json
from train.run_circular_shift import main as single_seed
from train.run_circular_shift_multiseed import main as multi_seed, paired_row, statistics, TEN_BUCKET_SOURCE_MIGRATIONS
from train.tests.fixture import make_fixture


def web_fixture(root):
    dataset = make_fixture(root)
    manifest = read_json(dataset / "manifest.json")
    for shard in manifest["shards"]:
        file = dataset / shard["path"]
        with np.load(file, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        keep = arrays["y_sequence"] == 1
        # Allocate enough string width for the recorded Web label.
        arrays["primary_attack"] = np.where(keep, "Brute_Force_-Web", "Benign")
        for index in np.flatnonzero(keep):
            active = 1 + index % 3
            value = arrays["X"][index, 0].copy()
            arrays["X"][index] = 0
            arrays["X"][index, :active] = value
            arrays["mask"][index] = 0
            arrays["mask"][index, :active] = 1
        arrays["y_bucket"] = arrays["y_sequence"][:, None] * arrays["mask"]
        arrays["flow_counts"] = arrays["mask"].astype(np.int64) * 2
        np.savez_compressed(file, **arrays)
        shard["sha256"] = sha256(file)
    write_json(dataset / "manifest.json", manifest)
    return dataset


class MultiSeedTests(unittest.TestCase):
    def test_reviewed_source_migration_preserves_ten_bucket_arrays_and_weights(self):
        from train.common import data as current
        base = Path(current.__file__).parent
        source = (base / "data.py").read_text(encoding="utf-8")
        source = source.replace(
            '        self.timesteps = m.get("timesteps")\n'
            '        if (type(self.timesteps) is not int or not 1 <= self.timesteps <= 127\n'
            '                or m.get("feature_width") != 65):\n'
            '            raise ValueError("Expected Scheme2 inputs [N,T,65] with 1 <= T <= 127")',
            '        if (m.get("timesteps"), m.get("feature_width")) != (10, 65):\n'
            '            raise ValueError("Expected Scheme2 inputs [N,10,65]")')
        source = source.replace("self.timesteps", "10").replace(
            '    if x.shape[1] != 10:\n'
            '        raise ValueError("The circular-shift experiment requires the original ten-bucket grid")\n', "")
        experiment = (base / "experiment.py").read_text(encoding="utf-8").replace(
            '    if start and data.timesteps != 10:\n'
            '        raise ValueError("Circular-shift experiments require the original ten-bucket grid")\n', "")
        for key, text in (("common/data.py", source), ("common/experiment.py", experiment)):
            candidates = (text.encode(), text.replace("\n", "\r\n").encode())
            self.assertIn(TEN_BUCKET_SOURCE_MIGRATIONS[key][0], {hashlib.sha256(value).hexdigest() for value in candidates})
        legacy = {"__package__": "train.common", "__name__": "train.common.legacy_fixture_data"}
        exec(compile(source, "legacy_fixture_data.py", "exec"), legacy)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = web_fixture(root / "dataset")
            for split_name in ("train", "validation"):
                old = legacy["Scheme2"](dataset, root / "old_cache").load(split_name)
                new = current.Scheme2(dataset, root / "new_cache").load(split_name)
                try:
                    self.assertEqual(old.info["cohort_sha256"], new.info["cohort_sha256"])
                    for key in new.arrays:
                        np.testing.assert_array_equal(old[key], new[key])
                    for start in (0, 5):
                        np.testing.assert_array_equal(legacy["flat_inputs"](old, start), current.flat_inputs(new, start))
                    if split_name == "train":
                        np.testing.assert_array_equal(legacy["training_weights"](old), current.training_weights(new))
                finally:
                    old.close()
                    new.close()

    def test_statistics_use_paired_differences_and_sample_sd(self):
        rows = [paired_row("lstm", seed, "ap", a, b) for seed, a, b in ((1, .2, .3), (2, .5, .4), (3, .7, .7))]
        row = statistics(rows, ("model", "metric"))[0]
        self.assertAlmostEqual(row["delta_mean"], 0)
        self.assertAlmostEqual(row["delta_sd"], 10)
        self.assertEqual((row["positive_seeds"], row["negative_seeds"], row["zero_seeds"]), (1, 1, 1))
        empty = statistics([paired_row("cnn", 1, "web", None, None)], ("model", "metric"))[0]
        self.assertIsNone(empty["delta_mean"])
        self.assertEqual(empty["n_nonempty_pairs"], 0)
        self.assertIsNone(statistics(rows[:1], ("model", "metric"))[0]["delta_sd"])

    def test_direct_cli_all_models_two_seeds_and_metadata_only_summary(self):
        project = Path(__file__).resolve().parents[2]
        script = project / "train" / "run_circular_shift_multiseed.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = web_fixture(root / "dataset")
            # No test shards are accessible to any fit or report.
            shard = data / "test" / "part_000000.npz"
            shard.rename(shard.with_suffix(".hidden"))
            command = [sys.executable, str(script), "--dataset", str(data), "--cache", str(root / "cache"),
                       "--out", str(root / "study"), "--seeds", "10", "11", "--device", "cpu",
                       "--threads", "1", "--epochs", "1", "--rounds", "2"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            study = root / "study"
            plan = read_json(study / "plan.json")
            self.assertEqual(len(plan["runs"]), 16)
            self.assertEqual({record["seed"] for record in plan["runs"]}, {10, 11})
            status = read_json(study / "status.json")
            self.assertEqual(status["status"], "complete")
            report = Path(status["latest_report"])
            summary = read_json(report / "summary.json")
            self.assertFalse(summary["test_used"])
            self.assertTrue(all(row["n_pairs"] == 2 for row in summary["aggregate"]))
            self.assertEqual({(row["group"], row["n"]) for row in summary["web_aggregate"]},
                             {("all", 24), ("one", 8), ("two", 8), ("three_or_more", 8)})
            raw = np.genfromtxt(report / "per_seed.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            for row in summary["aggregate"]:
                keep = (raw["model"] == row["model"]) & (raw["metric"] == row["metric"])
                self.assertAlmostEqual(row["delta_mean"], raw["delta"][keep].mean())
                self.assertAlmostEqual(row["delta_sd"], raw["delta"][keep].std(ddof=1))
            artifacts = {record["path"]: sha256(Path(record["path"]) / "predictions.npz") for record in plan["runs"]}
            with patch("train.run_circular_shift_multiseed.train_model", side_effect=AssertionError("Must not fit")):
                multi_seed(["--resume", str(study)])
                fresh_report = multi_seed(["--summarize", str(study)])
            self.assertNotEqual(report, fresh_report)
            self.assertEqual(artifacts, {folder: sha256(Path(folder) / "predictions.npz") for folder in artifacts})

    def test_reuse_plan_interruption_retry_and_incompatibility_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = web_fixture(root / "dataset")
            settings = ["--dataset", str(data), "--cache", str(root / "cache"), "--models", "lstm",
                        "--device", "cpu", "--threads", "1", "--epochs", "1"]
            old = single_seed(settings + ["--shift-only", "--out", str(root / "old")])
            reused = old / "lstm" / "start5"
            original_hash = sha256(reused / "predictions.npz")
            with patch("train.run_circular_shift_multiseed.train_model", side_effect=AssertionError("Must not fit")):
                study = multi_seed(settings + ["--seeds", "4266", "4267", "--reuse-batch", str(old),
                                               "--out", str(root / "study"), "--plan-only"])
            plan = read_json(study / "plan.json")
            self.assertEqual(sum(record["reused"] for record in plan["runs"]), 1)
            with patch("train.run_circular_shift_multiseed.train_model", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    multi_seed(["--resume", str(study)])
            # Simulate a partially written fit directory from the interrupted run.
            interrupted = Path(plan["runs"][0]["path"])
            interrupted.mkdir(parents=True, exist_ok=True)
            write_json(interrupted / "status.json", {"status": "failed"})
            calls = []
            def fit(name, argv):
                calls.append(argv)
                return train_model(name, argv)
            with patch("train.run_circular_shift_multiseed.train_model", side_effect=fit):
                multi_seed(["--resume", str(study)])
            self.assertEqual(len(calls), 3)
            final = read_json(study / "plan.json")
            self.assertEqual(final["runs"][0]["attempts"], [str(interrupted)])
            self.assertNotEqual(final["runs"][0]["path"], str(interrupted))
            self.assertEqual(read_json(interrupted / "status.json")["status"], "failed")
            self.assertEqual(sha256(reused / "predictions.npz"), original_hash)
            with self.assertRaisesRegex(ValueError, "training settings"):
                multi_seed([arg if arg != "1" else "2" for arg in settings] +
                           ["--seeds", "4266", "--reuse-batch", str(old), "--plan-only"])
            config_path = reused / "config.json"
            changed = read_json(config_path)
            for file, (legacy_hash, current_hash) in TEN_BUCKET_SOURCE_MIGRATIONS.items():
                key = str(Path(file))
                self.assertEqual(changed["source_sha256"][key], current_hash)
                changed["source_sha256"][key] = legacy_hash
            write_json(config_path, changed)
            compatible = multi_seed(settings + ["--seeds", "4266", "--reuse-batch", str(old),
                                                 "--out", str(root / "compatible"), "--plan-only"])
            reused_record = next(record for record in read_json(compatible / "plan.json")["runs"] if record["reused"])
            self.assertEqual(len(reused_record["source_compatibility"]), 2)
            changed["source_sha256"][str(Path("common/data.py"))] = "unknown-source"
            write_json(config_path, changed)
            with self.assertRaisesRegex(ValueError, "different training source"):
                multi_seed(settings + ["--seeds", "4266", "--reuse-batch", str(old), "--plan-only"])
            changed["environment"]["packages"]["torch"] = "incompatible-fixture-version"
            write_json(config_path, changed)
            with self.assertRaisesRegex(ValueError, "runtime environment"):
                multi_seed(settings + ["--seeds", "4266", "--reuse-batch", str(old), "--plan-only"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                multi_seed(["--resume", str(study), "--epochs", "1"])


if __name__ == "__main__":
    unittest.main()
