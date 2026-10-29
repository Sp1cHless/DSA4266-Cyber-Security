"""Small end-to-end train/validation-only temporal CNN fixture checks."""
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
import torch

from CIC_new.tests.fixtures import fixture_database
from train.common.experiment import main as train_model
from train.common.io import PROJECT, read_json, write_json, sha256
from train.evaluate import main as evaluate_model
from train.run_temporal_cnn import main as run_cnn_batch
from train.tests.fixture import make_fixture


class TemporalCNNTests(unittest.TestCase):
    MODEL = "cnn"
    run_batch = staticmethod(run_cnn_batch)

    def test_four_cases_reuse_buckets_match_reference_and_never_read_test(self):
        run_batch = self.run_batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = make_fixture(root / "baseline_dataset")
            (original / "test").rename(original / "test_not_available")
            manifest = read_json(original / "manifest.json")
            for split in ("train", "validation"):
                path = original / split / "part_000000.npz"
                with np.load(path) as saved:
                    payload = {key: saved[key] for key in saved.files}
                positive = np.flatnonzero(payload["y_sequence"])
                payload["primary_attack"] = payload["primary_attack"].astype("U80")
                payload["primary_attack"][positive[:len(positive)//2]] = "Infilteration"
                payload["primary_attack"][positive[len(positive)//2:]] = "Brute_Force_-Web"
                np.savez_compressed(path, **payload)
                next(record for record in manifest["shards"] if record["path"] == f"{split}/part_000000.npz")["sha256"] = sha256(path)
            write_json(original / "manifest.json", manifest)
            config = read_json(PROJECT / f"train/{self.MODEL}/config.json")
            config["training"].update(epochs=1, patience=1, batch_size=32)
            write_json(root / "fixture_detector.json", config)
            baseline = train_model(self.MODEL, ["--dataset", str(original), "--config", str(root / "fixture_detector.json"),
                       "--cache", str(root / "baseline_cache"), "--out", str(root / "baseline"),
                       "--device", "cpu", "--threads", "1"])
            db = root / "flows.sqlite"
            fixture_database(db)
            connection = sqlite3.connect(db)
            try:
                connection.execute("UPDATE network_flows SET attack='Infilteration' WHERE attack='Alpha'")
                connection.execute("UPDATE network_flows SET attack='Brute_Force_-Web' WHERE attack='Beta'")
                connection.commit()
            finally:
                connection.close()
            prepared = root / "prepared_batch"
            result = subprocess.run([sys.executable, str(PROJECT / "CIC_new/run_temporal_ablation.py"),
                         "--db", str(db), "--out", str(prepared)], cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            preparations = read_json(prepared / "batch.json")
            for item in preparations["cases"].values():
                path = Path(item["preprocess"]["path"])
                (path / "splits/test").rename(path / "splits/test_not_available")
            # Explicit fixture references reduce budgets; production references are never edited.
            encoder = root / "fixture_encoder.json"
            write_json(encoder, {"seed": 4266, "steps": 1, "validate_every": 1, "batch_size": 8,
                       "cpu_threads": 1, "category_dim": 4, "bucket_dim": 64, "val_limit": 16, "lr": .001})
            output = root / "training"
            arguments = ["--prepared-batch", str(prepared), "--baseline-run", str(baseline),
                         "--embedding-config", str(encoder), "--out", str(output), "--shard-size", "17"]
            run_batch(arguments + ["--dry-run"])
            self.assertFalse(output.exists())
            # Artificial historical metadata differ; use the current environment without an opt-in flag.
            historical_config = read_json(baseline / "config.json")
            historical_config["environment"]["packages"]["torch"] = "fixture-historical-version"
            write_json(baseline / "config.json", historical_config)
            run_batch(arguments)
            state = read_json(output / "batch.json")
            self.assertTrue(state["complete"])
            self.assertFalse(state["test_used"])
            self.assertEqual(len(state["cases"]), 4)
            self.assertEqual(state["settings"]["model"], self.MODEL)
            self.assertEqual(state["settings"]["runtime_differences"]["torch"]["baseline"],
                             "fixture-historical-version")
            self.assertEqual(state["settings"]["runtime_differences"]["torch"]["current"],
                             state["settings"]["packages"]["torch"])
            original_summary = read_json(baseline / "summary.json")
            for case in state["settings"]["cases"]:
                item = state["cases"][case["name"]]
                representation = Path(item["representation"]["path"])
                dataset = representation / "dataset"
                self.assertFalse((representation / "encoded_sequences").exists())
                self.assertTrue((representation / "bucket_embeddings/best.pt").is_file())
                self.assertFalse((representation / "encoded_sequences/test").exists())
                self.assertFalse((dataset / "test").exists())
                self.assertFalse((Path(case["prepared"]["path"]) / "mmap/test").exists())
                self.assertFalse((Path(case["prepared"]["path"]) / "mmap/train").exists())
                self.assertFalse((Path(case["prepared"]["path"]) / "mmap/validation").exists())
                self.assertEqual(set(read_json(dataset / "manifest.json")["counts"]), {"train", "validation"})
                self.assertEqual(set(read_json(dataset / "flow_metadata.json")["splits"]), {"validation"})
                run = Path(item[self.MODEL]["path"])
                self.assertFalse(run.with_name(run.name + "_cache").exists())
                saved = read_json(run / "config.json")
                self.assertEqual(saved["training"], config["training"])
                self.assertEqual(saved["architecture"], config["architecture"])
                self.assertEqual(saved["input_transform"], {"name": "identity"})
                audit = read_json(run / "data_audit.json")
                self.assertFalse(audit["test_opened"])
                totals = list(audit["train_group_weight_totals"].values())
                np.testing.assert_allclose(totals, np.repeat(sum(totals) / 4, 4), rtol=1e-6)
                summary = read_json(run / "summary.json")
                self.assertFalse(summary["test_used"])
                self.assertEqual(summary["fit"]["parameters"], original_summary["fit"]["parameters"])
                replay = evaluate_model(["--run", str(run), "--device", "cpu", "--threads", "1",
                                         "--cache", str(root / "replay_cache")])
                with np.load(run / "predictions.npz") as left, np.load(replay / "predictions.npz") as right:
                    np.testing.assert_allclose(left["score"], right["score"], rtol=1e-6, atol=1e-7)
            report = read_json(output / "validation_comparison.json")
            self.assertEqual(len(report["rows"]), 5)
            self.assertFalse(report["paired_cohorts"])
            self.assertEqual(report["rows"][0]["case"], "existing_baseline")
            self.assertEqual(report["model"], self.MODEL)
            self.assertEqual(len(report["focused_attacks"]), 20)
            self.assertTrue((output / "ATTACK_COMPARISON.md").is_file())
            self.assertTrue((output / "attack_comparison.csv").is_file())
            for row in report["focused_attacks"]:
                self.assertEqual(row["tp"] + row["fn"], row["n"])
                if row["n"]:
                    self.assertAlmostEqual(row["recall"], row["tp"] / row["n"])
                    self.assertAlmostEqual(row["delta_recall_pp_descriptive"],
                                           100 * (row["recall"] - row["baseline_recall"]))
            run_batch(arguments + ["--resume"])
            resumed = read_json(output / "batch.json")
            self.assertTrue(all(len(item["attempts"]) == 2 for item in resumed["cases"].values()))
            self.assertFalse((output / "RUNNING.lock").exists())
            run_batch(arguments + ["--resume", "--cleanup-only"])
            self.assertEqual(resumed, read_json(output / "batch.json"))

    def test_identity_matches_original_forward_without_resizing(self):
        import importlib
        from train.common.networks import pool, sequence_input
        torch.set_num_threads(1)
        model = importlib.import_module(f"train.{self.MODEL}.model").Model(dropout=.2).eval()
        x = torch.randn(3, 10, 65)
        mask = torch.tensor([[1, 0, 1, 0, 1, 1, 1, 0, 1, 1]] * 3, dtype=torch.float32)
        with torch.no_grad():
            # The original baseline performs these same operations with no rotation.
            if self.MODEL == "lstm":
                outputs, (hidden, _) = model.lstm(sequence_input(x, mask))
                expected = model.head(torch.cat((pool(outputs, mask), hidden[-1]), dim=-1)).squeeze(-1)
            else:
                h = sequence_input(x, mask).transpose(1, 2)
                h = model.dropout(model.relu(model.first(h))) * mask.unsqueeze(1)
                h = model.relu(model.second(h)).transpose(1, 2)
                expected = model.head(pool(h, mask)).squeeze(-1)
            torch.testing.assert_close(model(x, mask), expected, rtol=0, atol=0)
            for times in (20, 40):
                self.assertEqual(model(torch.randn(3, times, 65), torch.ones(3, times)).shape, (3,))


if __name__ == "__main__":
    unittest.main()
