"""Artificial checks for a new feature fit against a completed read-only baseline."""
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

from CIC_new.preprocessing.distribution_features import aggregate, NAMES
from CIC_new.preprocessing.run_preprocessing import prepare
from CIC_new.prepare_scheme2_dataset import build
from CIC_new.tests.fixtures import fixture_database, event
from CIC_new.preprocessing import bucket_statistics as bf
from train.common.io import read_json, write_json, sha256
from train.common.experiment import main as train_model
from train.run_distribution_feature_ablation import main as run_batch


class DistributionFeatureTest(unittest.TestCase):
    def test_known_statistics_and_label_independence(self):
        values = np.stack([event(1000, in_bytes=50, out_bytes=0, out_pkts=0),
                           event(2000, in_bytes=150, out_bytes=100, out_pkts=2, destination=2)])
        result = aggregate(values)
        self.assertEqual(result.shape, (10, 15))
        self.assertAlmostEqual(result[0, NAMES.index("in_bytes_std")], 50)
        self.assertAlmostEqual(result[0, NAMES.index("zero_reply_fraction")], .5)
        self.assertAlmostEqual(result[0, NAMES.index("destination_port_pair_unique_fraction")], 1)
        self.assertTrue(np.all(result[1:] == 0))
        values[:, bf.E["label"]] = 1
        np.testing.assert_array_equal(result, aggregate(values))

    def test_new_only_pipeline_preserves_cohorts_and_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = root / "source.sqlite"
            fixture_database(db)
            connection = sqlite3.connect(db)
            connection.execute("UPDATE network_flows SET attack='Infilteration' WHERE attack='Alpha'")
            connection.execute("UPDATE network_flows SET attack='Brute_Force_-Web' WHERE attack='Beta'")
            connection.commit()
            connection.close()
            source = root / "original"
            prepare(source, db=db, shard_size=128)
            (source / "splits/test").rename(source / "splits/test_unavailable")
            reference = root / "reference"
            detector = {"model": "lstm", "architecture": {"hidden_size": 8, "dropout": .2},
                "training": {"epochs": 1, "patience": 1, "batch_size": 32, "learning_rate": .001, "weight_decay": .0001},
                "seed": 4266, "threads": 1, "device": "cpu", "input_transform": {"name": "identity"}}
            detector_config = root / "detector.json"
            write_json(detector_config, detector)
            baseline_data = root / "reference_representation"
            build(SimpleNamespace(db=db, out=baseline_data, prepared=source, bucket_checkpoint=None,
                resume=False, device="cpu", seed=4266, cpu_threads=1, shard_size=128,
                batch_size=32, embedding_steps=1, validate_every=1, val_limit=32,
                category_dim=2, bucket_dim=64, sequence_seconds=100, bucket_seconds=10,
                splits=["train", "validation"]))
            train_model("lstm", ["--dataset", str(baseline_data / "dataset"), "--config", str(detector_config),
                "--out", str(reference), "--cache", str(root / "reference_cache"), "--device", "cpu",
                "--threads", "1", "--seed", "4266"])
            reference_hashes = {p.name: sha256(p) for p in reference.iterdir() if p.is_file()}
            encoder = root / "encoder.json"
            write_json(encoder, {"seed": 4266, "cpu_threads": 1, "batch_size": 32, "steps": 1,
                "validate_every": 1, "category_dim": 2, "bucket_dim": 64, "lr": .001, "val_limit": 32})
            arguments = ["--prepared", str(source), "--reference-run", str(reference), "--encoder-config", str(encoder), "--db", str(db)]
            output = root / "batch"
            run_batch(arguments + ["--out", str(output), "--dry-run"])
            self.assertFalse(output.exists())
            original_hash = sha256(source / "manifest.sqlite")
            run_batch(arguments + ["--out", str(output)])
            batch = read_json(output / "batch.json")
            self.assertEqual(batch["status"], "complete")
            self.assertEqual(sha256(source / "manifest.sqlite"), original_hash)
            self.assertEqual(set(batch["cases"]), {"distribution97"})
            self.assertFalse((output / "control82").exists())
            self.assertEqual(reference_hashes, {p.name: sha256(p) for p in reference.iterdir() if p.is_file()})
            extended = Path(batch["cases"]["distribution97"]["prepared"]["path"])
            self.assertEqual(len(read_json(extended / "schema.json")["numeric_names"]), 97)
            self.assertEqual(read_json(source / "scaler.json")["mean"], read_json(extended / "scaler.json")["mean"][:82])
            for split in ("train", "validation"):
                for a in (source / "splits" / split).glob("part_*.npz"):
                    with np.load(a) as old, np.load(extended / "splits" / split / a.name) as new:
                        for key in old.files:
                            np.testing.assert_array_equal(old[key], new[key][..., :82] if key == "numeric" else new[key])
            result = read_json(output / "comparison.json")
            self.assertTrue(result["same_donor_pairs_verified"])
            self.assertTrue(result["baseline_reused"])
            self.assertEqual(Path(result["control"]), reference)
            self.assertFalse(result["test_used"])
            self.assertEqual(len(result["rows"]), 4)
            case = batch["cases"]["distribution97"]
            checkpoint = torch.load(Path(case["representation"]["path"]) / "bucket_embeddings/best.pt", weights_only=True)
            self.assertEqual(checkpoint["model_args"].get("numeric_width", 82), 97)
            fitted = read_json(Path(case["lstm"]["path"]) / "config.json")
            self.assertEqual(fitted["training"], detector["training"])
            before = sha256(output / "comparison.json")
            run_batch(arguments + ["--resume", str(output)])
            self.assertEqual(before, sha256(output / "comparison.json"))


if __name__ == "__main__":
    unittest.main()
