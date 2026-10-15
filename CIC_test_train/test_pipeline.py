"""Focused checks plus a tiny synthetic CUDA integration run (not real results).

    python CIC_test_train/test_pipeline.py

Creates disposable SQLite fixtures in the OS temporary directory. Never trains
on or modifies the real CIC data. CUDA integration is skipped if CUDA is absent.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from common import (BIN_MS, DAY_MS, HERE, ROOT, SEQ_LEN, SPLITS, WINDOW_DTYPE,
                    WindowData, dump_json, feature_groups, read_json)
from prepare_data import balanced_sample, make_windows, split_blocks
from train_cnn_ae import MaskedCNNAutoencoder, MaskedTransformer, per_window_mse


class WindowTests(unittest.TestCase):
    def test_sparse_grid_preserves_gaps_and_labels(self):
        base = 1518652800000  # UTC day boundary
        block_ms = 1_800_000
        stamps = [base, base + 20_000, base + 90_000, base + 3_600_000]
        meta = [(i, t, int(i == 2), 1 if i == 2 else 0, int(i == 2)) for i, t in enumerate(stamps)]
        assignment = {t // block_ms: "test" for t in stamps}
        w, tags = make_windows(0, meta, 5, block_ms, assignment)
        first = w[w["start_ms"] == base][0]
        self.assertEqual(first["active"], 3)
        self.assertEqual(first["label"], 1)
        self.assertEqual(first["rows"].tolist(), [0, -1, 1, -1, -1, -1, -1, -1, -1, 2])
        self.assertFalse(any((row["rows"] == 2).any() and (row["rows"] == 3).any() for row in w))
        benign_meta = [(i, t, 0, 0, 0) for i, t in enumerate(stamps)]
        b, benign_tags = make_windows(0, benign_meta, 5, block_ms, assignment)
        np.testing.assert_array_equal(w["start_ms"], b["start_ms"])
        np.testing.assert_array_equal(w["rows"], b["rows"])
        np.testing.assert_array_equal(tags, benign_tags)

    def test_no_block_boundary_crossing_for_all_strides(self):
        block_ms = 1_800_000
        for stride in (1, 2, 3, 5, 10):
            meta = [(i, t, 0, 0, 0) for i, t in enumerate(range(block_ms - 90_000, block_ms + 100_000, BIN_MS))]
            w, _ = make_windows(0, meta, stride, block_ms, {0: "train", 1: "test"})
            self.assertTrue(len(w))
            np.testing.assert_array_equal(w["start_ms"] // block_ms,
                                          (w["start_ms"] + SEQ_LEN * BIN_MS - 1) // block_ms)

    def test_blocks_deterministic_and_label_independent(self):
        blocks = list(range(100, 130))
        a = split_blocks(blocks, 1_800_000, 0)
        self.assertEqual(a, split_blocks(reversed(blocks), 1_800_000, 0))
        self.assertEqual(set(a), set(blocks))
        self.assertEqual(set(a.values()), set(SPLITS))

    def test_masked_score_is_not_diluted_by_empty_bins(self):
        x = torch.zeros(2, 10, 88)
        mask = torch.zeros(2, 10)
        x[0, :1], mask[0, :1] = 2, 1
        x[1, :5], mask[1, :5] = 2, 1
        result = per_window_mse(x, torch.zeros_like(x), mask)
        torch.testing.assert_close(result, torch.tensor([4., 4.]))
        model = MaskedCNNAutoencoder(width=4, latent=3)
        self.assertEqual(tuple(model(x, mask).shape), (2, 10, 88))
        changed = x.clone()
        changed[mask == 0] = 12345
        torch.testing.assert_close(model(x, mask), model(changed, mask))

    def test_sampling_cap_and_repeatability(self):
        w = np.zeros(200, dtype=WINDOW_DTYPE)
        w["host"] = np.repeat(np.arange(10), 20)
        w["start_ms"] = np.tile(np.arange(20) * 50_000, 10)
        a = balanced_sample(w, 50, 0, 1_800_000, .1)
        b = balanced_sample(w, 50, 0, 1_800_000, .1)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 50)
        self.assertLessEqual(max(Counter(a["host"]).values()), 5)

    def test_transform_fit_unique_train_observations_only(self):
        class FakeData:
            buckets = np.array([[1, .3], [3, .5], [9999, .9]], dtype=np.float32)
            windows = {"train": np.zeros(2, dtype=WINDOW_DTYPE)}
        data = FakeData()
        data.windows["train"]["rows"] = -1
        data.windows["train"]["rows"][:, :2] = [0, 1]
        transform = MaskedTransformer(["flow_count", "protocol_tcp_ratio"]).fit(data)
        self.assertEqual(transform.n_fit_buckets, 2)
        self.assertAlmostEqual(transform.scaler.mean_[0], (np.log(2) + np.log(4)) / 2)
        x = np.zeros((1, 10, 2), dtype=np.float32)
        mask = np.zeros((1, 10), dtype=np.float32)
        x[0, 0], mask[0, 0] = [1, .3], 1
        out = transform.transform(x, mask)
        self.assertTrue(np.all(out[0, 1:] == 0))
        self.assertAlmostEqual(float(out[0, 0, 1]), .3, places=6)


def create_fixture(folder):
    schema = read_json(ROOT / "processed_binary_temporal/categorical_schema.json")
    dump_json(folder / "schema.json", schema)
    names = schema["feature_names"]
    log, bounded = feature_groups(names)
    conn = sqlite3.connect(folder / "fixture.db")
    cols = ",".join('"' + n + '" REAL' for n in names)
    conn.execute(f"CREATE TABLE cic_temporal_buckets(ipv4_src_addr TEXT,bucket_start_ms INTEGER,label INTEGER,{cols})")
    conn.execute("CREATE TABLE network_flows(ipv4_src_addr TEXT,flow_start_milliseconds INTEGER,label INTEGER,attack TEXT)")
    rng = np.random.default_rng(3)
    base = 1518652800000
    for host in range(4):
        ip = f"10.0.0.{host + 1}"
        for day in range(2):
            for block in range(12):
                for tick in range(180):
                    if (tick + host) % 7 == 0:
                        continue
                    t = base + day * DAY_MS + block * 1_800_000 + tick * BIN_MS
                    y = int(tick % 60 in (30, 31, 32))
                    x = np.zeros(len(names), dtype=np.float32)
                    x[log] = rng.uniform(1, 5, len(log)) + y * 8
                    x[bounded] = rng.uniform(0, 1, len(bounded))
                    x[names.index("flow_count")] = 1
                    conn.execute(f"INSERT INTO cic_temporal_buckets VALUES({','.join('?' for _ in range(3 + len(names)))})",
                                 (ip, t, y, *map(float, x)))
                    conn.execute("INSERT INTO network_flows VALUES(?,?,?,?)", (ip, t, y, "SyntheticAttack" if y else "Benign"))
    conn.execute("CREATE UNIQUE INDEX bucket_key ON cic_temporal_buckets(ipv4_src_addr,bucket_start_ms)")
    conn.commit()
    conn.close()


class IntegrationTests(unittest.TestCase):
    def test_preparation_and_cuda_training(self):
        with tempfile.TemporaryDirectory(prefix="cic_pipeline_test_") as tmp:
            folder = Path(tmp)
            create_fixture(folder)
            output = folder / "prepared"
            command = [sys.executable, str(HERE / "prepare_data.py"), "--db", str(folder / "fixture.db"),
                       "--schema", str(folder / "schema.json"), "--output", str(output),
                       "--train-budget", "160", "--validation-budget", "40", "--calibration-budget", "40",
                       "--max-host-fraction", "1"]
            subprocess.run(command + ["--check-only"], check=True, cwd=tmp)
            self.assertFalse(output.exists())
            subprocess.run(command, check=True, cwd=tmp)
            with WindowData(output) as data:
                self.assertTrue(data.manifest["complete"])
                self.assertEqual(data.manifest["stride"], 5)
                self.assertEqual(data.manifest["cross_split_shared_buckets"], 0)
                self.assertEqual(set(data.windows["test"]["label"]), {0, 1})
                self.assertTrue((data.windows["test"]["active"] < 10).any())
                for split in SPLITS[:-1]:
                    self.assertTrue(np.all(data.windows[split]["label"] == 0))
            if not torch.cuda.is_available():
                print("[test] CUDA unavailable: data checks passed; CUDA training integration skipped")
                return
            run_dir = folder / "run"
            subprocess.run([sys.executable, str(HERE / "train_cnn_ae.py"), "--data-dir", str(output),
                            "--outdir", str(run_dir), "--epochs", "2", "--patience", "2",
                            "--batch-size", "32", "--budget", "5000", "--latent", "4"], check=True, cwd=tmp)
            self.assertTrue(read_json(run_dir / "run_config.json")["completed"])
            result = read_json(run_dir / "test_metrics.json")
            self.assertTrue(np.isfinite(result["roc_auc"]))
            for name in ("metrics_by_attack.csv", "metrics_by_day.csv", "metrics_by_occupancy.csv",
                         "calibration_scores.csv", "test_scores.csv", "cnn_ae_checkpoint.pt", "scaler.npz"):
                self.assertTrue((run_dir / name).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
