"""Small independent-reference and direct-terminal tests; never scan the real DB."""
import hashlib
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.dataset import ModelInputs
from CIC_new.scheme1.statistics import (FIELDS, HISTOGRAMS, raw_bucket, merge_buckets,
                                       category_indices, project_histogram)
from CIC_new.scheme1.storage import arrays, read_json, digest, write_json
from CIC_new.scheme1.pipeline import build
from CIC_new.scheme1.validate import validate
from CIC_new.tests.fixtures import event, fixture_database

ROOT = Path(__file__).resolve().parents[2]


class SchemeOneTests(unittest.TestCase):
    def test_statistics_match_event_level_union(self):
        rng = np.random.default_rng(52)
        for trial in range(30):
            states, events = [], []
            for side in range(2):
                rows = []
                for j in range(int(rng.integers(1, 25))):
                    protocol = int(rng.choice([1, 6, 17]))
                    port = int(rng.choice([0, 53, 80, 443, 50000]))
                    fields = dict(l4_src_port=int(rng.integers(0, 8)), l4_dst_port=port,
                                  flow_duration_milliseconds=int(rng.integers(0, 500000)),
                                  in_bytes=int(rng.integers(0, 100000)), tcp_flags=int(rng.integers(0, 256)),
                                  client_tcp_flags=int(rng.integers(0, 256)), server_tcp_flags=int(rng.integers(0, 256)),
                                  dns_query_id=int(port == 53), dns_query_type=int(rng.choice([1, 28])) if port == 53 else 0,
                                  icmp_ipv4_type=int(rng.choice([0, 3, 8])) if protocol == 1 else 0,
                                  tcp_win_max_in=int(rng.integers(0, 65000)), duration_in=int(rng.integers(0, 1000)),
                                  min_ip_pkt_len=int(rng.integers(20, 80)), num_pkts_128_to_256_bytes=int(rng.integers(0, 20)))
                    # Many duplicate starts and interleaving timestamps exercise gap recomputation.
                    rows.append(event(int(rng.integers(0, 100)) * 1000, label=int(side == 1 and j % 3 == 0),
                                      protocol=protocol, l7=int(rng.integers(0, 5)), destination=int(rng.integers(1, 5)), **fields))
                matrix = np.stack(rows)
                matrix = matrix[np.argsort(matrix[:, bf.E["flow_start_milliseconds"]], kind="stable")]
                events.append(matrix)
                states.append(raw_bucket(matrix))
            combined = merge_buckets(*states)
            reference = raw_bucket(np.concatenate(events))
            np.testing.assert_allclose(combined["numeric"], reference["numeric"], rtol=1e-12, atol=1e-9,
                                       err_msg=f"trial {trial}")
            for key in ("mask", "flow_counts", "attack_flow_counts", "y_bucket", "packet_bins"):
                np.testing.assert_array_equal(combined[key], reference[key])
            for field in HISTOGRAMS:
                np.testing.assert_array_equal(combined["hist_" + field], reference["hist_" + field])

    def test_unknown_zero_and_weighted_category_composition(self):
        np.testing.assert_array_equal(category_indices(np.array([0, 999]), [0, 5]), [1, 0])
        table = np.array([[0., 0.], [1., 2.], [4., -1.]])
        a = np.array([[0, 0, 2]], np.int64)
        b = np.array([[0, 5, 1]], np.int64)
        combined = np.concatenate((a, b))
        result = project_histogram(combined, 1, [0, 5], table)
        np.testing.assert_allclose(result, [[2., 1.]])
        empty = project_histogram(np.empty((0, 3), np.int64), 1, [], None)
        np.testing.assert_array_equal(empty, [[0.]])

    def test_no_torch_in_data_processing(self):
        code = "import builtins; original=builtins.__import__; exec(\"def checked(name,*a,**k):\\n if name=='torch' or name.startswith('torch.'):\\n  raise AssertionError('data processing imported torch')\\n return original(name,*a,**k)\"); builtins.__import__=checked; import CIC_new.scheme1.pipeline; import CIC_new.scheme1.validate"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def terminal(self, root, name, *arguments):
        result = subprocess.run([sys.executable, str(ROOT / "CIC_new" / name), *map(str, arguments)],
                                cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_terminal_workflow_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "flows.sqlite"
            fixture_database(db)
            connection = sqlite3.connect(db)
            connection.execute("UPDATE network_flows SET l4_src_port=rowid%31,l4_dst_port=CASE rowid%3 WHEN 0 THEN 53 WHEN 1 THEN 443 ELSE 22 END")
            connection.execute("UPDATE network_flows SET dns_query_id=1,dns_query_type=CASE rowid%2 WHEN 0 THEN 1 ELSE 28 END WHERE l4_dst_port=53")
            connection.execute("UPDATE network_flows SET protocol=1,icmp_ipv4_type=CASE rowid%2 WHEN 0 THEN 0 ELSE 8 END WHERE ipv4_src_addr='10.0.0.1' AND rowid%3=0")
            connection.execute("UPDATE network_flows SET l4_src_port=65000,l4_dst_port=65001 WHERE l7_proto=999")
            connection.commit()
            connection.close()
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            out = root / "separate"
            prep_args = ("--db", db, "--out", out, "--shard-size", 17, "--train-pool", 4, "--val-pool", 32)
            self.terminal(root, "prepare_dataset.py", *prep_args, "--prepare-only")
            prepared = out / "prepared"
            schema = read_json(prepared / "schema.json")
            self.assertNotIn(999, schema["vocabularies"]["l7"])
            self.assertNotIn(65000, schema["vocabularies"]["src_port"])
            self.assertNotIn(65001, schema["vocabularies"]["dst_port"])
            self.assertIn(0, schema["vocabularies"]["src_port"])
            pool = arrays(prepared / "embedding_pools/train.npz")
            self.assertGreater(len(pool["numeric"]), 4)  # rare-category witnesses supplement the uniform pool
            for field in ("l7", "src_port", "dst_port"):
                np.testing.assert_array_equal(np.unique(pool[field + "_indices"]),
                                              np.arange(1, len(schema["vocabularies"][field]) + 1))
            embeddings = out / "custom_embeddings"
            self.terminal(root, "train_embeddings.py", "--data", prepared, "--out", embeddings,
                          "--steps", 6, "--validate-every", 3, "--batch-size", 8, "--device", "cpu",
                          "--cpu-threads", 1, "--l7-dim", 2, "--src-port-dim", 3, "--dst-port-dim", 4)
            saved = arrays(embeddings / "embeddings.npz")
            for field, dimension in (("l7", 2), ("src_port", 3), ("dst_port", 4)):
                self.assertEqual(saved[field].shape, (len(schema["vocabularies"][field]) + 1, dimension))
                self.assertTrue(np.all(saved[field][0] == 0))
            weight_hash = digest(embeddings / "embeddings.npz")
            self.terminal(root, "prepare_dataset.py", *prep_args, "--resume", "--embeddings", embeddings, "--skip-validation")
            self.terminal(root, "validate_dataset.py", "--data", out / "dataset", "--deep-samples", 3)
            report = read_json(out / "dataset/validation.json")
            self.assertTrue(report["passed"])
            self.assertGreater(report["independent_flow_reconstructions"], 0)
            self.assertEqual(validate(out / "dataset", deep_samples=0)["independent_flow_reconstructions"], 0)
            with ModelInputs(out / "dataset", "train") as model_inputs:
                batch = next(model_inputs.batches(8, 1))
                self.assertEqual(int(batch["y_sequence"].sum()), 4)
            self.assertEqual(weight_hash, digest(embeddings / "embeddings.npz"))
            # Direct downstream entry: scheme1 dimensions, CPU subset and auto/full validation.
            for mode, extra in (("subset", ("--device", "cpu", "--val-limit", 16)),
                                ("full_control", ("--real-only",))):
                run = root / ("ae_" + mode)
                self.terminal(root, "train_autoencoder.py", "--dataset", out / "dataset", "--out", run,
                              "--epochs", 1, "--steps-per-epoch", 2, "--batch-size", 8,
                              "--width", 4, "--latent", 3, "--log-every", 1, "--cpu-threads", 1, *extra)
                summary = read_json(run / "summary.json")
                self.assertTrue(summary["complete"])
                self.assertFalse(summary["test_used"])
                self.assertEqual(summary["sample_draws"], 16)
                configuration = read_json(run / "config.json")
                self.assertEqual(configuration["model"]["features"], read_json(out / "dataset/manifest.json")["feature_width"])
                self.assertEqual(digest(run / "embeddings.npz"), weight_hash)
                expected_validation = 16 if mode == "subset" else read_json(out / "dataset/manifest.json")["total_samples"]["validation"]
                self.assertEqual(summary["validation_examples"], expected_validation)
                self.assertFalse((out / "dataset/model_cache/test").exists())
                if mode == "subset":
                    # Frozen post-training evaluation is separate from AE fitting and never opens test.
                    before_evaluation = digest(run / "best.pt")
                    self.terminal(root, "evaluate_autoencoder.py", "--run", run, "--device", "cpu", "--batch-size", 32)
                    report_path = next((run / "evaluations").glob("*/report.json"))
                    evaluation = read_json(report_path)
                    self.assertTrue(evaluation["complete"])
                    self.assertTrue(evaluation["training_artifacts_unchanged"])
                    self.assertFalse(evaluation["test_used"])
                    self.assertEqual(digest(run / "best.pt"), before_evaluation)
                    self.assertEqual(evaluation["sampling"]["records"], 16)
                    self.assertFalse(evaluation["linear_probe"]["parameters"]["validation_used_for_fitting"])
            automatic = root / "automatic"
            settings = dict(db=db, out=automatic, shard_size=17, train_pool=4, val_pool=32,
                            embedding_steps=3, batch_size=8, validate_every=30, device="auto", cpu_threads=1)
            self.assertTrue(build(**settings)["complete"])
            self.assertTrue(build(**settings, resume=True)["complete"])
            with self.assertRaises(FileExistsError):
                build(**settings)
            plan_a, plan_b = arrays(out / "dataset/donor_plan.npz"), arrays(automatic / "dataset/donor_plan.npz")
            for key in plan_a:
                np.testing.assert_array_equal(plan_a[key], plan_b[key])
            self.assertEqual(before, hashlib.sha256(db.read_bytes()).hexdigest())
            # Modify a fitted statistic and update its checksum: validation must independently refit it.
            scaler_path = automatic / "dataset/scaler.json"
            scaler = read_json(scaler_path)
            scaler["mean"][0] += 1
            write_json(scaler_path, scaler)
            manifest_path = automatic / "dataset/manifest.json"
            manifest = read_json(manifest_path)
            manifest["scaler_sha256"] = digest(scaler_path)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "Scaler does not match"):
                validate(automatic / "dataset")
            with self.assertRaisesRegex(ValueError, "Scaler does not match"):
                build(**settings, resume=True)
            self.assertFalse(read_json(automatic / "workflow.json")["complete"])
            self.assertFalse(read_json(automatic / "dataset/validation.json")["passed"])
            # A changed final shard fails the integrity check without reading/training a model.
            path = next((out / "dataset/train").glob("part_*.npz"))
            payload = arrays(path)
            payload["X"][0, 0, 0] += 1
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "Shard checksum mismatch"):
                validate(out / "dataset")
            failed = subprocess.run([sys.executable, str(ROOT / "CIC_new/validate_dataset.py"),
                                     "--data", str(out / "dataset")], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(read_json(out / "dataset/validation.json")["passed"])


if __name__ == "__main__":
    unittest.main()
