"""Focused tests for labels, temporal positions, midnight tails and no overlap."""
import csv
import gzip
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("sequence_eda", Path(__file__).with_name("02_compare_sequence_lengths.py"))
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


class SequenceTests(unittest.TestCase):
    def test_temporal_shape(self):
        self.assertEqual(eda.temporal_shape([0, 1, 9], 10), (70, 20))
        self.assertEqual(eda.temporal_shape([0], 100), (990, 10))
        self.assertEqual(eda.temporal_shape([1, 2, 3], 5), (10, 30))
        with self.assertRaises(ValueError):
            eda.temporal_shape([1, 1], 5)

    def test_disjoint_windows_and_tails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            conn = sqlite3.connect(":memory:")
            conn.execute("""CREATE TABLE source(entity TEXT,bucket_start_ms INTEGER,flows INTEGER,
                attack_flows INTEGER,total_bytes INTEGER,total_packets INTEGER,first_ms INTEGER,
                last_ms INTEGER,max_flow_duration_ms INTEGER)""")
            # A has hybrid at 50/100s but pure attack in a later 50s window.
            # B has benign at 23:59:50, excluded by the 1000s day-end tail.
            # C has a mixed-label bucket. D spans midnight, never one sequence.
            rows = [("A", 0, 2, 0), ("A", 10000, 1, 1), ("A", 90000, 2, 2), ("A", 100000, 1, 0),
                    ("B", 0, 1, 0), ("B", 86390000, 2, 0), ("C", 0, 2, 1),
                    ("D", 86390000, 1, 0), ("D", 86400000, 1, 1)]
            conn.executemany("INSERT INTO source VALUES (?,?,?,?,?,?,?,?,?)",
                             [(ip, t, n, a, n * 10, n, t + 1, t + 2, 20) for ip, t, n, a in rows])
            conn.execute("CREATE TABLE network AS SELECT DISTINCT bucket_start_ms FROM source")
            lookup = {(ip, t): [("Attack", a, t + 1, t + 2)] for ip, t, n, a in rows if a}
            attacks = {"Attack": sum(r[3] for r in rows)}
            controls = {"flows": sum(r[2] for r in rows), "attack_flows": attacks["Attack"], "buckets": len(rows)}
            prefix = eda.global_prefix(conn)
            results = {length: eda.analyze_length(conn, length, out, lookup, attacks, prefix, controls)
                       for length in (50, 100, 200, 1000)}
            with gzip.open(out / "windows_100s.csv.gz", "rt", encoding="utf-8-sig", newline="") as f:
                windows = list(csv.DictReader(f))
            a0 = next(r for r in windows if r["source_ip"] == "A" and r["start_ms"] == "0")
            self.assertEqual(a0["activity_class"], "hybrid")
            self.assertEqual(int(a0["active_buckets"]), 3)
            self.assertEqual(a0["bucket_mask_hex"], hex((1 << 0) | (1 << 1) | (1 << 9)))
            self.assertEqual(int(a0["longest_empty_run_s"]), 70)
            self.assertEqual(int(a0["hybrid_buckets"]), 0)
            self.assertEqual(int(a0["attack_flows"]), 3)
            self.assertEqual(results[100]["summary"]["same_bucket_hybrid_windows"], 1)
            self.assertEqual(results[100]["summary"]["cross_bucket_only_hybrid_windows"], 1)
            self.assertEqual(results[1000]["summary"]["excluded_tail_flows"], 3)
            self.assertEqual(results[50]["summary"]["excluded_tail_flows"], 0)
            self.assertEqual(sum(int(r["flows"]) for r in windows), controls["flows"])
            self.assertTrue(all(int(r["flows"]) > 0 for r in windows))
            for r in windows:
                start, end = int(r["start_ms"]), int(r["end_ms_exclusive"])
                self.assertEqual(start // eda.DAY_MS, (end - 1) // eda.DAY_MS)
            for result in results.values():
                self.assertEqual(result["summary"]["flows"] + result["summary"]["excluded_tail_flows"], controls["flows"])
            conn.close()


if __name__ == "__main__":
    unittest.main()
