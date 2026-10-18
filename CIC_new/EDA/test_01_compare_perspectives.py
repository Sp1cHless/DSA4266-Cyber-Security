"""Focused accounting checks with known flows, a self-flow and a day boundary."""
import csv
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("perspectives", Path(__file__).with_name("01_compare_perspectives.py"))
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


class PerspectiveTests(unittest.TestCase):
    def test_views_and_self_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db, out = base / "raw.sqlite", base / "output"
            out.mkdir()
            c = sqlite3.connect(db)
            c.execute("""CREATE TABLE network_flows (
                ipv4_src_addr TEXT, ipv4_dst_addr TEXT, flow_start_milliseconds INTEGER,
                flow_end_milliseconds INTEGER,label INTEGER,attack TEXT,protocol INTEGER,
                in_bytes INTEGER,out_bytes INTEGER,in_pkts INTEGER,out_pkts INTEGER,
                flow_duration_milliseconds INTEGER)""")
            rows = [("A", "B", 0, 0), ("C", "B", 1, 1), ("A", "A", 10000, 0),
                    ("C", "B", 10001, 1), ("D", "E", 86400005, 0), ("C", "B", 86400010, 1)]
            c.executemany("INSERT INTO network_flows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          [(s, d, t, t + 20, a, "Attack" if a else "Benign", 6, 100, 10, 2, 1, 20)
                           for s, d, t, a in rows])
            c.commit()
            c.close()
            before = db.read_bytes()
            result = eda.analyze(db.resolve(), out, top_n=10)
            self.assertEqual(before, db.read_bytes())
            views = {v["summary"]["perspective"]: v for v in result["views"]}
            self.assertEqual(views["source"]["summary"]["flows"], 6)
            self.assertEqual(views["destination"]["summary"]["flows"], 6)
            self.assertEqual(views["endpoint"]["summary"]["flows"], 11)
            self.assertEqual(views["network"]["summary"]["flows"], 6)
            self.assertEqual(views["source"]["summary"]["hybrid_entities"], 0)
            self.assertEqual(views["destination"]["summary"]["hybrid_entities"], 1)
            self.assertEqual(views["endpoint"]["summary"]["hybrid_buckets"], 1)
            self.assertEqual(views["network"]["summary"]["hybrid_buckets"], 3)
            self.assertEqual(views["source"]["runs"]["max"], 2)
            self.assertEqual(views["network"]["summary"]["active_buckets"], 3)
            with (out / "endpoint_entities.csv").open(encoding="utf-8-sig", newline="") as f:
                entities = {r["entity"]: r for r in csv.DictReader(f)}
            self.assertEqual(int(entities["A"]["flows"]), 2)
            self.assertEqual(int(entities["B"]["flows"]), 4)
            self.assertEqual(int(entities["B"]["hybrid_buckets"]), 1)
            self.assertEqual(float(entities["B"]["attack_fraction"]), .75)

    def test_quantiles_and_classes(self):
        stats = eda.distribution({1: 2, 4: 1, 10: 1})
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["p50"], 2.5)
        self.assertEqual(stats["mean"], 4)
        self.assertIsNone(eda.distribution({})["p50"])
        self.assertEqual(eda.activity_class(1, 0), "benign_only")
        self.assertEqual(eda.activity_class(1, 1), "attack_only")
        self.assertEqual(eda.activity_class(2, 1), "hybrid")
        with self.assertRaises(ValueError):
            eda.activity_class(0, 0)


if __name__ == "__main__":
    unittest.main()
