"""Focused checks for numeric quality, protocol support and legacy reconciliation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from collections import Counter

import numpy as np

SPEC = importlib.util.spec_from_file_location("feature_audit", Path(__file__).with_name("03_audit_features.py"))
EDA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EDA)

NUMERIC = """flow_start_milliseconds flow_end_milliseconds l4_src_port l4_dst_port protocol l7_proto
in_bytes in_pkts out_bytes out_pkts tcp_flags client_tcp_flags server_tcp_flags flow_duration_milliseconds
duration_in duration_out min_ttl max_ttl longest_flow_pkt shortest_flow_pkt min_ip_pkt_len max_ip_pkt_len
src_to_dst_second_bytes dst_to_src_second_bytes retransmitted_in_bytes retransmitted_in_pkts
retransmitted_out_bytes retransmitted_out_pkts src_to_dst_avg_throughput dst_to_src_avg_throughput
num_pkts_up_to_128_bytes num_pkts_128_to_256_bytes num_pkts_256_to_512_bytes num_pkts_512_to_1024_bytes
num_pkts_1024_to_1514_bytes tcp_win_max_in tcp_win_max_out icmp_type icmp_ipv4_type dns_query_id
dns_query_type dns_ttl_answer ftp_command_ret_code src_to_dst_iat_min src_to_dst_iat_max src_to_dst_iat_avg
src_to_dst_iat_stddev dst_to_src_iat_min dst_to_src_iat_max dst_to_src_iat_avg dst_to_src_iat_stddev label""".split()


class AuditTests(unittest.TestCase):
    def test_ftp_support_uses_each_code_range(self):
        schema = json.loads((EDA.ROOT / "processed_binary_temporal/categorical_schema.json").read_text())
        counts = Counter({("ftp_command_ret_code", str(code), label): n
                          for code, label, n in ((0,0,10),(150,0,1),(200,0,2),(226,1,3),(550,1,4),(700,1,5))})
        rows = {r["feature"]:r for r in EDA.legacy_registry(schema, counts, Counter())}
        for name, support in (("ftp_flow_ratio",15),("ftp_code_nunique",15),("ftp_1xx_ratio",1),
                              ("ftp_2xx_ratio",5),("ftp_5xx_ratio",4),("ftp_other_code_ratio",5),("ftp_3xx_ratio",0)):
            self.assertEqual(rows[name]["supporting_flows"], support)

    def test_batch_moments_null_and_nonfinite(self):
        s = EDA.Stats()
        s.add(np.array([1., 0., np.nan, np.inf]), np.array([False, False, True, False]))
        s.add(np.array([3., -2.]), np.zeros(2, dtype=bool))
        r = s.row("example", "all")
        self.assertEqual((r["rows"], r["finite"], r["nulls"], r["nonfinite"], r["zeros"], r["negative"]), (6, 4, 1, 1, 1, 1))
        self.assertAlmostEqual(r["mean"], .5)
        self.assertAlmostEqual(r["std_population"], float(np.std([1, 0, 3, -2])))
        self.assertFalse(r["constant"])

    def test_full_analysis_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            db, out = folder / "raw.db", folder / "audit"
            out.mkdir()
            columns = NUMERIC + ["ipv4_src_addr", "ipv4_dst_addr", "attack"]
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE network_flows (" + ",".join(f'"{n}" ' + ("INTEGER" if n in NUMERIC else "TEXT") for n in columns) + ")")
            rows = []
            for i, (label, proto, l7) in enumerate(((0, 6, 88), (1, 6, 88), (0, 17, 5), (1, 1, 0))):
                r = dict.fromkeys(NUMERIC, 0)
                r.update(flow_start_milliseconds=i*10000, flow_end_milliseconds=i*10000+10,
                         flow_duration_milliseconds=10 if i != 3 else 11,
                         label=label, protocol=proto, l7_proto=l7, in_bytes=(i+1)*100,
                         in_pkts=2 if proto==6 else 1, out_pkts=0,
                         tcp_flags=2 if proto==6 else 255, client_tcp_flags=2 if proto==6 else 255,
                         ipv4_src_addr="A", ipv4_dst_addr="B", attack="Attack" if label else "Benign")
                if i == 2:
                    r.update(dns_query_id=3, dns_query_type=1, dns_ttl_answer=60)
                rows.append(tuple(r[n] for n in columns))
            conn.executemany("INSERT INTO network_flows VALUES (" + ",".join("?" for _ in columns) + ")", rows)
            conn.commit()
            conn.close()
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            schema = EDA.ROOT / "processed_binary_temporal/categorical_schema.json"
            result = EDA.analyze(db.resolve(), schema, out, sample_target=100, batch_size=2)
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)
            self.assertEqual(result["controls"]["flows"], 4)
            self.assertEqual(result["controls"]["attack"], 2)
            self.assertEqual(result["controls"]["l7_outside_legacy"], 2)
            self.assertEqual(result["metadata"]["sample_actual"], 4)
            self.assertEqual(len(result["registry"]), 55)
            self.assertEqual(len(result["legacy"]), 88)
            legacy = {r["feature"]: r for r in result["legacy"]}
            self.assertEqual(legacy["ftp_flow_ratio"]["supporting_flows"], 0)
            self.assertEqual(legacy["ftp_2xx_ratio"]["recommendation"], "review unsupported feature")
            flags = {(r["field"], r["flag"]): r for r in result["flags"]}
            self.assertEqual(flags["tcp_flags", "syn"]["support_flows"], 2)
            self.assertEqual(flags["tcp_flags", "fin"]["support_flows"], 0)  # non-TCP flags ignored
            p = {(r["field"], r["population"]): r for r in result["profiles"]}
            self.assertEqual(p["dns_query_type", "eligible"]["rows"], 1)
            self.assertEqual(p["src_to_dst_iat_min", "eligible"]["rows"], 2)
            self.assertEqual(p["in_bytes", "all"]["mean"], 250)
            d = {r["check"]:r for r in result["diagnostics"]}
            self.assertEqual(d["duration_differs_from_endpoint_span"]["affected"], 1)
            self.assertEqual(d["invalid_binary_label"]["affected"], 0)
            self.assertEqual(d["longest_packet_differs_from_max_ip_length"]["affected"], 0)
            self.assertEqual(json.loads((out/"results.json").read_text(encoding="utf-8"))["controls"], result["controls"])


if __name__ == "__main__":
    unittest.main()
