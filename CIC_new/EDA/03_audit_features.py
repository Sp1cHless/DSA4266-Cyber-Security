#!/usr/bin/env python3
"""Audit CIC raw features, legacy 88 inputs and mergeable aggregation rules.

Exact full-table quality/counts; seeded Bernoulli sample only for quantiles and
raw-feature redundancy. Does not fit a vocabulary, scaler, selector or model.
Requires numpy. Excel is authored by the bundled artifact-tool JS companion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("perspectives_eda", HERE / "01_compare_perspectives.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
BITS = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16, "urg": 32, "ece": 64, "cwr": 128}
CAT_FIELDS = ("protocol", "l7_proto", "l4_src_port", "l4_dst_port", "icmp_ipv4_type", "dns_query_type", "ftp_command_ret_code")
META = {"flow_start_milliseconds", "flow_end_milliseconds", "ipv4_src_addr", "ipv4_dst_addr", "dns_query_id", "label", "attack"}
OLD_RAW = {"l4_src_port", "l4_dst_port", "ipv4_dst_addr", "protocol", "l7_proto", "in_bytes", "out_bytes", "in_pkts", "out_pkts",
           "tcp_flags", "client_tcp_flags", "server_tcp_flags", "flow_duration_milliseconds", "icmp_ipv4_type", "dns_query_id", "dns_query_type", "dns_ttl_answer", "ftp_command_ret_code"}
NPB = "https://www.ntop.org/guides/nprobe/flow_information_elements.html"
UQ = "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def csv_out(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_rule(name):
    """Explicit candidate registry; describes proposals, not a fitted input schema."""
    if name in {"label", "attack"}:
        return "target", "label", "target only; sequence/bucket label recomputed after merge", "Never an input or donor-selection feature exposed to the model."
    if name in {"flow_start_milliseconds", "flow_end_milliseconds"}:
        return "timeline", "epoch ms", "shift both endpoints by the same offset; sort merged starts", "Absolute calendar time stays metadata; derive local flow-start gaps separately."
    if name in {"ipv4_src_addr", "ipv4_dst_addr"}:
        return "identity", "address", "source as donor metadata; destination frequency map for distinct/top1/entropy", "No literal IP input; cross-source mixes describe a synthetic source."
    if name == "dns_query_id":
        return "identity", "ID", "retain for old DNS-presence predicate; do not embed transaction IDs", "Zero is not a reliable missing-value indicator."
    if name in {"protocol", "l7_proto", "l4_src_port", "l4_dst_port", "icmp_type", "icmp_ipv4_type", "dns_query_type", "ftp_command_ret_code"}:
        return "category", "code", "merge category frequency maps; recompute ratios, distinct, top1 and normalized entropy", "Fit categorical vocabulary/embeddings on training donors only; ports start with distribution summaries."
    if "tcp_flags" in name:
        return "flags", "bitmask", "TCP-only bit counts / TCP flow count; retain denominator", "Do not add bitmask integers; no TCP means not applicable."
    if name in {"in_bytes", "out_bytes", "in_pkts", "out_pkts"} or name.startswith(("retransmitted_", "num_pkts_")):
        return "counter", "packets" if "pkts" in name else "bytes", "sum; add nonzero-flow ratio; retain per-flow moments if needed", "Whole-flow counts assigned to start bucket are not bytes observed during that 10s interval."
    if name in {"flow_duration_milliseconds", "duration_in", "duration_out"}:
        return "duration", "ms", "merge n/sum/sum-of-squares/min/max; derive mean/std", "Keep exported duration and endpoint span separate; no silent replacement."
    if "iat_" in name:
        return "packet IAT", "exported unit; verify", "summarize exported per-flow IAT with eligible count/moments/min/max", "Not inter-flow start gaps; cannot recover merged packet-level IAT from flow summaries."
    if name.endswith("second_bytes"):
        return "exporter rate", "exported value; verify", "hold for exporter-semantics review; preserve raw values", "Do not assume it is additive byte volume for the synthetic 10s bucket."
    if "throughput" in name:
        return "throughput", "bits/s", "per-flow moments; optionally derive separate bucket assigned bytes / 10s", "Sum of flow average rates is not bucket observed throughput."
    if "ttl" in name and name != "dns_ttl_answer":
        return "TTL", "hops", "per-flow extrema and moments; zero fraction", "Host/OS/path fingerprints can survive IP removal."
    if name == "dns_ttl_answer":
        return "DNS TTL", "seconds", "DNS-eligible count/moments/extrema; preserve validity mask", "Zero TTL can be valid; old presence predicate is a heuristic."
    if "tcp_win" in name:
        return "TCP window", "bytes", "TCP-eligible count/moments/extrema; zero fraction", "Protocol eligibility is not proof exporter populated the field."
    if "pkt" in name:
        return "packet size", "bytes", "per-flow extrema and moments; preserve zero fraction", "Pooling per-flow extrema does not reconstruct packet-size mean or variance."
    raise ValueError(f"Unregistered raw field: {name}")


class Stats:
    """Merge stable batch moments; exact counts/extrema of finite numeric values."""
    def __init__(self):
        self.rows = self.finite = self.nulls = self.nonfinite = self.zeros = self.negative = 0
        self.mean = self.m2 = 0.0
        self.minimum = self.maximum = None

    def add(self, values, nulls):
        self.rows += len(values)
        self.nulls += int(nulls.sum())
        ok = np.isfinite(values)
        self.nonfinite += int((~ok & ~nulls).sum())
        v = values[ok]
        if not len(v):
            return
        n = len(v)
        mu = float(v.mean())
        delta = mu - self.mean
        total = self.finite + n
        self.m2 += float(np.square(v - mu).sum()) + delta * delta * self.finite * n / total
        self.mean += delta * n / total
        self.finite = total
        self.zeros += int((v == 0).sum())
        self.negative += int((v < 0).sum())
        lo, hi = float(v.min()), float(v.max())
        self.minimum = lo if self.minimum is None else min(lo, self.minimum)
        self.maximum = hi if self.maximum is None else max(hi, self.maximum)

    def row(self, field, population):
        return {"field": field, "population": population, "rows": self.rows, "finite": self.finite,
                "nulls": self.nulls, "nonfinite": self.nonfinite, "zeros": self.zeros, "negative": self.negative,
                "zero_fraction": self.zeros / self.finite if self.finite else None,
                "minimum": self.minimum, "maximum": self.maximum,
                "mean": self.mean if self.finite else None,
                "std_population": float(np.sqrt(self.m2 / self.finite)) if self.finite else None,
                "constant": self.finite > 0 and self.minimum == self.maximum}


def eligible(name, get):
    if "tcp_flags" in name or "tcp_win" in name:
        return get("protocol") == 6
    if name in {"icmp_type", "icmp_ipv4_type"}:
        return get("protocol") == 1
    if name in {"dns_query_id", "dns_query_type", "dns_ttl_answer"}:
        return (get("dns_query_id") != 0) | (get("dns_query_type") != 0) | (get("dns_ttl_answer") != 0)
    if name == "ftp_command_ret_code":
        return get(name) != 0
    if name.startswith("src_to_dst_iat_"):
        return get("in_pkts") >= 2
    if name.startswith("dst_to_src_iat_"):
        return get("out_pkts") >= 2
    return None


def diagnostic_masks(get):
    all_rows = np.ones(len(get("label")), dtype=bool)
    checks = [
        ("end_before_start", get("flow_end_milliseconds") < get("flow_start_milliseconds"), all_rows, "invalid interval"),
        ("duration_differs_from_endpoint_span", get("flow_duration_milliseconds") != get("flow_end_milliseconds") - get("flow_start_milliseconds"), all_rows, "semantic discrepancy; do not auto-clean"),
        ("duration_over_10s", get("flow_duration_milliseconds") > 10000, all_rows, "start allocation includes whole long flow"),
        ("duration_over_100s", get("flow_duration_milliseconds") > 100000, all_rows, "flow can extend beyond sequence"),
        ("invalid_binary_label", ~np.isin(get("label"), [0, 1]), all_rows, "invalid target"),
        ("ttl_min_over_max", get("min_ttl") > get("max_ttl"), all_rows, "inspect exporter zero/sentinel semantics"),
        ("ip_packet_min_over_max", get("min_ip_pkt_len") > get("max_ip_pkt_len"), all_rows, "inspect exporter zero/sentinel semantics"),
        ("flow_packet_shortest_over_longest", get("shortest_flow_pkt") > get("longest_flow_pkt"), all_rows, "inspect exporter zero/sentinel semantics"),
        ("longest_packet_differs_from_max_ip_length", get("longest_flow_pkt") != get("max_ip_pkt_len"), all_rows, "tests sample-equal pair across full table; not an error condition"),
    ]
    for direction, packets in (("src_to_dst", "in_pkts"), ("dst_to_src", "out_pkts")):
        mask = get(packets) >= 2
        checks.extend([
            (direction + "_iat_min_over_max", get(direction + "_iat_min") > get(direction + "_iat_max"), mask, "eligible: direction has >=2 packets"),
            (direction + "_iat_avg_outside_extrema", (get(direction + "_iat_avg") < get(direction + "_iat_min")) | (get(direction + "_iat_avg") > get(direction + "_iat_max")), mask, "eligible: direction has >=2 packets"),
        ])
    for field, upper in (("l4_src_port", 65535), ("l4_dst_port", 65535), ("min_ttl", 255), ("max_ttl", 255), ("protocol", 255)):
        checks.append((field + "_out_of_range", (get(field) < 0) | (get(field) > upper), all_rows, f"expected 0..{upper}"))
    return checks


def legacy_registry(schema, categories, flags):
    rows = []
    for name in schema["feature_names"]:
        group, source, merge, support = "", "", "", None
        if name == "flow_count":
            group, source, merge = "volume", "all flows", "sum counts"
        elif name in {"in_bytes_sum", "out_bytes_sum", "in_pkts_sum", "out_pkts_sum"}:
            group, source, merge = "volume", name[:-4], "sum raw counters"
        elif name.startswith("duration_"):
            group, source, merge = "duration", "flow_duration_milliseconds", "merge count/sum/min/max; mean weighted by count"
        elif name.startswith("protocol_"):
            group, source, merge = "protocol", "protocol", "merge category counts; recompute ratio/distinct"
        elif name.startswith("l7_"):
            group, source, merge = "L7", "l7_proto", "merge category counts; replace NB15 vocabulary after train split"
            if name.endswith("_ratio") and name not in {"l7_other_ratio", "l7_top1_ratio"}:
                category = name[3:-6].replace("_", ".")
                support = sum(v for (field, value, label), v in categories.items() if field == "l7_proto" and value == category)
        elif "tcp_flag_" in name:
            group = "TCP flags"
            source = "client_tcp_flags" if name.startswith("client_") else "server_tcp_flags" if name.startswith("server_") else "tcp_flags"
            merge = "sum bit counts / sum TCP denominator"
            flag = name.split("_flag_")[1].removesuffix("_ratio")
            support = sum(v for (field, bit, label), v in flags.items() if field == source and bit == flag)
        else:
            groups = (("src_port_", "source ports", "l4_src_port"), ("dst_port_", "destination ports", "l4_dst_port"),
                      ("dst_ip_", "destination diversity", "ipv4_dst_addr"), ("icmp_", "ICMP", "protocol; icmp_ipv4_type"),
                      ("dns_", "DNS", "dns_query_id; dns_query_type; dns_ttl_answer"), ("ftp_", "FTP", "ftp_command_ret_code"))
            for prefix, group_name, raw in groups:
                if name.startswith(prefix):
                    group, source = group_name, raw
                    break
            merge = "merge full frequency maps and applicable denominators; recompute summary"
            if group == "FTP":
                def supported_code(value):
                    code = float(value)
                    if code == 0:
                        return False
                    if name == "ftp_other_code_ratio":
                        return not 100 <= code <= 599
                    if name in {f"ftp_{i}xx_ratio" for i in range(1, 6)}:
                        lower = int(name[4]) * 100
                        return lower <= code <= lower + 99
                    return True
                support = sum(v for (field, value, label), v in categories.items()
                              if field == "ftp_command_ret_code" and supported_code(value))
        if not group:
            raise ValueError(f"Unregistered legacy feature: {name}")
        rows.append({"feature": name, "family": group, "raw_dependencies": source, "merge_rule": merge,
                     "supporting_flows": support,
                     "recommendation": "review unsupported feature" if support == 0 else "replace vocabulary" if group == "L7" else "retain as baseline candidate",
                     "note": "Support is exact full-data occurrence, not bucket variance or predictive importance." if support is not None else "No automatic keep/drop decision from this raw audit."})
    return rows


def analyze(db, schema_path, out, sample_target=100000, seed=4266, batch_size=50000):
    start = time.monotonic()
    before = db.stat()
    schema_before = schema_path.read_bytes()
    schema = json.loads(schema_before)
    conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        info = conn.execute("PRAGMA table_info(network_flows)").fetchall()
        names = [r[1] for r in info]
        rules = {name: raw_rule(name) for name in names}
        numeric = [r[1] for r in info if r[2].upper() != "TEXT"]
        text = [r[1] for r in info if r[2].upper() == "TEXT"]
        count, attacks = conn.execute("SELECT COUNT(*),SUM(label) FROM network_flows NOT INDEXED").fetchone()
        if not count:
            raise ValueError("Empty network_flows table")
        probability = min(1, sample_target / count)
        stats = {(name, pop): Stats() for name in numeric for pop in ("all", "benign", "attack", "eligible")}
        texts = {name: {"nulls": 0, "empty": 0, "values": set()} for name in text}
        categories, flags, attack_counts = Counter(), Counter(), Counter()
        diagnostics = {}
        samples = []
        rng = np.random.default_rng(seed)
        numeric_index = [names.index(name) for name in numeric]
        text_index = {name: names.index(name) for name in text}
        ni = {name: i for i, name in enumerate(numeric)}
        cursor = conn.execute("SELECT * FROM network_flows NOT INDEXED ORDER BY rowid")
        seen, next_log = 0, 1000000
        while rows := cursor.fetchmany(batch_size):
            array = np.array([[r[i] for i in numeric_index] for r in rows], dtype=np.float64)
            nulls = np.array([[r[i] is None for i in numeric_index] for r in rows], dtype=bool)
            get = lambda name: array[:, ni[name]]
            benign, malicious = get("label") == 0, get("label") == 1
            if not np.all(benign | malicious):
                raise ValueError("Invalid binary labels")
            for name in numeric:
                j = ni[name]
                values, missing = array[:, j], nulls[:, j]
                stats[name, "all"].add(values, missing)
                stats[name, "benign"].add(values[benign], missing[benign])
                stats[name, "attack"].add(values[malicious], missing[malicious])
                mask = eligible(name, get)
                if mask is not None:
                    stats[name, "eligible"].add(values[mask], missing[mask])
            for name, index in text_index.items():
                vals = [r[index] for r in rows]
                texts[name]["nulls"] += sum(v is None for v in vals)
                texts[name]["empty"] += sum(v == "" for v in vals)
                texts[name]["values"].update(v for v in vals if v is not None)
            attack_counts.update(r[text_index["attack"]] for r in rows)
            for name in CAT_FIELDS:
                for label, mask in ((0, benign), (1, malicious)):
                    v, frequencies = np.unique(get(name)[mask & np.isfinite(get(name))], return_counts=True)
                    categories.update({(name, BASE.canonical_category(float(value)) if hasattr(BASE, "canonical_category") else format(float(value), ".15g"), label): int(n) for value, n in zip(v, frequencies)})
            tcp = get("protocol") == 6
            for name in ("tcp_flags", "client_tcp_flags", "server_tcp_flags"):
                values = np.nan_to_num(get(name)).astype(np.int64)
                for flag, bit in BITS.items():
                    for label, mask in ((0, benign), (1, malicious)):
                        flags[name, flag, label] += int((tcp & mask & ((values & bit) != 0)).sum())
            for check, mask, population, note in diagnostic_masks(get):
                record = diagnostics.setdefault(check, {"check": check, "affected": 0, "eligible": 0, "note": note})
                record["affected"] += int((mask & population).sum())
                record["eligible"] += int(population.sum())
            selected = rng.random(len(rows)) < probability
            samples.append(array[selected].copy())
            seen += len(rows)
            if seen >= next_log:
                print(f"[scan] {seen:,}/{count:,} flows ({time.monotonic()-start:.0f}s)", flush=True)
                next_log += 1000000
        if seen != count or stats["label", "attack"].rows != attacks:
            raise AssertionError("Full-table flow/attack reconciliation failed")
        sampled = np.concatenate(samples)
        profiles, quantiles = [], []
        for name in numeric:
            for pop in ("all", "benign", "attack", "eligible"):
                stat = stats[name, pop]
                if stat.rows or pop != "eligible":
                    row = stat.row(name, pop)
                    if pop == "eligible" or stat.rows:
                        profiles.append(row)
            for pop in ("all", "benign", "attack"):
                mask = np.ones(len(sampled), dtype=bool) if pop == "all" else sampled[:, ni["label"]] == (pop == "attack")
                vals = sampled[mask, ni[name]]
                vals = vals[np.isfinite(vals)]
                q = np.quantile(vals, [0.5, 0.9, 0.99]).tolist() if len(vals) else [None] * 3
                quantiles.append({"field": name, "population": pop, "sample_finite": len(vals), "p50": q[0], "p90": q[1], "p99": q[2]})
        registry = []
        for i, name in enumerate(names):
            family, unit, aggregation, note = rules[name]
            stat = stats[name, "all"] if name in numeric else None
            registry.append({"field": name, "sqlite_type": info[i][2], "family": family, "unit": unit,
                             "used_by_legacy": name in OLD_RAW, "model_role": "metadata / target" if name in META else "baseline input candidate" if name in OLD_RAW else "additional candidate",
                             "constant_full_table": stat.row(name, "all")["constant"] if stat else len(texts[name]["values"]) == 1,
                             "aggregation": aggregation, "note": note})
        category_rows = []
        for field in CAT_FIELDS:
            values = sorted({value for f, value, label in categories if f == field}, key=float)
            for value in values:
                b, a = categories[field, value, 0], categories[field, value, 1]
                category_rows.append({"field": field, "value": value, "flows": b+a, "benign": b, "attack": a,
                                      "flow_fraction": (b+a)/count, "attack_fraction": a/(b+a),
                                      "in_legacy_l7": value in schema["kept_l7_categories"] if field == "l7_proto" else None})
        # Pearson of log1p nonnegative raw scalars. These are diagnostics, not a fitted selector.
        candidates = [name for name in numeric if rules[name][0] not in {"target", "timeline", "identity", "category", "flags"}]
        correlations = []
        for i, left in enumerate(candidates):
            x = sampled[:, ni[left]]
            for right in candidates[i+1:]:
                y = sampled[:, ni[right]]
                mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
                if mask.sum() < 3:
                    continue
                lx, ly = np.log1p(x[mask]), np.log1p(y[mask])
                if np.ptp(lx) == 0 or np.ptp(ly) == 0:
                    continue
                corr = float(np.corrcoef(lx, ly)[0, 1])
                correlations.append({"left": left, "right": right, "paired_sample": int(mask.sum()),
                                     "pearson_log1p": corr, "abs_correlation": abs(corr),
                                     "equal_value_fraction": float((x[mask] == y[mask]).mean())})
        correlations.sort(key=lambda r: r["abs_correlation"], reverse=True)
        l7 = [r for r in category_rows if r["field"] == "l7_proto"]
        outside = [r for r in l7 if not r["in_legacy_l7"]]
        text_rows = [{"field": name, "rows": count, "nulls": d["nulls"], "empty_strings": d["empty"], "distinct_nonnull": len(d["values"])} for name, d in texts.items()]
        flag_rows = []
        for name in ("tcp_flags", "client_tcp_flags", "server_tcp_flags"):
            for flag in BITS:
                b, a = flags[name, flag, 0], flags[name, flag, 1]
                denominator = sum(v for (f, value, label), v in categories.items() if f == "protocol" and value == "6")
                flag_rows.append({"field": name, "flag": flag, "support_flows": b+a, "benign": b, "attack": a, "tcp_flows": denominator, "tcp_fraction": (b+a)/denominator if denominator else None})
        after = db.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or schema_path.read_bytes() != schema_before:
            raise RuntimeError("Input database/schema changed during analysis")
        legacy = legacy_registry(schema, categories, flags)
        result = {
            "metadata": {"dataset": "NF-CICIDS2018-v3", "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                         "raw_database": str(db), "raw_bytes": before.st_size, "raw_mtime_ns": str(before.st_mtime_ns),
                         "legacy_schema": str(schema_path), "legacy_sha256": hashlib.sha256(schema_before).hexdigest(),
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "python": sys.version.split()[0], "numpy": np.__version__, "sqlite": sqlite3.sqlite_version,
                         "sample_seed": seed, "sample_target": sample_target, "sample_probability": probability,
                         "sample_actual": len(sampled), "sample_benign": int((sampled[:, ni["label"]] == 0).sum()),
                         "sample_attack": int((sampled[:, ni["label"]] == 1).sum()), "batch_size": batch_size,
                         "sampling": "independent Bernoulli per row; PCG64; rowid order; numeric finite filtering per metric",
                         "analysis_seconds": round(time.monotonic() - start, 2), "sources": [UQ, NPB]},
            "controls": {"flows": count, "benign": count-attacks, "attack": attacks, "raw_fields": len(names), "numeric_fields": len(numeric),
                         "legacy_features": len(legacy), "l7_categories": len(l7), "l7_outside_legacy": sum(r["flows"] for r in outside),
                         "l7_outside_benign": sum(r["benign"] for r in outside), "l7_outside_attack": sum(r["attack"] for r in outside),
                         "unsupported_legacy_l7": sum(r["family"] == "L7" and r["supporting_flows"] == 0 for r in legacy)},
            "registry": registry, "profiles": profiles, "quantiles": quantiles, "text_profiles": text_rows,
            "categories": category_rows, "flags": flag_rows, "diagnostics": list(diagnostics.values()),
            "legacy": legacy, "correlations": correlations, "attack_counts": dict(attack_counts),
        }
        for key in ("registry", "profiles", "quantiles", "text_profiles", "categories", "flags", "diagnostics", "legacy", "correlations"):
            csv_out(out / f"{key}.csv", result[key])
        dump(out / "aggregation_registry.json", {"status": "candidate rules; not model input schema", "fields": registry, "legacy": legacy})
        dump(out / "results.json", result)
        return result
    finally:
        conn.close()


def render(out, node, modules):
    work = out / "_excel_runtime"
    work.mkdir(exist_ok=True)
    link = work / "node_modules"
    if not link.exists():
        if sys.platform == "win32":
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(modules)], check=True, stdout=subprocess.DEVNULL)
        else:
            link.symlink_to(modules, target_is_directory=True)
    elif link.resolve() != modules:
        raise ValueError("node_modules points at another runtime")
    builder = work / "render_feature_audit.mjs"
    shutil.copyfile(HERE / builder.name, builder)
    subprocess.run([str(node), str(builder), str(out / "results.json")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "datasets/cic_2018_traffic.db")
    parser.add_argument("--legacy-schema", type=Path, default=ROOT / "processed_binary_temporal/categorical_schema.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-target", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--render-only", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--node-modules", type=Path)
    args = parser.parse_args()
    if args.sample_target < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error("sample-target and batch-size must be positive; seed must be nonnegative")
    node, modules = BASE.find_excel_runtime(args.node, args.node_modules)
    if args.render_only:
        render(args.render_only.resolve(), node, modules)
        return
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = (args.output or HERE / "outputs" / f"03_features_{stamp}").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "run_status.json", {"complete": False})
    analyze(args.db.resolve(), args.legacy_schema.resolve(), out, args.sample_target, args.seed, args.batch_size)
    render(out, node, modules)
    dump(out / "run_status.json", {"complete": True, "workbook": "03_cic_feature_audit.xlsx"})
    print(f"[done] {out / '03_cic_feature_audit.xlsx'}", flush=True)


if __name__ == "__main__":
    main()
