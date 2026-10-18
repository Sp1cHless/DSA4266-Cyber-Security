#!/usr/bin/env python3
"""CIC source-IP EDA: nonoverlapping 50/100/200/500/1000-second windows.

Reuse the completed EDA 01 count cache; read raw attack annotations once.
Keep internal empty bins, omit wholly empty windows, never cross UTC midnight.
No augmentation, class balancing, model fitting or train/test selection.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
DAY_MS = 86_400_000
BIN_MS = 10_000
LENGTHS = (50, 100, 200, 500, 1000)
CLASSES = ("benign_only", "attack_only", "hybrid")
METRICS = {
    "flows": "flows", "active_buckets": "buckets", "occupancy": "fraction",
    "longest_empty_run_s": "seconds", "longest_active_run_s": "seconds",
    "flow_start_span_s": "seconds", "attack_flows": "flows",
    "attack_fraction": "fraction", "total_bytes": "bytes", "total_packets": "packets",
}
CATALOG_FIELDS = [
    "source_ip", "start_ms", "end_ms_exclusive", "activity_class", "flows", "benign_flows",
    "attack_flows", "attack_fraction", "active_buckets", "total_buckets", "occupancy",
    "bucket_mask_hex", "hybrid_buckets", "attack_buckets", "longest_empty_run_s",
    "longest_active_run_s", "first_flow_offset_s", "last_flow_offset_s", "flow_start_span_s",
    "first_attack_offset_s", "last_attack_offset_s", "global_active_start_buckets",
    "total_bytes", "total_packets", "max_flow_duration_ms", "attack_types",
]

spec = importlib.util.spec_from_file_location("cic_eda_01", HERE / "01_compare_perspectives.py")
eda01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda01)
log, save_json, utc = eda01.log, eda01.save_json, eda01.utc


def latest_eda01():
    candidates = []
    for path in (HERE / "outputs").glob("01_perspectives_*/results.json"):
        status = path.with_name("run_status.json")
        if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("complete"):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Run 01_compare_perspectives.py first, or pass --perspectives-dir.")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns).parent


def read_inputs(folder, db_override=None):
    first = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or first["metadata"]["bucket_seconds"] != 10:
        raise ValueError("Need a completed EDA 01 using 10-second buckets")
    db = (db_override or Path(first["metadata"]["database"])).resolve()
    stat = db.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
            first["metadata"]["database_bytes"], first["metadata"]["database_mtime_ns"]):
        raise ValueError("Raw database fingerprint changed; regenerate EDA 01 before reusing its cache")
    cache = folder / "perspective_counts.sqlite"
    conn = sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        metadata = {k: json.loads(v) for k, v in conn.execute("SELECT key,value FROM metadata")}
        for key in ("database_bytes", "database_mtime_ns", "bucket_seconds", "script_sha256"):
            if metadata[key] != first["metadata"][key]:
                raise ValueError(f"EDA 01 JSON/cache mismatch: {key}")
        totals = conn.execute("SELECT SUM(flows),SUM(attack_flows) FROM source").fetchone()
        if totals != (first["raw"]["flows"], first["raw"]["attack_flows"]):
            raise ValueError("Source cache does not match raw control totals")
    finally:
        conn.close()
    return first, db, cache


def attack_lookup(db):
    log("[attack labels] Reading raw attack counts and start times (one read-only scan)...")
    conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        lookup, totals = {}, Counter()
        for ip, stamp, name, n, first, last in conn.execute("""
            SELECT ipv4_src_addr,(flow_start_milliseconds/10000)*10000,attack,
                   COUNT(*),MIN(flow_start_milliseconds),MAX(flow_start_milliseconds)
            FROM network_flows NOT INDEXED WHERE label=1 GROUP BY 1,2,3
        """):
            lookup.setdefault((ip, stamp), []).append((name, n, first, last))
            totals[name] += n
        return lookup, totals
    finally:
        conn.close()


def global_prefix(conn):
    """Number of bins with ANY flow start; this is not a capture-availability mask."""
    flags = {}
    for (stamp,) in conn.execute("SELECT bucket_start_ms FROM network"):
        day, within = divmod(stamp, DAY_MS)
        flags.setdefault(day, bytearray(DAY_MS // BIN_MS))[within // BIN_MS] = 1
    result = {}
    for day, values in flags.items():
        prefix, total = array("I", [0]), 0
        for value in values:
            total += value
            prefix.append(total)
        result[day] = prefix
    return result


def temporal_shape(positions, n_bins):
    if not positions or positions != sorted(set(positions)) or positions[-1] >= n_bins or positions[0] < 0:
        raise ValueError("Invalid active bucket positions")
    longest_empty = max(positions[0], n_bins - 1 - positions[-1],
                        max((b - a - 1 for a, b in zip(positions, positions[1:])), default=0))
    longest_active = run = 1
    for a, b in zip(positions, positions[1:]):
        run = run + 1 if b == a + 1 else 1
        longest_active = max(longest_active, run)
    return longest_empty * 10, longest_active * 10


def new_window(ip, start):
    return {"ip": ip, "start": start, "flows": 0, "attack_flows": 0, "positions": [],
            "hybrid_buckets": 0, "attack_buckets": 0, "first_ms": None, "last_ms": None,
            "first_attack_ms": None, "last_attack_ms": None, "total_bytes": 0,
            "total_packets": 0, "max_flow_duration_ms": 0, "attack_types": Counter()}


def analyze_length(conn, length, out, lookup, raw_attack_totals, prefix, controls):
    n_bins, length_ms = length // 10, length * 1000
    hists = {cls: {metric: Counter() for metric in METRICS} for cls in CLASSES}
    class_counts, class_hosts = {cls: Counter() for cls in CLASSES}, {cls: set() for cls in CLASSES}
    occupancy_counts = Counter()
    flow_bands = Counter()
    daily, daily_hosts = defaultdict(Counter), defaultdict(set)
    host_windows, host_attacks = Counter(), Counter()
    attack_stats = {name: Counter() for name in raw_attack_totals}
    attack_hosts = defaultdict(set)
    excluded, total, state = Counter(), Counter(), None
    catalog = out / f"windows_{length}s.csv.gz"

    def flush(writer):
        if state is None:
            return
        n, a = state["flows"], state["attack_flows"]
        cls = eda01.activity_class(n, a)
        active = len(state["positions"])
        empty_s, active_s = temporal_shape(state["positions"], n_bins)
        day, within = divmod(state["start"], DAY_MS)
        position = within // BIN_MS
        global_active = prefix[day][position + n_bins] - prefix[day][position]
        row = {
            "source_ip": state["ip"], "start_ms": state["start"], "end_ms_exclusive": state["start"] + length_ms,
            "activity_class": cls, "flows": n, "benign_flows": n - a, "attack_flows": a,
            "attack_fraction": a / n, "active_buckets": active, "total_buckets": n_bins,
            "occupancy": active / n_bins, "bucket_mask_hex": hex(sum(1 << p for p in state["positions"])),
            "hybrid_buckets": state["hybrid_buckets"], "attack_buckets": state["attack_buckets"],
            "longest_empty_run_s": empty_s, "longest_active_run_s": active_s,
            "first_flow_offset_s": (state["first_ms"] - state["start"]) / 1000,
            "last_flow_offset_s": (state["last_ms"] - state["start"]) / 1000,
            "flow_start_span_s": (state["last_ms"] - state["first_ms"]) / 1000,
            "first_attack_offset_s": (state["first_attack_ms"] - state["start"]) / 1000 if a else None,
            "last_attack_offset_s": (state["last_attack_ms"] - state["start"]) / 1000 if a else None,
            "global_active_start_buckets": global_active, "total_bytes": state["total_bytes"],
            "total_packets": state["total_packets"], "max_flow_duration_ms": state["max_flow_duration_ms"],
            "attack_types": ";".join(sorted(state["attack_types"])),
        }
        writer.writerow(row)
        counts = class_counts[cls]
        counts.update(windows=1, flows=n, attack_flows=a, active_buckets=active,
                      global_gap_windows=int(global_active < n_bins),
                      same_bucket_hybrid_windows=int(state["hybrid_buckets"] > 0),
                      cross_bucket_only_hybrid_windows=int(cls == "hybrid" and state["hybrid_buckets"] == 0))
        class_hosts[cls].add(state["ip"])
        for metric in METRICS:
            hists[cls][metric][row[metric]] += 1
        occupancy_counts[cls, active] += 1
        band = "1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21-100" if n <= 100 else "101-1000" if n <= 1000 else "1001+"
        flow_bands[cls, band] += 1
        d = daily[day]
        d.update(windows=1, flows=n, attack_flows=a, active_buckets=active,
                 global_gap_windows=int(global_active < n_bins))
        d[cls] += 1
        daily_hosts[day].add(state["ip"])
        host_windows[state["ip"]] += 1
        if a:
            host_attacks[state["ip"]] += 1
        total.update(windows=1, flows=n, attack_flows=a, active_buckets=active,
                     single_active_bucket_windows=int(active == 1),
                     fully_active_windows=int(active == n_bins),
                     global_gap_windows=int(global_active < n_bins),
                     long_flow_windows=int(state["max_flow_duration_ms"] > length_ms))
        for name, count in state["attack_types"].items():
            attack_stats[name].update(windows=1, attack_flows=count,
                                      hybrid_windows=int(cls == "hybrid"), attack_only_windows=int(cls == "attack_only"))
            attack_hosts[name].add(state["ip"])

    with gzip.open(catalog, "wt", newline="", encoding="utf-8-sig", compresslevel=1) as handle:
        import csv
        writer = csv.DictWriter(handle, CATALOG_FIELDS)
        writer.writeheader()
        query = """SELECT entity,bucket_start_ms,flows,attack_flows,total_bytes,total_packets,
                          first_ms,last_ms,max_flow_duration_ms
                   FROM source ORDER BY entity,bucket_start_ms"""
        for ip, stamp, n, a, byte_count, packet_count, first, last, duration in conn.execute(query):
            day_start = (stamp // DAY_MS) * DAY_MS
            start = day_start + ((stamp - day_start) // length_ms) * length_ms
            if start + length_ms > day_start + DAY_MS:
                excluded.update(buckets=1, flows=n, attack_flows=a)
                continue
            if state is None or (state["ip"], state["start"]) != (ip, start):
                flush(writer)
                state = new_window(ip, start)
            state["flows"] += n
            state["attack_flows"] += a
            state["total_bytes"] += byte_count
            state["total_packets"] += packet_count
            state["max_flow_duration_ms"] = max(state["max_flow_duration_ms"], duration)
            state["positions"].append((stamp - start) // BIN_MS)
            state["hybrid_buckets"] += int(0 < a < n)
            state["attack_buckets"] += int(a > 0)
            state["first_ms"] = first if state["first_ms"] is None else min(first, state["first_ms"])
            state["last_ms"] = last
            if a:
                records = lookup[ip, stamp]
                if sum(r[1] for r in records) != a:
                    raise AssertionError(f"Raw attack/cache disagreement for {ip}, {stamp}")
                for name, count, attack_first, attack_last in records:
                    state["attack_types"][name] += count
                    state["first_attack_ms"] = attack_first if state["first_attack_ms"] is None else min(state["first_attack_ms"], attack_first)
                    state["last_attack_ms"] = attack_last if state["last_attack_ms"] is None else max(state["last_attack_ms"], attack_last)
        flush(writer)
    if total["flows"] + excluded["flows"] != controls["flows"]:
        raise AssertionError(f"{length}s flow coverage does not reconcile")
    if total["attack_flows"] + excluded["attack_flows"] != controls["attack_flows"]:
        raise AssertionError(f"{length}s attack coverage does not reconcile")
    if total["active_buckets"] + excluded["buckets"] != controls["buckets"]:
        raise AssertionError(f"{length}s bucket coverage does not reconcile")
    if sum(r["attack_flows"] for r in attack_stats.values()) != total["attack_flows"]:
        raise AssertionError("Attack-name coverage does not reconcile")
    all_hists = {metric: Counter() for metric in METRICS}
    for cls in CLASSES:
        for metric in METRICS:
            all_hists[metric].update(hists[cls][metric])
    distributions = []
    for cls, hist_group in [("all", all_hists), *hists.items()]:
        for metric, hist in hist_group.items():
            stats = eda01.distribution(hist)
            # n=0 is a known empty population; its quantiles remain unavailable.
            if not hist:
                stats["n"] = 0
            distributions.append({"length_s": length, "activity_class": cls, "metric": metric,
                                  "unit": METRICS[metric], **stats})
    classes = [{"length_s": length, "activity_class": cls,
                **{k: class_counts[cls][k] for k in ("windows", "flows", "attack_flows", "active_buckets",
                    "global_gap_windows", "same_bucket_hybrid_windows", "cross_bucket_only_hybrid_windows")},
                "source_hosts": len(class_hosts[cls])} for cls in CLASSES]
    summary = {"length_s": length, "buckets_per_window": n_bins, "stride_s": length,
               **dict(total), **{cls: class_counts[cls]["windows"] for cls in CLASSES},
               "source_hosts": len(host_windows), "attack_source_hosts": len(host_attacks),
               "same_bucket_hybrid_windows": class_counts["hybrid"]["same_bucket_hybrid_windows"],
               "cross_bucket_only_hybrid_windows": class_counts["hybrid"]["cross_bucket_only_hybrid_windows"],
               "top10_window_share": sum(n for _, n in host_windows.most_common(10)) / total["windows"],
               "top10_attack_window_share": sum(n for _, n in host_attacks.most_common(10)) / sum(host_attacks.values()) if host_attacks else None,
               "flow_coverage": total["flows"] / controls["flows"],
               "excluded_tail_flows": excluded["flows"], "excluded_tail_attack_flows": excluded["attack_flows"],
               "excluded_tail_buckets": excluded["buckets"], "tail_seconds_per_utc_day": 86400 % length}
    occupancy = [{"length_s": length, "activity_class": cls, "active_buckets": active,
                  "total_buckets": n_bins, "occupancy": active / n_bins, "windows": count,
                  "class_fraction": count / class_counts[cls]["windows"]}
                 for (cls, active), count in sorted(occupancy_counts.items())]
    bands = [{"length_s": length, "activity_class": cls, "flow_band": band, "windows": count}
             for (cls, band), count in sorted(flow_bands.items())]
    daily_rows = [{"length_s": length, "day_utc": utc(day * DAY_MS)[:10],
                   **{k: values[k] for k in ("windows", *CLASSES, "flows", "attack_flows", "active_buckets", "global_gap_windows")},
                   "source_hosts": len(daily_hosts[day]),
                   "mean_occupancy": values["active_buckets"] / (values["windows"] * n_bins)}
                  for day, values in sorted(daily.items())]
    attack_rows = [{"length_s": length, "attack_type": name,
                    **{k: attack_stats[name][k] for k in ("windows", "attack_flows", "hybrid_windows", "attack_only_windows")},
                    "source_hosts": len(attack_hosts[name]), "raw_attack_flows": raw_attack_totals[name],
                    "flow_coverage": attack_stats[name]["attack_flows"] / raw_attack_totals[name]}
                   for name in sorted(raw_attack_totals)]
    log(f"[{length}s] {total['windows']:,} nonempty windows; "
        f"B={summary['benign_only']:,}, M={summary['attack_only']:,}, H={summary['hybrid']:,}; "
        f"coverage={summary['flow_coverage']:.4%}")
    return {"summary": summary, "classes": classes, "distributions": distributions,
            "occupancy": occupancy, "flow_bands": bands, "daily": daily_rows, "attacks": attack_rows}


def analyze(cache, db, first, out, lengths):
    started = time.monotonic()
    cache_stat, raw_stat = cache.stat(), db.stat()
    lookup, raw_attack_totals = attack_lookup(db)
    if sum(raw_attack_totals.values()) != first["raw"]["attack_flows"]:
        raise AssertionError("Raw attack annotations do not match EDA 01")
    conn = sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-65536")
    try:
        controls = dict(zip(("buckets", "flows", "attack_flows"), conn.execute(
            "SELECT COUNT(*),SUM(flows),SUM(attack_flows) FROM source").fetchone()))
        if (controls["flows"], controls["attack_flows"]) != (first["raw"]["flows"], first["raw"]["attack_flows"]):
            raise AssertionError("Source count cache does not match original raw totals")
        prefix = global_prefix(conn)
        results = []
        for length in lengths:
            log(f"[{length}s] Building disjoint source-IP windows...")
            results.append(analyze_length(conn, length, out, lookup, raw_attack_totals, prefix, controls))
        for path, before in ((cache, cache_stat), (db, raw_stat)):
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Input changed during analysis: {path}")
        result = {
            "metadata": {"analysis_version": 1, "dataset": first["metadata"]["dataset"],
                         "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                         "perspective": "source", "bucket_seconds": 10, "lengths_seconds": lengths,
                         "stride_policy": "stride = length; nonoverlapping", "anchor": "UTC midnight, reset each date",
                         "day_tail_policy": "exclude windows ending after UTC midnight; count excluded raw flows",
                         "cache": str(cache), "raw_database": str(db), "cache_bytes": cache_stat.st_size,
                         "cache_mtime_ns": str(cache_stat.st_mtime_ns), "raw_database_bytes": raw_stat.st_size,
                         "raw_database_mtime_ns": str(raw_stat.st_mtime_ns),
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "eda01_script_sha256": first["metadata"]["script_sha256"],
                         "python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                         "analysis_seconds": round(time.monotonic() - started, 2)},
            "controls": controls, "lengths": results,
        }
        save_json(out / "results.json", result)
        for key in ("classes", "distributions", "occupancy", "flow_bands", "daily", "attacks"):
            rows = [row for length in results for row in length[key]]
            if rows:
                eda01.csv_write(out / f"{key}.csv", rows, list(rows[0]))
        summaries = [r["summary"] for r in results]
        fields = list(dict.fromkeys(k for row in summaries for k in row))
        eda01.csv_write(out / "summary.csv", summaries, fields)
        return result
    finally:
        conn.close()


def export_excel(out, node, modules):
    work = out / "_excel_runtime"
    work.mkdir(exist_ok=True)
    junction = work / "node_modules"
    if not junction.exists():
        if os.name == "nt":
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(junction), str(modules)],
                           check=True, stdout=subprocess.DEVNULL)
        else:
            junction.symlink_to(modules, target_is_directory=True)
    elif junction.resolve() != modules:
        raise ValueError("Existing runtime link points to a different node_modules directory")
    builder = work / "render_sequence_lengths.mjs"
    shutil.copyfile(HERE / "render_sequence_lengths.mjs", builder)
    subprocess.run([str(node), str(builder), str(out / "results.json")], check=True)
    save_json(out / "run_status.json", {"complete": True, "workbook": "02_cic_sequence_lengths.xlsx"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perspectives-dir", type=Path, help="Completed EDA 01 output; defaults to latest completed run")
    parser.add_argument("--db", type=Path, help="Relocated copy of the original raw SQLite; fingerprint must match EDA 01")
    parser.add_argument("--lengths-seconds", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--output", type=Path, help="New/empty output directory")
    parser.add_argument("--render-only", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--node-modules", type=Path)
    args = parser.parse_args()
    if any(n <= 0 or n > 86400 or n % 10 for n in args.lengths_seconds):
        parser.error("Each length must be a positive multiple of 10, no longer than a UTC day")
    lengths = sorted(set(args.lengths_seconds))
    node, modules = eda01.find_excel_runtime(args.node, args.node_modules)
    if args.render_only:
        out = args.render_only.resolve()
        if not (out / "results.json").is_file():
            parser.error("render-only needs an existing results.json")
        export_excel(out, node, modules)
        return
    folder = (args.perspectives_dir or latest_eda01()).resolve()
    first, db, cache = read_inputs(folder, args.db)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = (args.output or HERE / "outputs" / f"02_sequence_lengths_{stamp}").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output is nonempty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_status.json", {"complete": False})
    log(f"[output] {out}")
    analyze(cache, db, first, out, lengths)
    export_excel(out, node, modules)
    log(f"[done] {out / '02_cic_sequence_lengths.xlsx'}")


if __name__ == "__main__":
    main()
