#!/usr/bin/env python3
"""Compare CIC source, destination, endpoint and network perspectives.

Uses Python's standard library. Excel export calls the adjacent JS renderer
with the Codex bundled Node/artifact-tool runtime (see README.md).
The source SQLite is attached READ ONLY; all derived tables stay in --output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ("source", "destination", "endpoint", "network")
CLASSES = ("benign_only", "attack_only", "hybrid")
DAY_MS = 86_400_000
VERSION = 1
ENTITY_FIELDS = [
    "perspective", "entity", "activity_class", "flows", "benign_flows",
    "attack_flows", "attack_fraction", "active_days_utc", "first_start_utc",
    "last_start_utc", "calendar_span_s", "active_buckets", "benign_only_buckets",
    "attack_only_buckets", "hybrid_buckets", "hybrid_bucket_fraction",
    "flows_per_active_bucket", "longest_active_run_buckets",
    "max_unobserved_gap_within_day_s", "total_bytes", "total_packets",
    "max_flow_duration_ms", "outgoing_flows", "incoming_flows",
]


def log(message):
    print(message, flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds")


def activity_class(total, attack):
    if not 0 <= attack <= total or total <= 0:
        raise ValueError(f"Invalid class counts: total={total}, attack={attack}")
    return "benign_only" if attack == 0 else "attack_only" if attack == total else "hybrid"


def distribution(hist):
    """Exact linear-interpolated quantiles of a frequency histogram."""
    n = sum(hist.values())
    if not n:
        return {key: None for key in ("n", "mean", "p25", "p50", "p75", "p90", "p99", "max")}
    items = sorted(hist.items())

    def at_rank(rank):
        count = 0
        for value, frequency in items:
            count += frequency
            if rank < count:
                return value
        raise AssertionError("Rank outside histogram")

    def quantile(q):
        rank = (n - 1) * q
        low, high = math.floor(rank), math.ceil(rank)
        return at_rank(low) + (at_rank(high) - at_rank(low)) * (rank - low)

    return {"n": n, "mean": sum(v * k for v, k in items) / n,
            **{f"p{int(q * 100)}": quantile(q) for q in (.25, .5, .75, .9, .99)},
            "max": items[-1][0]}


def csv_write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect_raw(conn):
    required = {"ipv4_src_addr", "ipv4_dst_addr", "flow_start_milliseconds",
                "flow_end_milliseconds", "label", "attack", "protocol", "in_bytes",
                "out_bytes", "in_pkts", "out_pkts", "flow_duration_milliseconds"}
    columns = {r[1] for r in conn.execute("PRAGMA raw.table_info(network_flows)")}
    if required - columns:
        raise ValueError(f"Missing raw columns: {sorted(required - columns)}")
    fields = ["flows", "attack_flows", "self_flows", "self_attack_flows", "first_ms",
              "last_ms", "null_required_rows", "invalid_labels", "negative_values",
              "end_before_start", "duration_mismatch_rows", "flows_longer_than_10s"]
    null_check = " OR ".join(f"{x} IS NULL" for x in sorted(required))
    row = conn.execute(f"""
        SELECT COUNT(*),SUM(label),SUM(ipv4_src_addr=ipv4_dst_addr),
          SUM(CASE WHEN ipv4_src_addr=ipv4_dst_addr THEN label ELSE 0 END),
          MIN(flow_start_milliseconds),MAX(flow_start_milliseconds),
          SUM({null_check}),SUM(label NOT IN (0,1)),
          SUM(flow_start_milliseconds<0 OR flow_duration_milliseconds<0 OR
              in_bytes<0 OR out_bytes<0 OR in_pkts<0 OR out_pkts<0),
          SUM(flow_end_milliseconds<flow_start_milliseconds),
          SUM(flow_end_milliseconds-flow_start_milliseconds != flow_duration_milliseconds),
          SUM(flow_duration_milliseconds>10000)
        FROM raw.network_flows NOT INDEXED
    """).fetchone()
    result = dict(zip(fields, row))
    if not result["flows"]:
        raise ValueError("Raw table is empty")
    if any(result[k] for k in ("null_required_rows", "invalid_labels", "negative_values")):
        raise ValueError(f"Raw values require investigation before aggregation: {result}")
    return result


def create_buckets(conn, bucket_ms, raw):
    """Cache additive statistics, retaining both class counts (not MAX(label))."""
    numeric = ("flows", "attack_flows", "total_bytes", "total_packets")
    for view in VIEWS:
        log(f"[aggregate] {view}: building entity x {bucket_ms // 1000}s counts...")
        conn.execute(f"""CREATE TABLE {view} (
            entity TEXT NOT NULL, bucket_start_ms INTEGER NOT NULL,
            flows INTEGER NOT NULL, attack_flows INTEGER NOT NULL,
            total_bytes INTEGER NOT NULL, total_packets INTEGER NOT NULL,
            first_ms INTEGER NOT NULL, last_ms INTEGER NOT NULL,
            max_flow_duration_ms INTEGER NOT NULL,
            PRIMARY KEY(entity,bucket_start_ms)) WITHOUT ROWID""")
        if view in ("source", "destination"):
            column = "ipv4_src_addr" if view == "source" else "ipv4_dst_addr"
            query = f"""SELECT {column},(flow_start_milliseconds/{bucket_ms})*{bucket_ms},
                COUNT(*),SUM(label),SUM(in_bytes+out_bytes),SUM(in_pkts+out_pkts),
                MIN(flow_start_milliseconds),MAX(flow_start_milliseconds),
                MAX(flow_duration_milliseconds)
                FROM raw.network_flows NOT INDEXED GROUP BY 1,2"""
        else:
            if view == "network":
                union = "SELECT * FROM source"
                entity = "'ALL'"
            else:
                union = "SELECT * FROM source UNION ALL SELECT * FROM destination"
                entity = "entity"
                # Source+destination counts each self-flow twice. Subtract once.
                if raw["self_flows"]:
                    union += f""" UNION ALL SELECT ipv4_src_addr,
                        (flow_start_milliseconds/{bucket_ms})*{bucket_ms},
                        -COUNT(*),-SUM(label),-SUM(in_bytes+out_bytes),-SUM(in_pkts+out_pkts),
                        MIN(flow_start_milliseconds),MAX(flow_start_milliseconds),
                        MAX(flow_duration_milliseconds)
                        FROM raw.network_flows NOT INDEXED
                        WHERE ipv4_src_addr=ipv4_dst_addr GROUP BY 1,2"""
            query = f"""SELECT {entity},bucket_start_ms,
                {','.join('SUM(' + c + ')' for c in numeric)},
                MIN(first_ms),MAX(last_ms),MAX(max_flow_duration_ms)
                FROM ({union}) GROUP BY 1,2"""
        conn.execute(f"INSERT INTO {view} {query}")
        conn.commit()
        expected = raw["flows"] if view != "endpoint" else 2 * raw["flows"] - raw["self_flows"]
        attack_expected = (raw["attack_flows"] if view != "endpoint" else
                           2 * raw["attack_flows"] - raw["self_attack_flows"])
        n, total, attack = conn.execute(
            f"SELECT COUNT(*),SUM(flows),SUM(attack_flows) FROM {view}").fetchone()
        if (total, attack) != (expected, attack_expected):
            raise AssertionError(f"{view}: additive totals disagree with raw control")
        log(f"[aggregate] {view}: {n:,} active buckets, {total:,} flow memberships")


def summarize_view(conn, view, out, bucket_ms, top_n):
    daily = defaultdict(Counter)
    classes = defaultdict(Counter)
    bucket_hists = defaultdict(Counter)
    entity_hists = defaultdict(Counter)
    runs = Counter()
    totals = Counter()
    top_heap, attack_heap = [], []
    outgoing = dict(conn.execute("SELECT entity,SUM(flows) FROM source GROUP BY entity")) if view == "endpoint" else {}
    incoming = dict(conn.execute("SELECT entity,SUM(flows) FROM destination GROUP BY entity")) if view == "endpoint" else {}
    current = None
    state = None
    last_stamp = None
    run_length = 0

    def finish_run():
        nonlocal run_length
        if run_length:
            runs[run_length] += 1
            state["longest_active_run_buckets"] = max(state["longest_active_run_buckets"], run_length)
        run_length = 0

    def push(heap, row, field):
        item = (row[field], row["entity"], row)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    def finish_entity(writer):
        if state is None:
            return
        finish_run()
        n, a = state["flows"], state["attack_flows"]
        cls = activity_class(n, a)
        row = {"perspective": view, "entity": current, "activity_class": cls,
               **{k: state[k] for k in ("flows", "attack_flows", "active_days_utc", "active_buckets",
                    "benign_only_buckets", "attack_only_buckets", "hybrid_buckets", "total_bytes",
                    "total_packets", "longest_active_run_buckets", "max_unobserved_gap_within_day_s",
                    "max_flow_duration_ms")},
               "benign_flows": n - a, "attack_fraction": a / n,
               "first_start_utc": utc(state["first_ms"]), "last_start_utc": utc(state["last_ms"]),
               "calendar_span_s": (state["last_ms"] - state["first_ms"]) / 1000,
               "hybrid_bucket_fraction": state["hybrid_buckets"] / state["active_buckets"],
               "flows_per_active_bucket": n / state["active_buckets"],
               "outgoing_flows": outgoing.get(current) if view == "endpoint" else None,
               "incoming_flows": incoming.get(current) if view == "endpoint" else None}
        writer.writerow(row)
        classes[cls].update(entities=1, flows=n, attack_flows=a,
                            active_buckets=state["active_buckets"], hybrid_buckets=state["hybrid_buckets"])
        totals.update(entities=1, flows=n, attack_flows=a)
        entity_hists["flows"][n] += 1
        entity_hists["active_buckets"][state["active_buckets"]] += 1
        push(top_heap, row, "flows")
        if a:
            push(attack_heap, row, "attack_flows")

    path = out / f"{view}_entities.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, ENTITY_FIELDS)
        writer.writeheader()
        for entity, stamp, n, a, byte_count, packets, first, last, duration in conn.execute(
                f"SELECT * FROM {view} ORDER BY entity,bucket_start_ms"):
            if entity != current:
                finish_entity(writer)
                current, last_stamp = entity, None
                state = Counter()
                state["first_ms"] = first
            cls = activity_class(n, a)
            day = stamp // DAY_MS
            new_day = last_stamp is None or last_stamp // DAY_MS != day
            if new_day:
                state["active_days_utc"] += 1
                daily[day]["entities"] += 1
            if last_stamp is not None:
                if new_day or stamp - last_stamp != bucket_ms:
                    finish_run()
                if not new_day:
                    state["max_unobserved_gap_within_day_s"] = max(
                        state["max_unobserved_gap_within_day_s"], (stamp - last_stamp - bucket_ms) / 1000)
            run_length += 1
            state["last_ms"] = last
            state["max_flow_duration_ms"] = max(state["max_flow_duration_ms"], duration)
            state.update(flows=n, attack_flows=a, active_buckets=1, total_bytes=byte_count, total_packets=packets)
            state[f"{cls}_buckets"] += 1
            daily[day].update(flows=n, attack_flows=a, active_buckets=1)
            daily[day][f"{cls}_buckets"] += 1
            bucket_hists["all"][n] += 1
            bucket_hists[cls][n] += 1
            totals["active_buckets"] += 1
            totals[f"{cls}_buckets"] += 1
            totals["one_flow_buckets"] += int(n == 1)
            last_stamp = stamp
        finish_entity(writer)
    # Expected concentration among observed entities, not assumed internal machines.
    top_rows = [r[2] for r in sorted(top_heap, key=lambda x: x[:2], reverse=True)]
    attack_rows = [r[2] for r in sorted(attack_heap, key=lambda x: x[:2], reverse=True)]
    all_total = totals["flows"]
    all_attack = totals["attack_flows"]
    summary = {"perspective": view, **dict(totals),
               "benign_only_entities": classes["benign_only"]["entities"],
               "attack_only_entities": classes["attack_only"]["entities"],
               "hybrid_entities": classes["hybrid"]["entities"],
               "top1_flow_share": top_rows[0]["flows"] / all_total,
               "top10_flow_share": sum(r["flows"] for r in top_rows[:10]) / all_total,
               "top1_attack_share": attack_rows[0]["attack_flows"] / all_attack if all_attack else None,
               "top10_attack_share": sum(r["attack_flows"] for r in attack_rows[:10]) / all_attack if all_attack else None}
    return {
        "summary": summary,
        "entity_classes": [{"perspective": view, "activity_class": cls, **dict(classes[cls])} for cls in CLASSES],
        "density": [{"perspective": view, "bucket_class": cls, **distribution(bucket_hists[cls])}
                    for cls in ("all", *CLASSES)],
        "entity_distributions": [{"perspective": view, "metric": metric, **distribution(hist)}
                                 for metric, hist in entity_hists.items()],
        "runs": {"perspective": view, **distribution(runs)},
        "daily": [{"perspective": view, "day_utc": utc(day * DAY_MS)[:10], **dict(values)}
                  for day, values in sorted(daily.items())],
        "top_entities": [{"rank_basis": "flows", "rank": i + 1, **row} for i, row in enumerate(top_rows)]
                        + [{"rank_basis": "attack_flows", "rank": i + 1, **row} for i, row in enumerate(attack_rows)],
    }


def attack_summary(conn):
    # Source/destination counts measure entities, not independent attack episodes.
    rows = conn.execute("""SELECT attack,COUNT(*),COUNT(DISTINCT ipv4_src_addr),
        COUNT(DISTINCT ipv4_dst_addr),MIN(flow_start_milliseconds),MAX(flow_start_milliseconds)
        FROM raw.network_flows NOT INDEXED WHERE label=1 GROUP BY attack ORDER BY COUNT(*) DESC""").fetchall()
    return [{"attack_type": r[0], "flows": r[1], "source_entities": r[2],
             "destination_entities": r[3], "first_start_utc": utc(r[4]), "last_start_utc": utc(r[5])}
            for r in rows]


def find_excel_runtime(node_arg=None, modules_arg=None):
    base = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    node = Path(node_arg) if node_arg else base / "node/bin" / ("node.exe" if os.name == "nt" else "node")
    modules = Path(modules_arg) if modules_arg else base / "node/node_modules"
    if not node.is_file() or not (modules / "@oai/artifact-tool").is_dir():
        raise FileNotFoundError("Excel runtime missing. Pass --node and --node-modules; see README.md.")
    return node.resolve(), modules.resolve()


def render_excel(out, node, modules):
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
        raise ValueError("Existing node_modules link points at a different runtime")
    builder = work / "render_workbook.mjs"
    shutil.copyfile(HERE / "render_workbook.mjs", builder)
    subprocess.run([str(node), str(builder), str(out / "results.json")], check=True)


def analyze(db, out, bucket_ms=10000, top_n=50):
    started = time.monotonic()
    before = db.stat()
    conn = sqlite3.connect(out / "perspective_counts.sqlite", uri=True)
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("ATTACH DATABASE ? AS raw", (db.as_uri() + "?mode=ro",))
    try:
        log("[raw] Validating raw rows and reading independent control totals...")
        raw = inspect_raw(conn)
        log(f"[raw] {raw['flows']:,} flows; {raw['attack_flows']:,} attack flows")
        create_buckets(conn, bucket_ms, raw)
        views = []
        for view in VIEWS:
            log(f"[summarize] {view}...")
            views.append(summarize_view(conn, view, out, bucket_ms, top_n))
        attacks = attack_summary(conn)
        if sum(r["flows"] for r in attacks) != raw["attack_flows"]:
            raise AssertionError("Attack-name totals differ from binary attack count")
        after = db.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Raw database changed during analysis")
        result = {
            "metadata": {"analysis_version": VERSION, "dataset": "NF-CICIDS2018-v3",
                         "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                         "database": str(db), "database_bytes": before.st_size,
                         "database_mtime_ns": before.st_mtime_ns,
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                         "bucket_seconds": bucket_ms // 1000, "top_n": top_n,
                         "sequence_length": None, "sampling": "none; full raw table",
                         "analysis_seconds": round(time.monotonic() - started, 2)},
            "raw": raw, "views": views, "attacks": attacks,
        }
        save_json(out / "results.json", result)
        for key in ("density", "entity_classes", "entity_distributions", "daily", "top_entities"):
            rows = [row for v in views for row in v[key]]
            fields = list(dict.fromkeys(k for row in rows for k in row))
            csv_write(out / f"{key}.csv", rows, fields)
        csv_write(out / "attacks.csv", attacks, list(attacks[0]) if attacks else ["attack_type", "flows"])
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
        conn.executemany("INSERT INTO metadata VALUES (?,?)",
                         [(key, json.dumps(value, ensure_ascii=False)) for key, value in result["metadata"].items()])
        conn.commit()
        return result
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "datasets/cic_2018_traffic.db")
    parser.add_argument("--output", type=Path, help="New/empty output directory")
    parser.add_argument("--bucket-seconds", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=50, help="Excel examples per view and rank basis; CSV contains all entities")
    parser.add_argument("--node", type=Path)
    parser.add_argument("--node-modules", type=Path)
    parser.add_argument("--render-only", type=Path, help="Rebuild Excel from an existing results directory without rescanning raw data")
    args = parser.parse_args()
    if args.bucket_seconds <= 0 or DAY_MS % (args.bucket_seconds * 1000):
        parser.error("bucket-seconds must be positive and divide a UTC day")
    if args.top_n < 10:
        parser.error("top-n must be at least 10 for concentration statistics")
    node, modules = find_excel_runtime(args.node, args.node_modules)
    if args.render_only:
        out = args.render_only.resolve()
        if not (out / "results.json").is_file():
            parser.error("render-only needs an existing results.json")
        render_excel(out, node, modules)
        return
    db = args.db.resolve()
    if not db.is_file():
        parser.error(f"Missing database: {db}")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = (args.output or HERE / "outputs" / f"01_perspectives_{stamp}").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output is nonempty: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_status.json", {"complete": False})
    analyze(db, out, args.bucket_seconds * 1000, args.top_n)
    render_excel(out, node, modules)
    save_json(out / "run_status.json", {"complete": True, "workbook": "01_cic_perspectives.xlsx"})
    log(f"[done] {out / '01_cic_perspectives.xlsx'}")


if __name__ == "__main__":
    main()
