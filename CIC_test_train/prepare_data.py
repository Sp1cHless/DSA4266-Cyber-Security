"""Build sparse 100-second CIC windows from the EXISTING read-only bucket DB.

Defaults: 10-second bins, 10 bins/window, stride=5, 30-minute global time blocks.
Run this file from any working directory. No CSV import or bucket aggregation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import time

import numpy as np

from common import (BIN_MS, DAY_MS, HERE, ROOT, SEQ_LEN, SPLITS, WINDOW_DTYPE,
                    dump_json, empty_output, feature_groups, read_json, write_csv)

TABLE = "cic_temporal_buckets"


def split_blocks(block_ids, block_ms, seed):
    """Assign whole NETWORK time blocks within each UTC day, without labels."""
    days = defaultdict(list)
    for block in sorted(set(block_ids)):
        days[(int(block) * block_ms) // DAY_MS].append(int(block))
    assignment = {}
    for day, ids in sorted(days.items()):
        ordered = sorted(ids, key=lambda b: hashlib.sha256(
            f"{seed}|{day}|{b}".encode()).digest())
        n = len(ordered)
        if n < 4:
            # Do not split a tiny/incomplete capture day into adjacent windows.
            assignment.update({b: "test" for b in ordered})
            continue
        n_test, n_val, n_cal = max(1, round(n * .2)), max(1, round(n * .1)), max(1, round(n * .1))
        cuts = (n_test, n_test + n_val, n_test + n_val + n_cal)
        for i, block in enumerate(ordered):
            assignment[block] = ("test" if i < cuts[0] else "validation" if i < cuts[1]
                                 else "calibration" if i < cuts[2] else "train")
    return assignment


def make_windows(host, bucket_meta, stride, block_ms, assignment):
    """Candidate geometry is determined by timestamps, NEVER by attack labels."""
    # (feature row, timestamp, bucket label, attack-type bitset, attack flow count)
    a = np.asarray(bucket_meta, dtype=np.int64)
    rows, stamps, labels, attack_bits, attack_flows = a.T
    if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        raise ValueError("Host bucket timestamps must be strictly increasing")
    if np.any(stamps % BIN_MS):
        raise ValueError("Bucket timestamp is not on the 10-second grid")
    hop = stride * BIN_MS
    # Every active bin proposes the stride-grid starts whose window contains it.
    anchors = (stamps // hop) * hop
    starts = np.unique(np.concatenate([
        anchors - j * hop for j in range((SEQ_LEN + stride - 1) // stride)]))
    block_ids = starts // block_ms
    # A window cannot cross a 30-minute block or a capture-day boundary.
    valid = ((starts + SEQ_LEN * BIN_MS - 1) // block_ms == block_ids)
    starts, block_ids = starts[valid], block_ids[valid]
    targets = starts[:, None] + np.arange(SEQ_LEN, dtype=np.int64) * BIN_MS
    pos = np.searchsorted(stamps, targets)
    safe = np.minimum(pos, len(stamps) - 1)
    mask = (pos < len(stamps)) & (stamps[safe] == targets)
    nonempty = mask.any(axis=1)
    starts, block_ids, safe, mask = starts[nonempty], block_ids[nonempty], safe[nonempty], mask[nonempty]
    result = np.zeros(len(starts), dtype=WINDOW_DTYPE)
    result["host"], result["start_ms"] = host, starts
    result["rows"] = np.where(mask, rows[safe], -1)
    result["active"] = mask.sum(axis=1)
    result["label"] = np.where(mask, labels[safe], 0).max(axis=1)
    result["attack_bits"] = np.bitwise_or.reduce(
        np.where(mask, attack_bits[safe], 0).astype(np.uint64), axis=1)
    result["attack_flows"] = np.where(mask, attack_flows[safe], 0).sum(axis=1)
    if not np.array_equal(result["label"] > 0, result["attack_bits"] > 0):
        raise ValueError("Raw attack annotations disagree with binary bucket labels")
    splits = np.array([assignment[int(b)] for b in block_ids])
    return result, splits


def balanced_sample(windows, budget, seed, block_ms, max_host_fraction=.01):
    """Sample day -> host -> block -> window; cap one host at 1% of budget.

    Labels play no role here: input contains only benign candidate windows.
    A cap shortage is reported instead of silently allowing a dominant host.
    """
    if not len(windows):
        return windows
    rng = np.random.default_rng(seed)
    days = windows["start_ms"] // DAY_MS
    blocks = windows["start_ms"] // block_ms
    # Each group contributes windows in a random order. Cycling over days and
    # hosts avoids allocating most of the sample to dense servers.
    order = np.lexsort((blocks, windows["host"], days))
    d, h, b = days[order], windows["host"][order], blocks[order]
    cut = np.flatnonzero((np.diff(d) != 0) | (np.diff(h) != 0) | (np.diff(b) != 0)) + 1
    grouped = defaultdict(lambda: defaultdict(list))
    for inds in np.split(order, cut):
        grouped[int(days[inds[0]])][int(windows["host"][inds[0]])].append(list(rng.permutation(inds)))
    day_order = list(grouped)
    rng.shuffle(day_order)
    state = {}
    for day in day_order:
        hosts = list(grouped[day])
        rng.shuffle(hosts)
        state[day] = hosts
        for host in hosts:
            rng.shuffle(grouped[day][host])
    cap = max(1, int(np.ceil(budget * max_host_fraction)))
    used = Counter()
    chosen = []
    # Round-robin one draw from each day; rotate host and block queues.
    from collections import deque
    queues = {day: deque(hosts) for day, hosts in state.items()}
    block_queues = {(day, host): deque(groups) for day, hs in grouped.items() for host, groups in hs.items()}
    active_days = deque(day_order)
    while active_days and len(chosen) < budget:
        day = active_days.popleft()
        hosts = queues[day]
        while hosts:
            host = hosts.popleft()
            if used[host] >= cap:
                continue
            groups = block_queues[day, host]
            group = groups.popleft()
            chosen.append(group.pop())
            used[host] += 1
            if group:
                groups.append(group)
            if groups and used[host] < cap:
                hosts.append(host)
            break
        if hosts:
            active_days.append(day)
    return windows[np.sort(np.asarray(chosen, dtype=np.int64))]


def attack_lookup(conn, enabled=True):
    if not enabled:
        return [], {}
    print("[prepare] Reading raw attack annotations (one read-only scan)...", flush=True)
    query = ("SELECT ipv4_src_addr, (flow_start_milliseconds/10000)*10000, attack, COUNT(*) "
             "FROM network_flows WHERE label=1 GROUP BY 1,2,3")
    groups = conn.execute(query).fetchall()
    names = sorted({str(r[2]) for r in groups})
    if len(names) > 63:
        raise ValueError("Too many attack classes for the saved bitset")
    bits = {name: 1 << i for i, name in enumerate(names)}
    lookup = {}
    for ip, stamp, name, count in groups:
        key = (str(ip), int(stamp))
        old_bits, old_count = lookup.get(key, (0, 0))
        lookup[key] = (old_bits | bits[str(name)], old_count + int(count))
    return names, lookup


def metadata_rows(windows, hosts, attacks, block_ms):
    for i, w in enumerate(windows):
        start = int(w["start_ms"])
        yield {"window_id": i, "src_ip": hosts[int(w["host"])], "start_ms": start,
               "end_ms_exclusive": start + SEQ_LEN * BIN_MS,
               "day_utc": datetime.fromtimestamp(start / 1000, timezone.utc).date().isoformat(),
               "block_id": start // block_ms, "label": int(w["label"]),
               "active_buckets": int(w["active"]), "attack_flows": int(w["attack_flows"]),
               "attack_types": ";".join(n for j, n in enumerate(attacks) if int(w["attack_bits"]) & (1 << j))}


def prepare(args):
    started = time.monotonic()
    if not args.db.is_file():
        raise FileNotFoundError(args.db)
    schema = read_json(args.schema)
    names = schema["feature_names"]
    if len(names) != 88 or len(set(names)) != 88:
        raise ValueError("Expected the frozen 88-feature schema")
    log_cols, bounded_cols = feature_groups(names)
    block_ms = args.block_minutes * 60_000
    if args.stride < 1 or args.stride > SEQ_LEN:
        raise ValueError("stride must be between 1 and 10 (larger strides leave gaps)")
    if (args.block_minutes <= 0 or DAY_MS % block_ms
            or block_ms < SEQ_LEN * BIN_MS or block_ms % (args.stride * BIN_MS)):
        raise ValueError("Block length must divide a UTC day and align with the stride grid")
    if min(args.train_budget, args.validation_budget, args.calibration_budget) < 1:
        raise ValueError("Budgets must be positive")
    if not 0 < args.max_host_fraction <= 1:
        raise ValueError("max-host-fraction must be in (0,1]")
    conn = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True, timeout=120)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=FILE")
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE})")}
        if not set(names + ["ipv4_src_addr", "bucket_start_ms", "label"]).issubset(cols):
            raise ValueError("CIC bucket table/schema is missing; generate buckets first")
        n, max_id = conn.execute(f"SELECT COUNT(*),MAX(rowid) FROM {TABLE}").fetchone()
        if not n or n != max_id or n >= np.iinfo(np.int32).max:
            raise ValueError("Expected nonempty contiguous positive SQLite rowids fitting int32")
        block_ids = [r[0] for r in conn.execute(
            f"SELECT DISTINCT bucket_start_ms/{block_ms} FROM {TABLE}")]
        assignment = split_blocks(block_ids, block_ms, args.seed)
        print(f"[prepare] {n:,} existing buckets; {len(block_ids)} global time blocks", flush=True)
        if args.check_only:
            print({s: list(assignment.values()).count(s) for s in SPLITS})
            print("[check] Database/schema readable; no output files created.")
            return
        empty_output(args.output)
        dump_json(args.output / "manifest.json", {"complete": False})
        attacks, lookup = attack_lookup(conn)
        features = np.lib.format.open_memmap(args.output / "bucket_features.npy", mode="w+",
                                             dtype=np.float32, shape=(n, len(names)))
        pools = {s: [] for s in SPLITS}
        candidates = {s: Counter() for s in SPLITS}
        hosts = []
        meta = []
        current = None

        def finish_host():
            if not meta:
                return
            windows, tags = make_windows(len(hosts) - 1, meta, args.stride, block_ms, assignment)
            for split in SPLITS:
                subset = windows[tags == split]
                candidates[split].update({"all": len(subset), "benign": int((subset["label"] == 0).sum()),
                                          "attack": int((subset["label"] == 1).sum())})
                if split != "test":
                    subset = subset[subset["label"] == 0]
                if len(subset):
                    pools[split].append(subset)

        quoted = ",".join('"' + name + '"' for name in names)
        cursor = conn.execute(f"SELECT rowid,ipv4_src_addr,bucket_start_ms,label,{quoted} "
                              f"FROM {TABLE} ORDER BY ipv4_src_addr,bucket_start_ms")
        done = 0
        while True:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            row_ids = np.fromiter((int(r[0]) - 1 for r in batch), dtype=np.int64)
            x = np.asarray([r[4:] for r in batch], dtype=np.float32)
            if not np.isfinite(x).all() or np.any(x[:, log_cols] < 0):
                raise ValueError("Nonfinite/negative engineered bucket feature")
            if np.any(x[:, bounded_cols] < -1e-6) or np.any(x[:, bounded_cols] > 1 + 1e-6):
                raise ValueError("Ratio/entropy outside [0,1]")
            features[row_ids] = x
            for row, row_id in zip(batch, row_ids):
                ip, stamp, label = str(row[1]), int(row[2]), int(row[3])
                if label not in (0, 1):
                    raise ValueError("Nonbinary bucket label")
                if ip != current:
                    finish_host()
                    hosts.append(ip)
                    current, meta = ip, []
                bits, count = lookup.get((ip, stamp), (0, 0))
                if (label == 1) != bool(bits):
                    raise ValueError(f"Bucket/raw attack mismatch at {ip}, {stamp}")
                meta.append((int(row_id), stamp, label, bits, count))
            done += len(batch)
            if done // 250_000 != (done - len(batch)) // 250_000:
                print(f"[prepare] {done:,}/{n:,} buckets, {len(hosts):,} hosts", flush=True)
        finish_host()
        features.flush()
        del features
        del lookup
        budgets = {"train": args.train_budget, "validation": args.validation_budget,
                   "calibration": args.calibration_budget}
        summary, used_blocks, used_rows = {}, {}, {}
        for j, split in enumerate(SPLITS):
            windows = np.concatenate(pools[split]) if pools[split] else np.empty(0, dtype=WINDOW_DTYPE)
            pools[split].clear()
            if split != "test":
                windows = balanced_sample(windows, budgets[split], args.seed + j, block_ms,
                                          args.max_host_fraction)
            if not len(windows):
                raise ValueError(f"No usable {split} windows; inspect the capture and time blocks")
            order = np.lexsort((windows["host"], windows["start_ms"]))
            windows = windows[order]
            np.save(args.output / f"{split}_windows.npy", windows, allow_pickle=False)
            write_csv(args.output / f"{split}_metadata.csv",
                      metadata_rows(windows, hosts, attacks, block_ms))
            used_blocks[split] = set((windows["start_ms"] // block_ms).tolist())
            ids = np.unique(windows["rows"])
            used_rows[split] = ids[ids >= 0]
            per_host = np.unique(windows["host"], return_counts=True)[1]
            summary[split] = {
                "candidate_counts": dict(candidates[split]), "selected": len(windows),
                "benign": int((windows["label"] == 0).sum()), "attack": int((windows["label"] == 1).sum()),
                "unique_buckets": len(used_rows[split]), "hosts": len(per_host),
                "largest_host_fraction": float(per_host.max() / len(windows)),
                "occupancy_histogram": {str(k): int((windows["active"] == k).sum()) for k in range(1, 11)},
                "attack_type_windows": {name: int(((windows["attack_bits"] & (1 << i)) > 0).sum())
                                        for i, name in enumerate(attacks)},
            }
            if split in budgets:
                summary[split]["requested_budget"] = budgets[split]
                summary[split]["budget_shortfall"] = max(0, budgets[split] - len(windows))
            print(f"[{split}] {len(windows):,} windows; benign={summary[split]['benign']:,}; "
                  f"attack={summary[split]['attack']:,}", flush=True)
        for i, left in enumerate(SPLITS):
            for right in SPLITS[i + 1:]:
                if used_blocks[left] & used_blocks[right] or np.intersect1d(used_rows[left], used_rows[right]).size:
                    raise AssertionError(f"Shared block or raw bucket between {left} and {right}")
        if not summary["test"]["attack"] or not summary["test"]["benign"]:
            raise ValueError("Frozen test split lacks one class; no binary AUC possible. Do not tune seed using scores.")
        write_csv(args.output / "time_blocks.csv", (
            {"block_id": b, "start_ms": b * block_ms,
             "day_utc": datetime.fromtimestamp(b * block_ms / 1000, timezone.utc).date().isoformat(),
             "split": assignment[b]} for b in sorted(assignment)))
        dump_json(args.output / "hosts.json", hosts)
        dump_json(args.output / "categorical_schema.json", schema)
        manifest = {
            "complete": True, "format_version": 1, "dataset": "NF-CICIDS2018-v3 only",
            "database": str(args.db.resolve()), "database_bytes": args.db.stat().st_size,
            "database_mtime_ns": args.db.stat().st_mtime_ns,
            "feature_names": names, "n_features": len(names), "bucket_rows": n,
            "bucket_size_ms": BIN_MS, "sequence_length": SEQ_LEN, "stride": args.stride,
            "block_size_ms": block_ms, "seed": args.seed,
            "split_policy": "label-independent day-stratified network-wide time blocks, 60/10/10/20 approximately",
            "tiny_day_policy": "fewer than 4 active blocks -> test only",
            "evaluation_scope": "same environment, unseen time blocks; NOT host-disjoint or future-day generalization",
            "window_label": "any observed attack bucket -> attack; no label-based sequence segmentation",
            "silence": "no observed flow-start bucket; not asserted to mean true host inactivity",
            "sampling": "benign day/host/block round-robin; test retains all nonempty in-block windows",
            "max_host_fraction_of_budget": args.max_host_fraction,
            "attack_types": attacks, "splits": summary,
            "cross_split_shared_buckets": 0, "cross_split_shared_blocks": 0,
            "build_seconds": time.monotonic() - started,
            "warning": "flow statistics are allocated by flow start, so this is offline detection, not causal real-time latency",
        }
        dump_json(args.output / "manifest.json", manifest)
        print(f"[done] {args.output.resolve()} ({manifest['build_seconds']:.1f}s)", flush=True)
        print("[note] Test attacks remain imbalanced; missing attack types must be reported as unevaluated.")
    finally:
        conn.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", type=Path, default=ROOT / "datasets/cic_2018_traffic.db")
    p.add_argument("--schema", type=Path, default=ROOT / "processed_binary_temporal/categorical_schema.json")
    p.add_argument("--output", type=Path, default=HERE / "data_stride5")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--block-minutes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-budget", type=int, default=100_000)
    p.add_argument("--validation-budget", type=int, default=20_000)
    p.add_argument("--calibration-budget", type=int, default=20_000)
    p.add_argument("--max-host-fraction", type=float, default=.01)
    p.add_argument("--check-only", action="store_true", help="Check existing DB/schema; do not build data")
    return p.parse_args(argv)


if __name__ == "__main__":
    prepare(parse_args())
