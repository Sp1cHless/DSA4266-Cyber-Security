#!/usr/bin/env python3
"""Compare CIC split candidates without fitting features or changing the builder.

Reuse completed EDA 02 100s windows. One read-only raw scan reconciles every
window, records actual flow ends, and collects 10s attack activity. Candidate
episodes are gap-based diagnostics, not ground-truth independent campaigns.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
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

import numpy as np

HERE = Path(__file__).resolve().parent
WINDOW = 100_000
DAY = 86_400_000
SPLITS = ("train", "validation", "test")
GAPS = (60, 300, 900)
VERSION = 1
spec = importlib.util.spec_from_file_location("split_eda01", HERE / "01_compare_perspectives.py")
eda01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda01)


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def utc(ms):
    return datetime.fromtimestamp(int(ms)/1000, timezone.utc).isoformat(timespec="milliseconds")


def fingerprint(path):
    st = path.stat()
    return {"bytes": st.st_size, "mtime_ns": str(st.st_mtime_ns)}


def latest_sequences():
    paths = [p.parent for p in (HERE / "outputs").glob("02_sequence_lengths_*/results.json")
             if p.with_name("run_status.json").is_file()
             and json.loads(p.with_name("run_status.json").read_text())["complete"]]
    if not paths:
        raise FileNotFoundError("Run EDA 02 first, or pass --sequences-dir")
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def validate_inputs(folder, db_override=None):
    result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or 100 not in result["metadata"]["lengths_seconds"]:
        raise ValueError("Need completed EDA 02 with a 100s catalog")
    db = (db_override or Path(result["metadata"]["raw_database"])).resolve()
    expected = {"bytes": result["metadata"]["raw_database_bytes"],
                "mtime_ns": str(result["metadata"]["raw_database_mtime_ns"])}
    if fingerprint(db) != expected:
        raise ValueError("Raw database fingerprint differs from EDA 02")
    return result, db, folder / "windows_100s.csv.gz"


def load_catalog(path, controls):
    # Preserve EDA 02 source/time ordering, also used by the current builder.
    n = int(controls["windows"])
    values = {k: np.zeros(n, dtype=np.int64) for k in ("start", "flows", "attack_flows", "first", "last")}
    values["source"] = np.zeros(n, dtype=np.int32)
    names, name_ids, lookup, declared_attacks = [], {}, {}, {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                raise ValueError("Catalog has more rows than its control total")
            ip, start = row["source_ip"], int(row["start_ms"])
            if ip not in name_ids:
                name_ids[ip] = len(names)
                names.append(ip)
            key = (name_ids[ip], start)
            if key in lookup or int(row["end_ms_exclusive"]) != start + WINDOW:
                raise ValueError("Repeated or malformed 100s window")
            lookup[key] = i
            values["source"][i], values["start"][i] = key
            values["flows"][i], values["attack_flows"][i] = int(row["flows"]), int(row["attack_flows"])
            values["first"][i] = start + round(float(row["first_flow_offset_s"])*1000)
            values["last"][i] = start + round(float(row["last_flow_offset_s"])*1000)
            if row["attack_types"]:
                declared_attacks[i] = set(row["attack_types"].split(";"))
        if i + 1 != n:
            raise ValueError("Catalog row count does not match EDA 02")
    if int(values["flows"].sum()) != controls["flows"] or int(values["attack_flows"].sum()) != controls["attack_flows"]:
        raise ValueError("Catalog flow totals do not match EDA 02")
    values["source_names"] = np.asarray(names)
    log(f"Loaded {n:,} windows and {len(names):,} sources")
    return values, name_ids, lookup, declared_attacks


def scan_raw(db, values, name_ids, lookup, declared_attacks):
    n = len(values["start"])
    counts, positives = np.zeros(n, np.int64), np.zeros(n, np.int64)
    max_end = np.full(n, -1, np.int64)
    min_start = np.full(n, np.iinfo(np.int64).max, np.int64)
    max_start = np.full(n, -1, np.int64)
    # cells: (sequence index, attack name) -> exact malicious flow count
    cells, bins, invalid = Counter(), {}, 0
    read = 0
    before = fingerprint(db)
    conn = sqlite3.connect(db.as_uri()+"?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        cursor = conn.execute("""SELECT ipv4_src_addr,flow_start_milliseconds,
            flow_end_milliseconds,label,attack FROM network_flows NOT INDEXED""")
        while rows := cursor.fetchmany(100_000):
            for ip, start, end, label, attack in rows:
                if not isinstance(start, int) or not isinstance(end, int) or end < start or label not in (0, 1):
                    invalid += 1
                    continue
                sid = lookup[(name_ids[ip], (start//WINDOW)*WINDOW)]
                counts[sid] += 1
                positives[sid] += label
                if end > max_end[sid]:
                    max_end[sid] = end
                if start < min_start[sid]:
                    min_start[sid] = start
                if start > max_start[sid]:
                    max_start[sid] = start
                if label:
                    cells[(sid, attack)] += 1
                    key = (attack, start//10_000)
                    if key not in bins:
                        bins[key] = [start, start, 0, set()]
                    b = bins[key]
                    b[0], b[1], b[2] = min(b[0], start), max(b[1], start), b[2]+1
                    b[3].add(sid)
            read += len(rows)
            if read % 2_000_000 == 0:
                log(f"Raw timing scan: {read:,} flows")
    finally:
        conn.close()
    if invalid or not np.array_equal(counts, values["flows"]) or not np.array_equal(positives, values["attack_flows"]):
        raise AssertionError(f"Raw/catalog reconciliation failed; invalid timing rows={invalid}")
    if not np.array_equal(min_start, values["first"]) or not np.array_equal(max_start, values["last"]):
        raise AssertionError("Raw/catalog first/last flow starts differ")
    actual = defaultdict(set)
    for sid, attack in cells:
        actual[sid].add(attack)
    if dict(actual) != declared_attacks:
        raise AssertionError("Raw/catalog attack memberships differ")
    if fingerprint(db) != before:
        raise RuntimeError("Raw database changed during analysis")
    values["max_end"] = max_end
    log(f"Reconciled all {read:,} raw flows; {len(bins):,} attack activity bins")
    return cells, bins


def make_episodes(bins, gap_seconds):
    episodes, current = [], None
    for (attack, _), (first, last, count, members) in sorted(bins.items()):
        if (current is None or current["attack_type"] != attack
                or first//DAY != current["first_ms"]//DAY
                or first-current["last_ms"] > gap_seconds*1000):
            current = {"episode_id": len(episodes)+1, "attack_type": attack, "gap_seconds": gap_seconds,
                       "first_ms": first, "last_ms": last, "attack_flows": 0, "members": set(), "bins": 0}
            episodes.append(current)
        current["last_ms"] = max(current["last_ms"], last)
        current["attack_flows"] += count
        current["members"].update(members)
        current["bins"] += 1
    return episodes


def partition_groups(ids, weights, seed=None):
    """Two cuts nearest 60% and 80% cumulative window mass, without labels."""
    ids, weights = np.asarray(ids), np.asarray(weights)
    if len(ids) < 3:
        raise ValueError("At least three groups are required")
    order = np.arange(len(ids)) if seed is None else np.random.default_rng(seed).permutation(len(ids))
    cumulative = np.cumsum(weights[order])
    first = int(np.argmin(np.abs(cumulative[:len(ids)-2]-.6*cumulative[-1])))+1
    second = int(np.argmin(np.abs(cumulative[first:len(ids)-1]-.8*cumulative[-1])))+first+1
    codes = np.empty(len(ids), np.uint8)
    codes[order[:first]], codes[order[first:second]], codes[order[second:]] = 0, 1, 2
    return codes


def random_stratified(values, cells, seed):
    # Match build_buckets.py's stratum ordering, seed and rounding exactly.
    names = sorted({attack for _, attack in cells})
    primary = np.full(len(values["start"]), "Benign", dtype=object)
    best = {}
    for (sid, attack), count in cells.items():
        if sid not in best or (-count, attack) < best[sid]:
            best[sid] = (-count, attack)
            primary[sid] = attack
    codes = np.empty(len(primary), np.uint8)
    rng = np.random.default_rng(seed)
    for name in sorted(set(names)|{"Benign"}):
        indices = np.flatnonzero(primary == name)
        if not len(indices):
            continue
        rng.shuffle(indices)
        if len(indices) < 3:
            counts = [len(indices), 0, 0]
        else:
            expected = np.array([.6, .2, .2])*len(indices)
            counts = np.floor(expected).astype(int)
            for j in np.argsort(-(expected-counts), kind="stable")[:len(indices)-int(counts.sum())]:
                counts[j] += 1
            for j in range(3):
                if not counts[j]:
                    donor = int(np.argmax(counts))
                    counts[donor] -= 1
                    counts[j] += 1
        offset = 0
        for code, count in enumerate(counts):
            codes[indices[offset:offset+count]] = code
            offset += count
    return codes


def guarded_blocks(times, episodes, block_ms):
    blocks = np.unique(times//block_ms)
    parent = np.arange(len(blocks))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for ep in episodes:
        lo = np.searchsorted(blocks, ep["first_ms"]//block_ms)
        hi = np.searchsorted(blocks, ep["last_ms"]//block_ms, side="right")
        for j in range(int(lo)+1, int(hi)):
            parent[find(j)] = find(int(lo))
    groups = np.array([find(j) for j in range(len(blocks))])
    return groups[np.searchsorted(blocks, times//block_ms)]


def overlaps_other_split(values, codes):
    """Conservative interval envelope vs another fold's assigned 100s windows.

    Includes the actual inclusive last flow end, even when duration disagrees.
    Envelope overlap is a risk diagnostic, not evidence of duplicated packets.
    """
    overlap = np.zeros(len(codes), bool)
    for code in range(3):
        members = np.flatnonzero(codes == code)
        other = np.unique(values["start"][(codes < 3) & (codes != code)])
        if not len(other) or not len(members):
            continue
        indices = np.searchsorted(other, values["first"][members]-WINDOW, side="right")
        valid = indices < len(other)
        overlap[members[valid]] = other[indices[valid]] <= values["max_end"][members[valid]]
    return overlap


def purge(values, original, embargo_ms):
    times, inverse = np.unique(values["start"], return_inverse=True)
    global_codes = np.full(len(times), 255, np.uint8)
    global_codes[inverse] = original
    if not np.array_equal(global_codes[inverse], original):
        raise ValueError("Cannot time-purge a split with mixed fold assignments at one global time")
    boundaries = times[1:][global_codes[1:] != global_codes[:-1]]
    # Drop complete windows whose support intersects a boundary's guard band.
    left = values["start"]
    right = np.maximum(left+WINDOW, values["max_end"]+1)
    near = np.zeros(len(left), bool)
    if embargo_ms and len(boundaries):
        j = np.searchsorted(boundaries, left-embargo_ms, side="right")
        valid = j < len(boundaries)
        near[valid] = boundaries[j[valid]]-embargo_ms < right[valid]
    crossing = overlaps_other_split(values, original)
    codes = original.copy()
    codes[near | crossing] = 3
    return codes, {"embargo_removed": int(near.sum()), "interval_removed": int(crossing.sum()),
                   "removed_union": int((near|crossing).sum()), "boundaries": len(boundaries)}


def candidates(values, cells, episodes, seed, embargo):
    times, inverse, weights = np.unique(values["start"], return_inverse=True, return_counts=True)
    result = []
    result.append(("random_100s", "Current builder baseline; label-stratified windows; no purge", random_stratified(values, cells, seed), None))
    result.append(("chronological", "Global 100s times in order; cuts nearest 60/80% sequence mass", partition_groups(times, weights)[inverse], len(times)))
    for name, randomized in (("day_chronological", False), ("day_shuffled", True)):
        groups, gi, gw = np.unique(values["start"]//DAY, return_inverse=True, return_counts=True)
        result.append((name, "Whole UTC dates; "+("seeded order" if randomized else "chronological order"),
                       partition_groups(groups, gw, seed if randomized else None)[gi], len(groups)))
    for minutes in (15, 30, 60):
        groups, gi, gw = np.unique(values["start"]//(minutes*60_000), return_inverse=True, return_counts=True)
        result.append((f"blocks_{minutes}m", f"Whole global {minutes}min blocks; seeded order; no label stratification",
                       partition_groups(groups, gw, seed)[gi], len(groups)))
    guarded = guarded_blocks(values["start"], episodes[300], 1_800_000)
    groups, gi, gw = np.unique(guarded, return_inverse=True, return_counts=True)
    result.append(("guarded_30m", "Global 30min blocks unioned across each 300s-gap attack episode; seeded order",
                   partition_groups(groups, gw, seed)[gi], len(groups)))
    for name, method, original, group_count in result:
        if name == "random_100s":
            codes, drops = original.copy(), {"embargo_removed": 0, "interval_removed": 0, "removed_union": 0, "boundaries": None}
        else:
            codes, drops = purge(values, original, embargo*1000)
        yield {"candidate": name, "method": method, "codes": codes, "original_codes": original,
               "groups": group_count, **drops}


def summarize(values, cells, episodes, plans):
    composition, coverage, isolation, daily, group_rows = [], [], [], [], []
    n = len(values["start"])
    attack_names = sorted({a for _, a in cells})
    type_cells = {a: [(i, c) for (i, t), c in cells.items() if t == a] for a in attack_names}
    days = values["start"]//DAY
    # Catalog is ordered by source then start, which permits exact adjacent-window diagnostics.
    adjacent = np.flatnonzero((values["source"][1:] == values["source"][:-1]) & (np.diff(values["start"]) == WINDOW))
    for plan in plans:
        name, codes = plan["candidate"], plan["codes"]
        log(f"Summarizing {name}")
        for code, split in enumerate((*SPLITS, "excluded")):
            select = codes == code
            count = int(select.sum())
            composition.append({"candidate": name, "split": split, "sequences": count,
                                "positive_sequences": int(np.count_nonzero(values["attack_flows"][select])),
                                "flows": int(values["flows"][select].sum()), "attack_flows": int(values["attack_flows"][select].sum()),
                                "sources": len(np.unique(values["source"][select])), "days": len(np.unique(days[select]))})
            if count:
                for day in np.unique(days[select]):
                    mask = select & (days == day)
                    daily.append({"candidate": name, "split": split, "day_utc": utc(day*DAY)[:10],
                                  "sequences": int(mask.sum()), "positive_sequences": int(np.count_nonzero(values["attack_flows"][mask])),
                                  "flows": int(values["flows"][mask].sum()), "attack_flows": int(values["attack_flows"][mask].sum())})
        for attack in attack_names:
            entries = type_cells[attack]
            indices = np.asarray([i for i, _ in entries], np.int64)
            counts = np.asarray([c for _, c in entries], np.int64)
            row = {"candidate": name, "attack_type": attack}
            for code, split in enumerate((*SPLITS, "excluded")):
                kept = indices[codes[indices] == code]
                row[split+"_windows"] = len(kept)
                row[split+"_flows"] = int(counts[codes[indices] == code].sum())
                row[split+"_sources"] = len(np.unique(values["source"][kept]))
                row[split+"_episodes"] = sum(bool(np.intersect1d(kept, np.fromiter(ep["members"], np.int64)).size)
                                              for ep in episodes[300] if ep["attack_type"] == attack)
            row["test_unseen_in_train"] = row["test_windows"] > 0 and row["train_windows"] == 0
            row["test_only"] = row["test_unseen_in_train"] and row["validation_windows"] == 0
            row["validation_unseen_in_train"] = row["validation_windows"] > 0 and row["train_windows"] == 0
            row["present_in_all_three"] = all(row[s+"_windows"] > 0 for s in SPLITS)
            coverage.append(row)
        times = np.unique(values["start"])
        fold_bits = np.zeros(len(times), np.uint8)
        for code in range(3):
            at = np.searchsorted(times, values["start"][codes == code])
            fold_bits[at] |= 1 << code
        mixed_times = int(np.count_nonzero((fold_bits != 0) & ((fold_bits & (fold_bits-1)) != 0)))
        risk = {"candidate": name, "shared_global_100s": mixed_times,
                "adjacent_source_pairs_cross_split": int(np.count_nonzero((codes[adjacent] < 3) & (codes[adjacent+1] < 3) & (codes[adjacent] != codes[adjacent+1]))),
                "interval_envelope_overlap_sequences": int(overlaps_other_split(values, codes).sum()),
                "source_overlap_train_test": len(np.intersect1d(values["source"][codes == 0], values["source"][codes == 2])),
                "retained_sequences": int(np.count_nonzero(codes < 3)), "excluded_sequences": int(np.count_nonzero(codes == 3)),
                "embargo_removed": plan["embargo_removed"], "interval_removed": plan["interval_removed"], "boundaries": plan["boundaries"]}
        for gap, eps in episodes.items():
            cross, train_test, affected = 0, 0, set()
            for ep in eps:
                ids = np.fromiter(ep["members"], np.int64)
                present = set(codes[ids].tolist())-{3}
                if len(present) > 1:
                    cross += 1
                    affected.update(ids[codes[ids] < 3].tolist())
                train_test += (0 in present and 2 in present)
                if gap == 300:
                    group_rows.append({"candidate": name, "episode_id": ep["episode_id"], "attack_type": ep["attack_type"],
                                       "train_windows": int(np.count_nonzero(codes[ids] == 0)),
                                       "validation_windows": int(np.count_nonzero(codes[ids] == 1)),
                                       "test_windows": int(np.count_nonzero(codes[ids] == 2)),
                                       "excluded_windows": int(np.count_nonzero(codes[ids] == 3)), "cross_split": len(present) > 1})
            risk[f"cross_episodes_{gap}s"] = cross
            risk[f"train_test_episodes_{gap}s"] = int(train_test)
            risk[f"affected_attack_windows_{gap}s"] = len(affected)
        if name != "random_100s" and (mixed_times or risk["interval_envelope_overlap_sequences"]):
            raise AssertionError(f"Time isolation failed after purge for {name}")
        isolation.append(risk)
        if sum(r["sequences"] for r in composition if r["candidate"] == name) != n:
            raise AssertionError("Candidate sample reconciliation failed")
    return composition, coverage, isolation, daily, group_rows


def activity_tables(values, cells, episodes):
    rows, detail = [], []
    names = sorted({a for _, a in cells})
    for gap, eps in episodes.items():
        for name in names:
            selected = [ep for ep in eps if ep["attack_type"] == name]
            ids = sorted(set().union(*(ep["members"] for ep in selected)))
            rows.append({"attack_type": name, "gap_seconds": gap, "candidate_episodes": len(selected),
                         "days": len({ep["first_ms"]//DAY for ep in selected}),
                         "sources": len(np.unique(values["source"][ids])), "windows": len(ids),
                         "attack_flows": sum(ep["attack_flows"] for ep in selected),
                         "first_utc": utc(min(ep["first_ms"] for ep in selected)),
                         "last_utc": utc(max(ep["last_ms"] for ep in selected)),
                         "largest_episode_flow_share": max(ep["attack_flows"] for ep in selected)/sum(ep["attack_flows"] for ep in selected)})
        if gap == 300:
            for ep in eps:
                detail.append({k:v for k,v in ep.items() if k != "members"} | {
                    "first_utc": utc(ep["first_ms"]), "last_utc": utc(ep["last_ms"]),
                    "span_seconds": (ep["last_ms"]-ep["first_ms"])/1000,
                    "windows": len(ep["members"]), "sources": len(np.unique(values["source"][list(ep["members"])]))})
    return rows, detail


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_excel(out, reports, node, modules):
    work = out / "_excel_runtime"
    work.mkdir(exist_ok=True)
    link = work / "node_modules"
    if not link.exists():
        if os.name == "nt":
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(modules)], check=True, stdout=subprocess.DEVNULL)
        else:
            link.symlink_to(modules, target_is_directory=True)
    elif link.resolve() != modules:
        raise ValueError("Existing runtime link targets different packages")
    builder = work / "render_splits.mjs"
    shutil.copyfile(HERE / "render_splits.mjs", builder)
    reports.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(node), str(builder), str(out / "results.json"), str(reports)], check=True)
    for file in ("results.json",):
        shutil.copyfile(out / file, reports / file)
    shutil.copyfile(HERE / "04_metrics_and_analysis.md", reports / "metrics_explained.md")
    save_json(out / "run_status.json", {"complete": True, "reports": str(reports)})


def analyze(folder, db, catalog, previous, out, seed, embargo):
    began = time.monotonic()
    summary = next(r["summary"] for r in previous["lengths"] if r["summary"]["length_s"] == 100)
    values, names, lookup, declared = load_catalog(catalog, summary)
    cells, bins = scan_raw(db, values, names, lookup, declared)
    del lookup, declared, names
    eps = {gap: make_episodes(bins, gap) for gap in GAPS}
    plans = list(candidates(values, cells, eps, seed, embargo))
    composition, coverage, isolation, daily, episode_splits = summarize(values, cells, eps, plans)
    activity, detail = activity_tables(values, cells, eps)
    payload = {"version": VERSION, "metadata": {"generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "dataset": "NF-CICIDS2018-v3", "raw_database": str(db), "raw_fingerprint": fingerprint(db),
               "catalog": str(catalog), "catalog_fingerprint": fingerprint(catalog), "seed": seed,
               "window_seconds": 100, "embargo_seconds": embargo, "episode_gaps_seconds": list(GAPS),
               "raw_flows_reconciled": int(values["flows"].sum()), "raw_attack_flows_reconciled": int(values["attack_flows"].sum()),
               "source_sequences": len(values["start"]), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "analysis_seconds": round(time.monotonic()-began, 2), "raw_end_max_utc": utc(values["max_end"].max()),
               "exact_full_row_duplicate_audit": "not performed", "builder_changed": False,
               "assignments": "candidate_assignments.npz; row key=(source_names[source_id],start_ms); codes 0=train,1=validation,2=test,3=excluded"},
               "candidates": [{k:v for k,v in p.items() if k not in ("codes", "original_codes")} for p in plans],
               "composition": composition, "coverage": coverage, "isolation": isolation, "daily": daily,
               "activity": activity, "episodes": detail, "episode_splits": episode_splits}
    save_json(out / "results.json", payload)
    np.savez_compressed(out / "candidate_assignments.npz", source_names=values["source_names"], source_id=values["source"],
                        start_ms=values["start"], first_ms=values["first"], max_end_ms=values["max_end"],
                        **{p["candidate"]: p["codes"] for p in plans}, **{p["candidate"]+"_before_purge": p["original_codes"] for p in plans})
    for name in ("composition", "coverage", "isolation", "daily", "activity", "episodes", "episode_splits"):
        write_csv(out / f"{name}.csv", payload[name])
    log(f"Analysis complete in {payload['metadata']['analysis_seconds']}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences-dir", type=Path)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--embargo-seconds", type=int, default=100)
    parser.add_argument("--render-only", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--node-modules", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or args.embargo_seconds < 0:
        parser.error("Seed and embargo must be nonnegative")
    node, modules = eda01.find_excel_runtime(args.node, args.node_modules)
    if args.render_only:
        out = args.render_only.resolve()
        saved = json.loads((out / "report_config.json").read_text())
        export_excel(out, (args.reports_dir or Path(saved["reports"])).resolve(), node, modules)
        return
    folder = (args.sequences_dir or latest_sequences()).resolve()
    previous, db, catalog = validate_inputs(folder, args.db)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = (args.output or HERE / "outputs" / f"04_splits_{stamp}").resolve()
    reports = (args.reports_dir or HERE.parent / "reports" / "EDA" / f"04_splits_{stamp}").resolve()
    if (out.exists() and any(out.iterdir())) or (reports.exists() and any(reports.iterdir())):
        parser.error("Use new/empty output and report directories")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_status.json", {"complete": False})
    save_json(out / "report_config.json", {"reports": str(reports)})
    log(f"Local cache: {out}; Git-visible reports: {reports}")
    analyze(folder, db, catalog, previous, out, args.seed, args.embargo_seconds)
    export_excel(out, reports, node, modules)
    log(f"Done: {reports / '04_cic_split_audit.xlsx'}")


if __name__ == "__main__":
    main()
