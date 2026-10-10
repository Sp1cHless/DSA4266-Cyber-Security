#!/usr/bin/env python3
"""Independent UNSW + CIC-v3 temporal training-set builder (no baseline overwrite).

Run from the repository root, on Windows PowerShell:
  python .\temporal_eda\03_build_merged_temporal_dataset.py import
  python .\temporal_eda\03_build_merged_temporal_dataset.py buckets
  python .\temporal_eda\03_build_merged_temporal_dataset.py assemble

Original UNSW processed data is reused exactly (no unnecessary recomputation).
The CIC CSV is imported separately into datasets/cic_2018_traffic.db.
CIC buckets use the same feature definitions and frozen 88-dimensional schema.
Only benign windows are admitted to the merged train or validation NPZ.
All train/validation/test host choices are made before model training.

Large CIC datasets: allow ample disk space. The 'buckets' stage checkpoints each
completed host so an interrupted run can be restarted without losing progress.
This script never reads/writes the original datasets/network_traffic.db.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sqlite3
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

BIN_MS = 10_000
SEQ_LEN = 10
STRIDE = 1
SOURCE = "ipv4_src_addr"
STAMP = "bucket_start_ms"
TEMP_BUCKET_TABLE = "temporal_buckets_10s_full"
CIC_BUCKET_TABLE = "cic_temporal_buckets"
META_COLS = ["sequence_id", "src_ip", "segment_id", "start_bucket_ms",
             "end_bucket_ms", "label", "num_buckets", "split", "round", "dataset_id"]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def clean_column(name: str) -> str:
    col = str(name).lower()
    for old, new in [(" ", "_"), ("-", "_"), ("(", ""), (")", ""),
                     ("/", "_"), (".", "_"), ("%", "percent"),
                     ("&", "and"), ("+", "plus")]:
        col = col.replace(old, new)
    return col


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def open_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(str(path), timeout=300)
    conn.execute("PRAGMA busy_timeout=300000")
    conn.execute("PRAGMA temp_store=FILE")
    return conn


def ensure_schema_compatible(conn: sqlite3.Connection, unsw_db: Path) -> None:
    """Compare normalized raw CSV fields against the original SQLite schema."""
    raw_cols = set(columns(conn, "network_flows"))
    if not raw_cols:
        raise ValueError("CIC raw table network_flows is missing")
    with open_db(unsw_db) as u:
        old_cols = set(columns(u, "network_flows"))
    if not old_cols:
        raise ValueError("Original UNSW SQLite network_flows table missing")
    missing = sorted(old_cols - raw_cols)
    extra = sorted(raw_cols - old_cols)
    if missing or extra:
        raise ValueError(f"Raw dataset schema differs: missing={missing}, extra={extra}")
    required = {SOURCE, "flow_start_milliseconds", "label", "in_bytes", "out_bytes",
                "in_pkts", "out_pkts", "flow_duration_milliseconds", "protocol",
                "l7_proto", "tcp_flags", "client_tcp_flags", "server_tcp_flags",
                "l4_src_port", "l4_dst_port", "ipv4_dst_addr", "icmp_ipv4_type",
                "dns_query_id", "dns_query_type", "dns_ttl_answer", "ftp_command_ret_code"}
    if not required.issubset(raw_cols):
        raise ValueError(f"Missing features required by old temporal SQL: {sorted(required - raw_cols)}")
    log(f"[schema] identical normalized raw columns: {len(raw_cols)}")


def stage_import(args) -> None:
    csv_path = args.cic_csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CIC CSV not found: {csv_path}; override with --cic-csv")
    if args.cic_db.resolve() == args.unsw_db.resolve():
        raise ValueError("Refusing to overwrite original UNSW database")
    args.cic_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.cic_db), timeout=300)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute("PRAGMA temp_store=FILE")
        if table_exists(conn, "import_status"):
            row = conn.execute("SELECT csv_path, csv_size, n_rows FROM import_status WHERE completed=1").fetchone()
            if row:
                source, source_size, n = row
                if Path(source) != csv_path or source_size != csv_path.stat().st_size:
                    raise ValueError("Existing CIC import belongs to a different CSV. Use a new --cic-db")
                if conn.execute("SELECT COUNT(*) FROM network_flows").fetchone()[0] != n:
                    raise ValueError("CIC row count disagrees with saved import_status")
                ensure_schema_compatible(conn, args.unsw_db)
                log(f"[import] complete; reused {n:,} rows in {args.cic_db}")
                return
        if table_exists(conn, "network_flows") or table_exists(conn, "import_status"):
            raise RuntimeError("CIC database has an incomplete import. Use a NEW --cic-db path; "
                               "do not silently reimport over partial data")
        expected_raw = None
        with open_db(args.unsw_db) as old:
            expected_raw = set(columns(old, "network_flows"))
        total = 0
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=args.chunk_size,
                                              low_memory=False), start=1):
            chunk.columns = [clean_column(x) for x in chunk.columns]
            if len(chunk.columns) != len(set(chunk.columns)):
                raise ValueError("Duplicate columns after normalization")
            if set(chunk.columns) != expected_raw:
                raise ValueError(f"CIC CSV column mismatch: missing={sorted(expected_raw-set(chunk.columns))}; "
                                 f"extra={sorted(set(chunk.columns)-expected_raw)}")
            if chunk[SOURCE].isna().any() or chunk["flow_start_milliseconds"].isna().any():
                raise ValueError(f"Chunk {i}: missing src_ip or timestamp")
            chunk[SOURCE] = chunk[SOURCE].astype(str)
            t = pd.to_numeric(chunk["flow_start_milliseconds"], errors="raise")
            if not np.isfinite(t.to_numpy(dtype=float)).all():
                raise ValueError(f"Chunk {i}: non-finite timestamps")
            chunk["flow_start_milliseconds"] = t.astype(np.int64)
            y = pd.to_numeric(chunk["label"], errors="raise")
            if not y.isin([0, 1]).all():
                raise ValueError(f"Chunk {i}: label must be binary 0/1")
            chunk["label"] = y.astype(np.int8)
            chunk.to_sql("network_flows", conn, if_exists="append", index=False,
                         method=None)
            total += len(chunk)
            log(f"[import] chunk {i}, total={total:,}")
        if total == 0:
            raise ValueError("Empty CIC CSV")
        conn.execute("CREATE TABLE import_status (csv_path TEXT, csv_size INTEGER, "
                     "n_rows INTEGER, completed INTEGER)")
        conn.execute("INSERT INTO import_status VALUES (?, ?, ?, 1)",
                     (str(csv_path), csv_path.stat().st_size, total))
        conn.commit()
        log("[import] building source-IP index (this may take time and disk)...")
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cic_raw_src ON network_flows (ipv4_src_addr)')
        conn.commit()
        ensure_schema_compatible(conn, args.unsw_db)
        log(f"[import] finished: {total:,} flows; DB={args.cic_db}")
    finally:
        conn.close()


def original_builder(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Expected original temporal EDA script: {path}")
    spec = importlib.util.spec_from_file_location("official_bucket_eda_for_cic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frozen_l7_from_schema(mod, schema: dict):
    cats = []
    for cat in schema["kept_l7_categories"]:
        code = float(cat)
        cats.append((code, str(cat), mod.safe_category_name(str(cat))))
    names = mod.feature_names(
        mod.l7_ratio_expressions(cats)[1], mod.tcp_flag_expressions()[1]
    )
    if names != schema["feature_names"] or len(names) != 88:
        raise ValueError("Frozen original 88-dimensional feature schema mismatch")
    return cats


def stage_buckets(args) -> None:
    schema = load_json(args.unsw_processed / "categorical_schema.json")
    original = original_builder(args.bucket_script)
    cats = frozen_l7_from_schema(original, schema)
    if args.cic_db.resolve() == args.unsw_db.resolve():
        raise ValueError("Refusing to work inside original UNSW database")
    with open_db(args.cic_db) as conn:
        if not table_exists(conn, "import_status") or not conn.execute(
            "SELECT 1 FROM import_status WHERE completed=1").fetchone():
            raise ValueError("CIC raw import incomplete; run import stage first")
        ensure_schema_compatible(conn, args.unsw_db)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cic_raw_src ON network_flows (ipv4_src_addr)')
        conn.execute("CREATE TABLE IF NOT EXISTS bucket_progress (src_ip TEXT PRIMARY KEY, n_buckets INTEGER)")
        conn.commit()
        ips = [str(x[0]) for x in conn.execute(
            'SELECT DISTINCT ipv4_src_addr FROM network_flows ORDER BY ipv4_src_addr')]
        done = {str(x[0]) for x in conn.execute("SELECT src_ip FROM bucket_progress")}
        log(f"[buckets] hosts={len(ips)} completed={len(done)}; frozen 88 features")
        if args.limit_hosts is not None:
            ips = ips[:args.limit_hosts]
            log("[buckets] WARNING --limit-hosts is for testing only; output incomplete")
        original.RAW_TABLE = "network_flows"
        original.BUCKET_TABLE = TEMP_BUCKET_TABLE
        # Each host is processed separately: avoids copying all ~20M raw rows into
        # one massive temporary table; uses the *original* SQL feature definitions.
        for j, ip in enumerate(ips, 1):
            if ip in done:
                continue
            conn.execute("DROP TABLE IF EXISTS temp.flow_rows")
            conn.execute(f"CREATE TEMP TABLE flow_rows AS "
                         f"SELECT *, (flow_start_milliseconds / {BIN_MS}) * {BIN_MS} "
                         "AS bucket_start_ms FROM network_flows WHERE ipv4_src_addr = ?", (ip,))
            conn.execute("CREATE INDEX idx_flow_rows_key ON flow_rows (ipv4_src_addr, bucket_start_ms)")
            original.build_base_aggregates(conn, cats)
            original.build_distribution_helpers(conn)
            original.build_final_table(conn)
            actual = conn.execute(f"SELECT COUNT(*) FROM {TEMP_BUCKET_TABLE}").fetchone()[0]
            if not table_exists(conn, CIC_BUCKET_TABLE):
                conn.execute(f"CREATE TABLE {CIC_BUCKET_TABLE} AS SELECT * FROM {TEMP_BUCKET_TABLE} WHERE 0")
            conn.execute(f'INSERT INTO {CIC_BUCKET_TABLE} SELECT * FROM {TEMP_BUCKET_TABLE}')
            conn.execute("INSERT INTO bucket_progress VALUES (?, ?)", (ip, actual))
            conn.commit()
            # Free temporary working tables after each host.
            for t in ("flow_rows", "base_aggregates", "l7_distribution",
                      "src_port_distribution", "dst_port_distribution",
                      "dst_ip_distribution", "icmp_distribution", "dns_distribution"):
                conn.execute(f'DROP TABLE IF EXISTS temp."{t}"')
            if j <= 5 or j % 25 == 0 or j == len(ips):
                log(f"[buckets] {j}/{len(ips)} host={ip} buckets={actual:,}")
        if args.limit_hosts is not None:
            return
        if conn.execute("SELECT COUNT(*) FROM bucket_progress").fetchone()[0] != len(ips):
            raise RuntimeError("Host-wise bucket building incomplete")
        conn.execute(f"DROP TABLE IF EXISTS {TEMP_BUCKET_TABLE}")
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_cic_bucket_key "
                     f"ON {CIC_BUCKET_TABLE}(ipv4_src_addr, bucket_start_ms)")
        conn.commit()
        nflows = conn.execute("SELECT n_rows FROM import_status WHERE completed=1").fetchone()[0]
        ncovered = conn.execute(f"SELECT SUM(flow_count) FROM {CIC_BUCKET_TABLE}").fetchone()[0]
        if nflows != ncovered:
            raise AssertionError(f"CIC raw flows {nflows} != aggregated {ncovered}")
        missing = set(schema["feature_names"]) - set(columns(conn, CIC_BUCKET_TABLE))
        if missing:
            raise AssertionError(f"Missing engineered features: {missing}")
        log(f"[buckets] DONE: {len(ips)} hosts; {ncovered:,} flows accounted for; "
            f"{conn.execute(f'SELECT COUNT(*) FROM {CIC_BUCKET_TABLE}').fetchone()[0]:,} buckets")


def inspect_cic_hosts(conn, feature_names: list[str]) -> dict[str, dict]:
    """Count all valid benign-only windows, allowing mixed-label source hosts."""
    if not table_exists(conn, CIC_BUCKET_TABLE):
        raise ValueError("Missing CIC buckets; run buckets stage")
    rawhosts = conn.execute("SELECT COUNT(*) FROM bucket_progress").fetchone()[0]
    completed = conn.execute("SELECT COUNT(DISTINCT ipv4_src_addr) FROM network_flows").fetchone()[0]
    if rawhosts != completed:
        raise RuntimeError(f"CIC bucket build incomplete: {rawhosts}/{completed} hosts")
    counts = {}
    query = f"SELECT bucket_start_ms, label FROM {CIC_BUCKET_TABLE} "
    for host, flow_count in conn.execute(
        f"SELECT ipv4_src_addr, SUM(flow_count) FROM {CIC_BUCKET_TABLE} GROUP BY ipv4_src_addr ORDER BY ipv4_src_addr"
    ).fetchall():
        run = 0
        last_time = None
        n = 0
        saw_attack = False
        for t, label in conn.execute(query + "WHERE ipv4_src_addr=? ORDER BY bucket_start_ms", (host,)):
            t, label = int(t), int(label)
            if label not in (0, 1):
                raise ValueError(f"Nonbinary CIC bucket label for {host}: {label}")
            if label == 1:
                saw_attack = True
                run = 0
            elif last_time is not None and t - last_time == BIN_MS:
                run += 1
            else:
                run = 1
            if label == 0 and run >= SEQ_LEN:
                n += 1
            last_time = t
        counts[str(host)] = {"benign_windows": n, "raw_flows": int(flow_count),
                             "has_attack_buckets": saw_attack}
    return counts


def split_cic_hosts(counts: dict[str, dict], seed: int, val_fraction: float, test_fraction: float):
    eligible = [h for h, v in counts.items() if v["benign_windows"] > 0]
    if len(eligible) < 5:
        raise ValueError(f"Only {len(eligible)} CIC hosts yield benign sequences; "
                         "need >=5 to make host-disjoint train/val/test. "
                         "This is a DATASET feasibility issue; inspect host counts.")
    def rank(h):
        return hashlib.sha256(f"{seed}|CIC|{h}".encode()).hexdigest()
    ordered = sorted(eligible, key=rank)
    nval = max(2, round(len(eligible) * val_fraction))
    ntest = max(1, round(len(eligible) * test_fraction))
    if nval + ntest >= len(eligible):
        raise ValueError("Not enough CIC hosts after val/test split")
    val = sorted(ordered[:nval])
    test = sorted(ordered[nval:nval+ntest])
    train = sorted(ordered[nval+ntest:])
    return {"train": train, "validation": val, "holdout": test}


def choose_unsw_val_hosts(meta: pd.DataFrame, fraction: float = .15):
    """Hold out >=1 59 and >=1 149, but retain >=2 149 in TRUE train."""
    b = meta.loc[meta.label == 0]
    counts = {str(k): int(v) for k, v in b.groupby("src_ip").size().items()}
    hosts = sorted(counts)
    groups59 = {h for h in hosts if h.startswith("59.166.")}
    groups149 = {h for h in hosts if h.startswith("149.171.")}
    if len(groups59) < 2 or len(groups149) < 3:
        raise ValueError("Unexpected UNSW productive-host groups: cannot preserve both 59/149 groups")
    target = sum(counts.values()) * fraction
    best = None
    choice = None
    for k in range(2, len(hosts)):
        for candidate in itertools.combinations(hosts, k):
            val = set(candidate)
            if not (val & groups59 and val & groups149):
                continue
            if len(groups149 - val) < 2 or len(groups59 - val) < 2:
                continue
            score = (abs(sum(counts[h] for h in val) - target), len(val), candidate)
            if best is None or score < best:
                best, choice = score, candidate
    if choice is None:
        raise ValueError("Could not find host-disjoint UNSW validation subset")
    val = sorted(choice)
    train = sorted(set(hosts) - set(val))
    return train, val


def quotas(counts: dict[str, int], total: int, max_per_host: int, seed: int) -> dict[str, int]:
    """Host-balanced deterministic quotas, never exceed caps or global budget."""
    if not counts:
        return {}
    host_cap = {h: min(n, max_per_host) for h, n in counts.items()}
    hosts = list(host_cap)
    if total < len(hosts):
        raise ValueError(f"Budget={total} < productive host count={len(hosts)}; "
                         "increase --cic-{split}-budget to include every selected host")
    target = min(total, sum(host_cap.values()))
    lo, hi = 0, max(host_cap.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(x, mid) for x in host_cap.values()) <= target:
            lo = mid
        else:
            hi = mid - 1
    allot = {h: min(x, lo) for h, x in host_cap.items()}
    remaining = target - sum(allot.values())
    ordered = sorted(hosts, key=lambda h: hashlib.sha256(f"{seed}|quota|{h}".encode()).hexdigest())
    for h in ordered:
        if remaining <= 0:
            break
        if allot[h] < host_cap[h]:
            allot[h] += 1
            remaining -= 1
    if remaining != 0 or any(allot[h] < 1 for h in hosts):
        raise AssertionError("Invalid quota assignment")
    return allot


def select_windows(conn, host: str, n_available: int, n_select: int,
                   feature_names: list[str], split: str, run_label: str):
    """Stream frozen CIC bucket features; evenly spaced benign-only windows."""
    if n_select == 0:
        return [], []
    if not 0 < n_select <= n_available:
        raise ValueError(f"Invalid selection quota for {host}")
    # Linspace-based selection spreads samples across benign activity rather
    # than keeping only the first (highly overlapping) sequences.
    target_indices = np.linspace(0, n_available-1, n_select).astype(np.int64).tolist()
    if len(set(target_indices)) != n_select:
        raise AssertionError("Duplicate sampled window indices")
    names = ", ".join('"' + x + '"' for x in feature_names)
    query = (f"SELECT bucket_start_ms, label, {names} FROM {CIC_BUCKET_TABLE} "
             "WHERE ipv4_src_addr=? ORDER BY bucket_start_ms")
    buffer = deque(maxlen=SEQ_LEN)
    selected_X = []
    meta = []
    cur_index = 0
    target_ptr = 0
    last_t = None
    seg_id = 0
    for row in conn.execute(query, (host,)):
        t, label = int(row[0]), int(row[1])
        if last_t is None or t - last_t != BIN_MS:
            seg_id += 1
            buffer.clear()
        last_t = t
        if label == 1:
            buffer.clear()
            continue
        x = np.array(row[2:], dtype=np.float32)
        if x.shape != (len(feature_names),) or not np.isfinite(x).all():
            raise ValueError(f"Invalid feature vector for {host} at {t}")
        buffer.append((t, x))
        if len(buffer) < SEQ_LEN:
            continue
        if target_ptr < len(target_indices) and cur_index == target_indices[target_ptr]:
            v = np.stack([it[1] for it in buffer]).astype(np.float32)
            selected_X.append(v)
            meta.append({"sequence_id": f"CIC_{split}_{host}_{t}",
                         "src_ip": f"CIC::{host}", "segment_id": seg_id,
                         "start_bucket_ms": buffer[0][0], "end_bucket_ms": t,
                         "label": 0, "num_buckets": SEQ_LEN, "split": split,
                         "round": run_label, "dataset_id": "CIC2018"})
            target_ptr += 1
        cur_index += 1
    if cur_index != n_available or len(selected_X) != n_select:
        raise AssertionError(f"CIC {host}: counted {n_available} benign windows, "
                             f"generated {cur_index}, selected {len(selected_X)}/{n_select}")
    return selected_X, meta


def load_unsw_train(args, schema):
    trainpath = args.unsw_processed / "round_1" / "train.npz"
    metapath = args.unsw_processed / "round_1" / "train_metadata.csv"
    with np.load(trainpath, allow_pickle=False) as data:
        if set(data.files) != {"X", "y", "feature_names"}:
            raise ValueError("Unexpected original UNSW NPZ keys")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int8)
        names = [str(x) for x in data["feature_names"].tolist()]
    if names != schema["feature_names"] or X.shape[1:] != (10, 88):
        raise ValueError("UNSW original train NPZ and frozen schema mismatch")
    meta = pd.read_csv(metapath, dtype={"src_ip": str, "sequence_id": str})
    if len(X) != len(meta) or not np.array_equal(y, meta.label.to_numpy()):
        raise ValueError("UNSW original train NPZ / metadata inconsistent")
    b = y == 0
    X = X[b]
    meta = meta.loc[b].reset_index(drop=True).copy()
    train_ips, val_ips = choose_unsw_val_hosts(meta)
    if set(train_ips) & set(val_ips):
        raise AssertionError("UNSW train/val overlap")
    split = np.where(meta.src_ip.isin(val_ips), "validation", "train")
    meta["src_ip"] = "UNSW::" + meta["src_ip"].astype(str)
    meta["sequence_id"] = "UNSW_" + meta["sequence_id"].astype(str)
    meta["round"] = "merged"
    meta["split"] = split
    meta["dataset_id"] = "UNSW"
    return X, meta, {"train": train_ips, "validation": val_ips}


def assemble(args) -> None:
    out = args.output.resolve()
    if out.resolve() == args.unsw_processed.resolve():
        raise ValueError("Refusing to overwrite existing UNSW processed data")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output folder already nonempty: {out}; choose a new --output")
    schema = load_json(args.unsw_processed / "categorical_schema.json")
    if len(schema["feature_names"]) != 88:
        raise ValueError("Expected official UNSW 88-feature schema")
    with open_db(args.cic_db) as conn:
        ensure_schema_compatible(conn, args.unsw_db)
        counts = inspect_cic_hosts(conn, schema["feature_names"])
        groups = split_cic_hosts(counts, args.seed, args.cic_val_host_frac,
                                 args.cic_test_host_frac)
        log("[split] CIC productive hosts: " + ", ".join(
            f"{k}={len(v)}" for k, v in groups.items()))
        quotas_by_split = {}
        budgets = {"train": args.cic_train_budget,
                   "validation": args.cic_val_budget,
                   "holdout": args.cic_test_budget}
        for split, hosts in groups.items():
            eligible_counts = {h: counts[h]["benign_windows"] for h in hosts}
            quotas_by_split[split] = quotas(eligible_counts, budgets[split],
                                            args.max_cic_per_host, args.seed)
            log(f"[quota] CIC {split}: {sum(quotas_by_split[split].values()):,} "
                f"selected windows from {len(hosts)} hosts")
        unswX, unsw_meta, unsw_hostsets = load_unsw_train(args, schema)
        arrays = {}
        metadata = {}
        for split in ("train", "validation", "holdout"):
            rows = []
            features = []
            for j, h in enumerate(groups[split], 1):
                q = quotas_by_split[split][h]
                x, m = select_windows(conn, h, counts[h]["benign_windows"], q,
                                      schema["feature_names"], split, "merged")
                features.extend(x)
                rows.extend(m)
                if j <= 3 or j % 50 == 0 or j == len(groups[split]):
                    log(f"[assemble] CIC {split} host {j}/{len(groups[split])}: {h} / {q} windows")
            arr = np.stack(features) if features else np.empty((0, 10, 88), np.float32)
            mdf = pd.DataFrame(rows, columns=META_COLS)
            if split in ("train", "validation"):
                mask = unsw_meta.split.eq(split).to_numpy()
                arr = np.concatenate((unswX[mask], arr), axis=0)
                mdf = pd.concat([unsw_meta.loc[mask, META_COLS], mdf], ignore_index=True)
            if len(arr) != len(mdf) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid merged {split} data")
            arrays[split], metadata[split] = arr, mdf
        train_hosts = set(metadata["train"].src_ip.unique())
        val_hosts = set(metadata["validation"].src_ip.unique())
        hold_hosts = set(metadata["holdout"].src_ip.unique())
        if train_hosts & val_hosts or train_hosts & hold_hosts or val_hosts & hold_hosts:
            raise AssertionError("Cross-dataset host leakage between splits")
        orig_manifest = load_json(args.unsw_processed / "split_manifest.json")
        original_test_ips = {"UNSW::" + h for h in orig_manifest["benign_test_ips"]}
        for r in ("round_1", "round_2"):
            original_test_ips.add("UNSW::" + orig_manifest[r]["test_malicious_ip"])
        if (train_hosts | val_hosts | hold_hosts) & original_test_ips:
            raise AssertionError("Original UNSW test host leaked into merged data")
        out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "holdout"):
            arr, mdf = arrays[split], metadata[split]
            np.savez_compressed(out / f"{split}.npz", X=arr,
                                y=np.zeros(len(arr), dtype=np.int8),
                                feature_names=np.asarray(schema["feature_names"], dtype=str))
            mdf.to_csv(out / f"{split}_metadata.csv", index=False)
            log(f"[write] {split}: {len(arr):,} benign windows, {mdf.src_ip.nunique()} unique hosts")
        dump_json(out / "categorical_schema.json", schema)
        manifest = {
            "experiment": "UNSW + CIC2018 v3 benign-only host-balanced temporal extension",
            "seed": args.seed, "sequence_length": SEQ_LEN, "bucket_size_ms": BIN_MS,
            "stride": STRIDE, "label": "0=benign, 1=malicious; only benign sequences saved",
            "original_unsw_test_unchanged": True,
            "original_unsw_test_paths": {
                "round_1": str((args.unsw_processed / "round_1" / "test.npz").resolve()),
                "round_2": str((args.unsw_processed / "round_2" / "test.npz").resolve()),
            },
            "unsw_split_hosts": unsw_hostsets,
            "cic_split_hosts": groups,
            "cic_host_coverage": counts,
            "cic_selected_per_host": quotas_by_split,
            "selected_sequence_count_by_split": {
                k: {"all": int(len(arrays[k])),
                    "UNSW": int(metadata[k].dataset_id.eq("UNSW").sum()),
                    "CIC2018": int(metadata[k].dataset_id.eq("CIC2018").sum())}
                for k in arrays},
            "max_cic_per_host": args.max_cic_per_host,
            "cic_train_budget": args.cic_train_budget,
            "cic_val_budget": args.cic_val_budget,
            "cic_test_budget": args.cic_test_budget,
            "sample_policy": "host-balanced quota, then deterministic evenly spaced benign-only windows",
            "host_key": "dataset_id::src_ip",
            "feature_names": schema["feature_names"],
            "excludes_unlabeled_silence": True,
            "independent_test_warning": "CIC holdout here has benign-only sequences; not a binary test set",
        }
        dump_json(out / "merged_manifest.json", manifest)
        log("[done] merged data available at: " + str(out))
        log("[note] existing original CNN runner cannot load these merged split files unchanged; "
            "it assumes exactly the original UNSW split and <=20 productive hosts")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("stage", choices=["import", "buckets", "assemble", "all"])
    ap.add_argument("--cic-csv", type=Path,
                    default=Path("datasets/NF-CICIDS2018-v3.csv"))
    ap.add_argument("--cic-db", type=Path, default=Path("datasets/cic_2018_traffic.db"))
    ap.add_argument("--unsw-db", type=Path, default=Path("datasets/network_traffic.db"))
    ap.add_argument("--unsw-processed", type=Path, default=Path("processed_binary_temporal"))
    ap.add_argument("--output", type=Path, default=Path("processed_binary_temporal_unsw_cic"))
    ap.add_argument("--bucket-script", type=Path,
                    default=Path("temporal_eda/01_build_temporal_buckets.py"))
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cic-val-host-frac", type=float, default=0.2)
    ap.add_argument("--cic-test-host-frac", type=float, default=0.2)
    ap.add_argument("--cic-train-budget", type=int, default=120_000)
    ap.add_argument("--cic-val-budget", type=int, default=25_000)
    ap.add_argument("--cic-test-budget", type=int, default=25_000)
    ap.add_argument("--max-cic-per-host", type=int, default=2_000)
    ap.add_argument("--limit-hosts", type=int, default=None,
                    help="TEST ONLY: process first N CIC hosts; will not complete buckets")
    args = ap.parse_args()
    if args.chunk_size <= 0 or args.max_cic_per_host <= 0 or any(
        x < 1 for x in (args.cic_train_budget, args.cic_val_budget, args.cic_test_budget)):
        ap.error("Invalid nonpositive chunk/budget/per-host cap")
    if args.cic_val_host_frac <= 0 or args.cic_test_host_frac <= 0 or (
        args.cic_val_host_frac + args.cic_test_host_frac >= 1):
        ap.error("Host fractions must be positive and sum to < 1")
    return args


if __name__ == "__main__":
    a = parse_args()
    if a.stage in ("import", "all"):
        stage_import(a)
    if a.stage in ("buckets", "all"):
        stage_buckets(a)
    if a.stage in ("assemble", "all"):
        assemble(a)
