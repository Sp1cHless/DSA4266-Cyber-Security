#!/usr/bin/env python3
"""Build CIC buckets: global 15min split, temporal purge, train-only superposition.

Run from the repository root: python CIC_new/build_buckets.py
See CIC_new/README.md for outputs, stage-level resume, and dataset loading.
Only NumPy is required in Python. Excel export uses the adjacent JS companion
and the configured Codex artifact-tool runtime; its availability is preflighted.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, OrderedDict, defaultdict
from contextlib import closing
from datetime import datetime, timezone, timedelta
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

import bucket_features as bf
import split_policy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = ("train", "validation", "test")
STAGES = ("index", "split", "plan", "aggregate", "normalize", "report")
LOG = logging.getLogger("cic_buckets")


def write_json(path, value):
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(db):
    s = db.stat()
    return {"path": str(db), "bytes": s.st_size, "mtime_ns": str(s.st_mtime_ns)}


def code_signature():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE/"bucket_features.py", HERE/"split_policy.py")}


def connect(out, raw=None):
    conn = sqlite3.connect(out/"manifest.sqlite", uri=True)
    conn.execute("PRAGMA cache_size=-131072")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("ATTACH DATABASE ? AS work", (str(out/"work.sqlite"),))
    if raw:
        conn.execute("ATTACH DATABASE ? AS raw", (raw.as_uri()+"?mode=ro",))
    last = [time.monotonic()]
    def progress():
        now = time.monotonic()
        if now-last[0] >= 30:
            LOG.info("SQLite operation still running...")
            last[0] = now
        return 0
    conn.set_progress_handler(progress, 2_000_000)
    return conn


def index_source(out, db):
    """Create the group manifest and a clustered, numeric read-only event cache."""
    conn = connect(out, db)
    try:
        required_numeric = [n for n in bf.EVENT_COLUMNS if n not in {"raw_rowid", "destination_id"}]
        columns = {r[1] for r in conn.execute("PRAGMA raw.table_info(network_flows)")}
        required = set(required_numeric) | {"ipv4_src_addr", "ipv4_dst_addr", "attack"}
        if missing := required-columns:
            raise ValueError(f"Missing CIC columns: {sorted(missing)}")
        invalid = " OR ".join(f'(typeof("{n}") != \'integer\' OR "{n}" < 0)' for n in required_numeric)
        controls = conn.execute(f'''SELECT COUNT(*),SUM(label),
            SUM(({invalid}) OR label NOT IN(0,1) OR flow_end_milliseconds < flow_start_milliseconds
                OR ipv4_src_addr IS NULL OR ipv4_dst_addr IS NULL OR attack IS NULL)
            FROM raw.network_flows NOT INDEXED''').fetchone()
        if not controls[0] or controls[2]:
            raise ValueError(f"Input is empty or contains {controls[2]} invalid required rows")
        LOG.info("Input controls: %s flows, %s attack flows", f"{controls[0]:,}", f"{controls[1]:,}")
        conn.executescript('''
            CREATE TABLE original_sequences(
                sequence_id INTEGER PRIMARY KEY, source_ip TEXT NOT NULL, window_start_ms INTEGER NOT NULL,
                flows INTEGER NOT NULL, attack_flows INTEGER NOT NULL, active_buckets INTEGER NOT NULL,
                first_ms INTEGER NOT NULL,last_ms INTEGER NOT NULL,max_end_ms INTEGER NOT NULL,
                primary_attack TEXT NOT NULL DEFAULT 'Benign',split TEXT,assigned_split TEXT,
                exclusion_reason INTEGER NOT NULL DEFAULT 0,UNIQUE(source_ip,window_start_ms));
            INSERT INTO original_sequences(source_ip,window_start_ms,flows,attack_flows,active_buckets,first_ms,last_ms,max_end_ms)
            SELECT ipv4_src_addr,(flow_start_milliseconds/100000)*100000,COUNT(*),SUM(label),
                COUNT(DISTINCT flow_start_milliseconds/10000),MIN(flow_start_milliseconds),MAX(flow_start_milliseconds),MAX(flow_end_milliseconds)
            FROM raw.network_flows NOT INDEXED GROUP BY 1,2 ORDER BY 1,2;
            CREATE TABLE sequence_attacks(sequence_id INTEGER NOT NULL,attack TEXT NOT NULL,flows INTEGER NOT NULL,
                PRIMARY KEY(sequence_id,attack)) WITHOUT ROWID;
            INSERT INTO sequence_attacks
            SELECT s.sequence_id,r.attack,COUNT(*) FROM raw.network_flows AS r NOT INDEXED
            JOIN original_sequences s ON s.source_ip=r.ipv4_src_addr
                AND s.window_start_ms=(r.flow_start_milliseconds/100000)*100000
            WHERE r.label=1 GROUP BY s.sequence_id,r.attack;
            UPDATE original_sequences SET primary_attack=(SELECT a.attack FROM sequence_attacks a
                WHERE a.sequence_id=original_sequences.sequence_id ORDER BY a.flows DESC,a.attack LIMIT 1)
                WHERE attack_flows>0;
            CREATE TABLE destinations(destination_id INTEGER PRIMARY KEY,address TEXT UNIQUE NOT NULL);
            INSERT INTO destinations(address) SELECT DISTINCT ipv4_dst_addr FROM raw.network_flows ORDER BY 1;
        ''')
        # Integer IDs substitute addresses only for equality/diversity, never as features.
        selects = ["r.rowid AS raw_rowid" if n == "raw_rowid" else "d.destination_id AS destination_id" if n == "destination_id" else f'r."{n}"' for n in bf.EVENT_COLUMNS]
        LOG.info("Building sorted event cache; the original database remains read-only")
        conn.execute(f'''CREATE TABLE work.events AS SELECT s.sequence_id,{','.join(selects)}
            FROM raw.network_flows AS r NOT INDEXED JOIN original_sequences s
              ON s.source_ip=r.ipv4_src_addr AND s.window_start_ms=(r.flow_start_milliseconds/100000)*100000
            JOIN destinations d ON d.address=r.ipv4_dst_addr
            ORDER BY s.sequence_id,r.flow_start_milliseconds,r.rowid''')
        conn.execute("CREATE INDEX work.events_sequence ON events(sequence_id)")
        conn.execute("CREATE UNIQUE INDEX work.events_rowid ON events(raw_rowid)")
        totals = conn.execute("SELECT SUM(flows),SUM(attack_flows),COUNT(*) FROM original_sequences").fetchone()
        cached = conn.execute("SELECT COUNT(*),SUM(label) FROM work.events").fetchone()
        if totals[:2] != controls[:2] or cached != controls[:2]:
            raise AssertionError("Raw/group/cache flow reconciliation failed")
        conn.commit()
        write_json(out/"source_controls.json", {"flows": controls[0], "attack_flows": controls[1], "original_sequences": totals[2], "invalid_required_rows": controls[2]})
        LOG.info("Indexed %s complete original sequences", f"{totals[2]:,}")
    finally:
        conn.close()


def split_originals(out, seed):
    conn = connect(out)
    try:
        n = conn.execute("SELECT COUNT(*) FROM original_sequences").fetchone()[0]
        data = {k: np.empty(n, np.int64) for k in ("ids", "start", "first", "end", "source", "attacks")}
        sources = {}
        for i, (sid, start, first, end, source, attacks) in enumerate(conn.execute(
                "SELECT sequence_id,window_start_ms,first_ms,max_end_ms,source_ip,attack_flows FROM original_sequences ORDER BY sequence_id")):
            source_id = sources.setdefault(source, len(sources))
            for key, value in zip(data, (sid, start, first, end, source_id, attacks)):
                data[key][i] = value
        cells = {}
        for sid, attack, count in conn.execute("SELECT sequence_id,attack,flows FROM sequence_attacks ORDER BY sequence_id,attack"):
            index = int(np.searchsorted(data["ids"], sid))
            if index == n or data["ids"][index] != sid:
                raise AssertionError("Attack membership has no original sequence")
            cells[(index, attack)] = count
        kept, original, reasons, blocks, assigned, reserved, notes = sp.split(
            data["start"], data["first"], data["end"], data["source"], data["attacks"], cells, seed)
        conn.executemany("UPDATE original_sequences SET split=?,assigned_split=?,exclusion_reason=? WHERE sequence_id=?",
                         ((sp.NAMES[int(kept[i])], sp.NAMES[int(original[i])], int(reasons[i]), int(sid)) for i,sid in enumerate(data["ids"])))
        conn.execute("DROP TABLE IF EXISTS split_groups")
        conn.execute("CREATE TABLE split_groups(block_start_ms INTEGER PRIMARY KEY,assigned_split TEXT NOT NULL,coverage_reserved INTEGER NOT NULL)")
        conn.executemany("INSERT INTO split_groups VALUES(?,?,?)",
                         ((int(block), sp.NAMES[int(assigned[i])], int(reserved[i] >= 0)) for i,block in enumerate(blocks)))
        for witness in notes["coverage_reservations"]:
            witness["sequence_id"] = int(data["ids"][witness.pop("sequence_index")])
        notes["excluded_flows"], notes["excluded_attack_flows"] = conn.execute(
            "SELECT COALESCE(SUM(flows),0),COALESCE(SUM(attack_flows),0) FROM original_sequences WHERE split='excluded'").fetchone()
        notes["original_composition"] = []
        for name in sp.NAMES:
            counts = conn.execute('''SELECT COUNT(*),COALESCE(SUM(attack_flows>0),0),COALESCE(SUM(flows),0),
                COALESCE(SUM(attack_flows),0),COUNT(DISTINCT source_ip) FROM original_sequences WHERE split=?''',(name,)).fetchone()
            notes["original_composition"].append(dict(zip(("split","sequences","positive_sequences","flows","attack_flows","sources"),(name,*counts))))
        if conn.execute("SELECT COUNT(*) FROM original_sequences WHERE split IS NULL").fetchone()[0]:
            raise AssertionError("Unassigned original sequence")
        conn.execute("CREATE INDEX IF NOT EXISTS original_split ON original_sequences(split)")
        conn.commit()
        write_json(out/"split_notes.json", notes)
        LOG.info("15min global split after purge: %s; excluded %s sequences", notes["retained_sequences"], f"{notes['excluded_sequences']:,}")
        LOG.info("Attack types retained in train/validation/test: %s", [sum(r[s+"_sequences"] > 0 for r in notes["coverage"]) for s in SPLITS])
    finally:
        conn.close()


class Pool:
    """Uniform nonempty stratum, then source, then sequence selection."""
    def __init__(self, groups):
        self.groups = groups
        self.sources = {k: list(v) for k, v in groups.items() if v}
        self.indices = {k: {s:i for i,s in enumerate(v)} for k,v in self.sources.items()}

    def pick(self, rng, exclude=None, group=None):
        eligible = [g for g, sources in self.sources.items()
                    if (group is None or g == group) and (len(sources)>1 or sources[0] != exclude)]
        if not eligible:
            raise ValueError("No donor satisfies the different-source constraint")
        g = eligible[int(rng.integers(len(eligible)))]
        sources = self.sources[g]
        excluded_index = self.indices[g].get(exclude)
        n = len(sources)-(excluded_index is not None)
        i = int(rng.integers(n))
        if excluded_index is not None and i >= excluded_index:
            i += 1
        source = sources[i]
        ids = self.groups[g][source]
        return int(ids[int(rng.integers(len(ids)))]), source, g


def plan_samples(out, seed):
    conn = connect(out)
    try:
        # Recreating the plan is safe on a stage restart; all IDs are deterministic.
        conn.execute("DROP TABLE IF EXISTS samples")
        conn.execute('''CREATE TABLE samples(sample_id INTEGER PRIMARY KEY,split TEXT NOT NULL,sample_type TEXT NOT NULL,
            left_id INTEGER NOT NULL,right_id INTEGER,right_shift_ms INTEGER NOT NULL DEFAULT 0,target_attack TEXT,
            flows INTEGER NOT NULL,attack_flows INTEGER NOT NULL,active_buckets INTEGER,mixed_buckets INTEGER,
            shard TEXT,shard_row INTEGER)''')
        conn.execute('''INSERT INTO samples(sample_id,split,sample_type,left_id,flows,attack_flows)
            SELECT sequence_id,split,CASE WHEN attack_flows=0 THEN 'real_benign' ELSE 'real_malicious' END,
                sequence_id,flows,attack_flows FROM original_sequences WHERE split IN ('train','validation','test')''')
        vocabulary = [r[0] for r in conn.execute('''SELECT DISTINCT e.l7_proto FROM work.events e
            JOIN original_sequences s USING(sequence_id) WHERE s.split='train' ORDER BY 1''')]
        schema = bf.make_schema(vocabulary)
        write_json(out/"feature_schema.json", schema)
        bg = defaultdict(lambda: defaultdict(lambda: array("q")))
        positive = defaultdict(lambda: defaultdict(lambda: array("q")))
        for sid, source, active in conn.execute("SELECT sequence_id,source_ip,active_buckets FROM original_sequences WHERE split='train' AND attack_flows=0 ORDER BY sequence_id"):
            band = "1" if active == 1 else "2-4" if active <= 4 else "5-10"
            bg[band][source].append(sid)
        for attack, source, sid in conn.execute('''SELECT a.attack,s.source_ip,s.sequence_id FROM sequence_attacks a
            JOIN original_sequences s USING(sequence_id) WHERE s.split='train' ORDER BY a.attack,s.source_ip,s.sequence_id'''):
            positive[attack][source].append(sid)
        n, p = conn.execute("SELECT COUNT(*),SUM(attack_flows>0) FROM original_sequences WHERE split='train'").fetchone()
        if not p or len({s for group in bg.values() for s in group}) < 2:
            raise ValueError("Need attacks and at least two distinct benign training sources")
        backgrounds, attacks = Pool(bg), Pool(positive)
        rng = np.random.default_rng(seed+1)
        last_id = conn.execute("SELECT MAX(sequence_id) FROM original_sequences").fetchone()[0]
        fetch = lambda sid: conn.execute("SELECT window_start_ms,first_ms,last_ms,flows,attack_flows FROM original_sequences WHERE sequence_id=?", (sid,)).fetchone()
        plans = []
        for sample_type in ("synthetic_malicious", "synthetic_benign"):
            for _ in range(p):
                if sample_type == "synthetic_malicious":
                    right, source, target = attacks.pick(rng)
                    left, _, _ = backgrounds.pick(rng, exclude=source)
                else:
                    left, source, _ = backgrounds.pick(rng)
                    right, _, _ = backgrounds.pick(rng, exclude=source)
                    target = None
                l, r = fetch(left), fetch(right)
                lo, hi = -(r[1]-r[0]), bf.WINDOW_MS-1-(r[2]-r[0])
                shift = int(rng.integers(lo,hi+1))
                last_id += 1
                plans.append((last_id,"train",sample_type,left,right,shift,target,l[3]+r[3],l[4]+r[4]))
        conn.executemany("INSERT INTO samples(sample_id,split,sample_type,left_id,right_id,right_shift_ms,target_attack,flows,attack_flows) VALUES(?,?,?,?,?,?,?,?,?)", plans)
        conn.execute("CREATE INDEX samples_split ON samples(split)")
        conn.commit()
        if conn.execute("SELECT COUNT(*) FROM samples WHERE split='train'").fetchone()[0] != n+2*p:
            raise AssertionError("N+2P sample plan failed")
        write_json(out/"augmentation_plan.json", {"original_train_N": n, "original_train_P": p, "synthetic_positive": p,
                   "synthetic_negative": p, "final_train": n+2*p, "seed": seed+1,
                   "attack_types": sorted(positive), "background_density_bands": sorted(bg),
                   "feature_count": schema["feature_count"], "model_width_with_mask": schema["model_width_with_mask"]})
        LOG.info("Plan: original train N=%s, P=%s; final train=%s; features=%s + mask", f"{n:,}", f"{p:,}", f"{n+2*p:,}", schema["feature_count"])
    finally:
        conn.close()


def remove_shards(directory):
    """Only remove this builder's incomplete-stage shard names, never a tree."""
    directory = directory.resolve()
    if not directory.is_dir():
        return
    for file in directory.iterdir():
        if re.fullmatch(r"part_[0-9]{6}\.npz(?:\.partial)?", file.name):
            if file.is_symlink() or file.resolve().parent != directory:
                raise ValueError("Unsafe shard cleanup target")
            file.unlink()


class ShardWriter:
    def __init__(self, out, split, size, conn):
        self.directory = out/"raw_shards"/split
        self.directory.mkdir(parents=True, exist_ok=True)
        remove_shards(self.directory)
        self.size, self.conn, self.split = size, conn, split
        self.buffer, self.part = [], 0

    def append(self, sid, result):
        self.buffer.append((sid, result))
        if len(self.buffer) >= self.size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        name = f"part_{self.part:06d}.npz"
        path = self.directory/name
        ids = np.asarray([r[0] for r in self.buffer], dtype=np.int64)
        data = [r[1] for r in self.buffer]
        y = np.stack([r[2] for r in data])
        with path.with_name(name+".partial").open("wb") as handle:
            np.savez_compressed(handle, X_raw=np.stack([r[0] for r in data]), mask=np.stack([r[1] for r in data]),
                                y_bucket=y, y_sequence=y.max(axis=1), sample_id=ids,
                                flow_counts=np.stack([r[3] for r in data]), attack_flow_counts=np.stack([r[4] for r in data]))
        path.with_name(name+".partial").replace(path)
        updates = []
        for i, (sid, result) in enumerate(self.buffer):
            _, mask, _, flows, attacks = result
            updates.append((int(mask.sum()),int(((attacks>0)&(attacks<flows)).sum()), f"splits/{self.split}/{name}",i,sid))
        self.conn.executemany("UPDATE samples SET active_buckets=?,mixed_buckets=?,shard=?,shard_row=? WHERE sample_id=?", updates)
        self.conn.commit()
        self.part += 1
        self.buffer.clear()


class DonorCache:
    def __init__(self, conn, megabytes):
        self.conn = conn
        self.limit = megabytes*1024*1024
        self.items, self.bytes = OrderedDict(), 0

    def load(self, sid):
        if sid in self.items:
            self.items.move_to_end(sid)
            return self.items[sid]
        start = self.conn.execute("SELECT window_start_ms FROM original_sequences WHERE sequence_id=?", (sid,)).fetchone()[0]
        rows = self.conn.execute("SELECT "+",".join(bf.EVENT_COLUMNS)+" FROM work.events WHERE sequence_id=? ORDER BY flow_start_milliseconds,raw_rowid", (sid,)).fetchall()
        values = bf.local_events(np.asarray(rows, dtype=np.int64), start)
        values.setflags(write=False)
        if values.nbytes <= self.limit:
            while self.items and self.bytes+values.nbytes > self.limit:
                _, evicted = self.items.popitem(last=False)
                self.bytes -= evicted.nbytes
            self.items[sid] = values
            self.bytes += values.nbytes
        return values


def aggregate_all(out, shard_size, cache_mb):
    conn = connect(out)
    try:
        schema = read_json(out/"feature_schema.json")
        conn.execute("UPDATE samples SET active_buckets=NULL,mixed_buckets=NULL,shard=NULL,shard_row=NULL")
        conn.commit()
        writers = {split: ShardWriter(out,split,shard_size,conn) for split in SPLITS}
        moments = bf.Moments(schema["feature_count"])
        metadata = iter(conn.execute("SELECT sequence_id,window_start_ms,split,flows,attack_flows FROM original_sequences ORDER BY sequence_id"))
        total = read_json(out/"source_controls.json")["original_sequences"]
        query = "SELECT sequence_id,"+",".join(bf.EVENT_COLUMNS)+" FROM work.events ORDER BY sequence_id,flow_start_milliseconds,raw_rowid"
        done, flow_total = 0, 0
        for sid, group in itertools.groupby(conn.execute(query), key=lambda r:r[0]):
            meta = next(metadata)
            if meta[0] != sid:
                raise AssertionError("Sorted event/group manifest mismatch")
            if meta[2] == "excluded":
                count = sum(1 for _ in group)
                if count != meta[3]:
                    raise AssertionError("Excluded flow coverage mismatch")
                done += 1
                flow_total += count
                continue
            events = bf.local_events(np.asarray([row[1:] for row in group], dtype=np.int64), meta[1])
            result = bf.aggregate(events, schema)
            if int(result[3].sum()) != meta[3] or int(result[4].sum()) != meta[4]:
                raise AssertionError("Original flow/attack conservation failed")
            writers[meta[2]].append(sid, result)
            if meta[2] == "train":
                moments.add(bf.transform_log(result[0][result[1] != 0], schema))
            done += 1
            flow_total += len(events)
            if done % 5000 == 0:
                LOG.info("Real sequences: %s/%s; flows aggregated: %s", f"{done:,}", f"{total:,}", f"{flow_total:,}")
        if done != total or flow_total != read_json(out/"source_controls.json")["flows"]:
            raise AssertionError("Original sequence/flow coverage mismatch")
        scaler = moments.finish()
        write_json(out/"scaler.json", {**scaler,"population":"original train active buckets", "log1p_columns":schema["log1p_columns"]})
        LOG.info("Scaler fitted on %s original training active buckets", f"{moments.n:,}")
        cache = DonorCache(conn,cache_mb)
        synthetic = conn.execute("SELECT sample_id,left_id,right_id,right_shift_ms,flows,attack_flows FROM samples WHERE right_id IS NOT NULL ORDER BY sample_id").fetchall()
        for i, (sid,left,right,shift,expected_flows,expected_attacks) in enumerate(synthetic,1):
            events = bf.superimpose(cache.load(left),cache.load(right),shift)
            result = bf.aggregate(events,schema)
            if int(result[3].sum()) != expected_flows or int(result[4].sum()) != expected_attacks:
                raise AssertionError("Synthetic flow/attack conservation failed")
            writers["train"].append(sid,result)
            if i % 1000 == 0:
                LOG.info("Synthetic sequences: %s/%s", f"{i:,}",f"{len(synthetic):,}")
        for writer in writers.values():
            writer.flush()
        if conn.execute("SELECT COUNT(*) FROM samples WHERE shard IS NULL").fetchone()[0]:
            raise AssertionError("Unwritten sample")
    finally:
        conn.close()


def normalize_all(out):
    schema, scaler = read_json(out/"feature_schema.json"),read_json(out/"scaler.json")
    counts = Counter()
    with closing(sqlite3.connect(out/"manifest.sqlite")) as conn:
        expected = dict(conn.execute("SELECT split,COUNT(*) FROM samples GROUP BY split"))
        for split in SPLITS:
            destination = out/"splits"/split
            destination.mkdir(parents=True,exist_ok=True)
            remove_shards(destination)
            for number, file in enumerate(sorted((out/"raw_shards"/split).glob("part_*.npz")),1):
                with np.load(file,allow_pickle=False) as source:
                    values = {k:source[k] for k in source.files if k != "X_raw"}
                    raw = source["X_raw"]
                    values["X"] = bf.normalize(raw,values["mask"],schema,scaler)
                    if not np.array_equal(values["y_sequence"],values["y_bucket"].max(axis=1)):
                        raise AssertionError("Sequence/bucket label mismatch")
                    if not np.array_equal(values["y_bucket"],(values["attack_flow_counts"]>0).astype(np.uint8)):
                        raise AssertionError("Bucket/flow label mismatch")
                    if np.any(values["X"][values["mask"]==0] != 0):
                        raise AssertionError("Empty normalized bucket is nonzero")
                    total, positives = len(values["sample_id"]),int(values["y_sequence"].sum())
                    ids = values["sample_id"]
                    for i,sid in enumerate(ids):
                        row = conn.execute("SELECT split,flows,attack_flows FROM samples WHERE sample_id=?",(int(sid),)).fetchone()
                        if row != (split,int(values["flow_counts"][i].sum()),int(values["attack_flow_counts"][i].sum())):
                            raise AssertionError("Exported shard/manifest mismatch")
                    final = destination/file.name
                    temporary = final.with_name(final.name+".partial")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle,**values)
                    temporary.replace(final)
                    counts[split] += total
                    counts[split+"_positives"] += positives
                if number % 20 == 0:
                    LOG.info("Normalized %s: %s shards, %s sequences", split,number,f"{counts[split]:,}")
            if counts[split] != expected.get(split,0):
                raise AssertionError(f"Normalized count mismatch in {split}")
        expected_labels = dict(conn.execute("SELECT split,SUM(attack_flows>0) FROM samples GROUP BY split"))
        if any(counts[s+"_positives"] != expected_labels.get(s,0) for s in SPLITS):
            raise AssertionError("Normalized positive-label totals mismatch")
    write_json(out/"export_checks.json",{"counts":dict(counts),"finite":True,"empty_buckets_zero":True,
               "labels_reconciled":True,"manifest_reconciled":True,"allow_pickle":False})


def describe_feature(name):
    if name.startswith("l7_code_") or name == "l7_UNK_ratio":
        return "L7 ratios", "Category flow count / all bucket flow starts; UNK contains training-unseen codes."
    if "flag_" in name:
        return "TCP flags", "TCP flows containing the named bit / TCP flow count; zero when no TCP flow."
    if name.endswith("_entropy"):
        return "Diversity", "Normalized Shannon entropy: -sum(p*ln(p))/ln(k); zero for k<=1."
    if name.endswith("_reuse_ratio"):
        return "Diversity", "1 - distinct category count / total bucket flow count."
    if name.endswith("_top1_ratio"):
        return "Diversity", "Most frequent category count / all applicable flow counts; DNS/ICMP use their eligible population."
    if name.endswith("_nunique"):
        return "Diversity", "Distinct original values in the bucket; DNS/ICMP only use eligible flows."
    if name.startswith("num_pkts_"):
        return "Packet-size bins", "Sum of this exported packet-size bin / sum of all five exported bins; zero denominator gives zero."
    if name.startswith("flow_start_"):
        return "Start timing", "Mean/max adjacent sorted flow-start gap or last-first span in milliseconds; <2 starts gives zero."
    if name.startswith("tcp_win_"):
        return "TCP window", "Mean/max of the exported per-flow window field over TCP flows only; bytes."
    if name.startswith("duration"):
        return "Duration", "Named mean/min/max of the exported per-flow duration in milliseconds; stored zeros retained."
    if "ip_pkt_len" in name:
        return "Packet size", "Named statistic of the exported per-flow packet-length extremum, in bytes; not a pooled packet mean."
    if name.startswith("retransmitted"):
        return "Retransmissions", "Sum of the named exported per-flow retransmission counter."
    if name.endswith("_sum") or name == "flow_count":
        return "Volume", "Whole-flow counter sum or flow-start count assigned to this bucket; not interval-measured traffic."
    return "Protocol presence", "Protocol or legacy DNS-presence flow fraction over all bucket starts."


def make_summary(out, reports, config):
    schema, scaler = read_json(out/"feature_schema.json"),read_json(out/"scaler.json")
    plan, controls = read_json(out/"augmentation_plan.json"),read_json(out/"source_controls.json")
    checks_file = read_json(out/"export_checks.json")
    conn = sqlite3.connect(out/"manifest.sqlite")
    try:
        composition=[]
        for split,kind,n,positive,flows,attack_flows,active,mixed in conn.execute('''SELECT split,sample_type,COUNT(*),SUM(attack_flows>0),
            SUM(flows),SUM(attack_flows),SUM(active_buckets),SUM(mixed_buckets) FROM samples GROUP BY split,sample_type ORDER BY split,sample_type'''):
            composition.append({"split":split,"sample_type":kind,"sequences":n,"positive_sequences":positive,"bucket_positions":n*10,
                                "flow_presentations":flows,"attack_flow_presentations":attack_flows,"active_buckets":active,
                                "mean_occupancy":active/(n*10),"mixed_buckets":mixed})
        attacks=[]
        names = [r[0] for r in conn.execute("SELECT DISTINCT attack FROM sequence_attacks ORDER BY attack")]
        for name in names:
            for split in SPLITS:
                n,hosts,flows=conn.execute('''SELECT COUNT(*),COUNT(DISTINCT s.source_ip),COALESCE(SUM(a.flows),0)
                    FROM sequence_attacks a JOIN original_sequences s USING(sequence_id) WHERE a.attack=? AND s.split=?''',(name,split)).fetchone()
                selected = conn.execute("SELECT COUNT(*) FROM samples WHERE sample_type='synthetic_malicious' AND target_attack=?",(name,)).fetchone()[0] if split=="train" else 0
                containing = conn.execute('''SELECT COUNT(*) FROM samples p JOIN sequence_attacks a ON a.sequence_id=p.right_id
                    WHERE p.sample_type='synthetic_malicious' AND a.attack=?''',(name,)).fetchone()[0] if split=="train" else 0
                attacks.append({"attack_type":name,"split":split,"real_sequences":n,"real_sources":hosts,"real_attack_flows":flows,
                                "synthetic_selected":selected,"synthetic_containing":containing})
        features=[]
        log_columns=set(schema["log1p_columns"])
        for i,name in enumerate(schema["feature_names"]):
            family,definition=describe_feature(name)
            features.append({"index":i,"feature":name,"family":family,"definition":definition,"log1p":i in log_columns,
                             "train_mean":scaler["mean"][i],"train_scale":scaler["scale"][i],"train_constant":scaler["constant"][i]})
        actual_train=conn.execute("SELECT COUNT(*) FROM samples WHERE split='train'").fetchone()[0]
        actual_positive=conn.execute("SELECT COUNT(*) FROM samples WHERE split='train' AND attack_flows>0").fetchone()[0]
        donor_violations=conn.execute('''SELECT COUNT(*) FROM samples p JOIN original_sequences l ON l.sequence_id=p.left_id
            JOIN original_sequences r ON r.sequence_id=p.right_id WHERE p.right_id IS NOT NULL
            AND (p.split!='train' OR l.split!='train' OR r.split!='train' OR l.source_ip=r.source_ip OR l.sequence_id=r.sequence_id)''').fetchone()[0]
        position_violations=conn.execute('''SELECT COUNT(*) FROM samples p JOIN original_sequences r ON r.sequence_id=p.right_id
            WHERE r.first_ms-r.window_start_ms+p.right_shift_ms<0 OR r.last_ms-r.window_start_ms+p.right_shift_ms>=100000''').fetchone()[0]
        raw_total=conn.execute("SELECT SUM(flows),SUM(attack_flows) FROM original_sequences").fetchone()
        split_notes=read_json(out/"split_notes.json")
        real_output=conn.execute("SELECT COALESCE(SUM(flows),0),COALESCE(SUM(attack_flows),0) FROM samples WHERE right_id IS NULL").fetchone()
        excluded_samples=conn.execute("SELECT COUNT(*) FROM samples p JOIN original_sequences s ON s.sequence_id=p.left_id WHERE s.split='excluded'").fetchone()[0]
        block_violations=conn.execute('''SELECT COUNT(*) FROM original_sequences s LEFT JOIN split_groups g
            ON g.block_start_ms=(s.window_start_ms/900000)*900000
            WHERE g.block_start_ms IS NULL OR s.assigned_split!=g.assigned_split OR (s.split!='excluded' AND s.split!=g.assigned_split)''').fetchone()[0]
        checks=[{"check":"Final train N+2P","expected":plan["final_train"],"observed":actual_train},
                {"check":"Final train positive 2P","expected":2*plan["original_train_P"],"observed":actual_positive},
                {"check":"Original unique flow coverage","expected":controls["flows"],"observed":raw_total[0]},
                {"check":"Original attack flow coverage","expected":controls["attack_flows"],"observed":raw_total[1]},
                {"check":"Retained real plus excluded flow coverage","expected":controls["flows"],"observed":real_output[0]+split_notes["excluded_flows"]},
                {"check":"Retained real plus excluded attack coverage","expected":controls["attack_flows"],"observed":real_output[1]+split_notes["excluded_attack_flows"]},
                {"check":"Excluded originals in samples","expected":0,"observed":excluded_samples},
                {"check":"Inconsistent global 15min block assignments","expected":0,"observed":block_violations},
                {"check":"Shared global 100s times","expected":0,"observed":split_notes["shared_global_100s"]},
                {"check":"Retained interval overlap sequences","expected":0,"observed":split_notes["retained_interval_overlap_sequences"]},
                {"check":"Cross-split adjacent source pairs","expected":0,"observed":split_notes["adjacent_source_pairs_cross_split"]},
                {"check":"Attack types missing from train","expected":0,"observed":sum(not r["train_sequences"] for r in split_notes["coverage"])},
                {"check":"Invalid/cross-split/same-source synthetic donors","expected":0,"observed":donor_violations},
                {"check":"Shifted starts outside 100s","expected":0,"observed":position_violations}]
        for split in SPLITS:
            expected=sum(r["sequences"] for r in composition if r["split"]==split)
            checks.append({"check":f"{split} normalized sequence count","expected":expected,"observed":checks_file["counts"].get(split,0)})
        for key in ("finite","empty_buckets_zero","labels_reconciled","manifest_reconciled"):
            checks.append({"check":key,"expected":True,"observed":checks_file[key]})
        checks.append({"check":"Raw database fingerprint unchanged","expected":config["source"],"observed":signature(Path(config["source"]["path"]))})
        for check in checks:
            check["passed"]=check["expected"]==check["observed"]
        if not all(r["passed"] for r in checks):
            raise AssertionError("Final report reconciliation failed")
        return {"metadata":{"generated_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),"config":config,
                            "output_directory":str(out),"report_directory":str(reports),"python":sys.version.split()[0],"numpy":np.__version__},
                "controls":controls,"plan":plan,"schema":schema,"scaler_active_buckets":scaler["n"],
                "split_notes":split_notes,"composition":composition,"attacks":attacks,"features":features,"checks":checks}
    finally:
        conn.close()


def excel_runtime(node=None, modules=None):
    base = Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies"
    executable = Path(node) if node else base/"node/bin"/("node.exe" if os.name == "nt" else "node")
    packages = Path(modules) if modules else base/"node/node_modules"
    if not executable.is_file() or not (packages/"@oai/artifact-tool").is_dir():
        raise FileNotFoundError("Excel runtime missing. Configure --node and --node-modules as explained in CIC_new/README.md. No long scan was started.")
    return executable.resolve(),packages.resolve()


def render_report(out, reports, node, modules):
    work=out/"_excel_runtime"
    work.mkdir(exist_ok=True)
    link=work/"node_modules"
    if not link.exists():
        if os.name == "nt":
            subprocess.run(["cmd.exe","/c","mklink","/J",str(link),str(modules)],check=True,stdout=subprocess.DEVNULL)
        else:
            link.symlink_to(modules,target_is_directory=True)
    elif link.resolve() != modules:
        raise ValueError("Existing Excel runtime link points elsewhere")
    builder=work/"render_bucket_summary.mjs"
    shutil.copyfile(HERE/builder.name,builder)
    subprocess.run([str(node),str(builder),str(reports/"summary.json"),str(work/"previews")],check=True)


def report_stage(out,reports,config,node,modules):
    reports.mkdir(parents=True,exist_ok=True)
    summary=make_summary(out,reports,config)
    write_json(reports/"summary.json",summary)
    for filename in ("feature_schema.json","scaler.json","split_notes.json"):
        shutil.copyfile(out/filename,reports/filename)
    metrics=(HERE/"summary_metrics.md").read_text(encoding="utf-8")
    introduction=(f"# Dataset build results\n\nGenerated: {summary['metadata']['generated_at']}\n\n"
                  f"Original training sequences N: **{summary['plan']['original_train_N']:,}**. "
                  f"Original attack-bearing training sequences P: **{summary['plan']['original_train_P']:,}**. "
                  f"Final training sequences N+2P: **{summary['plan']['final_train']:,}**.\n\n"
                  f"Each sequence has ten buckets and {summary['schema']['feature_count']} features per bucket; activity mask is separate. "
                  f"All {len(summary['checks'])} final reconciliation checks passed.\n\n"
                  f"Split: global 15min blocks, 100s guard on both sides, actual flow-end exclusions. "
                  f"Excluded original sequences: **{summary['split_notes']['excluded_sequences']:,}**. "
                  f"Attack types in train/validation/test: **{' / '.join(str(sum(r[s+'_sequences']>0 for r in summary['split_notes']['coverage'])) for s in SPLITS)}**.\n\n")
    (reports/"metrics_explained.md").write_text(introduction+metrics,encoding="utf-8")
    render_report(out,reports,node,modules)


def configure_logging(out):
    LOG.setLevel(logging.INFO)
    for handler in LOG.handlers:
        handler.close()
    LOG.handlers.clear()
    formatter=logging.Formatter("%(asctime)s %(levelname)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    # Explicit client timezone, independent of the machine's timezone.
    formatter.converter=lambda seconds: datetime.fromtimestamp(seconds,timezone(timedelta(hours=8))).timetuple()
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(out/"run.log",encoding="utf-8")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def restart_index_files(out):
    for name in ("manifest.sqlite","work.sqlite"):
        for suffix in ("","-journal","-wal","-shm"):
            path=out/(name+suffix)
            if path.exists():
                if path.is_symlink() or path.resolve().parent != out.resolve():
                    raise ValueError("Unsafe owned-cache restart path")
                path.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db",type=Path,default=ROOT/"datasets/cic_2018_traffic.db")
    parser.add_argument("--output",type=Path,help="Local dataset/cache directory; default CIC_new/outputs/buckets_v2_<time>")
    parser.add_argument("--reports-dir",type=Path,help="Git-visible small report directory; default CIC_new/reports/<run name>")
    parser.add_argument("--seed",type=int,default=4266)
    parser.add_argument("--shard-sequences",type=int,default=2048)
    parser.add_argument("--donor-cache-mb",type=int,default=128)
    parser.add_argument("--resume",action="store_true",help="Resume completed stages in --output; restart the first incomplete stage")
    parser.add_argument("--report-only",type=Path,help="Rebuild final reports from an already normalized output directory")
    parser.add_argument("--node",type=Path)
    parser.add_argument("--node-modules",type=Path)
    args=parser.parse_args(argv)
    if args.seed<0 or args.shard_sequences<1 or args.donor_cache_mb<0:
        parser.error("seed/cache must be nonnegative and shard-sequences positive")
    node,modules=excel_runtime(args.node,args.node_modules)
    if args.report_only:
        out=args.report_only.resolve()
        state=read_json(out/"run_status.json")
        if state["config"].get("pipeline_version") != 2:
            parser.error("This report requires pipeline v2 time-block output; old random-split runs must use their original code or a fresh build")
        if "normalize" not in state["completed_stages"]:
            parser.error("report-only requires a completed normalize stage")
        reports=(args.reports_dir or Path(state["config"]["reports_directory"])).resolve()
        configure_logging(out)
        try:
            report_stage(out,reports,state["config"],node,modules)
            LOG.info("Reports rebuilt: %s",reports)
        finally:
            for handler in LOG.handlers:
                handler.close()
            LOG.handlers.clear()
        return 0
    db=args.db.resolve()
    if not db.is_file():
        parser.error(f"Missing source database: {db}")
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out=(args.output or HERE/"outputs"/f"buckets_v2_{stamp}").resolve()
    if args.resume and not args.output:
        parser.error("--resume requires the original --output directory")
    out.mkdir(parents=True,exist_ok=True)
    status_path=out/"run_status.json"
    if args.resume:
        state=read_json(status_path)
        previous=state["config"]
        reports=(args.reports_dir or Path(previous["reports_directory"])).resolve()
    else:
        if any(out.iterdir()):
            parser.error("Output must be new/empty. Use --resume with the original directory to continue.")
        reports=(args.reports_dir or HERE/"reports"/out.name).resolve()
        if reports.exists() and any(reports.iterdir()):
            parser.error("Report directory must be new/empty")
    config={"pipeline_version":2,"source":signature(db),"code_sha256":code_signature(),"seed":args.seed,
            "split_ratios":[.6,.2,.2],"window_seconds":100,"bucket_seconds":10,
            "split_policy":sp.POLICY,
            "shard_sequences":args.shard_sequences,"donor_cache_mb":args.donor_cache_mb,
            "reports_directory":str(reports)}
    if args.resume:
        if previous != config:
            parser.error("Resume config/source/code differs. Use exactly the original parameters and files, or start a new output directory.")
        if state.get("complete"):
            print(f"Already complete: {out}\nReports: {reports}")
            return 0
    else:
        state={"config":config,"complete":False,"completed_stages":[],"stage_seconds":{}}
        write_json(status_path,state)
    lock=out/"run.lock"
    try:
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        parser.error(f"Run lock exists: {lock}. Stop the other process first; after a killed process, remove only this lock file before resuming.")
    os.write(descriptor,str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    configure_logging(out)
    temp=out/"sqlite_temp"
    temp.mkdir(exist_ok=True)
    old_temp=os.environ.get("SQLITE_TMPDIR")
    os.environ["SQLITE_TMPDIR"]=str(temp)
    try:
        LOG.info("Dataset output: %s",out)
        LOG.info("Git-visible report output: %s",reports)
        LOG.info("Available output-disk space: %.1f GiB",shutil.disk_usage(out).free/1024**3)
        jobs={"index":lambda:index_source(out,db),"split":lambda:split_originals(out,args.seed),
              "plan":lambda:plan_samples(out,args.seed),"aggregate":lambda:aggregate_all(out,args.shard_sequences,args.donor_cache_mb),
              "normalize":lambda:normalize_all(out),"report":lambda:report_stage(out,reports,config,node,modules)}
        for stage in STAGES:
            if stage in state["completed_stages"]:
                LOG.info("Reusing completed stage: %s",stage)
                continue
            state.update(current_stage=stage,error=None)
            write_json(status_path,state)
            LOG.info("Starting stage: %s",stage)
            if stage == "index":
                restart_index_files(out)
            started=time.monotonic()
            jobs[stage]()
            if signature(db) != config["source"]:
                raise RuntimeError("Source database changed during the run")
            state["stage_seconds"][stage]=round(time.monotonic()-started,3)
            state["completed_stages"].append(stage)
            write_json(status_path,state)
            LOG.info("Completed %s in %.1fs",stage,state["stage_seconds"][stage])
        state.update(complete=True,current_stage=None,error=None)
        write_json(status_path,state)
        LOG.info("DONE. Model-ready shards: %s",out/"splits")
        LOG.info("Workbook and English explanation: %s",reports)
        return 0
    except BaseException as exc:
        state.update(complete=False,error=f"{type(exc).__name__}: {exc}")
        write_json(status_path,state)
        LOG.exception("Build interrupted. Resume the same output and parameters; completed stages are retained.")
        raise
    finally:
        lock.unlink(missing_ok=True)
        if old_temp is None:
            os.environ.pop("SQLITE_TMPDIR",None)
        else:
            os.environ["SQLITE_TMPDIR"]=old_temp
        for handler in LOG.handlers:
            handler.close()
        LOG.handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())
