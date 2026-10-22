"""Bounded shards and sparse category storage; arrays are never globally densified."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import logging
from pathlib import Path
import sqlite3

import numpy as np

from CIC_new.preprocessing.sequence_index import read_json, write_json
from .statistics import DENSE, HISTOGRAMS, LEARNED


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(prepared):
    return digest(Path(prepared) / "manifest.json")


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def shard_files(prepared):
    return sorted((Path(prepared) / "raw_originals").glob("part_*.npz"))


class RawWriter:
    def __init__(self, folder, connection, size):
        self.folder = Path(folder)
        self.folder.mkdir()
        self.connection, self.size = connection, size
        self.buffer, self.part = [], 0
        connection.execute("CREATE TABLE bucket_locations(sample_id INTEGER PRIMARY KEY,shard TEXT,row_index INTEGER)")

    def append(self, sid, state):
        self.buffer.append((sid, state))
        if len(self.buffer) >= self.size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        payload = {key: np.stack([state[key] for _, state in self.buffer]) for key in DENSE}
        payload["sample_id"] = np.array([sid for sid, _ in self.buffer], np.int64)
        for key in ["hist_" + f for f in HISTOGRAMS] + ["starts"]:
            parts = []
            for i, (_, state) in enumerate(self.buffer):
                part = state[key].copy()
                part[:, 0] += 10 * i
                parts.append(part)
            payload[key] = np.concatenate(parts)
        name = f"part_{self.part:06d}.npz"
        np.savez_compressed(self.folder / name, **payload)
        self.connection.executemany("INSERT INTO bucket_locations VALUES(?,?,?)",
                                    [(int(sid), name, i) for i, sid in enumerate(payload["sample_id"])])
        self.connection.commit()
        self.buffer.clear()
        self.part += 1


def state_at(shard, index):
    result = {key: shard[key][index].copy() for key in DENSE}
    for key in ["hist_" + f for f in HISTOGRAMS] + ["starts"]:
        coo = shard[key]
        lo = np.searchsorted(coo[:, 0], index * 10)
        hi = np.searchsorted(coo[:, 0], (index + 1) * 10)
        part = coo[lo:hi].copy()
        part[:, 0] -= index * 10
        result[key] = part
    return result


class RawStore:
    def __init__(self, prepared, cache_size=3):
        self.prepared = Path(prepared)
        self.connection = readonly(self.prepared / "manifest.sqlite")
        self.cache, self.cache_size = OrderedDict(), cache_size
        self.donors = None

    def prime(self, ids, path):
        """Read each large raw shard once, cache only selected donor states on disk."""
        self.donors = sqlite3.connect(path)
        self.donors.execute("CREATE TABLE donors(sample_id INTEGER PRIMARY KEY,payload BLOB NOT NULL)")
        needed = set(map(int, ids))
        for number, file in enumerate(shard_files(self.prepared), 1):
            shard = arrays(file)
            for row, sid in enumerate(shard["sample_id"]):
                if int(sid) in needed:
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **state_at(shard, row))
                    self.donors.execute("INSERT INTO donors VALUES(?,?)", (int(sid), buffer.getvalue()))
                    needed.remove(int(sid))
            self.donors.commit()
            if number % 50 == 0:
                logging.getLogger(__name__).info("Donor cache: %s raw shards, %s donors remaining", number, len(needed))
            if not needed:
                break
        if needed:
            raise AssertionError("Donor IDs missing from bucket storage")

    def get(self, sid):
        if self.donors is not None:
            row = self.donors.execute("SELECT payload FROM donors WHERE sample_id=?", (int(sid),)).fetchone()
            if row is not None:
                with np.load(io.BytesIO(row[0]), allow_pickle=False) as archive:
                    return {key: archive[key] for key in archive.files}
        row = self.connection.execute("SELECT shard,row_index FROM bucket_locations WHERE sample_id=?", (int(sid),)).fetchone()
        if row is None:
            raise ValueError(f"Unknown original ID {sid}")
        name, index = row
        if name not in self.cache:
            self.cache[name] = arrays(self.prepared / "raw_originals" / name)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return state_at(self.cache[name], index)

    def events(self, sid):
        from CIC_new.preprocessing import bucket_statistics as bf
        window = self.connection.execute("SELECT window_start_ms FROM original_sequences WHERE sequence_id=?", (int(sid),)).fetchone()[0]
        connection = readonly(self.prepared / "work.sqlite")
        try:
            rows = connection.execute("SELECT " + ",".join(bf.EVENT_COLUMNS) +
                                      " FROM events WHERE sequence_id=? ORDER BY flow_start_milliseconds,raw_rowid", (int(sid),)).fetchall()
            return bf.local_events(np.asarray(rows, np.int64), window)
        finally:
            connection.close()

    def close(self):
        self.cache.clear()
        if self.donors is not None:
            self.donors.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def original_catalog(prepared):
    """Dense ID-indexed metadata; zero is a sentinel, original IDs begin at one."""
    connection = readonly(Path(prepared) / "manifest.sqlite")
    try:
        maximum = connection.execute("SELECT MAX(sequence_id) FROM original_sequences").fetchone()[0]
        result = {"split": np.full(maximum + 1, 3, np.uint8), "label": np.zeros(maximum + 1, np.uint8),
                  "flows": np.zeros(maximum + 1, np.int64), "attacks": np.zeros(maximum + 1, np.int64),
                  "active": np.zeros(maximum + 1, np.uint8), "ip": np.empty(maximum + 1, dtype="U15"),
                  "attack": np.empty(maximum + 1, dtype="U64"), "start": np.zeros(maximum + 1, np.int64),
                  "first": np.zeros(maximum + 1, np.int64), "end": np.zeros(maximum + 1, np.int64)}
        names = ("train", "validation", "test", "excluded")
        for sid, split, attack_flows, flows, active, ip, attack, start, first, end in connection.execute(
                "SELECT sequence_id,split,attack_flows,flows,active_buckets,source_ip,primary_attack,window_start_ms,first_ms,max_end_ms FROM original_sequences"):
            for key, value in (("split", names.index(split)), ("label", attack_flows > 0), ("flows", flows),
                               ("attacks", attack_flows), ("active", active), ("ip", ip), ("attack", attack),
                               ("start", start), ("first", first), ("end", end)):
                result[key][sid] = value
        return result
    finally:
        connection.close()


def load_tables(prepared, folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if not manifest.get("complete") or not manifest.get("frozen"):
        raise ValueError("Completed frozen category tables required")
    if manifest["prepared_identity"] != identity(prepared):
        raise ValueError("Category tables belong to a different preparation")
    if manifest["embedding_sha256"] != digest(folder / "embeddings.npz"):
        raise ValueError("Category weights checksum mismatch")
    schema = read_json(Path(prepared) / "schema.json")
    tables = arrays(folder / "embeddings.npz")
    for field in LEARNED:
        expected = (len(schema["vocabularies"][field]) + 1, manifest["dimensions"][field])
        if tables[field].shape != expected or not np.isfinite(tables[field]).all() or np.any(tables[field][0] != 0):
            raise ValueError(f"Invalid embedding table: {field}")
    return tables, manifest
