"""Frozen training-only donor plans for Scheme2 vector superimposition."""
from collections import defaultdict
from pathlib import Path
import sqlite3

import numpy as np

from .dataset import Pool


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


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


def make_plan(prepared, catalog, seed):
    bg = defaultdict(lambda: defaultdict(list))
    attacks = defaultdict(lambda: defaultdict(list))
    train = np.flatnonzero(catalog["split"] == 0)
    for sid in train[catalog["label"][train] == 0]:
        active = catalog["active"][sid]
        band = 0 if active == 1 else 1 if active <= 4 else 2
        bg[band][str(catalog["ip"][sid])].append(int(sid))
    connection = readonly(Path(prepared) / "manifest.sqlite")
    try:
        for sid, attack in connection.execute("SELECT sequence_id,attack FROM sequence_attacks ORDER BY attack,sequence_id"):
            if catalog["split"][sid] == 0:
                attacks[attack][str(catalog["ip"][sid])].append(sid)
    finally:
        connection.close()
    positive_count = int(catalog["label"][train].sum())
    if not positive_count or len({host for group in bg.values() for host in group}) < 2:
        raise ValueError("Need real attacks and at least two pure-benign training sources")
    backgrounds, positives = Pool(bg), Pool(attacks)
    rng = np.random.default_rng(seed)
    lefts, rights, targets, bands, kinds = [], [], [], [], []
    for malicious in (True, False):
        for _ in range(positive_count):
            if malicious:
                right, host, target = positives.pick(rng)
                left, _, band = backgrounds.pick(rng, exclude=host)
            else:
                left, host, band = backgrounds.pick(rng)
                right, _, _ = backgrounds.pick(rng, exclude=host)
                target = ""
            lefts.append(left)
            rights.append(right)
            targets.append(target)
            bands.append(band)
            kinds.append("synthetic_malicious" if malicious else "synthetic_benign")
    maximum = len(catalog["split"]) - 1
    return dict(sample_id=np.arange(maximum + 1, maximum + 1 + 2 * positive_count, dtype=np.int64),
                left_id=np.array(lefts, np.int64), right_id=np.array(rights, np.int64),
                right_shift_buckets=np.zeros(2 * positive_count, np.int8), target_attack=np.array(targets, dtype="U64"),
                background_band=np.array(bands, np.uint8), sample_type=np.array(kinds, dtype="U20"))
