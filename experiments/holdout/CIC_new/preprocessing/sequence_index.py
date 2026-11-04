"""Index original source-IP sequences and assign global time-block splits.

No augmentation or neural-model training runs here.
"""
import json
import logging
import sqlite3
import time

import numpy as np

from . import bucket_statistics as bf
from . import split_policy as sp

SPLITS = ("train", "validation", "test")
LOG = logging.getLogger(__name__)


def write_json(path, value):
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(db):
    s = db.stat()
    return {"path": str(db), "bytes": s.st_size, "mtime_ns": str(s.st_mtime_ns)}


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


def split_originals(out, seed, holdout=()):
    conn = connect(out)
    try:
        n = conn.execute("SELECT COUNT(*) FROM original_sequences").fetchone()[0]
        data = {k: np.empty(n, np.int64) for k in ("ids", "start", "first", "end", "source", "attacks")}
        primary = np.empty(n, dtype="U96")
        sources = {}
        for i, (sid, start, first, end, source, attacks, attack) in enumerate(conn.execute(
                "SELECT sequence_id,window_start_ms,first_ms,max_end_ms,source_ip,attack_flows,primary_attack FROM original_sequences ORDER BY sequence_id")):
            source_id = sources.setdefault(source, len(sources))
            for key, value in zip(data, (sid, start, first, end, source_id, attacks)):
                data[key][i] = value
            primary[i] = attack
        cells = {}
        for sid, attack, count in conn.execute("SELECT sequence_id,attack,flows FROM sequence_attacks ORDER BY sequence_id,attack"):
            index = int(np.searchsorted(data["ids"], sid))
            if index == n or data["ids"][index] != sid:
                raise AssertionError("Attack membership has no original sequence")
            cells[(index, attack)] = count
        kept, original, reasons, blocks, assigned, reserved, notes = sp.split(
            data["start"], data["first"], data["end"], data["source"], data["attacks"], cells, seed,
            primary=primary, holdout_types=holdout)
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
        if holdout:
            residual = []
            for name in holdout:
                retained = conn.execute("SELECT COUNT(*) FROM original_sequences WHERE split='train' AND primary_attack=?",(name,)).fetchone()[0]
                if retained:
                    raise AssertionError(f"Holdout attack label retained as a training primary label: {name}")
                sequences, flows = conn.execute('''SELECT COUNT(*),COALESCE(SUM(a.flows),0) FROM sequence_attacks a
                    JOIN original_sequences s ON s.sequence_id=a.sequence_id
                    WHERE a.attack=? AND s.split='train' AND s.primary_attack<>?''',(name,name)).fetchone()
                residual.append({"attack_type": name, "train_sequences_not_primary": sequences,
                                 "train_flows_not_primary": flows,
                                 "scope": "flows of this label inside retained training sequences whose primary label is another attack or Benign"})
            notes["train_holdout_residual"] = residual
        if conn.execute("SELECT COUNT(*) FROM original_sequences WHERE split IS NULL").fetchone()[0]:
            raise AssertionError("Unassigned original sequence")
        conn.execute("CREATE INDEX IF NOT EXISTS original_split ON original_sequences(split)")
        conn.commit()
        write_json(out/"split_notes.json", notes)
        LOG.info("15min global split after purge: %s; excluded %s sequences", notes["retained_sequences"], f"{notes['excluded_sequences']:,}")
        LOG.info("Attack types retained in train/validation/test: %s", [sum(r[s+"_sequences"] > 0 for r in notes["coverage"]) for s in SPLITS])
    finally:
        conn.close()
