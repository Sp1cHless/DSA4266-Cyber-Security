#!/usr/bin/env python3
r"""Read-only Data Coverage Audit for DSA4266's official 10-second pipeline.

Run from the repository root (PowerShell/CMD):
    python .\analysis\audit_data_coverage.py

Or specify paths:
    python .\analysis\audit_data_coverage.py --db .\datasets\network_traffic.db --data-dir .\processed_binary_temporal --outdir .\experiments\data_coverage_audit

No ML training, CUDA, NPZ loads, preprocessing imports, or data mutation.
Only Python's standard library is used.

Definitions:
- An ACTIVE bucket is a stored (source IP, 10-second interval) row.
- A SEGMENT is a run of buckets whose timestamps differ by exactly 10 seconds.
- A VALID segment has >=10 active buckets (official sequence length).
- COVERED flows are raw flows aggregated into at least one bucket contained in a
  valid segment. This counts each raw flow ONCE, not once per overlapping window.
  It does not imply that original per-flow attributes survived aggregation.
- Sequence count follows the official stride=1 rule: length - 10 + 1 per
  valid segment. Overlapping sequences are not independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', type=Path, default=ROOT / 'datasets' / 'network_traffic.db')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'processed_binary_temporal')
    p.add_argument('--outdir', type=Path, default=ROOT / 'experiments' / 'data_coverage_audit')
    p.add_argument('--validation-hosts', type=Path,
                   default=ROOT / 'experiments' / 'cnn_ae_temporal_control_seed0' / 'validation_hosts.json',
                   help='Optional previous AE validation_hosts.json; missing file is allowed')
    return p.parse_args()


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, obj):
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def safe_pct(num, denom):
    return round(100.0 * num / denom, 4) if denom else None


def identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def check_columns(db, table, expected):
    existing = {r[1] for r in db.execute(f'PRAGMA table_info({identifier(table)})')}
    if not existing:
        raise ValueError(f'SQLite table does not exist: {table}')
    missing = sorted(set(expected) - existing)
    if missing:
        raise ValueError(f'{table} missing columns: {missing}')


def length_bin(n):
    if n <= 9:
        return str(n)
    for low, high in ((10, 19), (20, 49), (50, 99), (100, 499), (500, 999)):
        if low <= n <= high:
            return f'{low}-{high}'
    return '1000+'


@dataclass
class Host:
    ip: str
    raw_flows: int = 0
    raw_benign_flows: int = 0
    raw_attack_flows: int = 0
    raw_invalid_labels: int = 0
    raw_missing_start_times: int = 0
    raw_start_ms: int | None = None
    raw_end_ms: int | None = None
    active_buckets: int = 0
    bucket_flows: int = 0
    benign_buckets: int = 0
    attack_buckets: int = 0
    invalid_bucket_labels: int = 0
    segments: int = 0
    short_segments: int = 0
    valid_segments: int = 0
    short_buckets: int = 0
    valid_buckets: int = 0
    short_segment_flows: int = 0
    covered_flows: int = 0
    sequences: int = 0
    max_segment_buckets: int = 0
    gap_breaks: int = 0
    missing_intervals_between_active_buckets: int = 0
    max_gap_seconds: float = 0.0
    timestamp_anomalies: int = 0
    misaligned_buckets: int = 0
    last_bucket_ms: int | None = None
    current_len: int = 0
    current_flows: int = 0
    hist: Counter = field(default_factory=Counter)
    # Hist keys are length bins; each value is a triple (segments, buckets, flows).

    def finish_segment(self, sequence_length):
        n = self.current_len
        if not n:
            return
        flows = self.current_flows
        self.segments += 1
        self.max_segment_buckets = max(self.max_segment_buckets, n)
        key = length_bin(n)
        a, b, c = self.hist[key] if key in self.hist else (0, 0, 0)
        self.hist[key] = (a + 1, b + n, c + flows)
        if n >= sequence_length:
            self.valid_segments += 1
            self.valid_buckets += n
            self.covered_flows += flows
            self.sequences += n - sequence_length + 1
        else:
            self.short_segments += 1
            self.short_buckets += n
            self.short_segment_flows += flows
        self.current_len = 0
        self.current_flows = 0

    def add_bucket(self, ts, flow_count, label, bucket_ms, sequence_length):
        if ts is None:
            raise ValueError(f'NULL bucket time for host {self.ip}')
        ts = int(ts)
        flow_count = int(flow_count)
        if flow_count < 1:
            raise ValueError(f'Non-positive flow_count for host {self.ip} at {ts}: {flow_count}')
        if ts % bucket_ms:
            self.misaligned_buckets += 1
        if self.last_bucket_ms is not None:
            delta = ts - self.last_bucket_ms
            if delta != bucket_ms:
                self.finish_segment(sequence_length)
                self.gap_breaks += 1
                if delta > bucket_ms:
                    self.missing_intervals_between_active_buckets += delta // bucket_ms - 1
                    self.max_gap_seconds = max(self.max_gap_seconds, delta / 1000)
                else:
                    self.timestamp_anomalies += 1
        self.last_bucket_ms = ts
        self.active_buckets += 1
        self.bucket_flows += flow_count
        if label == 0:
            self.benign_buckets += 1
        elif label == 1:
            self.attack_buckets += 1
        else:
            self.invalid_bucket_labels += 1
        self.current_len += 1
        self.current_flows += flow_count

    @property
    def class_name(self):
        if self.raw_invalid_labels:
            return 'invalid_raw_label'
        if self.raw_benign_flows and self.raw_attack_flows:
            return 'mixed'
        if self.raw_benign_flows:
            return 'benign'
        if self.raw_attack_flows:
            return 'malicious'
        return 'unknown'


def load_raw_host_counts(db, hosts):
    print('[1/4] Scanning raw network_flows by source IP ...', flush=True)
    sql = '''SELECT ipv4_src_addr, COUNT(*) AS n,
        SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS benign,
        SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS malicious,
        SUM(CASE WHEN label IS NULL OR label NOT IN (0,1) THEN 1 ELSE 0 END) AS bad,
        SUM(CASE WHEN flow_start_milliseconds IS NULL THEN 1 ELSE 0 END) AS null_time,
        MIN(flow_start_milliseconds), MAX(flow_start_milliseconds)
        FROM network_flows GROUP BY ipv4_src_addr'''
    for ip, n, ben, mal, bad, null_time, start, end in db.execute(sql):
        ip = str(ip) if ip is not None else '<NULL_IP>'
        h = hosts.setdefault(ip, Host(ip))
        h.raw_flows = int(n)
        h.raw_benign_flows = int(ben or 0)
        h.raw_attack_flows = int(mal or 0)
        h.raw_invalid_labels = int(bad or 0)
        h.raw_missing_start_times = int(null_time or 0)
        h.raw_start_ms = int(start) if start is not None else None
        h.raw_end_ms = int(end) if end is not None else None


def load_buckets(db, table, hosts, bucket_ms, seq_len):
    print(f'[2/4] Scanning {table} in source/time order ...', flush=True)
    sql = (f'SELECT ipv4_src_addr, bucket_start_ms, flow_count, label '
           f'FROM {identifier(table)} ORDER BY ipv4_src_addr, bucket_start_ms')
    previous_ip = None
    for ip, ts, fc, label in db.execute(sql):
        ip = str(ip) if ip is not None else '<NULL_IP>'
        if previous_ip is not None and ip != previous_ip:
            hosts[previous_ip].finish_segment(seq_len)
        h = hosts.setdefault(ip, Host(ip))
        h.add_bucket(ts, fc, label, bucket_ms, seq_len)
        previous_ip = ip
    if previous_ip is not None:
        hosts[previous_ip].finish_segment(seq_len)


def audit_metadata(data_dir, manifest, hosts):
    print('[3/4] Comparing generated sequence counts with metadata CSV ...', flush=True)
    checks = []
    details = []
    for round_name in ('round_1', 'round_2'):
        for split in ('train', 'test'):
            path = data_dir / round_name / f'{split}_metadata.csv'
            if not path.is_file():
                checks.append({'name': f'{round_name}/{split} metadata', 'status': 'not_checked',
                               'details': f'Not found: {path}; no NPZ load required'})
                continue
            found = Counter()
            labels = Counter()
            with path.open('r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if not {'src_ip', 'label'}.issubset(set(reader.fieldnames or [])):
                    raise ValueError(f'Metadata missing src_ip/label: {path}')
                for row in reader:
                    ip = row['src_ip']
                    found[ip] += 1
                    labels[row['label']] += 1
            expected_ips = set(manifest[round_name][f'{split}_ips'])
            all_ips = expected_ips | set(found)
            mismatches = [(ip, int(found[ip]), hosts.get(ip, Host(ip)).sequences)
                          for ip in sorted(all_ips)
                          if int(found[ip]) != hosts.get(ip, Host(ip)).sequences]
            expected_counts = manifest[round_name][f'{split}_sequence_counts']
            label_mismatch = (int(labels.get('0', 0)) != int(expected_counts['benign']) or
                              int(labels.get('1', 0)) != int(expected_counts['malicious']))
            extra_ips = sorted(set(found) - expected_ips)
            status = 'PASS' if not mismatches and not label_mismatch and not extra_ips else 'FAIL'
            checks.append({'name': f'{round_name}/{split} metadata', 'status': status,
                           'details': f'rows={sum(found.values())}; bad_host_counts={len(mismatches)}; '
                                      f'label_mismatch={label_mismatch}; unexpected_hosts={len(extra_ips)}'})
            details.append({'round': round_name, 'split': split, 'metadata_rows': sum(found.values()),
                            'per_host_sequence_count_mismatches': mismatches[:30],
                            'label_counts': dict(labels), 'unexpected_hosts': extra_ips})
    return checks, details


def host_role(ip, cls, manifest, val_hosts):
    benign_test = set(manifest.get('benign_test_ips', []))
    benign_train = set(manifest.get('benign_train_ips', []))
    r1 = manifest.get('round_1', {}).get('test_malicious_ip')
    r2 = manifest.get('round_2', {}).get('test_malicious_ip')
    if cls == 'benign':
        if ip in benign_test:
            return 'benign_test'
        if ip in benign_train:
            if val_hosts is None:
                return 'benign_train_pool'
            return 'benign_validation' if ip in val_hosts else 'benign_true_train'
        return 'benign_unassigned'
    if cls == 'malicious':
        if ip == r1:
            return 'malicious_round_1_test'
        if ip == r2:
            return 'malicious_round_2_test'
        return 'malicious_other'
    return cls


CSV_FIELDS = [
    'src_ip', 'class', 'split_role', 'raw_flows', 'raw_benign_flows', 'raw_attack_flows',
    'raw_invalid_labels', 'raw_missing_start_times', 'active_buckets', 'bucket_flows',
    'raw_to_bucket_flow_pct', 'segments', 'short_segments_lt_10', 'valid_segments_ge_10',
    'max_segment_buckets', 'short_buckets', 'covered_buckets', 'covered_buckets_pct',
    'flows_in_short_segments', 'covered_flows', 'covered_raw_flows_pct',
    'valid_sequences', 'gap_breaks', 'missing_10s_intervals_between_active_buckets',
    'max_gap_seconds', 'avg_window_reuse_of_covered_bucket', 'raw_start_ms', 'raw_end_ms',
    'benign_buckets', 'malicious_buckets', 'invalid_bucket_labels', 'timestamp_anomalies',
    'misaligned_buckets',
]


def to_row(h, role, seq_len):
    return {
        'src_ip': h.ip, 'class': h.class_name, 'split_role': role,
        'raw_flows': h.raw_flows, 'raw_benign_flows': h.raw_benign_flows,
        'raw_attack_flows': h.raw_attack_flows, 'raw_invalid_labels': h.raw_invalid_labels,
        'raw_missing_start_times': h.raw_missing_start_times,
        'active_buckets': h.active_buckets, 'bucket_flows': h.bucket_flows,
        'raw_to_bucket_flow_pct': safe_pct(h.bucket_flows, h.raw_flows),
        'segments': h.segments, 'short_segments_lt_10': h.short_segments,
        'valid_segments_ge_10': h.valid_segments,
        'max_segment_buckets': h.max_segment_buckets, 'short_buckets': h.short_buckets,
        'covered_buckets': h.valid_buckets,
        'covered_buckets_pct': safe_pct(h.valid_buckets, h.active_buckets),
        'flows_in_short_segments': h.short_segment_flows,
        'covered_flows': h.covered_flows,
        'covered_raw_flows_pct': safe_pct(h.covered_flows, h.raw_flows),
        'valid_sequences': h.sequences, 'gap_breaks': h.gap_breaks,
        'missing_10s_intervals_between_active_buckets': h.missing_intervals_between_active_buckets,
        'max_gap_seconds': round(h.max_gap_seconds, 3),
        'avg_window_reuse_of_covered_bucket': round(h.sequences * seq_len / h.valid_buckets, 4)
                                             if h.valid_buckets else None,
        'raw_start_ms': h.raw_start_ms, 'raw_end_ms': h.raw_end_ms,
        'benign_buckets': h.benign_buckets, 'malicious_buckets': h.attack_buckets,
        'invalid_bucket_labels': h.invalid_bucket_labels,
        'timestamp_anomalies': h.timestamp_anomalies,
        'misaligned_buckets': h.misaligned_buckets,
    }


def aggregate(rows, group_by):
    totals = defaultdict(lambda: defaultdict(int))
    sums = ['raw_flows', 'raw_benign_flows', 'raw_attack_flows', 'active_buckets',
            'bucket_flows', 'segments', 'short_segments_lt_10', 'valid_segments_ge_10',
            'short_buckets', 'covered_buckets', 'flows_in_short_segments',
            'covered_flows', 'valid_sequences']
    for row in rows:
        a = totals[row[group_by]]
        a['hosts'] += 1
        if row['valid_sequences']:
            a['productive_hosts'] += 1
        for k in sums:
            a[k] += int(row[k])
    out = []
    for key, a in sorted(totals.items()):
        r = {group_by: key, **a}
        r['raw_flows_covered_pct'] = safe_pct(a['covered_flows'], a['raw_flows'])
        r['active_buckets_covered_pct'] = safe_pct(a['covered_buckets'], a['active_buckets'])
        out.append(r)
    return out


def main():
    args = parse_args()
    db_path = args.db.resolve()
    manifest = read_json(args.data_dir / 'split_manifest.json')
    schema = read_json(args.data_dir / 'categorical_schema.json')
    bucket_table = manifest.get('bucket_table') or schema.get('bucket_table')
    bucket_ms = int(manifest.get('bucket_size_ms', schema.get('bucket_size_ms', 10000)))
    seq_len = int(manifest.get('sequence_length', 10))
    stride = int(manifest.get('stride', 1))
    if stride != 1:
        raise ValueError(f'This audit implements the official stride=1 only (manifest stride={stride})')
    if not bucket_table or bucket_ms <= 0 or seq_len <= 0:
        raise ValueError('Bad bucket_table, bucket_size_ms, or sequence_length in manifest')
    if int(schema.get('bucket_size_ms', bucket_ms)) != bucket_ms:
        raise ValueError('Manifest and schema disagree on bucket size')
    if not db_path.is_file():
        raise FileNotFoundError(f'SQLite database not found: {db_path}')

    val_hosts = None
    if args.validation_hosts.is_file():
        vh = read_json(args.validation_hosts)
        val_hosts = set(map(str, vh['validation_hosts']))
        print(f'[split] Validation host file loaded: {args.validation_hosts} ({len(val_hosts)} hosts)', flush=True)
    else:
        print('[split] Validation file missing; benign training pool will not be subdivided.', flush=True)

    uri = db_path.as_uri() + '?mode=ro'
    db = sqlite3.connect(uri, uri=True)
    hosts = {}
    try:
        db.execute('PRAGMA query_only = ON')
        check_columns(db, 'network_flows', ['ipv4_src_addr', 'label', 'flow_start_milliseconds'])
        check_columns(db, bucket_table, ['ipv4_src_addr', 'bucket_start_ms', 'flow_count', 'label'])
        load_raw_host_counts(db, hosts)
        load_buckets(db, bucket_table, hosts, bucket_ms, seq_len)
    finally:
        db.close()

    metadata_checks, metadata_details = audit_metadata(args.data_dir, manifest, hosts)
    print('[4/4] Writing per-host and coverage summaries ...', flush=True)
    roles = {ip: host_role(ip, h.class_name, manifest, val_hosts) for ip, h in hosts.items()}
    rows = [to_row(h, roles[ip], seq_len) for ip, h in hosts.items()]
    rows.sort(key=lambda r: (-r['raw_flows'], r['src_ip']))
    class_summary = aggregate(rows, 'class')
    role_summary = aggregate(rows, 'split_role')
    zero_rows = [r for r in rows if r['raw_flows'] > 0 and r['valid_sequences'] == 0]
    zero_rows.sort(key=lambda r: (-r['raw_flows'], r['src_ip']))

    # Explicit integrity checks, including generated metadata if locally available.
    checks = list(metadata_checks)
    def add_check(name, valid, details):
        checks.append({'name': name, 'status': 'PASS' if valid else 'FAIL', 'details': details})

    bad_raw_bucket = [(h.ip, h.raw_flows, h.bucket_flows) for h in hosts.values()
                      if h.raw_flows != h.bucket_flows]
    add_check('Raw flows = summed per-bucket flow_count', not bad_raw_bucket,
              f'mismatched hosts: {bad_raw_bucket[:10]}')
    bad_partition = [h.ip for h in hosts.values()
                     if h.short_buckets + h.valid_buckets != h.active_buckets or
                        h.short_segment_flows + h.covered_flows != h.bucket_flows or
                        h.short_segments + h.valid_segments != h.segments]
    add_check('All buckets and flows belong to exactly one segment', not bad_partition,
              f'mismatched hosts: {bad_partition[:10]}')
    bad_timestamps = [h.ip for h in hosts.values() if h.timestamp_anomalies or h.misaligned_buckets]
    add_check('Sorted aligned timestamps, no duplicate bucket keys', not bad_timestamps,
              f'problematic hosts: {bad_timestamps[:10]}')
    mixed = [h.ip for h in hosts.values() if h.class_name not in ('benign', 'malicious')]
    add_check('Host-level constant binary labels', not mixed, f'mixed/invalid host labels: {mixed}')
    bad_host_counts = [(ip, int(expected), hosts.get(ip, Host(ip)).sequences)
                       for ip, expected in manifest.get('benign_sequence_counts_by_ip', {}).items()
                       if int(expected) != hosts.get(ip, Host(ip)).sequences]
    add_check('Recomputed benign sequence count = manifest', not bad_host_counts,
              f'mismatched hosts: {bad_host_counts[:10]}')
    if val_hosts is not None:
        invalid_val = sorted(val_hosts - set(manifest.get('benign_train_ips', [])))
        add_check('Validation hosts belong to benign train pool', not invalid_val,
                  f'invalid hosts: {invalid_val}')

    hist_rows = []
    for h in hosts.values():
        for b, (seg, buckets, flows) in h.hist.items():
            hist_rows.append({'src_ip': h.ip, 'class': h.class_name,
                              'split_role': roles[h.ip], 'segment_length_bin': b,
                              'segments': seg, 'active_buckets': buckets, 'raw_flows': flows})
    hist_rows.sort(key=lambda x: (x['class'], x['src_ip'], x['segment_length_bin']))

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / 'per_host_coverage.csv', rows, CSV_FIELDS)
    write_csv(args.outdir / 'zero_sequence_hosts.csv', zero_rows, CSV_FIELDS)
    sum_fields = ['class', 'split_role', 'hosts', 'productive_hosts', 'raw_flows',
                  'raw_benign_flows', 'raw_attack_flows', 'active_buckets', 'bucket_flows',
                  'segments', 'short_segments_lt_10', 'valid_segments_ge_10', 'short_buckets',
                  'covered_buckets', 'flows_in_short_segments', 'covered_flows', 'valid_sequences',
                  'raw_flows_covered_pct', 'active_buckets_covered_pct']
    write_csv(args.outdir / 'summary_by_class.csv', class_summary, sum_fields)
    write_csv(args.outdir / 'summary_by_split_role.csv', role_summary, sum_fields)
    write_csv(args.outdir / 'segment_length_histogram_by_host.csv', hist_rows,
              ['src_ip', 'class', 'split_role', 'segment_length_bin',
               'segments', 'active_buckets', 'raw_flows'])

    totals = {
        'raw_hosts': sum(bool(h.raw_flows) for h in hosts.values()),
        'bucket_hosts': sum(bool(h.active_buckets) for h in hosts.values()),
        'total_raw_flows': sum(h.raw_flows for h in hosts.values()),
        'total_active_buckets': sum(h.active_buckets for h in hosts.values()),
        'total_sequences_across_unique_hosts_not_rounds': sum(h.sequences for h in hosts.values()),
        'raw_flows_covered_by_valid_segments': sum(h.covered_flows for h in hosts.values()),
        'raw_flows_not_covered': sum(h.raw_flows - h.covered_flows for h in hosts.values()),
        'raw_flows_covered_pct': safe_pct(sum(h.covered_flows for h in hosts.values()),
                                         sum(h.raw_flows for h in hosts.values())),
        'active_buckets_covered_pct': safe_pct(sum(h.valid_buckets for h in hosts.values()),
                                               sum(h.active_buckets for h in hosts.values())),
        'zero_sequence_hosts': len(zero_rows),
        'zero_sequence_hosts_raw_flows': sum(r['raw_flows'] for r in zero_rows),
        'zero_sequence_hosts_raw_flows_pct': safe_pct(sum(r['raw_flows'] for r in zero_rows),
                                                     sum(h.raw_flows for h in hosts.values())),
    }
    summary = {
        'definitions': {
            'active_bucket': 'An observed source-IP/10-second bin with >=1 raw flow',
            'segment': 'Maximal consecutive active buckets, broken at any missing 10-second bin',
            'valid_segment': f'Segment with >= {seq_len} consecutive active buckets',
            'covered_flows': 'Raw flows aggregated into valid-segment buckets, each counted once',
            'important_caveat': 'Covered flows are aggregated, not preserved at per-flow detail; overlapping windows repeat buckets',
            'raw_to_bucket_flow_pct': 'Summed flow_count over active buckets divided by raw flow count (not number of features retained)',
            'unrepresented_raw_flows': 'Flows in short segments (or any raw/bucket count mismatch)',
            'silence_intervals': 'Missing observed buckets BETWEEN active timestamps only; neither zero-fill nor true inactivity duration measurement',
        },
        'input': {'database': str(db_path), 'bucket_table': bucket_table,
                  'bucket_size_ms': bucket_ms, 'sequence_length': seq_len,
                  'stride': stride, 'manifest': str(args.data_dir / 'split_manifest.json'),
                  'validation_hosts_file_used': str(args.validation_hosts) if val_hosts is not None else None},
        'totals': totals, 'by_class': class_summary, 'by_split_role': role_summary,
        'checks': checks, 'metadata_details': metadata_details,
    }
    write_json(args.outdir / 'audit_summary.json', summary)

    print('\n' + '=' * 74)
    print('COVERAGE AUDIT — OFFICIAL 10s / 10-BUCKET / STRIDE-1 PIPELINE')
    print('=' * 74)
    print(f"Hosts raw / bucket: {totals['raw_hosts']} / {totals['bucket_hosts']}")
    print(f"Raw flows: {totals['total_raw_flows']:,}")
    print(f"Active buckets: {totals['total_active_buckets']:,}")
    print(f"Flows in valid segments: {totals['raw_flows_covered_by_valid_segments']:,} "
          f"({totals['raw_flows_covered_pct']}% of raw)")
    print(f"Hosts with raw flows but no sequences: {totals['zero_sequence_hosts']}; "
          f"they account for {totals['zero_sequence_hosts_raw_flows']:,} flows "
          f"({totals['zero_sequence_hosts_raw_flows_pct']}% of all raw flows)")
    print('\nBY HOST CLASS:')
    for r in class_summary:
        print(f"  {r['class']:>12}  hosts={r['hosts']:>3} productive={r['productive_hosts']:>3} "
              f"raw_flows={r['raw_flows']:>11,} "
              f"covered={str(r['raw_flows_covered_pct']) + '%':>9} "
              f"sequences={r['valid_sequences']:>9,}")
    print('\nBY SPLIT ROLE:')
    for r in role_summary:
        print(f"  {r['split_role']:>24} hosts={r['hosts']:>3} "
              f"raw_flows={r['raw_flows']:>11,} "
              f"covered={str(r['raw_flows_covered_pct']) + '%':>9} "
              f"sequences={r['valid_sequences']:>9,}")
    print('\nHIGHEST-FLOW HOSTS WITH ZERO VALID SEQUENCES:')
    for r in zero_rows[:12]:
        print(f"  {r['src_ip']:<19} {r['class']:<10} "
              f"flows={r['raw_flows']:>10,} active_buckets={r['active_buckets']:>8,} "
              f"longest_segment={r['max_segment_buckets']:>3}")
    if not zero_rows:
        print('  None')
    print('\nCHECKS:')
    for c in checks:
        print(f"  [{c['status']}] {c['name']} — {c['details'][:130]}")
    print(f'\nSaved to: {args.outdir.resolve()}')
    if any(c['status'] == 'FAIL' for c in checks):
        print('ATTENTION: Some integrity checks FAILED. Inspect audit_summary.json.')
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, sqlite3.Error, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
