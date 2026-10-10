#!/usr/bin/env python3
"""Audit CIC-IDS2018-v3 benign-only 10s/10-bucket sequence coverage.

Read-only: scans an existing CIC bucket SQLite and the merged_manifest.json.
Distinguishes loss from sequence eligibility and deterministic window quotas.
Run from the repository root: python .\analysis\audit_cic_coverage.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, deque
from pathlib import Path

import numpy as np

BIN_MS = 10_000
SEQ_LEN = 10
TABLE = "cic_temporal_buckets"


def pct(n, d):
    return (100.0 * n / d) if d else None


def fmt_pct(n, d):
    p = pct(n, d)
    return "n/a" if p is None else f"{p:.4f}%"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stats_dict():
    return Counter()


def assign_split(groups, counts, quota):
    split_by_host = {}
    selected_by_host = {}
    for split in ("train", "validation", "holdout"):
        for host in groups.get(split, []):
            if host in split_by_host:
                raise ValueError(f"Host appears in multiple CIC splits: {host}")
            split_by_host[host] = split
            selected_by_host[host] = int(quota.get(split, {}).get(host, 0))
    expected = {host for host, info in counts.items() if int(info["benign_windows"]) > 0}
    if set(split_by_host) != expected:
        raise ValueError(f"CIC eligible/split mismatch: missing={len(expected-set(split_by_host))}, "
                         f"unexpected={len(set(split_by_host)-expected)}")
    if any(selected_by_host[h] < 1 for h in expected):
        raise ValueError("Some productive CIC hosts have zero selected windows")
    return split_by_host, selected_by_host


def selected_indices(n_available, n_select):
    if n_select < 1 or n_select > n_available:
        raise ValueError(f"Invalid quota: selected={n_select}, available={n_available}")
    indices = np.linspace(0, n_available - 1, n_select).astype(np.int64).tolist()
    if len(set(indices)) != n_select:
        raise ValueError("Selected window indices are not unique")
    return indices


def audit(db, manifest, out):
    for key in ("cic_host_coverage", "cic_split_hosts", "cic_selected_per_host"):
        if key not in manifest:
            raise ValueError(f"Missing {key} in merged_manifest.json")
    if (int(manifest.get("sequence_length", 10)), int(manifest.get("bucket_size_ms", 10000)),
            int(manifest.get("stride", 1))) != (10, 10000, 1):
        raise ValueError("Manifest sequence geometry is not 10s / 10 buckets / stride 1")

    counts = manifest["cic_host_coverage"]
    split_by_host, selected_by_host = assign_split(
        manifest["cic_split_hosts"], counts, manifest["cic_selected_per_host"])
    out.mkdir(parents=True, exist_ok=True)
    if not db.is_file():
        raise FileNotFoundError(db)
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=120)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=FILE")
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone():
            raise ValueError(f"Missing SQLite table {TABLE}")
        raw_by_label = {int(y): int(n) for y, n in conn.execute(
            "SELECT label, COUNT(*) FROM network_flows GROUP BY label")}
        if set(raw_by_label) - {0, 1}:
            raise ValueError(f"Unexpected raw label values: {raw_by_label}")
        print(f"[raw] benign={raw_by_label.get(0,0):,}; malicious={raw_by_label.get(1,0):,}", flush=True)
        n_raw = sum(raw_by_label.values())
        bucket_count, sum_flows = conn.execute(
            f"SELECT COUNT(*), SUM(flow_count) FROM {TABLE}").fetchone()
        if int(sum_flows) != n_raw:
            raise ValueError(f"Raw {n_raw} != aggregated bucket flow_count {sum_flows}")
        print(f"[buckets] {bucket_count:,} rows; {sum_flows:,} flows accounted for", flush=True)

        columns = ["src_ip", "split", "has_attack_buckets", "eligible", "total_raw_flows",
                   "all_buckets", "benign_only_buckets", "benign_only_bucket_flows",
                   "attack_labeled_bucket_flows", "benign_segments", "valid_benign_segments",
                   "longest_benign_segment_buckets", "short_benign_segment_flows",
                   "valid_benign_windows", "valid_segment_covered_buckets",
                   "valid_segment_covered_flows", "selected_windows", "selected_covered_buckets",
                   "selected_covered_flows", "window_selection_fraction_pct"]
        all_sum = stats_dict()
        group_sum = {}
        zeros = []
        top_excluded = []
        current = None
        hs = {}
        flowbuf = deque(maxlen=SEQ_LEN)
        seq_targets = []
        seq_target_ptr = 0
        seq_index = 0
        last_selected_end = -1
        bucket_position = -1
        last_stamp = None
        run_len = 0
        run_flow = 0

        def finish_run():
            nonlocal run_len, run_flow
            if run_len:
                hs["benign_segments"] += 1
                hs["longest_benign_segment_buckets"] = max(
                    hs["longest_benign_segment_buckets"], run_len)
                if run_len >= SEQ_LEN:
                    hs["valid_benign_segments"] += 1
                    hs["valid_segment_covered_buckets"] += run_len
                    hs["valid_segment_covered_flows"] += run_flow
                else:
                    hs["short_benign_segment_flows"] += run_flow
            run_len = 0
            run_flow = 0
            flowbuf.clear()

        def finish_host(writer):
            if current is None:
                return
            finish_run()
            info = counts.get(current)
            if info is None:
                raise ValueError(f"Unexpected host {current} not in manifest")
            if hs["total_raw_flows"] != int(info["raw_flows"]):
                raise ValueError(f"{current} flow count differs from manifest")
            if hs["valid_benign_windows"] != int(info["benign_windows"]):
                raise ValueError(f"{current} window count differs from manifest")
            expected_selected = int(selected_by_host.get(current, 0))
            if hs["selected_windows"] != expected_selected:
                raise ValueError(f"{current} selected count {hs['selected_windows']} != {expected_selected}")
            if seq_target_ptr != len(seq_targets):
                raise ValueError(f"{current} did not find all selected window indices")
            eligible = hs["valid_benign_windows"] > 0
            group = split_by_host.get(current, "ineligible")
            row = {"src_ip": current, "split": group,
                   "has_attack_buckets": bool(info.get("has_attack_buckets", False)),
                   "eligible": eligible, **hs,
                   "window_selection_fraction_pct": pct(hs["selected_windows"],
                                                           hs["valid_benign_windows"])}
            writer.writerow(row)
            cohort = "eligible" if eligible else "ineligible"
            for k, v in hs.items():
                if isinstance(v, int):
                    all_sum[k] += v
            for tag in dict.fromkeys((cohort, group)):
                bucket = group_sum.setdefault(tag, Counter())
                bucket["hosts"] += 1
                for k, v in hs.items():
                    if isinstance(v, int):
                        bucket[k] += v
            if not eligible:
                zeros.append((current, hs["total_raw_flows"], hs["benign_only_bucket_flows"],
                              hs["all_buckets"], hs["longest_benign_segment_buckets"],
                              bool(info.get("has_attack_buckets", False))))
            if eligible:
                top_excluded.append((current, hs["valid_benign_windows"],
                                     hs["selected_windows"],
                                     hs["valid_segment_covered_flows"],
                                     hs["selected_covered_flows"], group))

        outcsv = out / "cic_per_host_coverage.csv"
        with outcsv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            query = (f"SELECT ipv4_src_addr, bucket_start_ms, label, flow_count "
                     f"FROM {TABLE} ORDER BY ipv4_src_addr, bucket_start_ms")
            for host, t, label, flow_count in conn.execute(query):
                host = str(host)
                t, label, flow_count = int(t), int(label), int(flow_count)
                if label not in (0, 1) or flow_count <= 0 or t % BIN_MS != 0:
                    raise ValueError(f"Invalid bucket for {host} at {t}: label={label}, flows={flow_count}")
                if host != current:
                    finish_host(writer)
                    current = host
                    hs = {k: 0 for k in columns if k not in
                          ("src_ip", "split", "has_attack_buckets", "eligible", "window_selection_fraction_pct")}
                    last_stamp = None
                    run_len = run_flow = 0
                    flowbuf.clear()
                    seq_index = 0
                    seq_target_ptr = 0
                    last_selected_end = -1
                    bucket_position = -1
                    n_avail = int(counts[current]["benign_windows"])
                    n_sel = int(selected_by_host.get(current, 0))
                    seq_targets = selected_indices(n_avail, n_sel) if n_sel else []
                bucket_position += 1
                hs["all_buckets"] += 1
                hs["total_raw_flows"] += flow_count
                if label == 0:
                    hs["benign_only_buckets"] += 1
                    hs["benign_only_bucket_flows"] += flow_count
                    if last_stamp is None or t - last_stamp != BIN_MS or run_len == 0:
                        finish_run()
                    run_len += 1
                    run_flow += flow_count
                    flowbuf.append((bucket_position, flow_count))
                    if run_len >= SEQ_LEN:
                        hs["valid_benign_windows"] += 1
                        if seq_target_ptr < len(seq_targets) and seq_index == seq_targets[seq_target_ptr]:
                            hs["selected_windows"] += 1
                            for position, fcnt in flowbuf:
                                if position > last_selected_end:
                                    hs["selected_covered_buckets"] += 1
                                    hs["selected_covered_flows"] += fcnt
                            last_selected_end = bucket_position
                            seq_target_ptr += 1
                        seq_index += 1
                else:
                    hs["attack_labeled_bucket_flows"] += flow_count
                    finish_run()
                last_stamp = t
            finish_host(writer)

        n_hosts = int(group_sum.get("eligible", {}).get("hosts", 0) +
                      group_sum.get("ineligible", {}).get("hosts", 0))
        if n_hosts != len(counts):
            raise ValueError(f"Host count mismatch: scanned={n_hosts}; manifest={len(counts)}")
        if all_sum["total_raw_flows"] != n_raw or all_sum["all_buckets"] != bucket_count:
            raise ValueError("Aggregate bucket totals mismatch")
        benign_raw = int(raw_by_label.get(0, 0))
        pure_benign = all_sum["benign_only_bucket_flows"]
        covered = all_sum["valid_segment_covered_flows"]
        selected = all_sum["selected_covered_flows"]
        if not (0 <= selected <= covered <= pure_benign <= benign_raw):
            raise ValueError(f"Benign coverage inconsistent: {selected} <= {covered} <= "
                             f"{pure_benign} <= {benign_raw}")
        summary = {
            "database": str(db.resolve()),
            "manifest": "processed_binary_temporal_unsw_cic/merged_manifest.json",
            "all_raw_flows": n_raw,
            "raw_benign_flows": benign_raw,
            "raw_attack_flows": int(raw_by_label.get(1, 0)),
            "source_hosts_all": n_hosts,
            "source_hosts_eligible_benign_only_10b": group_sum.get("eligible", {}).get("hosts", 0),
            "source_hosts_no_eligible_window": group_sum.get("ineligible", {}).get("hosts", 0),
            "total_active_buckets": int(bucket_count),
            "pure_benign_bucket_flows": pure_benign,
            "benign_flows_in_buckets_labeled_attack": benign_raw - pure_benign,
            "benign_flows_in_valid_contiguous_10b_segments": covered,
            "benign_flows_covered_by_actually_selected_windows": selected,
            "raw_benign_flow_valid_segment_coverage_pct": pct(covered, benign_raw),
            "raw_benign_flow_selected_window_coverage_pct": pct(selected, benign_raw),
            "loss_benign_flows_in_short_segments": pure_benign - covered,
            "sampling_loss_benign_flows_not_in_selected_windows": covered - selected,
            "possible_benign_flow_loss_mixed_label_buckets": benign_raw - pure_benign,
            "all_valid_benign_windows_stride1": all_sum["valid_benign_windows"],
            "selected_benign_windows": all_sum["selected_windows"],
            "selected_window_fraction_pct": pct(all_sum["selected_windows"], all_sum["valid_benign_windows"]),
            "groups": {k: dict(v) for k, v in group_sum.items()},
            "checks": {"raw_flows_equal_buckets": True, "manifest_all_host_counts_match": True,
                       "manifest_all_window_counts_match": True, "all_selected_quotas_match": True,
                       "no_valid_host_excluded": True},
        }
        with (out / "cic_coverage_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        group_keys = ["eligible", "ineligible", "train", "validation", "holdout"]
        with (out / "cic_groups.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "hosts", "all_raw_flows", "benign_only_bucket_flows",
                             "benign_flows_valid_segments", "benign_flows_selected_windows",
                             "valid_windows", "selected_windows"])
            for tag in group_keys:
                g = group_sum.get(tag, Counter())
                writer.writerow([tag, g["hosts"], g["total_raw_flows"],
                                 g["benign_only_bucket_flows"], g["valid_segment_covered_flows"],
                                 g["selected_covered_flows"], g["valid_benign_windows"],
                                 g["selected_windows"]])
        with (out / "cic_largest_ineligible_hosts.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src_ip", "all_raw_flows", "pure_benign_bucket_flows", "active_buckets",
                        "longest_benign_segment_buckets", "has_attack_buckets"])
            for row in sorted(zeros, key=lambda x: x[2], reverse=True)[:100]:
                w.writerow(row)
        with (out / "cic_largest_sampling_losses.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src_ip", "valid_windows", "selected_windows",
                        "valid_segment_covered_flows", "selected_covered_flows", "split"])
            for row in sorted(top_excluded, key=lambda x: x[3] - x[4], reverse=True)[:100]:
                w.writerow(row)
        print("\n=== CIC DATA COVERAGE AUDIT ===")
        print(f"Raw hosts: {n_hosts:,}; productive benign hosts: {summary['source_hosts_eligible_benign_only_10b']:,} "
              f"({fmt_pct(summary['source_hosts_eligible_benign_only_10b'], n_hosts)})")
        print(f"Raw benign flows: {benign_raw:,}")
        print(f"Benign flows in label-0 buckets: {pure_benign:,} "
              f"({fmt_pct(pure_benign, benign_raw)})")
        print(f"Benign flows covered by ANY valid 10-bucket segment: {covered:,} "
              f"({fmt_pct(covered, benign_raw)})")
        print(f"Benign flows covered by SELECTED windows: {selected:,} "
              f"({fmt_pct(selected, benign_raw)})")
        print(f"All eligible benign windows: {all_sum['valid_benign_windows']:,}")
        print(f"Selected benign windows: {all_sum['selected_windows']:,} "
              f"({fmt_pct(all_sum['selected_windows'], all_sum['valid_benign_windows'])})")
        print(f"Benign flow loss due to short segments: {pure_benign-covered:,}")
        print(f"Benign flows in mixed/attack-labeled buckets: {benign_raw-pure_benign:,}")
        print("Note: label-1 buckets may also contain benign flows; they cannot be used for benign-only sequences.")
        for tag in ("train", "validation", "holdout"):
            group = group_sum.get(tag, Counter())
            print(f"{tag}: hosts={group['hosts']:,}, available_windows={group['valid_benign_windows']:,}, "
                  f"selected={group['selected_windows']:,}, selected_flows={group['selected_covered_flows']:,}")
        print("[PASS] All per-host raw-flow, window-count and selection-quota checks")
        print(f"[done] {out.resolve()}")
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cic-db", type=Path, default=Path("datasets/cic_2018_traffic.db"))
    p.add_argument("--merged-manifest", type=Path,
                   default=Path("processed_binary_temporal_unsw_cic/merged_manifest.json"))
    p.add_argument("--outdir", type=Path, default=Path("experiments/cic_coverage_audit"))
    args = p.parse_args()
    manifest = load_json(args.merged_manifest)
    audit(args.cic_db, manifest, args.outdir)


if __name__ == "__main__":
    main()
