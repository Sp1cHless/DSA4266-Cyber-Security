"""Train-only feature screening and separately identified validation collision diagnostics."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sqlite3
import sys
import time

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.run_preprocessing import raw_bucket, service_ids
from train.common.io import ROOT, PROJECT, fresh, read_json, write_json, sha256, unique_name, environment
from train.run_circular_shift import write_csv

PREPARED = PROJECT / "CIC_new/outputs/scheme2_runs/20261022_115847_160650_scheme2_bucket64_shift0_seed4266_0fhh3gjt/preprocessed"
AUDIT = ROOT / "runs/20261024_122800_754357_pure_infilteration_raw_flow_audit_6146f6"
IAT = [f"{direction}_iat_{stat}" for direction in ("src_to_dst", "dst_to_src") for stat in ("min", "max", "avg", "stddev")]
RATES = [f"{direction}_{suffix}" for direction in ("src_to_dst", "dst_to_src") for suffix in ("second_bytes", "avg_throughput")]
TTL = ["min_ttl", "max_ttl"]
OMITTED = IAT + RATES + TTL + ["longest_flow_pkt", "shortest_flow_pkt", "icmp_type", "ftp_command_ret_code"]
FAMILY_FIELDS = {"iat": IAT, "rates": RATES, "ttl": TTL}
DNS_TYPES = (0, 1, 2, 5, 6, 12, 15, 16, 28, 255)
ICMP_TYPES = (0, 3, 5, 8, 11, 12)
LIMITATIONS = [
    "Diagnostic reuse of real training data; no formal Scheme2 model, encoder, or bucket export is trained here.",
    "Training probes classify primary Infilteration versus benign only. Other attacks and synthetic examples are excluded.",
    "All original training Infilteration sequences are included; benign controls combine fixed random sampling and same-host/occupancy/volume matching.",
    "The diagnostic cohort has artificial prevalence and does not estimate deployment precision or full-validation FPR.",
    "Feature definitions and probe settings are fixed before extraction; internal train-only time-block and source-host splits are used.",
    "The baseline probe uses pre-encoder bucket statistics/histograms, not the trained LSTM or the final Scheme2 representation.",
    "The original full-training category vocabulary is retained for the baseline representation; internal holdouts are screening splits, not independent external validation.",
    "No absolute timestamps, source/destination address values, labels, attack names or row identifiers are model features.",
    "Validation is used only for previously identified collision-pair diagnostics, never for candidate ranking, probe fitting or thresholds.",
    "A new feature separating one selected pair need not separate the attack from its other benign counterparts.",
    "Zeros in omitted fields can mean zero or an inapplicable/unavailable statistic; positive fractions are reported without assuming their meaning.",
    "TTL and rate families are separate diagnostics because host/path signatures and short-duration extraction effects may create shortcuts.",
    "No test flow records or feature shards are read. Results are exploratory and do not establish significance or recall improvement.",
]


def connect(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA cache_size=-131072")
    return connection


def cohort(manifest, random_benign):
    names = ("id", "host", "start", "flows", "attack_flows", "active", "attack", "split")
    fields = "sequence_id,source_ip,window_start_ms,flows,attack_flows,active_buckets,primary_attack,split"
    positives = [dict(zip(names, r)) for r in manifest.execute(
        f"SELECT {fields} FROM original_sequences WHERE split='train' AND primary_attack='Infilteration' ORDER BY sequence_id")]
    hosts = set(r["host"] for r in positives)
    benign = [dict(zip(names, r)) for r in manifest.execute(
        f"SELECT {fields} FROM original_sequences WHERE split='train' AND attack_flows=0 ORDER BY sequence_id")]
    pools = defaultdict(list)
    for i, row in enumerate(benign):
        if row["host"] in hosts:
            pools[(row["host"], row["active"], int(np.floor(np.log2(row["flows"]))))].append(i)
    rng = np.random.default_rng(4266)
    selected = set(map(int, rng.choice(len(benign), min(random_benign, len(benign)), replace=False)))
    random_ids = set(selected)
    matched = set()
    positive_with_match = 0
    for row in positives:
        pool_ids = pools.get((row["host"], row["active"], int(np.floor(np.log2(row["flows"])))), [])
        if pool_ids:
            positive_with_match += 1
            chosen = rng.choice(pool_ids, min(2, len(pool_ids)), replace=False)
            matched.update(map(int, chosen))
            selected.update(map(int, chosen))
    rows = []
    for row in positives:
        rows.append({**row, "target": 1, "role": "all_training_infilteration"})
    for i in sorted(selected):
        role = "random_and_matched" if i in random_ids and i in matched else "random_benign" if i in random_ids else "matched_benign"
        rows.append({**benign[i], "target": 0, "role": role})
    rows.sort(key=lambda r: r["id"])
    return rows, {"positive_n": len(positives), "benign_n": len(selected), "random_benign_n": len(random_ids),
        "matched_unique_benign_n": len(matched), "positives_with_exact_host_active_log2volume_pool": positive_with_match}


def candidate_names():
    result = {family: [f"{field}_{stat}" for field in fields for stat in ("mean", "max", "positive_fraction")]
              for family, fields in FAMILY_FIELDS.items()}
    result["distribution"] = [f"{field}_{stat}" for field in
        ("in_bytes", "out_bytes", "in_pkts", "out_pkts", "flow_duration_milliseconds") for stat in ("std", "median", "q90")]
    result["distribution"] += ["zero_reply_fraction", "zero_duration_fraction", "one_ms_or_less_duration_fraction",
        "in_byte_share_mean", "in_byte_share_std", "bytes_per_packet_mean", "bytes_per_packet_std",
        "destination_port_pair_unique_fraction", "longest_flow_pkt_mean", "shortest_flow_pkt_mean"]
    result["categories"] = [f"dns_type_{code}_fraction" for code in DNS_TYPES] + ["dns_type_other_fraction"]
    result["categories"] += [f"icmp_ipv4_type_{code}_fraction" for code in ICMP_TYPES] + ["icmp_ipv4_type_other_fraction"]
    return result


def extract(rows, start, schema):
    """Build old and candidate features for the identical 100s/10s grid."""
    if not rows:
        raise ValueError("Empty source sequence")
    get = lambda name: np.array([row[name] for row in rows], dtype=np.float64)
    destination = {value: i for i, value in enumerate(sorted(set(row["ipv4_dst_addr"] for row in rows)))}
    events = np.zeros((len(rows), len(bf.EVENT_COLUMNS)), np.int64)
    for name, index in bf.E.items():
        events[:, index] = ([destination[row["ipv4_dst_addr"]] for row in rows] if name == "destination_id"
                             else [row[name] for row in rows])
    events = bf.local_events(events, start)
    old, hist = raw_bucket(events)
    pieces = [old["numeric"].reshape(-1)]
    for field in ("protocol", "l7", "service"):
        vocab = schema["vocabularies"][field]
        index = {int(code): i + 1 for i, code in enumerate(vocab)}
        dense = np.zeros((10, len(vocab) + 1), np.float64)
        for bucket, code, count in hist[field]:
            dense[int(bucket), index.get(int(code), 0)] += count
        total = dense.sum(1, keepdims=True)
        np.divide(dense, total, out=dense, where=total > 0)
        pieces.append(dense.reshape(-1))
    pieces.append(old["mask"].astype(np.float64))
    base = np.concatenate(pieces)
    names = candidate_names()
    extra = {family: np.zeros((10, len(fields)), np.float64) for family, fields in names.items()}
    bucket_index = events[:, bf.E["flow_start_milliseconds"]] // 10000
    for bucket in np.unique(bucket_index):
        selected = bucket_index == bucket
        values = lambda name: get(name)[selected]
        for family, fields in FAMILY_FIELDS.items():
            features = []
            for field in fields:
                array = values(field)
                array = np.where(np.isfinite(array) & (array >= 0), array, 0)
                features.extend((array.mean(), array.max(), (array > 0).mean()))
            extra[family][bucket] = features
        features = []
        for field in ("in_bytes", "out_bytes", "in_pkts", "out_pkts", "flow_duration_milliseconds"):
            v = values(field)
            features.extend((v.std(), np.median(v), np.quantile(v, .9)))
        in_bytes, out_bytes = values("in_bytes"), values("out_bytes")
        byte_share = in_bytes / np.maximum(in_bytes + out_bytes, 1)
        bpp = (in_bytes + out_bytes) / np.maximum(values("in_pkts") + values("out_pkts"), 1)
        pairs = {(row["ipv4_dst_addr"], row["l4_dst_port"]) for row, take in zip(rows, selected) if take}
        features.extend(((values("out_pkts") == 0).mean(), (values("flow_duration_milliseconds") == 0).mean(),
            (values("flow_duration_milliseconds") <= 1).mean(), byte_share.mean(), byte_share.std(), bpp.mean(), bpp.std(),
            len(pairs) / selected.sum(), values("longest_flow_pkt").mean(), values("shortest_flow_pkt").mean()))
        extra["distribution"][bucket] = features
        dns = ((values("dns_query_id") != 0) | (values("dns_query_type") != 0) |
               (values("dns_ttl_answer") != 0) | (values("l4_dst_port") == 53))
        icmp = values("protocol") == 1
        categories = []
        for field, eligible, vocabulary in (("dns_query_type", dns, DNS_TYPES), ("icmp_ipv4_type", icmp, ICMP_TYPES)):
            codes = values(field)
            categories.extend(((codes == code) & eligible).mean() for code in vocabulary)
            categories.append((~np.isin(codes, vocabulary) & eligible).mean())
        extra["categories"][bucket] = categories
    for array in (base, *extra.values()):
        if not np.isfinite(array).all() or np.any(array < 0):
            raise AssertionError("Nonfinite or negative candidate features")
    return base, extra, old


def source_rows(work, raw, sequence_id):
    expected = [int(row[0]) for row in work.execute(
        "SELECT raw_rowid FROM events WHERE sequence_id=? ORDER BY flow_start_milliseconds,raw_rowid", (int(sequence_id),))]
    found = {}
    for offset in range(0, len(expected), 800):
        ids = expected[offset:offset + 800]
        for row in raw.execute("SELECT rowid AS raw_rowid,* FROM network_flows WHERE rowid IN (" + ",".join("?" for _ in ids) + ")", ids):
            found[int(row["raw_rowid"])] = dict(row)
    if len(found) != len(expected):
        raise AssertionError("Source row reconciliation failed")
    return [found[value] for value in expected]


def update_quality(quality, rows, target):
    group = "Infilteration_sequence" if target else "Benign_sequence"
    for field in OMITTED:
        values = np.array([r[field] if r[field] is not None else np.nan for r in rows], dtype=float)
        key = (group, field)
        result = quality.setdefault(key, {"group": group, "field": field, "flows": 0, "zero": 0,
            "positive": 0, "negative": 0, "nonfinite": 0, "min": None, "max": None})
        result["flows"] += len(values)
        result["zero"] += int((values == 0).sum())
        result["positive"] += int((values > 0).sum())
        result["negative"] += int((values < 0).sum())
        result["nonfinite"] += int((~np.isfinite(values)).sum())
        valid = values[np.isfinite(values)]
        if len(valid):
            result["min"] = float(valid.min()) if result["min"] is None else min(result["min"], float(valid.min()))
            result["max"] = float(valid.max()) if result["max"] is None else max(result["max"], float(valid.max()))


def split_indices(rows, mode):
    y = np.array([r["target"] for r in rows])
    groups = np.array([r["start"] // 900000 if mode == "time_block" else r["host"] for r in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=4266)
    fit, hold = next(splitter.split(np.zeros(len(rows)), y, groups))
    if len(np.unique(y[fit])) != 2 or len(np.unique(y[hold])) != 2:
        raise ValueError("The fixed internal group split lacks a class")
    assert not set(groups[fit]) & set(groups[hold])
    return fit, hold


def probes(rows, base, extras, out, trees):
    y = np.array([r["target"] for r in rows])
    names = candidate_names()
    results, ranks, memberships = [], [], []
    settings = [("old_aggregates", []), *[("old_plus_" + k, [k]) for k in names],
                ("old_plus_robust_candidates", ["iat", "distribution", "categories"]),
                ("old_plus_all_candidates", list(names))]
    arrays = {k: v.reshape(len(rows), -1).astype(np.float32) for k, v in extras.items()}
    base = base.astype(np.float32)
    for mode in ("time_block", "source_host"):
        fit, hold = split_indices(rows, mode)
        fit_set = set(fit)
        memberships.extend({"sample_id": row["id"], "mode": mode, "partition": "fit" if i in fit_set else "holdout"}
                           for i, row in enumerate(rows))
        for family, values in extras.items():
            active = np.array([r["active"] for r in rows])
            summary = values.sum(1) / active[:, None]
            for j, name in enumerate(names[family]):
                train_auc = roc_auc_score(y[fit], summary[fit, j])
                direction = 1 if train_auc >= .5 else -1
                ranks.append({"mode": mode, "family": family, "feature": name,
                    "fit_auc_oriented": max(train_auc, 1 - train_auc),
                    "holdout_auc_fit_direction": roc_auc_score(y[hold], direction * summary[hold, j]),
                    "positive_sequence_zero_fraction": float((summary[y == 1, j] == 0).mean()),
                    "benign_sequence_zero_fraction": float((summary[y == 0, j] == 0).mean())})
        for label, families in settings:
            x = np.concatenate([base] + [arrays[k] for k in families], axis=1)
            model = ExtraTreesClassifier(n_estimators=trees, max_depth=12, min_samples_leaf=8,
                max_features="sqrt", class_weight="balanced", n_jobs=4, random_state=4266)
            model.fit(x[fit], y[fit])
            score = model.predict_proba(x[hold])[:, 1]
            results.append({"mode": mode, "features": label, "dimensions": x.shape[1],
                "fit_n": len(fit), "holdout_n": len(hold), "fit_positive_n": int(y[fit].sum()),
                "holdout_positive_n": int(y[hold].sum()), "holdout_roc_auc": roc_auc_score(y[hold], score),
                "source_host_overlap_n": len({rows[i]["host"] for i in fit} & {rows[i]["host"] for i in hold})})
            print(f"probe {mode} {label}: AUC={results[-1]['holdout_roc_auc']:.4f}", flush=True)
    write_csv(out / "probe_results.csv", results)
    write_csv(out / "univariate_screening.csv", ranks)
    write_csv(out / "internal_partitions.csv", memberships)
    return results


def collision_pairs(audit):
    with (audit / "stage_exact_collisions.csv").open() as stream:
        return [row for row in csv.DictReader(stream) if row["stage"] == "scaled_pre_encoder"]


def collision_diagnostic(manifest, work, raw, schema, audit, out):
    pairs = collision_pairs(audit)
    cache = {}
    result = []
    for pair in pairs:
        ids = (int(pair["sample_id"]), int(pair["example_benign_id"]))
        for sid in ids:
            if sid not in cache:
                split, start = manifest.execute("SELECT split,window_start_ms FROM original_sequences WHERE sequence_id=?", (sid,)).fetchone()
                if split != "validation":
                    raise AssertionError("Collision pair is outside validation")
                rows = source_rows(work, raw, sid)
                cache[sid] = (rows, *extract(rows, start, schema)[:2])
        (left_rows, left, extra_left), (right_rows, right, extra_right) = [cache[sid] for sid in ids]
        if not np.array_equal(left, right):
            raise AssertionError("Previously exact pre-encoder pair no longer matches")
        changed = [field for field in OMITTED if [r[field] for r in left_rows] != [r[field] for r in right_rows]]
        families = [key for key in extra_left if not np.array_equal(extra_left[key], extra_right[key])]
        result.append({"sample_id": ids[0], "benign_id": ids[1], "outcome": pair["outcome"],
            "old_aggregate_equal": True, "differing_omitted_fields": ";".join(changed),
            "separating_candidate_families": ";".join(families),
            "attack_recorded_durations_ms": ";".join(str(r["flow_duration_milliseconds"]) for r in left_rows),
            "benign_recorded_durations_ms": ";".join(str(r["flow_duration_milliseconds"]) for r in right_rows),
            "attack_endpoint_durations_ms": ";".join(str(r["flow_end_milliseconds"]-r["flow_start_milliseconds"]) for r in left_rows),
            "benign_endpoint_durations_ms": ";".join(str(r["flow_end_milliseconds"]-r["flow_start_milliseconds"]) for r in right_rows)})
    write_csv(out / "validation_collision_pair_diagnostic.csv", result)
    return result


def write_report(out, info, quality, results, collisions):
    baseline = {row["mode"]: row["holdout_roc_auc"] for row in results if row["features"] == "old_aggregates"}
    lines = ["# Infilteration feature audit", "", "Training-data screening and separate validation-selected collision diagnostics. Not a model improvement claim.", "",
        f"Real train positives: {info['positive_n']}; sampled real benign: {info['benign_n']}.",
        f"Positives with a same-host, same-occupancy, same-log2-volume benign pool: {info['positives_with_exact_host_active_log2volume_pool']}.", "",
        "## Fixed train-only probes", "", "ROC-AUC uses an artificial Infilteration-versus-benign cohort and raw pre-encoder inputs.", "",
        "| Internal holdout | Features | Dimensions | ROC-AUC | Delta from old aggregates |", "|---|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['mode']} | {row['features']} | {row['dimensions']} | {row['holdout_roc_auc']:.4f} | {row['holdout_roc_auc']-baseline[row['mode']]:+.4f} |")
    lines += ["", "## Omitted-field availability", "", "These are all flows inside selected sequence groups, not necessarily individually Infilteration-labelled flows.", "",
        "| Field | Positive fraction in Infilteration sequences | Positive fraction in benign sequences |", "|---|---:|---:|"]
    by_key = {(r["group"], r["field"]): r for r in quality}
    for field in OMITTED:
        a, b = [by_key[(g, field)] for g in ("Infilteration_sequence", "Benign_sequence")]
        lines.append(f"| {field} | {a['positive']/a['flows']:.2%} | {b['positive']/b['flows']:.2%} |")
    fn = [r for r in collisions if r["outcome"] == "FN"]
    lines += ["", "## Previously exact validation collision pairs", "",
        f"{len(fn)} selected pure-miss pairs were checked. {sum(bool(r['separating_candidate_families']) for r in fn)} differ under at least one candidate family.",
        "This checks selected counterparts only; it is not a renewed search against every benign validation sequence.", "",
        "| Attack sample | Candidate families differing | Recorded duration(s), attack / benign (ms) | Endpoint duration(s), attack / benign (ms) |", "|---|---|---|---|"]
    for row in fn:
        lines.append(f"| {row['sample_id']} | {row['separating_candidate_families'] or 'none'} | {row['attack_recorded_durations_ms']} / {row['benign_recorded_durations_ms']} | {row['attack_endpoint_durations_ms']} / {row['benign_endpoint_durations_ms']} |")
    lines += ["", "## Boundaries", ""] + ["- " + item for item in LIMITATIONS]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--db", type=Path, default=PROJECT / "datasets/cic_2018_traffic.db")
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--random-benign", type=int, default=6000)
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.random_benign < 1 or args.trees < 1:
        raise ValueError("Counts must be positive")
    schema = read_json(args.prepared / "schema.json")
    if schema.get("timesteps") != 10:
        raise ValueError("The original ten-bucket grid is required")
    out = fresh(args.out or ROOT / "runs" / unique_name("infilteration_feature_audit"))
    write_json(out / "status.json", {"status": "running", "test_used": False})
    manifest, work, raw = [connect(path) for path in (args.prepared / "manifest.sqlite", args.prepared / "work.sqlite", args.db)]
    raw.row_factory = sqlite3.Row
    try:
        columns = [row[1] for row in raw.execute("PRAGMA table_info(network_flows)")]
        if set(OMITTED) - set(columns):
            raise ValueError("Source lacks candidate columns")
        rows, info = cohort(manifest, args.random_benign)
        print("Selected cohort: " + str(info), flush=True)
        write_csv(out / "training_cohort.csv", rows)
        quality, base, extras = {}, [], {key: [] for key in candidate_names()}
        begun = time.monotonic()
        for i, row in enumerate(rows):
            original = source_rows(work, raw, row["id"])
            if len(original) != row["flows"] or sum(r["label"] for r in original) != row["attack_flows"]:
                raise AssertionError("Raw flow counts or labels differ from manifest")
            if any(r["ipv4_src_addr"] != row["host"] for r in original):
                raise AssertionError("Raw source differs from manifest")
            old, additions, _ = extract(original, row["start"], schema)
            base.append(old)
            for key in extras:
                extras[key].append(additions[key])
            update_quality(quality, original, row["target"])
            if (i + 1) % 500 == 0:
                print(f"Extracted {i+1}/{len(rows)} sequences in {time.monotonic()-begun:.1f}s", flush=True)
        base = np.stack(base)
        extras = {key: np.stack(values) for key, values in extras.items()}
        np.savez_compressed(out / "training_feature_cache.npz", sample_id=np.array([r["id"] for r in rows]), old_aggregates=base, **extras)
        write_json(out / "candidate_definitions.json", {"candidate_names": candidate_names(), "original_numeric_names": bf.NUMERIC_NAMES,
            "families": FAMILY_FIELDS, "dns_type_codes": DNS_TYPES, "icmp_type_codes": ICMP_TYPES,
            "dns_eligibility": "nonzero query metadata OR destination port 53", "icmp_eligibility": "protocol 1",
            "geometry": "100 seconds, ten 10-second buckets", "new_scalar_summary": "mean over active buckets for univariate screening"})
        quality_rows = list(quality.values())
        write_csv(out / "omitted_field_quality.csv", quality_rows)
        results = probes(rows, base, extras, out, args.trees)
        collisions = collision_diagnostic(manifest, work, raw, schema, args.audit, out)
        write_report(out, info, quality_rows, results, collisions)
        write_json(out / "provenance.json", {"cohort": info, "environment": environment(), "limitations": LIMITATIONS,
            "source": {"path": str(args.db.resolve()), "bytes": args.db.stat().st_size, "mtime_ns": args.db.stat().st_mtime_ns},
            "prepared_manifest_sha256": sha256(args.prepared / "manifest.json"), "script_sha256": sha256(Path(__file__)),
            "training_reused": True, "test_used": False, "formal_model_trained": False, "validation_used_for_selection": False})
        write_json(out / "status.json", {"status": "complete", "test_used": False})
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error), "test_used": False})
        raise
    finally:
        for connection in (manifest, work, raw):
            connection.close()
    print(str(out), flush=True)
    return out


if __name__ == "__main__":
    main()
