"""Trace all pure Infilteration validation misses to flows and frozen features."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "train/cache/diagnostic_plot_dependencies"))

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.common.io import fresh, read_json, sha256, unique_name, write_json
from CIC_new.preprocessing.dataset import Originals
from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.run_preprocessing import raw_bucket, NUMERIC_SCHEMA
from CIC_new.representation_learning.bucket_autoencoder import BucketAE

FIELDS = ("protocol", "l7", "service")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pre_features(original, indices):
    arrays = original.arrays
    return np.concatenate([arrays[key][indices].reshape(len(indices), -1) for key in
                           ("numeric", "hist_protocol", "hist_l7", "hist_service", "mask")], axis=1)


def pre_hash(row, counts):
    digest = hashlib.sha256(canonical_float_bytes(row))
    digest.update(counts.tobytes())
    return digest.digest()


def canonical_float_bytes(row):
    values = row.copy()
    values[values == 0] = 0  # +0 and -0 are numerically equal model inputs.
    return values.tobytes()


def normalize_rows(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def main():
    run = PROJECT / "train/lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc"
    dataset = PROJECT / "datasets/dataset"
    summary = read_json(run / "summary.json")
    threshold = read_json(run / "thresholds.json")["values"]["validation_fpr_1pct"]
    if sha256(run / "model.pt") != summary["model_sha256"] or sha256(run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Frozen source changed")
    with np.load(run / "predictions.npz", allow_pickle=False) as a:
        ids, labels, attacks, scores, hosts = [a[k].copy() for k in ("sample_id", "y", "primary_attack", "score", "source_host")]
    flow_index = read_json(dataset / "flow_metadata.json")
    record = flow_index["splits"]["validation"]
    if sha256(dataset / record["path"]) != record["sha256"]:
        raise ValueError("Flow sidecar changed")
    with np.load(dataset / record["path"], allow_pickle=False) as a:
        order = np.argsort(a["sample_id"])
        positions = np.searchsorted(a["sample_id"][order], ids)
        if not np.array_equal(a["sample_id"][order][positions], ids):
            raise ValueError("Flow sidecar alignment mismatch")
        total, malicious = [a[k][order][positions].copy() for k in ("total_flows", "attack_flows")]
    pure = (attacks == "Infilteration") & (total == malicious) & (labels == 1)
    tp, fn = pure & (scores >= threshold), pure & (scores < threshold)
    if (int(pure.sum()), int(fn.sum()), int(tp.sum())) != (138, 34, 104):
        raise ValueError("Unexpected frozen pure-Infilteration cohort")
    output = fresh(PROJECT / "train/runs" / unique_name("pure_infilteration_raw_flow_audit"))
    write_json(output / "status.json", {"complete": False})
    prepared = Path(flow_index["prepared"])
    if not (prepared / "mmap/validation/identity.json").exists():
        raise ValueError("Expected an existing pre-encoder validation cache")
    original = Originals(prepared, "validation")
    try:
        original_positions = np.searchsorted(original.meta["id"], ids)
        if not np.array_equal(original.meta["id"][original_positions], ids):
            raise ValueError("Original cohort alignment mismatch")
        cache = PROJECT / f"train/cache/v1_{summary['dataset_manifest_sha256'][:20]}/validation"
        if not np.array_equal(np.load(cache / "ids.npy", mmap_mode="r"), ids):
            raise ValueError("Input cache alignment mismatch")
        x, mask = [np.load(cache / f"{key}.npy", mmap_mode="r") for key in ("x", "mask")]
        active = mask.sum(axis=1)
        target_indices = np.flatnonzero(pure)
        target_pre = pre_features(original, original_positions[target_indices])
        target_inputs = np.concatenate((x[target_indices].reshape(len(target_indices), -1), mask[target_indices]), axis=1)
        stage_targets = {"scaled_pre_encoder": defaultdict(list), "scheme2_input": defaultdict(list)}
        for i, position in enumerate(target_indices):
            stage_targets["scaled_pre_encoder"][pre_hash(target_pre[i], original.arrays["flow_counts"][original_positions[position]])].append(int(ids[position]))
            stage_targets["scheme2_input"][hashlib.sha256(canonical_float_bytes(target_inputs[i])).digest()].append(int(ids[position]))
        matches = {name: {int(ids[i]): [] for i in target_indices} for name in stage_targets}
        for start in range(0, len(ids), 2048):
            selected = np.flatnonzero(labels[start:start + 2048] == 0) + start
            p = original_positions[selected]
            matrix = pre_features(original, p)
            counts = original.arrays["flow_counts"][p]
            encoded = np.concatenate((x[selected].reshape(len(selected), -1), mask[selected]), axis=1)
            for j, position in enumerate(selected):
                for name, digest in (("scaled_pre_encoder", pre_hash(matrix[j], counts[j])),
                                     ("scheme2_input", hashlib.sha256(canonical_float_bytes(encoded[j])).digest())):
                    for sid in stage_targets[name].get(digest, []):
                        matches[name][sid].append(int(ids[position]))
            if start % (25 * 2048) == 0:
                print(f"Exact-feature search: {min(start + 2048, len(ids)):,}/{len(ids):,} validation rows", flush=True)

        # Match on coarse observation conditions, never on attack scores/features.
        controls = []
        lookup = {int(sid): i for i, sid in enumerate(ids)}
        true_positions = np.flatnonzero(tp)
        for position in np.flatnonzero(fn):
            exact = true_positions[(hosts[true_positions] == hosts[position]) & (active[true_positions] == active[position]) &
                                   (np.abs(np.log2(total[true_positions] / total[position])) <= 1)]
            same_active = true_positions[active[true_positions] == active[position]]
            candidates = exact if len(exact) else (same_active if len(same_active) else true_positions)
            cost = np.abs(np.log2(total[candidates] / total[position]))
            chosen = int(candidates[np.argmin(cost)])
            controls.append({"fn_id": int(ids[position]), "tp_id": int(ids[chosen]),
                             "strict_same_host_active_and_volume_factor2": bool(len(exact)),
                             "same_host": bool(hosts[position] == hosts[chosen]),
                             "same_active_buckets": bool(active[position] == active[chosen]),
                             "fn_flows": int(total[position]), "tp_flows": int(total[chosen]),
                             "fn_score": float(scores[position]), "tp_score": float(scores[chosen])})

        # Reuse the exact earlier visualization cohort to keep candidate sampling fixed.
        vis = PROJECT / "train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/lstm"
        with (vis / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
            vis_ids = np.array([int(r["sample_id"]) for r in csv.DictReader(stream, delimiter="\t")])
        vis_positions = np.array([lookup[int(sid)] for sid in vis_ids])
        hidden = np.loadtxt(vis / "vectors.tsv", delimiter="\t", dtype=np.float32)
        query_vis = np.flatnonzero(np.isin(vis_ids, ids[pure]))
        matrices = {"scaled_pre_encoder": normalize_rows(pre_features(original, original_positions[vis_positions]).astype(np.float64)),
                    "scheme2_input": normalize_rows(np.concatenate((x[vis_positions].reshape(len(vis_positions), -1), mask[vis_positions]), axis=1).astype(np.float64)),
                    "lstm_hidden": normalize_rows(hidden.astype(np.float64))}
        neighbor_rows, nearest_benign = [], {}
        for name, matrix in matrices.items():
            model = NearestNeighbors(n_neighbors=11, algorithm="brute", n_jobs=4).fit(matrix)
            indices = model.kneighbors(matrix[query_vis], return_distance=False)
            benign_model = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=4).fit(matrix[labels[vis_positions] == 0])
            benign_vis = np.flatnonzero(labels[vis_positions] == 0)
            distance, closest = benign_model.kneighbors(matrix[query_vis])
            for j, (own, peers) in enumerate(zip(query_vis, indices)):
                peers = peers[peers != own][:10]
                sid = int(vis_ids[own])
                nearest_benign[(sid, name)] = int(vis_ids[benign_vis[closest[j, 0]]])
                neighbor_rows.append({"sample_id": sid, "outcome": "FN" if fn[lookup[sid]] else "TP", "stage": name,
                                      "benign_fraction_in_10nn": float((labels[vis_positions[peers]] == 0).mean()),
                                      "nearest_benign_id": nearest_benign[(sid, name)], "nearest_benign_distance": float(distance[j, 0])})

        raw_connection = sqlite3.connect((prepared / "work.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        meta_connection = sqlite3.connect((prepared / "manifest.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        raw_cache = {}
        def raw_sequence(sid):
            if sid in raw_cache:
                return raw_cache[sid]
            info = meta_connection.execute("SELECT window_start_ms,split,flows,attack_flows FROM original_sequences WHERE sequence_id=?", (sid,)).fetchone()
            if info is None or info[1] != "validation":
                raise ValueError("Raw audit attempted to read outside validation")
            rows = raw_connection.execute("SELECT " + ",".join(bf.EVENT_COLUMNS) + " FROM events WHERE sequence_id=? ORDER BY flow_start_milliseconds,raw_rowid", (sid,)).fetchall()
            events = bf.local_events(np.asarray(rows, dtype=np.int64), info[0])
            values, hist = raw_bucket(events)
            p = int(np.searchsorted(original.meta["id"], sid))
            rebuilt = bf.normalize(values["numeric"], values["mask"], NUMERIC_SCHEMA, original.scaler if hasattr(original, "scaler") else read_json(prepared / "scaler.json"))
            if not np.array_equal(rebuilt, original.arrays["numeric"][p]):
                raise ValueError(f"Re-aggregated/scaled numeric mismatch: {sid}")
            if len(events) != info[2] or int(events[:, bf.E["label"]].sum()) != info[3]:
                raise ValueError("Raw labels or flow counts do not reconcile")
            for field in FIELDS:
                dense = np.zeros_like(original.arrays["hist_" + field][p])
                mapping = {int(code): i + 1 for i, code in enumerate(original.schema["vocabularies"][field])}
                for bucket, code, count in hist[field]:
                    dense[int(bucket), mapping.get(int(code), 0)] += count
                denom = dense.sum(axis=1, keepdims=True)
                np.divide(dense, denom, out=dense, where=denom > 0)
                if not np.array_equal(dense, original.arrays["hist_" + field][p]):
                    raise ValueError("Rebuilt histogram mismatch")
            raw_cache[sid] = (events, values, hist)
            return raw_cache[sid]

        raw_rows, collision_rows, field_differences = [], [], []
        original_schema = original.schema
        selected_raw = set(map(int, ids[pure]))
        selected_raw.update(nearest_benign.values())
        for name in matches:
            selected_raw.update(v[0] for v in matches[name].values() if v)
        try:
            for number, sid in enumerate(sorted(selected_raw)):
                events, values, hist = raw_sequence(sid)
                column = lambda key: events[:, bf.E[key]]
                dns = (column("dns_query_id") != 0) | (column("dns_query_type") != 0) | (column("dns_ttl_answer") != 0)
                position = lookup[sid]
                group = "pure_FN" if fn[position] else ("pure_TP" if tp[position] else "benign_reference")
                raw_rows.append({"sample_id": sid, "group": group, "score": float(scores[position]),
                                 "source_host_hash": hashlib.sha256(str(hosts[position]).encode()).hexdigest()[:16],
                                 "flows": len(events), "active_buckets": int(values["mask"].sum()),
                                 "tcp_fraction": float((column("protocol") == 6).mean()),
                                 "udp_fraction": float((column("protocol") == 17).mean()),
                                 "icmp_fraction": float((column("protocol") == 1).mean()),
                                 "dns_fraction": float(dns.mean()),
                                 "dst_port_53_fraction": float((column("l4_dst_port") == 53).mean()),
                                 "web_port_fraction": float(np.isin(column("l4_dst_port"), [80, 443]).mean()),
                                 "dst_ports": json.dumps(sorted(set(map(int, column("l4_dst_port"))))),
                                 "l7_codes": json.dumps(sorted(set(map(int, column("l7_proto"))))),
                                 "bytes_total": int((column("in_bytes") + column("out_bytes")).sum()),
                                 "packets_total": int((column("in_pkts") + column("out_pkts")).sum()),
                                 "zero_reply_flows": int((column("out_pkts") == 0).sum()),
                                 "mean_duration_ms": float(column("flow_duration_milliseconds").mean()),
                                 "rst_flow_fraction": float(((column("tcp_flags") & 4) > 0).mean()),
                                 "syn_flow_fraction": float(((column("tcp_flags") & 2) > 0).mean()),
                                 "unique_destinations": len(np.unique(column("destination_id"))),
                                 "unique_source_ports": len(np.unique(column("l4_src_port")))})
                if number % 50 == 0:
                    print(f"Raw-flow reconstruction: {number + 1}/{len(selected_raw)} selected validation sequences", flush=True)
            for sid in map(int, ids[pure]):
                for name in matches:
                    peers = matches[name][sid]
                    if not peers:
                        continue
                    benign_id = peers[0]
                    a, av, ah = raw_sequence(sid)
                    b, bv, bh = raw_sequence(benign_id)
                    numeric_equal = bool(np.array_equal(av["numeric"], bv["numeric"]))
                    hist_equal = all(np.array_equal(ah[f], bh[f]) for f in FIELDS)
                    differences = [bf.NUMERIC_NAMES[i] for i in range(82) if np.any(av["numeric"][:, i] != bv["numeric"][:, i])]
                    collision_rows.append({"sample_id": sid, "outcome": "FN" if fn[lookup[sid]] else "TP", "stage": name,
                                           "benign_match_n": len(peers), "example_benign_id": benign_id,
                                           "unscaled_numeric_exact": numeric_equal, "original_category_counts_exact": hist_equal,
                                           "unscaled_aggregate_and_histograms_exact": numeric_equal and hist_equal,
                                           "differing_unscaled_numeric_names": json.dumps(differences)})
                    if numeric_equal and hist_equal and a.shape == b.shape:
                        for field in bf.EVENT_COLUMNS:
                            if not np.array_equal(a[:, bf.E[field]], b[:, bf.E[field]]):
                                field_differences.append({"sample_id": sid, "benign_id": benign_id, "stage": name,
                                                          "field": field, "infilteration_values": json.dumps(a[:, bf.E[field]].tolist()),
                                                          "benign_values": json.dumps(b[:, bf.E[field]].tolist())})
        finally:
            raw_connection.close()
            meta_connection.close()

        # Re-encode existing numeric/histogram inputs with the frozen bucket AE.
        checkpoint_path = dataset / "frozen_bucket_encoder.pt"
        if sha256(checkpoint_path) != read_json(dataset / "manifest.json")["embedding_checkpoint_sha256"]:
            raise ValueError("Bucket encoder hash mismatch")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        bucket_model = BucketAE(**saved["model_args"]).eval()
        bucket_model.load_state_dict(saved["model_state"])
        torch.set_num_threads(4)
        p = original_positions[target_indices]
        with torch.no_grad():
            z = bucket_model.encode(torch.from_numpy(np.array(original.arrays["numeric"][p])),
                                    {f: torch.from_numpy(np.array(original.arrays["hist_" + f][p])) for f in FIELDS})
            z = F.normalize(z, p=2, dim=-1, eps=1e-8).numpy() * mask[target_indices, :, None]
        z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
        encoder_error = float(np.max(np.abs(z - x[target_indices, :, :64])))
        if encoder_error > 1e-5:
            raise ValueError(f"Frozen encoder reconstruction mismatch: {encoder_error}")

        all_models = read_json(PROJECT / "train/comparisons/review_20261024_four_models_seed4266/comparison.json")["rows"]
        model_decisions = {}
        for row in all_models:
            other = Path(row["run"])
            t = read_json(other / "thresholds.json")["values"]["validation_fpr_1pct"]
            with np.load(other / "predictions.npz", allow_pickle=False) as a:
                if not np.array_equal(a["sample_id"], ids):
                    raise ValueError("Cross-model cohort mismatch")
                model_decisions[row["model"]] = a["score"] >= t
        case_rows = []
        raw_by_id = {r["sample_id"]: r for r in raw_rows}
        for sid in map(int, ids[fn]):
            position = lookup[sid]
            record = dict(raw_by_id[sid])
            record.update({f"benign_collision_n_{name}": len(matches[name][sid]) for name in matches})
            record.update({f"detected_{name}": bool(decision[position]) for name, decision in model_decisions.items()})
            record["all_four_models_missed"] = all(not decision[position] for decision in model_decisions.values())
            case_rows.append(record)
        collision_summary = []
        for group, keep in (("pure_FN", fn), ("pure_TP", tp)):
            selected_ids = ids[keep].tolist()
            for name in matches:
                collision_summary.append({"group": group, "stage": name, "n": len(selected_ids),
                                          "with_exact_benign_counterpart": sum(bool(matches[name][sid]) for sid in selected_ids),
                                          "counterpart_search_benign_n": int((labels == 0).sum())})
        neighbor_summary = []
        for name in matrices:
            for outcome in ("FN", "TP"):
                selected = [r for r in neighbor_rows if r["stage"] == name and r["outcome"] == outcome]
                neighbor_summary.append({"stage": name, "outcome": outcome, "n": len(selected),
                                         "mean_benign_fraction_in_10nn": float(np.mean([r["benign_fraction_in_10nn"] for r in selected]))})
        raw_summary = []
        for group in ("pure_FN", "pure_TP"):
            group_rows = [r for r in raw_rows if r["group"] == group]
            raw_summary.append({"group": group, "n": len(group_rows),
                                "source_hosts": len(set(r["source_host_hash"] for r in group_rows)),
                                "median_flows": float(np.median([r["flows"] for r in group_rows])),
                                "median_active_buckets": float(np.median([r["active_buckets"] for r in group_rows])),
                                "all_tcp_sequence_n": sum(r["tcp_fraction"] == 1 for r in group_rows),
                                "all_udp_sequence_n": sum(r["udp_fraction"] == 1 for r in group_rows),
                                "all_dns_sequence_n": sum(r["dns_fraction"] == 1 for r in group_rows),
                                "all_web_ports_sequence_n": sum(r["web_port_fraction"] == 1 for r in group_rows),
                                "median_bytes": float(np.median([r["bytes_total"] for r in group_rows])),
                                "median_packets": float(np.median([r["packets_total"] for r in group_rows])),
                                "median_saved_score": float(np.median([r["score"] for r in group_rows]))})
        for name, rows in (("pure_fn_cases.csv", case_rows), ("pure_fn_matched_tp.csv", controls),
                           ("raw_flow_profiles.csv", raw_rows), ("stage_exact_collisions.csv", collision_rows),
                           ("raw_fields_lost_by_aggregation.csv", field_differences), ("stage_neighbors.csv", neighbor_rows)):
            if rows:
                csv_write(output / name, rows)
        strict_controls = sum(r["strict_same_host_active_and_volume_factor2"] for r in controls)
        report = {"complete": True, "scope": "Exploratory raw-flow audit after validation selection; not independent test or causal attribution",
                  "test_rows_read": False, "training_rows_used": False, "model_or_threshold_refitted": False,
                  "pure_fn_n": int(fn.sum()), "pure_tp_n": int(tp.sum()), "collision_summary": collision_summary,
                  "raw_behavior_summary": raw_summary, "neighbor_summary": neighbor_summary,
                  "strict_matched_control_n": strict_controls, "control_n": len(controls),
                  "unique_tp_controls": len(set(r["tp_id"] for r in controls)),
                  "all_four_missed_among_pure_lstm_fn": sum(r["all_four_models_missed"] for r in case_rows),
                  "raw_sequences_reaggregated_and_verified": len(raw_cache),
                  "frozen_bucket_encoder_max_absolute_error": encoder_error,
                  "model_sha256": summary["model_sha256"], "thresholds_sha256": summary["thresholds_sha256"],
                  "collisions": collision_rows, "limitations": [
                      "Complete validation benign cohort is used for exact collisions; no conclusion is made about raw label correctness without attack-ground-truth verification.",
                      "Continuous nearest-neighbor comparisons use the prior fixed visualization cohort of 4,000 benign plus 917 attacks; rates depend on its artificial class ratio, metric and feature scales.",
                      "Numeric scaling uses log1p and standardization without clipping; float32 rounding can still merge very close values.",
                      "Matched TP examples are selected conditional on outcome, can be reused and include fallback matches. They are descriptive controls, not independent pairs or causal estimates.",
                      "Raw profiling reports recorded NetFlow fields, not payload evidence or verified phases of an intrusion campaign."]}
        write_json(output / "audit.json", report)
        lines = ["# Pure Infilteration misses: raw-flow and feature audit", "",
                 "All 34 pure Infilteration misses and all 104 pure Infilteration detections at the frozen LSTM 1% validation-FPR threshold. Every flow in these 138 sequences is labeled malicious in the existing data. Labels are audited for consistency, not redefined.",
                 "", "## Full validation exact-feature collisions", "",
                 "The counterpart search covers all 250,229 real validation benign sequences, not just the visualization sample.",
                 "| Group | Stage | N | Has identical benign counterpart |", "|---|---|---:|---:|"]
        for r in collision_summary:
            lines.append(f"| {r['group']} | {r['stage']} | {r['n']} | {r['with_exact_benign_counterpart']} |")
        lines += ["", "## Raw behavior", "", "```json", json.dumps(raw_summary, indent=2), "```", "",
                  "## Stage-wise neighborhoods", "", "| Stage | Group | Mean benign fraction among 10 neighbors |", "|---|---|---:|"]
        for r in neighbor_summary:
            lines.append(f"| {r['stage']} | {r['outcome']} | {r['mean_benign_fraction_in_10nn']:.2%} |")
        lines += ["", "## Controls and integrity", "",
                  f"{strict_controls}/34 controls match source host, active-bucket count and total flows within a factor of two. Fallbacks match active buckets where possible, then nearest log volume. There are {report['unique_tp_controls']} unique TP controls; repeated controls are not independent observations.",
                  f"All {len(raw_cache)} queried sequences belong to validation. Their scaled numeric features, category histograms, masks and label/flow counts were rebuilt from exact staged flow rows and reconciled with frozen artifacts. Frozen bucket encoder error: {encoder_error:.3g}.",
                  f"{report['all_four_missed_among_pure_lstm_fn']}/34 pure LSTM misses are also missed by CNN, hybrid and XGBoost at their own frozen 1% thresholds.",
                  "Exact input collisions with opposite labels are unresolvable for a deterministic classifier using only those inputs. This is evidence about available representation, not automatic proof of labeling error. See raw_fields_lost_by_aggregation.csv to identify event-level differences hidden by aggregation.",
                  "", "## Limitations", ""]
        lines += ["- " + v for v in report["limitations"]]
        lines += ["", "CSV files preserve all pure FN cases, raw profiles, matched controls, exact collisions, changed raw fields and per-case neighborhoods. audit.json records all collision details.", ""]
        (output / "PURE_INFILTERATION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
        stages = list(matrices)
        yy = np.arange(len(stages))
        for offset, outcome, color in ((-.18, "TP", "#168774"), (.18, "FN", "#dc4c55")):
            means = [100 * next(r["mean_benign_fraction_in_10nn"] for r in neighbor_summary if r["stage"] == s and r["outcome"] == outcome) for s in stages]
            axes[0].barh(yy + offset, means, height=.34, label=outcome, color=color)
        axes[0].set_yticks(yy, stages)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Mean benign share among 10 neighbors (%)")
        axes[0].set_title("All pure Infilteration sequences")
        axes[0].legend()
        for group, color in (("pure_TP", "#168774"), ("pure_FN", "#dc4c55")):
            values = [r["score"] for r in raw_rows if r["group"] == group]
            axes[1].hist(values, bins=np.linspace(0, 1, 26), histtype="step", density=True, label=group, lw=2, color=color)
        axes[1].axvline(threshold, color="#444444", ls="--", label="Frozen 1% FPR threshold")
        axes[1].set_xlabel("Saved LSTM attack score")
        axes[1].set_ylabel("Density within group")
        axes[1].set_title("Pure attack labels do not guarantee separability")
        axes[1].legend(fontsize=8)
        fig.savefig(output / "pure_infilteration_audit.png", dpi=170)
        plt.close(fig)
        write_json(output / "status.json", {"complete": True, "test_rows_read": False, "training_rows_used": False})
        print(json.dumps({"output": str(output), **{k: v for k, v in report.items() if k != "collisions"}}, indent=2), flush=True)
    finally:
        original.close()


if __name__ == "__main__":
    main()
