"""Trace changed Infilteration validation decisions with frozen-model diagnostics."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.distribution_features import NAMES, aggregate, logged
from CIC_new.preprocessing.run_preprocessing import raw_bucket, NUMERIC_SCHEMA, selected_distribution
from CIC_new.representation_learning.bucket_autoencoder import BucketAE
from train.common.io import ROOT, fresh, read_json, write_json, sha256, unique_name
from train.common.torch_runner import load
from train.run_circular_shift import write_csv
from train.run_distribution_feature_ablation import verified

POINTS = ("validation_fpr_1pct", "validation_fpr_0.1pct")
FIELDS = ("protocol", "l7", "service")


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def predictions(folder):
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def components(events, schema, scaler):
    if len(events):
        values, hist = raw_bucket(events)
        extra = aggregate(events)
        histograms = {f: selected_distribution(hist[f], np.array([0]), schema["vocabularies"][f], 1)[0]
                      for f in FIELDS}
    else:
        values = {"numeric": np.zeros((10, 82)), "mask": np.zeros(10, np.uint8),
                  "flow_counts": np.zeros(10, np.int64), "attack_flow_counts": np.zeros(10, np.int64)}
        extra = np.zeros((10, 15))
        histograms = {f: np.zeros((10, schema["category_slots"][f]), np.float32) for f in FIELDS}
    old_scaler = {key: scaler[key][:82] for key in ("mean", "scale")}
    old = bf.normalize(values["numeric"], values["mask"], NUMERIC_SCHEMA, old_scaler)
    added = (logged(extra) - np.asarray(scaler["mean"])[82:]) / np.asarray(scaler["scale"])[82:]
    added[values["mask"] == 0] = 0
    return {"numeric": np.concatenate((old, added.astype(np.float32)), axis=-1),
            "mask": values["mask"].astype(np.float32), "flow_counts": values["flow_counts"],
            "attack_flow_counts": values["attack_flow_counts"], **{"hist_" + f: histograms[f] for f in FIELDS}}


def stack(records):
    return {key: np.stack([r[key] for r in records]) for key in records[0]}


@torch.inference_mode()
def encode(encoder, values, width, scaler):
    device = next(encoder.parameters()).device
    numeric = torch.from_numpy(values["numeric"][..., :width]).to(device)
    hist = {f: torch.from_numpy(values["hist_" + f]).to(device) for f in FIELDS}
    mask = torch.from_numpy(values["mask"]).to(device)
    z = F.normalize(encoder.encode(numeric, hist), p=2, dim=-1, eps=1e-8) * mask.unsqueeze(-1)
    intensity = ((np.log1p(values["flow_counts"].astype(np.float32)) - scaler["mean"][0]) /
                 scaler["scale"][0]) * values["mask"]
    x = torch.cat((z, torch.from_numpy(intensity).to(device).unsqueeze(-1)), dim=-1)
    return x, mask


@torch.inference_mode()
def score(model, encoded):
    return torch.sigmoid(model(*encoded)).cpu().numpy().astype(np.float64)


def benign_tail(scores, benign_scores):
    ordered = np.sort(benign_scores)
    return (len(ordered) - np.searchsorted(ordered, scores, side="left")) / len(ordered)


def outcome(old, new):
    return "retained_tp" if old and new else "lost" if old else "recovered" if new else "persistent_fn"


def median(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.median(values)) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args(argv)
    batch = args.batch.resolve()
    state = read_json(batch / "batch.json")
    if state["status"] != "complete":
        raise ValueError("Use a completed distribution-feature batch")
    old_run = verified(state["settings"]["reference"])
    new_run = verified(state["cases"]["distribution97"]["lstm"])
    prepared = Path(state["settings"]["prepared"])
    extended = verified(state["cases"]["distribution97"]["prepared"])
    representation = verified(state["cases"]["distribution97"]["representation"])
    schema, scaler = read_json(extended / "schema.json"), read_json(extended / "scaler.json")
    old, new = predictions(old_run), predictions(new_run)
    for key in ("sample_id", "y", "primary_attack", "source_host"):
        np.testing.assert_array_equal(old[key], new[key])
    selected = np.flatnonzero((old["y"] == 1) & (old["primary_attack"] == "Infilteration"))
    ids = old["sample_id"][selected]
    thresholds = [read_json(p / "thresholds.json")["values"] for p in (old_run, new_run)]
    tails = [benign_tail(pred["score"][selected], pred["score"][pred["y"] == 0]) for pred in (old, new)]
    connection = readonly(prepared / "manifest.sqlite")
    try:
        meta = {}
        attack_membership = {}
        locations = defaultdict(list)
        for sid in ids:
            meta[int(sid)] = connection.execute("SELECT source_ip,window_start_ms,flows,attack_flows,active_buckets,split FROM original_sequences WHERE sequence_id=?", (int(sid),)).fetchone()
            attack_membership[int(sid)] = dict(connection.execute("SELECT attack,flows FROM sequence_attacks WHERE sequence_id=?", (int(sid),)))
            split, shard, position = connection.execute("SELECT split,shard,row_index FROM v3_locations WHERE sample_id=?", (int(sid),)).fetchone()
            if split != "validation" or meta[int(sid)][-1] != "validation":
                raise ValueError("Non-validation flow access refused")
            locations[shard].append((int(sid), position))
    finally:
        connection.close()
    work = readonly(prepared / "work.sqlite")
    grouped = defaultdict(list)
    try:
        fields = ",".join(bf.EVENT_COLUMNS)
        query = "SELECT sequence_id," + fields + " FROM events WHERE sequence_id IN (" + ",".join("?" for _ in ids) + ") ORDER BY sequence_id,flow_start_milliseconds,raw_rowid"
        for row in work.execute(query, list(map(int, ids))):
            grouped[int(row[0])].append(row[1:])
    finally:
        work.close()
    print(f"Selected {len(ids)} real-validation sequences and {sum(map(len, grouped.values()))} staged flows", flush=True)
    variants = {key: [] for key in ("original", "attack_only", "benign_only")}
    sequence_rows, bucket_rows, flow_rows = [], [], []
    for sid, position in zip(ids, selected):
        host, start, total, attacks, active, _ = meta[int(sid)]
        events = bf.local_events(np.asarray(grouped[int(sid)], np.int64), start)
        attack = events[:, bf.E["label"]] == 1
        if len(events) != total or attack.sum() != attacks:
            raise ValueError("Staged flow labels/counts differ from original metadata")
        flow_profile = {}
        for name, keep in (("attack", attack), ("benign", ~attack)):
            flows = events[keep]
            record = {"sample_id": int(sid), "flow_label_group": name, "n": len(flows)}
            for family, proto, port in (("udp_53", 17, 53), ("tcp_3389", 6, 3389)):
                count = int(((flows[:, bf.E["protocol"]] == proto) & (flows[:, bf.E["l4_dst_port"]] == port)).sum())
                record[family + "_flows"] = count
                flow_profile[name + "_" + family + "_fraction"] = count / len(flows) if len(flows) else None
            for field in ("in_bytes", "out_bytes", "in_pkts", "out_pkts", "flow_duration_milliseconds"):
                record[field + "_median"] = float(np.median(flows[:, bf.E[field]])) if len(flows) else None
            flow_rows.append(record)
        family = "UDP/53" if flow_profile["attack_udp_53_fraction"] > .5 else "TCP/3389" if flow_profile["attack_tcp_3389_fraction"] > .5 else "other"
        for name, keep in (("original", np.ones(len(events), bool)), ("attack_only", attack), ("benign_only", ~attack)):
            variants[name].append(components(events[keep], schema, scaler))
        original = variants["original"][-1]
        counts, ac = original["flow_counts"], original["attack_flow_counts"]
        mixed = (ac > 0) & (ac < counts)
        attack_buckets = ac > 0
        sequence_rows.append({"sample_id": int(sid), "host": host, "window_start_ms": start,
            "attack_flow_family": family, **flow_profile,
            "infilteration_labeled_flows": attack_membership[int(sid)].get("Infilteration", 0),
            "total_flows": total, "attack_flows": attacks, "attack_fraction": attacks / total,
            "active_buckets": active, "attack_buckets": int(attack_buckets.sum()),
            "mixed_buckets": int(mixed.sum()), "benign_only_buckets": int(((counts > 0) & ~attack_buckets).sum()),
            "pure_attack_buckets": int(((ac == counts) & attack_buckets).sum()),
            "mixed_bucket_attack_fraction": float(ac[mixed].sum() / counts[mixed].sum()) if mixed.any() else None,
            "attack_bytes_fraction": float(events[attack][:, [bf.E["in_bytes"], bf.E["out_bytes"]]].sum(dtype=np.float64) /
                                           max(events[:, [bf.E["in_bytes"], bf.E["out_bytes"]]].sum(dtype=np.float64), 1)),
            "old_score": float(old["score"][position]), "new_score": float(new["score"][position]),
            "old_benign_tail": float(tails[0][len(sequence_rows)]), "new_benign_tail": float(tails[1][len(sequence_rows)]),
            "added_feature_abs_z_attack_bucket_mean": float(np.abs(original["numeric"][attack_buckets, 82:]).mean())})
        np.testing.assert_equal(active, int(original["mask"].sum()))
        for bucket in np.flatnonzero(counts):
            row = {"sample_id": int(sid), "bucket": int(bucket), "total_flows": int(counts[bucket]),
                   "attack_flows": int(ac[bucket]), "attack_fraction": float(ac[bucket] / counts[bucket])}
            row.update({name: float(original["numeric"][bucket, 82 + j]) for j, name in enumerate(NAMES)})
            bucket_rows.append(row)
    variants = {name: stack(records) for name, records in variants.items()}
    order = {int(sid): i for i, sid in enumerate(ids)}
    # Reconcile all recomputed fields with both completed preparations.
    for shard, values in locations.items():
        with np.load(prepared / "splits/validation" / shard, allow_pickle=False) as a, np.load(extended / "splits/validation" / shard, allow_pickle=False) as b:
            for sid, p in values:
                i = order[sid]
                np.testing.assert_equal(a["sample_id"][p], sid)
                np.testing.assert_equal(b["sample_id"][p], sid)
                for key in variants["original"]:
                    actual = variants["original"][key][i]
                    np.testing.assert_allclose(actual, b[key][p], atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(actual[..., :82] if key == "numeric" else actual, a[key][p], atol=1e-6, rtol=1e-6)
    print("Original/extended preparation reconciliation passed", flush=True)
    torch.set_num_threads(2)
    encoders, detectors, encoded, scored = {}, {}, {}, {}
    checkpoints = {"old": prepared.parent / "bucket_embeddings/best.pt", "new": representation / "bucket_embeddings/best.pt"}
    old_manifest = read_json(Path(read_json(old_run / "config.json")["dataset"]) / "manifest.json")
    if sha256(checkpoints["old"]) != old_manifest["embedding_checkpoint_sha256"]:
        raise ValueError("Original encoder does not match the reference dataset")
    reproduction = {}
    for label, run, width, pred in (("old", old_run, 82, old), ("new", new_run, 97, new)):
        device = read_json(run / "config.json")["device"]
        saved = torch.load(checkpoints[label], map_location="cpu", weights_only=True)
        encoder = BucketAE(**saved["model_args"]).to(device)
        encoder.load_state_dict(saved["model_state"])
        encoder.eval().requires_grad_(False)
        encoders[label] = encoder
        detectors[label] = load("lstm", read_json(run / "config.json"), run / "model.pt", device).eval().requires_grad_(False)
        encoded[label] = {name: encode(encoder, value, width, scaler) for name, value in variants.items()}
        scored[label] = {}
        for name in ("original", "attack_only", "benign_only"):
            x, mask = encoded[label][name]
            # Pure attacks have no benign-only input; leave those scores absent.
            valid = mask.sum(1).cpu().numpy() > 0
            result = np.full(len(ids), np.nan)
            result[valid] = score(detectors[label], (x[valid], mask[valid]))
            scored[label][name] = result
        # Selected-only batches can change floating-point reduction order.
        # Require tight score agreement and exact saved threshold decisions.
        np.testing.assert_allclose(scored[label]["original"], pred["score"][selected], atol=1e-5, rtol=0)
        reproduction[label] = {"max_absolute_error": float(np.max(np.abs(scored[label]["original"] - pred["score"][selected])))}
        for point in POINTS:
            np.testing.assert_array_equal(scored[label]["original"] >= thresholds[0 if label == "old" else 1][point], pred["score"][selected] >= thresholds[0 if label == "old" else 1][point])
        x, mask = encoded[label]["original"]
        keep = torch.from_numpy((variants["original"]["attack_flow_counts"] > 0).astype(np.float32)).to(x.device)
        scored[label]["drop_benign_only_buckets"] = score(detectors[label], (x * keep.unsqueeze(-1), mask * keep))
    # Compare mixture geometry only inside each encoder's own coordinate space.
    geometry = defaultdict(list)
    for row in bucket_rows:
        i, b = order[row["sample_id"]], row["bucket"]
        if not 0 < row["attack_fraction"] < 1:
            continue
        for label in ("old", "new"):
            z = {name: encoded[label][name][0][i, b, :64].cpu().numpy() for name in variants}
            row[label + "_cos_attack_only"] = float(np.dot(z["original"], z["attack_only"]))
            row[label + "_cos_benign_only"] = float(np.dot(z["original"], z["benign_only"]))
            row[label + "_benign_cos_advantage"] = row[label + "_cos_benign_only"] - row[label + "_cos_attack_only"]
            geometry[i, label].append(row[label + "_benign_cos_advantage"])
    # Every intervention is an oracle/OOD stress diagnostic, not new held-out recall.
    for name, channels in [("all_added_features_at_training_mean", list(range(82, 97)))] + [("mean_" + name, [82 + j]) for j, name in enumerate(NAMES)]:
        altered = dict(variants["original"])
        altered["numeric"] = altered["numeric"].copy()
        altered["numeric"][..., channels] = 0
        scored["new"][name] = score(detectors["new"], encode(encoders["new"], altered, 97, scaler))
    print("Frozen encoders/LSTMs reproduced saved decisions; stress scoring complete", flush=True)
    for label in ("old", "new"):
        all_benign = (old if label == "old" else new)["score"][old["y"] == 0]
        for name, scores in scored[label].items():
            rank = benign_tail(scores, all_benign)
            for i, row in enumerate(sequence_rows):
                row[label + "_" + name + "_score"] = float(scores[i]) if np.isfinite(scores[i]) else None
                row[label + "_" + name + "_benign_tail"] = float(rank[i]) if np.isfinite(scores[i]) else None
        for i, row in enumerate(sequence_rows):
            row[label + "_mixture_benign_cos_advantage"] = float(np.mean(geometry[i, label])) if geometry[i, label] else None
    for point in POINTS:
        for row in sequence_rows:
            row[point + "_outcome"] = outcome(row["old_score"] >= thresholds[0][point], row["new_score"] >= thresholds[1][point])
        for row in bucket_rows:
            row[point + "_outcome"] = sequence_rows[order[row["sample_id"]]][point + "_outcome"]
    for row in sequence_rows:
        for point in POINTS:
            row[point + "_baseline_seed_votes"] = 0
    for run in sorted(old_run.parents[2].glob("seed*/lstm/start0")):
        pred = predictions(run)
        np.testing.assert_array_equal(pred["sample_id"], old["sample_id"])
        limits = read_json(run / "thresholds.json")["values"]
        for point in POINTS:
            for i, row in enumerate(sequence_rows):
                row[point + "_baseline_seed_votes"] += int(pred["score"][selected[i]] >= limits[point])
    profile_keys = ("total_flows", "attack_flows", "attack_fraction", "attack_bytes_fraction", "active_buckets", "attack_buckets",
                    "mixed_buckets", "benign_only_buckets", "mixed_bucket_attack_fraction", "old_benign_tail", "new_benign_tail",
                    "added_feature_abs_z_attack_bucket_mean", "old_mixture_benign_cos_advantage", "new_mixture_benign_cos_advantage")
    profiles, stress, feature_profiles = [], [], []
    for point in POINTS:
        for group in ("lost", "recovered", "retained_tp", "persistent_fn"):
            rows = [r for r in sequence_rows if r[point + "_outcome"] == group]
            profile = {"operating_point": point, "group": group, "n": len(rows), "hosts": len({r["host"] for r in rows}),
                       "pure_sequences": sum(r["attack_fraction"] == 1 for r in rows),
                       "no_benign_only_bucket": sum(r["benign_only_buckets"] == 0 for r in rows),
                       "mixed_bucket_present": sum(r["mixed_buckets"] > 0 for r in rows),
                       "baseline_all_five_seeds_detected": sum(r[point + "_baseline_seed_votes"] == 5 for r in rows),
                       **{"median_" + key: median(rows, key) for key in profile_keys}}
            profiles.append(profile)
            for label in ("old", "new"):
                limit = thresholds[0 if label == "old" else 1][point]
                for name in scored[label]:
                    eligible = [r for r in rows if r[label + "_" + name + "_score"] is not None]
                    stress.append({"operating_point": point, "group": group, "model": label, "intervention": name,
                        "eligible_n": len(eligible), "above_original_threshold": sum(r[label + "_" + name + "_score"] >= limit for r in eligible),
                        "median_benign_tail": median(eligible, label + "_" + name + "_benign_tail")})
            for j, name in enumerate(NAMES):
                values = [np.abs(variants["original"]["numeric"][order[r["sample_id"]], :, 82 + j][variants["original"]["attack_flow_counts"][order[r["sample_id"]]] > 0]).mean() for r in rows]
                feature_profiles.append({"operating_point": point, "group": group, "feature": name,
                                         "median_abs_standardized_value_attack_buckets": float(np.median(values)) if values else None})
    out = fresh(batch / "reports" / unique_name("infilteration_changed_cases"))
    write_csv(out / "sequence_profiles.csv", sequence_rows)
    write_csv(out / "flow_label_profiles.csv", flow_rows)
    # Non-mixed buckets have no within-bucket counterfactual comparison.
    geometry_keys = [label + suffix for label in ("old", "new") for suffix in ("_cos_attack_only", "_cos_benign_only", "_benign_cos_advantage")]
    for row in bucket_rows:
        for key in geometry_keys:
            row.setdefault(key, None)
    write_csv(out / "bucket_profiles.csv", bucket_rows)
    write_csv(out / "group_profiles.csv", profiles)
    write_csv(out / "stress_diagnostics.csv", stress)
    write_csv(out / "added_feature_profiles.csv", feature_profiles)
    hosts = []
    for point in POINTS:
        for (group, host), count in Counter((r[point + "_outcome"], r["host"]) for r in sequence_rows).items():
            hosts.append({"operating_point": point, "group": group, "host": host, "n": count})
    write_csv(out / "host_profiles.csv", hosts)
    family_rows = []
    for point in POINTS:
        for group in ("lost", "recovered", "retained_tp", "persistent_fn"):
            for family in sorted({r["attack_flow_family"] for r in sequence_rows}):
                rows = [r for r in sequence_rows if r[point + "_outcome"] == group and r["attack_flow_family"] == family]
                family_rows.append({"operating_point": point, "group": group, "attack_flow_family": family,
                    "n": len(rows), "median_attack_fraction": median(rows, "attack_fraction"),
                    "median_attack_flows": median(rows, "attack_flows"),
                    "median_active_buckets": median(rows, "active_buckets"),
                    "old_attack_only_above_original_threshold": sum(r["old_attack_only_score"] >= thresholds[0][point] for r in rows),
                    "new_attack_only_above_original_threshold": sum(r["new_attack_only_score"] >= thresholds[1][point] for r in rows)})
    write_csv(out / "flow_family_profiles.csv", family_rows)
    result = {"complete": True, "n": len(ids), "staged_flows": sum(map(len, grouped.values())),
              "source_batch": str(batch), "score_reproduction": reproduction, "groups": profiles,
              "flow_families": family_rows,
              "stress_diagnostics": stress, "test_used": False, "training_started": False,
              "code_sha256": sha256(Path(__file__)),
              "limitations": ["Previously inspected real validation only; no fitting or test-feature access.",
                  "Completed original and new models reused the original training cohort.",
                  "Oracle deletion uses flow labels; altered input distributions and fixed original thresholds are stress diagnostics, not deployable recall improvements.",
                  "Setting features to standardized zero means training means and can produce off-manifold inputs; single-feature effects are not independent causal attributions.",
                  "Cosine comparisons use each encoder's own coordinates; no cross-encoder coordinate distance is interpreted.",
                  "Benign-tail ranks use original saved full-validation benign scores with ties counted above; altered benign distributions are not evaluated.",
                  "Pure/mixed refers to binary flow labels, not verified attack semantics.",
                  "One extended-feature encoder and detector seed cannot isolate stable feature effects from refitting."]}
    write_json(out / "analysis.json", result)
    lines = ["# Changed Infilteration validation cases", "", f"Inspected {len(ids)} original validation sequences and {result['staged_flows']} staged flows.",
             "All recomputed prepared features match both completed preparations; frozen inference on each run's recorded device reproduces saved decisions at both operating points.", "",
             "| FPR budget | Outcome | N | Hosts | Pure | Median attack fraction | Median active buckets | Mixed-bucket cases | No benign-only bucket | Old five-seed unanimous TP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in profiles:
        lines.append(f"| {r['operating_point']} | {r['group']} | {r['n']} | {r['hosts']} | {r['pure_sequences']} | {r['median_attack_fraction']:.3f} | {r['median_active_buckets']:.1f} | {r['mixed_bucket_present']} | {r['no_benign_only_bucket']} | {r['baseline_all_five_seeds_detected']} |")
    lines += ["", "## Oracle and feature-mean stress diagnostics", "",
              "Counts below use original model thresholds and altered attack inputs. They are not held-out detection metrics.", "",
              "| FPR budget | Group | Model | Intervention | Eligible N | Above original threshold | Median original-benign tail fraction |",
              "|---|---|---|---|---:|---:|---:|"]
    for r in stress:
        if r["group"] == "lost" and not r["intervention"].startswith("mean_"):
            tail = f"{r['median_benign_tail']:.6f}" if r["median_benign_tail"] is not None else "N/A"
            lines.append(f"| {r['operating_point']} | {r['group']} | {r['model']} | {r['intervention']} | {r['eligible_n']} | {r['above_original_threshold']} | {tail} |")
    lines += ["", "## Flow-family profiles of newly missed sequences", "",
              "Families are descriptive post-hoc slices based on the majority of attack-labeled flow protocol/destination-port pairs. They do not establish attack semantics.", "",
              "| FPR budget | Family | N | Median attack fraction | Median attack flows | Median active buckets | Old attack-only above threshold | New attack-only above threshold |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in family_rows:
        if r["group"] == "lost" and r["n"]:
            lines.append(f"| {r['operating_point']} | {r['attack_flow_family']} | {r['n']} | {r['median_attack_fraction']:.3f} | {r['median_attack_flows']:.1f} | {r['median_active_buckets']:.1f} | {r['old_attack_only_above_original_threshold']} | {r['new_attack_only_above_original_threshold']} |")
    lines += ["", "## Limits", ""] + ["- " + item for item in result["limitations"]]
    (out / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out, flush=True)
    for row in family_rows:
        if row["group"] == "lost":
            print(row)
    for row in profiles:
        print(row)
    for row in stress:
        if row["group"] == "lost":
            print(row)
    return out


if __name__ == "__main__":
    main()
