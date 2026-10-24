"""Exploratory frozen-LSTM diagnostics: sample count, composition and diversity."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
PLOT_DEPS = PROJECT / "train/cache/diagnostic_plot_dependencies"
if PLOT_DEPS.exists():
    sys.path.insert(0, str(PLOT_DEPS))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from train.common.io import fresh, read_json, sha256, unique_name, write_json

CORE = ["Infilteration", "Brute_Force_-Web", "Bot", "DDOS_attack-HOIC",
        "DDOS_attack-LOIC-UDP", "DDoS_attacks-LOIC-HTTP"]
WEAK = CORE[:2]
DDOS = CORE[3:]


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def dispersion(values):
    # Trace of the unbiased sample covariance; unlike a plot's occupied area,
    # its expectation is not increased merely by drawing more iid observations.
    return float(np.var(values, axis=0, ddof=1).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 100:
        raise ValueError("At least 100 diagnostic resamples are required")
    run = PROJECT / "train/lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc"
    dataset = PROJECT / "datasets/dataset"
    projector = PROJECT / "train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/lstm"
    summary = read_json(run / "summary.json")
    thresholds = read_json(run / "thresholds.json")["values"]
    if sha256(run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Frozen thresholds changed")
    with np.load(run / "predictions.npz", allow_pickle=False) as a:
        ids, y, scores, attack, host = [a[k].copy() for k in ("sample_id", "y", "score", "primary_attack", "source_host")]
    if len(ids) != summary["metrics"]["n"] or int(y.sum()) != summary["metrics"]["positives"]:
        raise ValueError("Validation population mismatch")
    index = read_json(dataset / "flow_metadata.json")
    record = index["splits"]["validation"]
    if sha256(dataset / record["path"]) != record["sha256"]:
        raise ValueError("Flow metadata hash mismatch")
    with np.load(dataset / record["path"], allow_pickle=False) as a:
        order = np.argsort(a["sample_id"])
        positions = np.searchsorted(a["sample_id"][order], ids)
        if not np.array_equal(a["sample_id"][order][positions], ids) or not np.array_equal(a["labels"][order][positions], y):
            raise ValueError("Flow metadata alignment mismatch")
        total_flows, attack_flows = [a[k][order][positions].copy() for k in ("total_flows", "attack_flows")]
    fraction = attack_flows / total_flows
    if not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("Invalid attack-flow fractions")
    cache = PROJECT / f"train/cache/v1_{summary['dataset_manifest_sha256'][:20]}"
    train_ready = read_json(cache / "train/ready.json")
    if train_ready["cohort_sha256"] != summary["train_cohort_sha256"]:
        raise ValueError("Training metadata cohort mismatch")
    train = {k: np.load(cache / f"train/{k}.npy", mmap_mode="r") for k in ("attack", "types", "host")}
    if not np.array_equal(np.load(cache / "validation/ids.npy", mmap_mode="r"), ids):
        raise ValueError("Validation cache alignment mismatch")
    mask = np.load(cache / "validation/mask.npy", mmap_mode="r")
    active = np.asarray(mask.sum(axis=1))
    db = Path(index["prepared"]) / "manifest.sqlite"
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        original = {row[0]: row[1:] for row in connection.execute(
            "SELECT sequence_id,window_start_ms,active_buckets FROM original_sequences WHERE split='validation'")}
        membership = dict(connection.execute("SELECT a.sequence_id,COUNT(*) FROM sequence_attacks a "
                                             "JOIN original_sequences s USING(sequence_id) "
                                             "WHERE s.split='validation' GROUP BY a.sequence_id"))
    finally:
        connection.close()
    if any(original[int(sid)][1] != active[i] for i, sid in enumerate(ids)):
        raise ValueError("Original active-bucket metadata mismatch")
    timestamps = np.array([original[int(sid)][0] for sid in ids])
    vectors = np.loadtxt(projector / "vectors.tsv", delimiter="\t", dtype=np.float64)
    with (projector / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    export_manifest = read_json(projector.parent / "manifest.json")
    for filename, key in (("vectors.tsv", "vectors_sha256"), ("metadata.tsv", "metadata_sha256")):
        if sha256(projector / filename) != export_manifest["models"]["lstm"][key]:
            raise ValueError("Projector source changed")
    id_position = {int(sid): i for i, sid in enumerate(ids)}
    vis_index = np.array([id_position[int(row["sample_id"])] for row in metadata])
    if len(vectors) != len(vis_index) or set(ids[vis_index[y[vis_index] == 1]]) != set(ids[y == 1]):
        raise ValueError("Visualization does not contain all validation attacks")
    representation = {int(ids[i]): vectors[j] for j, i in enumerate(vis_index) if y[i] == 1}
    # Explicitly record this transform; it is not claimed to reproduce the site's
    # potentially approximate PCA or its implementation of Sphereize data.
    unit_vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    unit_representation = {int(ids[i]): unit_vectors[j] for j, i in enumerate(vis_index) if y[i] == 1}
    positive_vis = np.flatnonzero(y[vis_index] == 1)
    neighbor_benign = {}
    for name, matrix in (("raw", vectors), ("unit", unit_vectors)):
        knn = NearestNeighbors(n_neighbors=11, algorithm="brute", metric="euclidean", n_jobs=4).fit(matrix)
        neighbors = knn.kneighbors(matrix[positive_vis], return_distance=False)
        proportions = []
        for own, row in zip(positive_vis, neighbors):
            peers = row[row != own][:10]
            proportions.append(float((y[vis_index[peers]] == 0).mean()))
        neighbor_benign[name] = dict(zip(ids[vis_index[positive_vis]].tolist(), proportions))
    output = fresh(args.out or PROJECT / "train/runs" / unique_name("lstm_attack_overlap_diagnostics"))
    rng = np.random.default_rng(4266)
    one, strict = scores >= thresholds["validation_fpr_1pct"], scores >= thresholds["validation_fpr_0.1pct"]
    group_count = train_ready["groups"]
    real_counts = Counter(train["attack"][train["types"] == "real_malicious"].tolist())
    synthetic_counts = Counter(train["attack"][train["types"] == "synthetic_malicious"].tolist())
    all_rows, host_rows, fraction_rows, size_rows, sparsity_rows = [], [], [], [], []
    draws = {}
    fraction_bin = np.digitize(fraction, [0.01, 0.1, 0.5, 1.0], right=True)
    fraction_names = ["(0,1%]", "(1%,10%]", "(10%,50%]", "(50%,100%)", "above100_invalid"]
    # Treat exactly pure sequences as a separate, interpretable last bin.
    fraction_bin[fraction == 1] = 4
    fraction_names[4] = "100%"
    for name in sorted(np.unique(attack[y == 1])):
        keep = (y == 1) & (attack == name)
        positions = np.flatnonzero(keep)
        values = np.stack([representation[int(ids[i])] for i in positions])
        normalized = np.stack([unit_representation[int(ids[i])] for i in positions])
        unique_hosts, inverse = np.unique(host[keep], return_inverse=True)
        hn = np.bincount(inverse)
        htp = np.bincount(inverse, weights=one[keep])
        htp_strict = np.bincount(inverse, weights=strict[keep])
        macro = float((htp / hn).mean())
        bootstrap = rng.integers(0, len(hn), size=(args.resamples, len(hn)))
        bootstrap_recall = htp[bootstrap].sum(axis=1) / hn[bootstrap].sum(axis=1)
        real_hosts = set(train["host"][(train["types"] == "real_malicious") & (train["attack"] == name)].tolist())
        seen = np.isin(host[keep], list(real_hosts))
        row = {"primary_attack": str(name), "validation_n": len(positions), "validation_hosts": len(hn),
               "real_train_n": real_counts[name], "synthetic_train_n": synthetic_counts[name],
               "positive_objective_share": (real_counts[name] / group_count["real_malicious"] + synthetic_counts[name] / group_count["synthetic_malicious"]) / 2,
               "recall_1pct": float(one[keep].mean()), "recall_0_1pct": float(strict[keep].mean()),
               "host_macro_recall_1pct": macro,
               "host_bootstrap_p025": quantiles(bootstrap_recall)[0] if len(hn) > 1 else None,
               "host_bootstrap_p975": quantiles(bootstrap_recall)[2] if len(hn) > 1 else None,
               "median_attack_flow_fraction": float(np.median(fraction[keep])),
               "fraction_le_1pct_n": int(np.count_nonzero(fraction[keep] <= 0.01)),
               "pure_attack_n": int(np.count_nonzero(fraction[keep] == 1)),
               "median_active_buckets": float(np.median(active[keep])),
               "median_total_flows": float(np.median(total_flows[keep])),
               "median_attack_flows": float(np.median(attack_flows[keep])),
               "tp_median_total_flows": float(np.median(total_flows[keep & one])) if np.any(keep & one) else None,
               "fn_median_total_flows": float(np.median(total_flows[keep & ~one])) if np.any(keep & ~one) else None,
               "tp_median_active_buckets": float(np.median(active[keep & one])) if np.any(keep & one) else None,
               "fn_median_active_buckets": float(np.median(active[keep & ~one])) if np.any(keep & ~one) else None,
               "tp_median_fraction": float(np.median(fraction[keep & one])) if np.any(keep & one) else None,
               "fn_median_fraction": float(np.median(fraction[keep & ~one])) if np.any(keep & ~one) else None,
               "within_type_mixed_attack_n": sum(membership.get(int(ids[i]), 0) > 1 for i in positions),
               "raw_hidden_covariance_trace": dispersion(values), "unit_hidden_covariance_trace": dispersion(normalized),
               "mean_benign_neighbors_of_10_raw": float(np.mean([neighbor_benign['raw'][int(ids[i])] for i in positions])),
               "mean_benign_neighbors_of_10_unit": float(np.mean([neighbor_benign['unit'][int(ids[i])] for i in positions])),
               "validation_seen_in_real_train_same_type_host_n": int(seen.sum()),
               "recall_seen_host": float(one[keep][seen].mean()) if seen.any() else None,
               "recall_unseen_host": float(one[keep][~seen].mean()) if (~seen).any() else None,
               "time_blocks": len(np.unique(timestamps[keep]))}
        all_rows.append(row)
        if row["validation_n"] != summary["metrics"]["per_primary_attack"][name]["n"] or row["recall_1pct"] != summary["metrics"]["per_primary_attack"][name]["recall_at_validation_fpr_1pct"]:
            raise ValueError("Per-type recall failed reconciliation")
        for h, n, tp, tp_strict in zip(unique_hosts, hn, htp, htp_strict):
            host_rows.append({"primary_attack": name, "host_hash": hashlib.sha256(str(h).encode()).hexdigest()[:16],
                              "n": int(n), "tp_1pct": int(tp), "recall_1pct": float(tp / n),
                              "tp_0_1pct": int(tp_strict), "recall_0_1pct": float(tp_strict / n)})
        for b, label in enumerate(fraction_names):
            subset = keep & (fraction_bin == b)
            if subset.any():
                fraction_rows.append({"primary_attack": name, "attack_flow_fraction": label, "n": int(subset.sum()),
                                      "hosts": len(np.unique(host[subset])), "tp_1pct": int(one[subset].sum()),
                                      "recall_1pct": float(one[subset].mean()), "recall_0_1pct": float(strict[subset].mean())})
        for variable, data, bins, bin_labels in (
                ("active_buckets", active, [1, 5], ["1", "2-5", "6-10"]),
                ("total_flows", total_flows, [2, 10, 100], ["1-2", "3-10", "11-100", "101+"])):
            cells_for_variable = np.digitize(data, bins, right=True)
            for b, label in enumerate(bin_labels):
                subset = keep & (cells_for_variable == b)
                if subset.any():
                    sparsity_rows.append({"primary_attack": name, "variable": variable, "bin": label,
                                          "n": int(subset.sum()), "hosts": len(np.unique(host[subset])),
                                          "tp_1pct": int(one[subset].sum()), "recall_1pct": float(one[subset].mean()),
                                          "recall_0_1pct": float(strict[subset].mean())})
        if name in CORE:
            samples = np.stack([rng.choice(len(positions), 50, replace=False) for _ in range(args.resamples)])
            recalls = one[positions][samples].mean(axis=1)
            strict_recalls = strict[positions][samples].mean(axis=1)
            raw_variance = np.array([dispersion(values[s]) for s in samples])
            unit_variance = np.array([dispersion(normalized[s]) for s in samples])
            # Match source-host count as a second diagnostic: one random sequence
            # from each of eight random hosts, not eight independent campaigns.
            per_host = [positions[host[positions] == h] for h in unique_hosts]
            host_recalls = []
            for _ in range(args.resamples):
                selected_hosts = rng.choice(len(per_host), 8, replace=False)
                selected = [rng.choice(per_host[h]) for h in selected_hosts]
                host_recalls.append(float(one[selected].mean()))
            draws[str(name)] = {"recall": recalls, "raw_dispersion": raw_variance, "unit_dispersion": unit_variance}
            size_rows.append({"primary_attack": name, "sample_n": 50, "iterations": args.resamples,
                              "recall_mean": float(recalls.mean()), "recall_p025": quantiles(recalls)[0], "recall_p975": quantiles(recalls)[2],
                              "strict_recall_mean": float(strict_recalls.mean()), "strict_recall_p025": quantiles(strict_recalls)[0], "strict_recall_p975": quantiles(strict_recalls)[2],
                              "raw_dispersion_median": quantiles(raw_variance)[1], "raw_dispersion_p025": quantiles(raw_variance)[0], "raw_dispersion_p975": quantiles(raw_variance)[2],
                              "unit_dispersion_median": quantiles(unit_variance)[1], "unit_dispersion_p025": quantiles(unit_variance)[0], "unit_dispersion_p975": quantiles(unit_variance)[2],
                              "eight_host_one_sequence_recall_mean": float(np.mean(host_recalls))})
    duplicate_rows = []
    for name in DDOS:
        keep = (y == 1) & (attack == name)
        original = np.flatnonzero(keep)
        duplicated = np.stack([rng.choice(original, 366, replace=True) for _ in range(args.resamples)])
        duplicate_rows.append({"primary_attack": name, "original_unique_n": len(original), "duplicated_n": 366,
                               "iterations": args.resamples, "recall_1pct_min": float(one[duplicated].mean(axis=1).min()),
                               "recall_1pct_max": float(one[duplicated].mean(axis=1).max()),
                               "recall_0_1pct_min": float(strict[duplicated].mean(axis=1).min()),
                               "recall_0_1pct_max": float(strict[duplicated].mean(axis=1).max()),
                               "interpretation": "Multiplicity-only diagnostic; no new independent examples or model fitting."})
    matched_rows = []
    reference = (y == 1) & np.isin(attack, DDOS)
    active_bin = np.digitize(active, [1, 5], right=True)
    cells = fraction_bin * 3 + active_bin
    for name in WEAK:
        target = (y == 1) & (attack == name)
        blocks = []
        for cell in np.unique(cells[target | reference]):
            left, right = np.flatnonzero(target & (cells == cell)), np.flatnonzero(reference & (cells == cell))
            n = min(len(left), len(right))
            if n:
                blocks.append((left, right, n))
        matched_n = sum(n for _, _, n in blocks)
        left_recalls, right_recalls = [], []
        if matched_n:
            for _ in range(args.resamples):
                left = np.concatenate([rng.choice(a, n, replace=False) for a, b, n in blocks])
                right = np.concatenate([rng.choice(b, n, replace=False) for a, b, n in blocks])
                left_recalls.append(float(one[left].mean()))
                right_recalls.append(float(one[right].mean()))
        matched_rows.append({"primary_attack": name, "reference": "pooled three DDoS types", "matched_n_per_group": matched_n,
                             "target_coverage": matched_n / int(target.sum()), "reference_coverage": matched_n / int(reference.sum()),
                             "strata": "attack-flow fraction (<=1%, <=10%, <=50%, <100%, 100%) x active buckets (1, 2-5, 6-10)",
                             "target_recall_mean": float(np.mean(left_recalls)) if matched_n else None,
                             "reference_recall_mean": float(np.mean(right_recalls)) if matched_n else None})
    for filename, rows in (("attack_profiles.csv", all_rows), ("source_host_recalls.csv", host_rows),
                           ("attack_fraction_slices.csv", fraction_rows), ("equal_n50_resampling.csv", size_rows),
                           ("ddos_duplication_diagnostic.csv", duplicate_rows), ("composition_matching.csv", matched_rows),
                           ("sparsity_slices.csv", sparsity_rows)):
        save_csv(output / filename, rows)
    findings = {"complete": True, "scope": "Exploratory diagnostics after validation-based checkpoint and threshold selection",
                "seed": 4266, "resamples": args.resamples, "test_opened": False,
                "training_metadata_inspected": True, "training_examples_used_as_evaluation_samples": False,
                "model_or_threshold_refitted": False, "model_sha256": summary["model_sha256"],
                "predictions_sha256": sha256(run / "predictions.npz"), "flow_metadata_sha256": record["sha256"],
                "profiles": all_rows, "equal_n50": size_rows, "ddos_duplication": duplicate_rows,
                "composition_matching": matched_rows,
                "limitations": ["Sequence resampling bands are variation across subsets, not independent confidence intervals.",
                    "Source-host bootstrap conditions on the fixed selected model; it does not account for training seed or campaign dependence.",
                    "Perfect detection on every observed example gives degenerate resampling bands; unseen attack patterns remain untested.",
                    "Neighbor composition uses a fixed attack-enriched visualization cohort of 4,000 benign plus 917 attack sequences.",
                    "Flow fraction is labeled-flow count divided by total-flow count; it is not byte/packet intensity or attack severity.",
                    "Matched observational slices do not establish causality; strata may have limited overlap."]}
    write_json(output / "diagnostics.json", findings)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ordered = [{r["primary_attack"]: r for r in size_rows}[name] for name in CORE]
    yy = np.arange(len(CORE))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    for ax, field, lo, hi, xlabel, scale in (
            (axes[0], "recall_mean", "recall_p025", "recall_p975", "Recall at frozen 1% validation-FPR threshold (%)", 100),
            (axes[1], "unit_dispersion_median", "unit_dispersion_p025", "unit_dispersion_p975", "Unit-vector covariance trace (64D; unbiased estimate)", 1)):
        means = np.array([r[field] for r in ordered]) * scale
        lower, upper = np.array([r[lo] for r in ordered]) * scale, np.array([r[hi] for r in ordered]) * scale
        ax.errorbar(means, yy, xerr=np.maximum(0, np.stack((means - lower, upper - means))), fmt="o", capsize=4, color="#176c91")
        ax.set_yticks(yy, CORE if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_xlim(0, 105)
    axes[0].set_title("Equal sample count: 50 per attack type")
    axes[1].set_title("Dispersion after equalizing sample count")
    fig.suptitle(f"Frozen LSTM | {args.resamples:,} subsets without replacement | bars: central 95% of subset results")
    fig.savefig(output / "equal_sample_size.png", dpi=170)
    fig.savefig(output / "equal_sample_size.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, name in zip(axes, WEAK):
        keep = (y == 1) & (attack == name)
        for found, color, label in ((False, "#ce4c48", "Missed at 1% threshold"), (True, "#17816e", "Detected at 1% threshold")):
            subset = keep & (one == found)
            ax.scatter(fraction[subset], scores[subset], c=color, label=label, alpha=.7, s=20)
        ax.axhline(thresholds["validation_fpr_1pct"], color="#555555", ls="--", lw=1, label="Frozen 1% threshold")
        ax.set_xscale("log")
        ax.set_xlabel("Attack flows / total flows (log scale)")
        ax.set_title(name)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Saved LSTM attack score (uncalibrated)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Full real validation attack slices; composition association is not a causal test")
    fig.savefig(output / "attack_fraction_vs_score.png", dpi=170)
    plt.close(fig)
    lines = ["# Why do LSTM attack types differ? Exploratory diagnostics", "",
             "Frozen model and original saved validation scores. No retraining, threshold adjustment, test scoring or new independent samples. Training metadata is inspected only for coverage and weighting; training examples are never scored as evaluation examples.",
             "", "## Main profiles", "",
             "| Attack | Real train | Synthetic train | Validation N / hosts | Recall at 1% | Recall at 0.1% | Median attack-flow fraction | 10-NN benign fraction, unit 64D |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        lines.append(f"| {r['primary_attack']} | {r['real_train_n']} | {r['synthetic_train_n']} | {r['validation_n']} / {r['validation_hosts']} | {r['recall_1pct']:.2%} | {r['recall_0_1pct']:.2%} | {r['median_attack_flow_fraction']:.4%} | {r['mean_benign_neighbors_of_10_unit']:.2%} |")
    lines += ["", "## Equal sample-size test", "",
              f"Each of six attack types supplies 50 distinct sequences per iteration, repeated {args.resamples:,} times at seed 4266. Thresholds and all vectors remain fixed. Dispersion is measured in 64D, before PCA; unit vectors use explicit row L2 normalization without centering. These are subset variation bands, not generalization confidence intervals.",
              "", "| Attack | Mean recall | Central 95% subset range | Median unit dispersion | One example per eight sampled hosts: mean recall |",
              "|---|---:|---:|---:|---:|"]
    for r in ordered:
        lines.append(f"| {r['primary_attack']} | {r['recall_mean']:.2%} | {r['recall_p025']:.2%} - {r['recall_p975']:.2%} | {r['unit_dispersion_median']:.5f} | {r['eight_host_one_sequence_recall_mean']:.2%} |")
    lines += ["", "## Multiplicity-only DDoS diagnostic", "",
              "Each of the three observed DDoS types was resampled WITH replacement to 366 rows in every iteration. Recall remained 100% at both frozen thresholds in all iterations. Every observed DDoS example was already detected: duplication cannot create a missed pattern. This is a mathematical/data-integrity control, not evidence from 366 independent attacks.",
              "", "## Composition matching", ""]
    for r in matched_rows:
        lines.append(json.dumps(r))
    pure = {r["primary_attack"]: r for r in fraction_rows if r["attack_flow_fraction"] == "100%"}
    profiles = {r["primary_attack"]: r for r in all_rows}
    lines += ["", "## Measured explanatory clues", "",
              f"Pure Web sequences still detect only {pure['Brute_Force_-Web']['tp_1pct']}/{pure['Brute_Force_-Web']['n']} ({pure['Brute_Force_-Web']['recall_1pct']:.2%}). Pure Infilteration detects {pure['Infilteration']['tp_1pct']}/{pure['Infilteration']['n']} ({pure['Infilteration']['recall_1pct']:.2%}). Background dilution cannot explain all misses.",
              f"Web median total flows / active buckets: {profiles['Brute_Force_-Web']['median_total_flows']} / {profiles['Brute_Force_-Web']['median_active_buckets']}. Infilteration: {profiles['Infilteration']['median_total_flows']} / {profiles['Infilteration']['median_active_buckets']}. Sparsity and volume slices are recorded separately.",
              "All validation DDoS source hosts have real training examples of the same type. These are temporal-block results, not unseen-host/campaign evaluation. Source IPs are not model inputs, but repeated behavioral profiles may still make these examples easier.",
              f"Infilteration micro recall is {profiles['Infilteration']['recall_1pct']:.2%}, while giving each observed source host equal weight yields {profiles['Infilteration']['host_macro_recall_1pct']:.2%}. Host composition materially affects the descriptive average; this does not prove causation.",
              "The observed angular-dispersion difference is not the same as raw variance. Raw hidden covariance trace for DDoS LOIC-HTTP exceeds both weak types, while unit-normalized dispersion is much smaller. Avoid claiming that generic variance alone causes low recall.",
              "10-NN overlap is measured in full 64D on one fixed visualization sample, so it does not rely on a 2D PCA projection. Its value still depends on the chosen benign cohort, class composition, metric and normalization.",
              "Training counts include both real and synthetic sequences, and four training groups have equal objective weight. positive_objective_share in attack_profiles.csv is the share within the combined positive training objective, not an unweighted sample fraction across the whole dataset.",
              "", "## Next experiments and interpretation", "",
              "1. Distinguish evaluation count from training count. More validation points can reveal tails and rare hosts, but do not inherently lower expected recall or increase unbiased covariance. Downsampling weak types controls visible point count, not their generating distribution.",
              "2. Inspect attack-flow-fraction slices, host recalls and high-dimensional neighbors. A sequence label is positive when at least one flow is labeled malicious; low fractions can make a 100-second aggregate resemble benign traffic. Association alone cannot distinguish dilution, attack behavior, labeling errors and host effects.",
              "3. For a controlled sensitivity test, vary malicious/benign flow mixture while fixing attack donor, benign donor, total volume, masks and timestamps where feasible. Recompute original aggregate features through the frozen encoder and LSTM. Do not add Gaussian noise to final hidden vectors and interpret it as a new network attack. Donor reuse, including any training donor reuse, must be disclosed; generated mixtures are stress diagnostics, not a held-out test.",
              "4. To test training-data effects, freeze a new experiment protocol first. Match real malicious sample counts or per-type objective weights, use at least three seeds and one fixed validation/FPR policy, then compare per-type recalls and false alarms. Different counts, class weights and synthetic donor allocations are different interventions and must be varied separately. Existing outputs must be preserved.",
              "5. To assess generalization, obtain new independent DDoS hosts/campaigns/time blocks and score them with the frozen pipeline; never import training DDoS sequences into validation/test. A newly collected or reserved holdout is needed after validation-informed development.",
              "6. Prioritize a representation ablation for sparse Web traffic: compare original aggregate features, frozen 64D bucket encoding and final 64D LSTM features using the same real cohorts. Inspect whether HTTP behavior differences survive aggregation and compression; consider destination/request-level information only if available in the source. This is a hypothesis, not a measured representation failure.",
              "The dataset authors describe Web attacks as automated DVWA interactions and Infiltration as a multi-stage exploit/backdoor/scanning scenario: https://www.unb.ca/cic/datasets/ids-2018.html . This motivates behavioral hypotheses, but does not validate every label or sequence in this NetFlow conversion.",
              "", "## Limitations", ""]
    lines += ["- " + item for item in findings["limitations"]]
    lines += ["", "Figures: equal_sample_size.png and attack_fraction_vs_score.png. Detailed counts, host hashes, resampling bands, matching coverage and hashes are in the CSV and JSON files.", ""]
    (output / "DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "profiles": [r for r in all_rows if r['primary_attack'] in CORE],
                      "equal_n50": size_rows, "composition_matching": matched_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
