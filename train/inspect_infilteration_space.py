"""Inspect frozen Infilteration features without fitting a detector or using test."""
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "train/cache/diagnostic_plot_dependencies"))

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.common.io import fresh, read_json, sha256, unique_name, write_json


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main():
    run = PROJECT / "train/lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc"
    source = PROJECT / "train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/lstm"
    summary = read_json(run / "summary.json")
    export = read_json(source.parent / "manifest.json")["models"]["lstm"]
    for filename, key in (("vectors.tsv", "vectors_sha256"), ("metadata.tsv", "metadata_sha256")):
        if sha256(source / filename) != export[key]:
            raise ValueError("Projector source changed")
    if sha256(run / "model.pt") != summary["model_sha256"] or sha256(run / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Frozen model or thresholds changed")
    hidden = np.loadtxt(source / "vectors.tsv", delimiter="\t", dtype=np.float32)
    with (source / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    ids = np.array([int(row["sample_id"]) for row in metadata])
    labels = np.array([row["label"] == "Attack" for row in metadata])
    attacks = np.array([row["primary_attack"] for row in metadata])
    scores = np.array([float(row["saved_score"]) for row in metadata])
    with np.load(run / "predictions.npz", allow_pickle=False) as saved:
        lookup = {int(sid): i for i, sid in enumerate(saved["sample_id"])}
        positions = np.array([lookup[int(sid)] for sid in ids])
        if not np.array_equal(saved["y"][positions], labels) or not np.array_equal(saved["primary_attack"][positions], attacks):
            raise ValueError("Saved score metadata mismatch")
        if set(ids[attacks == "Infilteration"]) != set(saved["sample_id"][saved["primary_attack"] == "Infilteration"]):
            raise ValueError("Not all validation Infilteration sequences are included")
        scores = saved["score"][positions].copy()
        hosts = saved["source_host"][positions].copy()
    cache = PROJECT / f"train/cache/v1_{summary['dataset_manifest_sha256'][:20]}/validation"
    cached_ids = np.load(cache / "ids.npy", mmap_mode="r")
    if not np.array_equal(cached_ids[positions], ids):
        raise ValueError("Input feature alignment mismatch")
    x = np.load(cache / "x.npy", mmap_mode="r")[positions]
    mask = np.load(cache / "mask.npy", mmap_mode="r")[positions]
    inputs = np.concatenate((x.reshape(len(ids), -1), mask), axis=1)
    threshold = read_json(run / "thresholds.json")["values"]["validation_fpr_1pct"]
    target = attacks == "Infilteration"
    benign = ~labels
    tp, fn = target & (scores >= threshold), target & (scores < threshold)
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    w = checkpoint["state_dict"]["head.3.weight"].numpy().reshape(-1).astype(np.float64)
    bias = float(checkpoint["state_dict"]["head.3.bias"].numpy()[0])
    logits = hidden.astype(np.float64) @ w + bias
    if not np.allclose(expit(logits), np.array([float(row["reloaded_score"]) for row in metadata]), atol=2e-6, rtol=0):
        raise ValueError("Final classifier reconstruction failed")
    if not np.array_equal(expit(logits)[target] >= threshold, scores[target] >= threshold):
        raise ValueError("Reload differences changed Infilteration decisions")
    centered_unit = unit(hidden.astype(np.float64) - hidden.mean(axis=0, dtype=np.float64))
    representations = {"Scheme2 input, row L2 normalized": unit(inputs.astype(np.float64)),
                       "LSTM hidden, raw": hidden.astype(np.float64),
                       "LSTM hidden, row L2 normalized": unit(hidden.astype(np.float64)),
                       "LSTM hidden, centered then row L2 normalized": centered_unit}
    query = np.flatnonzero(target)
    neighbors_by_stage, summaries = {}, []
    for name, matrix in representations.items():
        search = NearestNeighbors(n_neighbors=11, algorithm="brute", metric="euclidean", n_jobs=4).fit(matrix)
        nearest = search.kneighbors(matrix[query], return_distance=False)
        fractions, nearest_benign_ids = [], []
        for own, row in zip(query, nearest):
            peers = row[row != own][:10]
            fractions.append(float(benign[peers].mean()))
            nearest_benign_ids.append(int(ids[peers[benign[peers]][0]]) if benign[peers].any() else None)
        neighbors_by_stage[name] = np.array(fractions)
        for outcome, keep in (("TP", tp[query]), ("FN", fn[query])):
            summaries.append({"representation": name, "outcome_1pct": outcome, "n": int(keep.sum()),
                              "mean_benign_fraction_in_10nn": float(np.mean(np.array(fractions)[keep])),
                              "median_benign_fraction_in_10nn": float(np.median(np.array(fractions)[keep])),
                              "majority_benign_neighbor_n": int((np.array(fractions)[keep] > .5).sum())})
    exact_collisions = {}
    for name, matrix in (("Scheme2_input", inputs), ("LSTM_hidden", hidden)):
        benign_hashes = {hashlib.sha256(row.tobytes()).digest() for row in matrix[benign]}
        exact_collisions[name] = [int(ids[i]) for i in query if hashlib.sha256(matrix[i].tobytes()).digest() in benign_hashes]
    flow_manifest = read_json(PROJECT / "datasets/dataset/flow_metadata.json")
    flow_index = flow_manifest["splits"]["validation"]
    flow_path = PROJECT / "datasets/dataset" / flow_index["path"]
    if sha256(flow_path) != flow_index["sha256"]:
        raise ValueError("Flow metadata changed")
    with np.load(flow_path, allow_pickle=False) as flow:
        order = np.argsort(flow["sample_id"])
        p = np.searchsorted(flow["sample_id"][order], ids)
        if not np.array_equal(flow["sample_id"][order][p], ids):
            raise ValueError("Flow metadata alignment mismatch")
        total, malicious = [flow[key][order][p] for key in ("total_flows", "attack_flows")]
    collision_audit = []
    if exact_collisions["Scheme2_input"]:
        from CIC_new.preprocessing.dataset import Originals
        prepared = Path(flow_manifest["prepared"])
        if not (prepared / "mmap/validation/identity.json").exists():
            raise ValueError("Expected the existing read-only pre-encoder cache")
        with Originals(prepared, "validation") as original:
            for sid in exact_collisions["Scheme2_input"]:
                i = int(np.flatnonzero(ids == sid)[0])
                peers = np.flatnonzero(benign & np.all(inputs == inputs[i], axis=1))
                for peer in peers:
                    raw_positions = np.searchsorted(original.meta["id"], [sid, int(ids[peer])])
                    if not np.array_equal(original.meta["id"][raw_positions], [sid, int(ids[peer])]):
                        raise ValueError("Collision pre-encoder row alignment mismatch")
                    differences = {}
                    for key in ("numeric", "hist_protocol", "hist_l7", "hist_service", "mask", "flow_counts"):
                        a, b = original.arrays[key][raw_positions]
                        differences[key] = {"different_values": int(np.count_nonzero(a != b)),
                                            "max_absolute_difference": float(np.max(np.abs(a - b)))}
                    numeric_a, numeric_b = original.arrays["numeric"][raw_positions]
                    changed = np.any(numeric_a != numeric_b, axis=0)
                    collision_audit.append({"infilteration_id": sid, "benign_id": int(ids[peer]),
                                            "infilteration_saved_score": float(scores[i]),
                                            "benign_saved_score": float(scores[peer]),
                                            "input_vectors_exactly_equal": True,
                                            "hidden_vectors_exactly_equal": bool(np.array_equal(hidden[i], hidden[peer])),
                                            "pre_encoder_all_model_fields_equal": all(v["different_values"] == 0 for v in differences.values()),
                                            "pre_encoder_differences": differences,
                                            "changed_numeric_features": np.array(original.schema["numeric_names"])[changed].tolist()})
    norms = np.linalg.norm(hidden, axis=1)
    alignment = (logits - bias) / np.maximum(norms * np.linalg.norm(w), 1e-12)
    cases = []
    for j, i in enumerate(query):
        cases.append({"sample_id": int(ids[i]), "outcome_1pct": "TP" if tp[i] else "FN", "saved_score": float(scores[i]),
                      "attack_flow_fraction": float(malicious[i] / total[i]), "total_flows": int(total[i]),
                      "active_buckets": int(mask[i].sum()), "hidden_l2_norm": float(norms[i]),
                      "cosine_to_classifier_weight": float(alignment[i]),
                      "source_host_hash": hashlib.sha256(str(hosts[i]).encode()).hexdigest()[:16],
                      **{f"benign_10nn_fraction_stage_{number}": float(neighbors_by_stage[name][j])
                         for number, name in enumerate(representations, 1)}})
    outcome_profiles = []
    for outcome, keep in (("TP", tp), ("FN", fn)):
        outcome_profiles.append({"outcome": outcome, "n": int(keep.sum()),
                                 "median_score": float(np.median(scores[keep])),
                                 "median_attack_fraction": float(np.median(malicious[keep] / total[keep])),
                                 "median_flows": float(np.median(total[keep])), "median_active_buckets": float(np.median(mask[keep].sum(axis=1))),
                                 "median_hidden_norm": float(np.median(norms[keep])),
                                 "median_classifier_alignment": float(np.median(alignment[keep]))})
    output = fresh(PROJECT / "train/runs" / unique_name("infilteration_embedding_geometry"))
    csv_write(output / "stage_neighbor_overlap.csv", summaries)
    csv_write(output / "infilteration_cases.csv", cases)
    csv_write(output / "outcome_profiles.csv", outcome_profiles)
    colors = {"Benign": "#c6cbd0", "Infilteration TP": "#138775", "Infilteration FN": "#dc4d54"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    projection_stats = {}
    for ax, matrix, title in ((axes[0], hidden.astype(np.float64), "Raw hidden features: PCA"),
                              (axes[1], centered_unit, "Centered + unit-normalized: PCA")):
        pca = PCA(n_components=2, svd_solver="full")
        coordinates = pca.fit_transform(matrix)
        projection_stats[title] = float(pca.explained_variance_ratio_.sum())
        for label, keep, size, alpha in (("Benign", benign, 7, .25), ("Infilteration TP", tp, 19, .8), ("Infilteration FN", fn, 23, .85)):
            ax.scatter(*coordinates[keep].T, c=colors[label], s=size, alpha=alpha, label=label, edgecolors="none")
        ax.set_title(title + f"\n2D explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    bins = np.linspace(0, 1, 31)
    for label, keep in (("Benign", benign), ("Infilteration TP", tp), ("Infilteration FN", fn)):
        axes[2].hist(scores[keep], bins=bins, histtype="step", density=True, color=colors[label], lw=2, label=label)
    axes[2].axvline(threshold, ls="--", c="#333333", label="Frozen 1% FPR threshold")
    axes[2].set_title("Actual classifier scores")
    axes[2].set_xlabel("Saved attack score (uncalibrated)")
    axes[2].set_ylabel("Density within group")
    axes[2].legend(fontsize=8)
    fig.suptitle("Frozen LSTM | all 366 validation Infilteration + fixed 4,000 benign | green: detected; red: missed")
    fig.savefig(output / "infilteration_geometry.png", dpi=160)
    fig.savefig(output / "infilteration_geometry.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    keys = list(representations)
    for offset, outcome, color in ((-.18, "TP", colors["Infilteration TP"]), (.18, "FN", colors["Infilteration FN"])):
        values = [100 * next(r["mean_benign_fraction_in_10nn"] for r in summaries if r["representation"] == key and r["outcome_1pct"] == outcome) for key in keys]
        ax.barh(np.arange(len(keys)) + offset, values, height=.34, label=outcome, color=color)
    ax.set_yticks(np.arange(len(keys)), keys)
    ax.invert_yaxis()
    ax.set_xlabel("Mean benign fraction among 10 nearest neighbors (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Infilteration overlap before and after LSTM (fixed 4,917-point candidate cohort)")
    ax.legend()
    fig.savefig(output / "high_dimensional_overlap.png", dpi=160)
    plt.close(fig)
    result = {"complete": True, "scope": "Exploratory validation geometry, not causal feature attribution or independent test",
              "test_opened": False, "model_refitted": False, "threshold_refitted": False, "training_examples_reused": False,
              "infilteration_n": int(target.sum()), "benign_visualization_n": int(benign.sum()),
              "knn_candidate_n": len(ids), "threshold": threshold, "outcome_profiles": outcome_profiles,
              "stage_neighbor_overlap": summaries, "exact_benign_collisions": exact_collisions,
              "collision_audit": collision_audit,
              "false_negative_score_lt_0_5_n": int((fn & (scores < .5)).sum()),
              "false_negative_score_ge_0_5_n": int((fn & (scores >= .5)).sum()),
              "projection_variance": projection_stats, "model_sha256": summary["model_sha256"],
              "classifier_weight_norm": float(np.linalg.norm(w)), "classifier_bias": bias,
              "final_logit_reconstruction_max_score_difference": float(np.max(np.abs(expit(logits) - scores))),
              "limitations": ["The 4,000 benign and 917 attack candidate rows are attack-enriched; nearest-neighbor fractions are not natural-prior probabilities.",
                              "Stage-wise neighbor comparisons depend on dimensionality, masks, metric and normalization, and cannot prove information loss.",
                              "Both PCA fits use the same 4,917 rows, including other attacks not drawn; independently fitted axes are not directly comparable.",
                              "Centered-then-unit normalization is explicit local visualization preprocessing, not a claim of exact website implementation equivalence.",
                              "These validation rows already selected checkpoints and thresholds. Recorded scores and model decisions are preserved."]}
    write_json(output / "geometry.json", result)
    lines = ["# Infilteration: representation geometry and missed detections", "",
             "All 366 real validation Infilteration sequences and the same 4,000 benign visualization sequences. Full 4,917-point candidate cohort (including other attacks) is used for PCA/nearest-neighbor calculations. No retraining, threshold tuning, training-row evaluation or test reads.",
             "", "## Outcome profiles", "", json.dumps(outcome_profiles, indent=2),
             "", "## Neighbor composition before and after LSTM", "",
             "| Representation | Outcome | N | Mean benign share among 10 neighbors | Median share |",
             "|---|---|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['representation']} | {r['outcome_1pct']} | {r['n']} | {r['mean_benign_fraction_in_10nn']:.2%} | {r['median_benign_fraction_in_10nn']:.2%} |")
    lines += ["", "## Interpretation", "",
              "A ring is a shape of the chosen transformation and two-dimensional projection. Its horizontal direction has no attack-semantic meaning. Visual overlap does not imply equal original 64D vectors, and normalization can discard feature magnitude used by the classifier.",
              "The frozen final decision is sigmoid(w dot h + b), where h is the original 64D hidden vector. It uses both magnitude and alignment with w. PCA chooses variance directions, not the classifier's decision direction.",
              "Compare TP and FN rather than treating Infilteration as one cluster. Higher benign-neighbor fractions among FN support local representation overlap for those examples; they do not establish whether the source labels, aggregation, bucket encoder or supervised model caused the overlap.",
              "Exact benign collisions in the sampled input/hidden spaces: " + json.dumps(exact_collisions),
              "A pre-encoder audit of these exact collisions is in geometry.json. These examples illustrate an ambiguity in this particular representation; they are not a claim that all Infilteration labels are wrong or all misses have the same cause.",
              f"Among the {int(fn.sum())} misses at the 1% FPR threshold, {int((fn & (scores >= .5)).sum())} still have saved score >=0.5, and {int((fn & (scores < .5)).sum())} have score <0.5. The operating threshold ({threshold:.6f}) is much stricter than 0.5. Lowering it trades recall against benign alarms and is not evidence of improved discrimination.",
              "", "## Limitations", ""]
    lines += ["- " + value for value in result["limitations"]]
    lines += ["", "Files: infilteration_geometry.png/pdf, high_dimensional_overlap.png, stage_neighbor_overlap.csv, infilteration_cases.csv, outcome_profiles.csv and geometry.json.", ""]
    (output / "GEOMETRY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
