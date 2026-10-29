"""Export matched historical/holdout LSTM representations for Embedding Projector."""
import argparse
from collections import Counter
import csv
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

if __package__ in (None, ""):
    script_folder = Path(__file__).resolve().parent
    sys.path = [str(script_folder.parent)] + [
        value for value in sys.path if Path(value or ".").resolve() != script_folder
    ]

from train.common.data import Scheme2
from train.common.io import ROOT, DEFAULT_DATASET, fresh, read_json, sha256, unique_name, write_json
from train.export_projector import neural_vectors, export_files


def read_metadata(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_metadata(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_pair(folder, vectors, metadata):
    folder = fresh(folder)
    np.savetxt(folder / "vectors.tsv", vectors, delimiter="\t", fmt="%.9g", encoding="utf-8")
    write_metadata(folder / "metadata.tsv", metadata)
    restored = np.loadtxt(folder / "vectors.tsv", delimiter="\t", dtype=np.float32, ndmin=2)
    if not np.array_equal(restored, vectors) or read_metadata(folder / "metadata.tsv") != metadata:
        raise ValueError("Projector TSV round-trip failed")
    return {"path": str(folder.resolve()), "rows": len(metadata), "dimensions": vectors.shape[1],
            "vectors_sha256": sha256(folder / "vectors.tsv"),
            "metadata_sha256": sha256(folder / "metadata.tsv"), "round_trip_verified": True}


def geometry(vectors, labels, attacks):
    # Use native 64-D Euclidean geometry, not distances in a nonlinear 2-D plot.
    # X=None excludes each sample itself, including when other vectors are identical.
    neighbors = NearestNeighbors(n_neighbors=20, algorithm="brute", n_jobs=4).fit(vectors)
    indices = neighbors.kneighbors(return_distance=False)
    benign_mix = (labels[indices] == 0).mean(axis=1)
    same_type = (attacks[indices] == attacks[:, None]).mean(axis=1)
    benign = vectors[labels == 0].astype(np.float64)
    center = benign.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((benign - center) ** 2, axis=1))))
    result = {}
    for attack in sorted(set(attacks[labels == 1])):
        keep = (labels == 1) & (attacks == attack)
        distance = float(np.linalg.norm(vectors[keep].mean(axis=0, dtype=np.float64) - center))
        result[str(attack)] = {"n": int(keep.sum()),
                              "benign_fraction_among_20_neighbors": float(benign_mix[keep].mean()),
                              "same_type_fraction_among_20_neighbors": float(same_type[keep].mean()),
                              "centroid_distance_in_benign_rms_radii": distance / radius if radius else None}
    return result


def plot_pca(output, matrices, attacks, labels, heldout):
    if not importlib.util.find_spec("matplotlib"):
        sys.path.append(str(ROOT / "cache" / "diagnostic_plot_dependencies"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(set(attacks[labels == 1]))
    colors = {name: plt.get_cmap("tab20")(i) for i, name in enumerate(names)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    projections = {}
    for ax, (condition, vectors) in zip(axes, matrices.items()):
        pca = PCA(n_components=2, svd_solver="full")
        points = pca.fit_transform(vectors)
        projections[condition] = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                                  "independent_pca_per_condition": True, "sphereized": False}
        keep = labels == 0
        ax.scatter(points[keep, 0], points[keep, 1], s=5, c="#bbbbbb", alpha=.24, label="Benign")
        for name in names:
            keep = (labels == 1) & (attacks == name)
            ax.scatter(points[keep, 0], points[keep, 1], s=17, color=colors[name], alpha=.8,
                       marker="x" if name in heldout else "o", label=name)
        ax.set_title(condition.replace("_", " ").title())
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.grid(alpha=.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("LSTM validation representations: matched sequences, independent PCA\n"
                 "Crosses: types withheld from new training; axes across panels are not aligned", fontsize=12)
    fig.tight_layout(rect=(0, .19, 1, .93))
    fig.savefig(output / "pca_comparison.png", dpi=180)
    plt.close(fig)
    return projections


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_holdout_xss_slowhttp_hoic")
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc")
    parser.add_argument("--reference-projector", type=Path,
                        default=ROOT / "runs/20261024_104043_364574_projector_four_models_validation_805efb/lstm")
    parser.add_argument("--reference-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    summary = read_json(args.run / "summary.json")
    reference_summary = read_json(args.reference_run / "summary.json")
    for run, report in ((args.run, summary), (args.reference_run, reference_summary)):
        if not report.get("complete") or report["model"] != "lstm" or report["split"] != "validation":
            raise ValueError("Expected completed validation LSTM runs")
        for filename, key in (("model.pt", "model_sha256"), ("thresholds.json", "thresholds_sha256")):
            if sha256(run / filename) != report[key]:
                raise ValueError(f"Frozen source changed: {run / filename}")
    reference_export = read_json(args.reference_projector.parent / "manifest.json")
    reference_info = reference_export["models"]["lstm"]
    if not reference_export.get("complete") or reference_info["model_sha256"] != reference_summary["model_sha256"]:
        raise ValueError("Reference export/model mismatch")
    for filename, key in (("vectors.tsv", "vectors_sha256"), ("metadata.tsv", "metadata_sha256")):
        if sha256(args.reference_projector / filename) != reference_info[key]:
            raise ValueError("Reference Projector files changed")
    data = Scheme2(args.dataset)
    if data.identity != summary["dataset_manifest_sha256"]:
        raise ValueError("Holdout model/dataset mismatch")
    if sha256(args.reference_dataset / "manifest.json") != reference_summary["dataset_manifest_sha256"]:
        raise ValueError("Reference dataset mismatch")
    heldout = set(data.manifest["train_holdout_attacks"])
    output = fresh(args.out or ROOT / "runs" / unique_name("lstm_holdout_projector_comparison"))
    write_json(output / "status.json", {"complete": False, "test_opened": False})
    validation = data.load("validation")
    training = data.load("train")
    try:
        if validation.info["cohort_sha256"] != summary["validation_cohort_sha256"]:
            raise ValueError("Holdout validation identity mismatch")
        train_counts = dict(Counter(training["attack"].tolist()))
        if any(train_counts.get(name, 0) for name in heldout):
            raise ValueError("A withheld primary attack is present in training")
        # Keep exactly the historical visualization sample, rather than resampling.
        old_metadata = read_metadata(args.reference_projector / "metadata.tsv")
        old_vectors = np.loadtxt(args.reference_projector / "vectors.tsv", delimiter="\t", dtype=np.float32, ndmin=2)
        if old_vectors.shape != (len(old_metadata), 64) or not np.isfinite(old_vectors).all():
            raise ValueError("Invalid historical representation matrix")
        lookup = {int(value): i for i, value in enumerate(validation["ids"])}
        indices = np.asarray([lookup[int(row["sample_id"])] for row in old_metadata])
        if len(np.unique(indices)) != len(indices) or len(old_vectors) != len(indices):
            raise ValueError("Duplicate or missing visualization samples")
        for row, index in zip(old_metadata, indices):
            if (row["primary_attack"] != validation["attack"][index]
                    or row["label"] != ("Attack" if validation["y"][index] else "Benign")
                    or int(row["active_buckets"]) != int(validation["mask"][index].sum())
                    or int(row["flow_count"]) != int(validation["flows"][index])):
                raise ValueError("Reference/holdout sample metadata differs")
        if set(indices[validation["y"][indices] == 1]) != set(np.flatnonzero(validation["y"] == 1)):
            raise ValueError("Expected all real validation attacks")
        with np.load(args.run / "predictions.npz", allow_pickle=False) as predictions:
            for key, split_key in (("sample_id", "ids"), ("y", "y"), ("primary_attack", "attack")):
                if not np.array_equal(predictions[key], validation[split_key]):
                    raise ValueError("Holdout predictions/sample mismatch")
            scores = predictions["score"][indices].copy()
        with np.load(args.reference_run / "predictions.npz", allow_pickle=False) as reference_predictions:
            for key, split_key in (("sample_id", "ids"), ("y", "y"), ("primary_attack", "attack")):
                if not np.array_equal(reference_predictions[key], validation[split_key]):
                    raise ValueError("Full validation populations differ between conditions")
        vectors, fresh_scores, details = neural_vectors(
            "lstm", read_json(args.run / "config.json"), args.run,
            np.array(validation["x"][indices]), np.array(validation["mask"][indices]), args.device)
        score_delta = float(np.max(np.abs(scores - fresh_scores)))
        if not np.allclose(scores, fresh_scores, atol=.001, rtol=0):
            raise ValueError(f"Reloaded scores differ: {score_delta}")
        staging = fresh(output / "holdout_lstm")
        files = export_files(staging, vectors, validation, indices, scores, fresh_scores,
                             read_json(args.run / "thresholds.json")["values"])
        new_metadata = read_metadata(staging / "metadata.tsv")
        labels = np.array(validation["y"][indices])
        attacks = np.array(validation["attack"][indices])
        for condition, rows in (("historical_baseline", old_metadata), ("holdout_lstm", new_metadata)):
            for row in rows:
                row["condition"] = condition
                row["training_status"] = ("Benign" if row["label"] == "Benign" else
                    "Withheld from new training" if row["primary_attack"] in heldout else "Retained attack type")
        # Read immediately before replacing the just-created metadata with added columns.
        write_metadata(staging / "metadata.tsv", new_metadata)
        if read_metadata(staging / "metadata.tsv") != new_metadata:
            raise ValueError("Annotated metadata round-trip failed")
        files["metadata_sha256"] = sha256(staging / "metadata.tsv")
        matrices = {"historical_baseline": old_vectors, "holdout_lstm": vectors}
        metadata = {"historical_baseline": old_metadata, "holdout_lstm": new_metadata}
        exports = {"holdout_lstm": files,
                   "historical_baseline": write_pair(output / "historical_baseline", old_vectors, old_metadata)}
        subset_exports = {}
        for condition, matrix in matrices.items():
            subsets = {}
            for name in sorted(set(attacks[labels == 1])):
                keep = (labels == 0) | (attacks == name)
                safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
                subsets[name] = write_pair(output / "by_attack" / condition / safe_name, matrix[keep],
                                          [metadata[condition][i] for i in np.flatnonzero(keep)])
            keep = ~np.isin(attacks, sorted(heldout))
            subsets["retained_types_and_benign"] = write_pair(
                output / "retained_types" / condition, matrix[keep],
                [metadata[condition][i] for i in np.flatnonzero(keep)])
            subset_exports[condition] = subsets
        geometry_results = {name: geometry(matrix, labels, attacks) for name, matrix in matrices.items()}
        comparison_rows = []
        for name in sorted(geometry_results["holdout_lstm"]):
            row = {"primary_attack": name, "withheld_from_new_training": name in heldout,
                   "n": geometry_results["holdout_lstm"][name]["n"]}
            for condition in matrices:
                stats = geometry_results[condition][name]
                row.update({f"{condition}_{key}": value for key, value in stats.items() if key != "n"})
                model_summary = summary if condition == "holdout_lstm" else reference_summary
                row[f"{condition}_recall_at_1pct"] = model_summary["metrics"]["per_primary_attack"][name]["recall_at_validation_fpr_1pct"]
            row["delta_benign_neighbor_fraction"] = (row["holdout_lstm_benign_fraction_among_20_neighbors"]
                - row["historical_baseline_benign_fraction_among_20_neighbors"])
            comparison_rows.append(row)
        with (output / "geometry_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
        projections = plot_pca(output, matrices, attacks, labels, heldout)
        old_manifest = read_json(args.reference_dataset / "manifest.json")
        preprocessing = {key: {"historical": old_manifest.get(key), "holdout": data.manifest.get(key),
                              "equal": old_manifest.get(key) == data.manifest.get(key)}
                         for key in ("schema_sha256", "scaler_sha256", "embedding_checkpoint_sha256", "donor_plan_sha256")}
        manifest = {"complete": True, "split": "validation", "test_opened": False,
                    "training_data_reused_for_visualization": False,
                    "training_data_reused_for_new_fit": "Retained original training sequences are reused; synthetic donor plan differs.",
                    "validation_reused_for_checkpoint_and_threshold_selection": True,
                    "heldout_primary_attack_training_counts": {name: train_counts.get(name, 0) for name in sorted(heldout)},
                    "training_primary_attack_counts": train_counts,
                    "raw_flow_attack_membership_independently_audited": False,
                    "rows": len(indices), "dimensions": vectors.shape[1],
                    "attacks": int(labels.sum()), "benign": int((labels == 0).sum()),
                    "identical_full_validation_ids_labels_primary_attacks": True,
                    "identical_selected_sample_order_flow_counts_and_active_bucket_counts": True,
                    "dataset_manifest_sha256": data.identity,
                    "run": str(args.run.resolve()), "model_sha256": summary["model_sha256"],
                    "reference_run": str(args.reference_run.resolve()),
                    "reference_model_sha256": reference_summary["model_sha256"],
                    "reference_projector": str(args.reference_projector.resolve()),
                    "reference_projector_manifest_sha256": sha256(args.reference_projector.parent / "manifest.json"),
                    "preprocessing_comparison": preprocessing, "exports": exports, "subsets": subset_exports,
                    "representation_checks": details, "reload_score_max_absolute_difference": score_delta,
                    "threshold_decision_disagreements": {key: int(np.count_nonzero((scores >= value) != (fresh_scores >= value)))
                        for key, value in read_json(args.run / "thresholds.json")["values"].items()},
                    "geometry": geometry_results, "pca": projections, "script_sha256": sha256(__file__)}
        write_json(output / "manifest.json", manifest)
        lines = ["# LSTM embeddings after withholding XSS, SlowHTTPTest and HOIC", "",
                 "## Open in TensorFlow Embedding Projector", "",
                 "Open https://projector.tensorflow.org/ and click Load. Select holdout_lstm/vectors.tsv in Step 1 and holdout_lstm/metadata.tsv in Step 2. The historical_baseline directory is the original-model reference on exactly the same sequence IDs and in the same order.",
                 "Set Color by = primary_attack; use training_status to distinguish retained and withheld attack types. Set Label by = sample_id. Begin with PCA; record Sphereize data and projection settings. The website receives original 64-D hidden vectors, not precomputed PCA coordinates.",
                 "retained_types/ contains matched views of the retained attack types and benign examples. by_attack/ contains one attack type plus the same benign sample per condition. Independently loaded subsets and conditions have independently fitted projections; their coordinates, orientation and scales cannot be compared as literal movement.", "",
                 "## Cohort and representation", "",
                 f"Both views contain all {int(labels.sum())} real validation attacks and exactly the historical {int((labels == 0).sum())} benign examples, sampled with seed {reference_export['sampling_seed']}. The full validation IDs, labels and primary attack types match. Selected flow counts and active-bucket counts also match.",
                 "Vectors are the 64-unit ReLU representation entering the final binary linear layer, with dropout disabled. Labels, attack names and outcomes are metadata only. Original weights and experiment outputs are preserved.",
                 "The new fit uses the existing LSTM recipe: seed 4266, chronological buckets, hidden size 64, dropout 0.2, batch 512, AdamW learning rate 0.001, at most 30 epochs, patience 5, and equal total weights for the four training sample groups.",
                 f"New fit: best epoch {summary['fit']['best_epoch']} of {summary['fit']['epochs_completed']}; validation AP {summary['metrics']['average_precision']:.6f}. Historical validation AP {reference_summary['metrics']['average_precision']:.6f}.", "",
                 "## Interpretation limits", "",
                 "This is an exploratory comparison of two complete pipelines, not an isolated causal ablation of LSTM training labels. The scaler, frozen bucket encoder and synthetic donor plan differ. The reference model was trained historically, using a different Torch version. The same architecture and seed do not remove these differences.",
                 "The three withheld labels have zero primary_attack occurrences in the actual new training shards, including synthetic samples. Upstream raw-flow attack memberships and the bucket encoder fitting cohort were not independently reconstructed from this portable export. Mixed sequences may contain more than their primary label.",
                 "Validation was reused for checkpoint selection and threshold fitting, including the withheld types. This is not a strict unseen-attack test, untouched test evaluation or evidence of statistical significance. No test shards or training examples are included in the visualization. Retained training data were reused for the new model fit; a different synthetic donor plan was supplied with the new dataset.",
                 "The plot sample is attack-enriched. Geometry statistics depend on this fixed sample composition. Small types, repeated hosts/campaigns, and duplicate vectors limit interpretation.", "",
                 "## Distribution comparison", "",
                 "The table measures native 64-D Euclidean geometry using the same 4,917 sequences. Benign-neighbor fraction is the mean proportion of benign samples among each attack sample's 20 nearest neighbors, excluding itself. Higher means more local mixing with benign in this sample. It is not recall, an FPR, or a natural-prevalence estimate. Ties between duplicate vectors can affect individual neighborhoods. geometry_comparison.csv also records same-type neighbor fractions, centroid distances normalized by benign RMS radius, and full-validation recall at model-specific 1% FPR thresholds.", "",
                 "| Primary attack | New training | N | Historical benign neighbors | Holdout benign neighbors | Change (pp) |",
                 "|---|---|---:|---:|---:|---:|"]
        for row in comparison_rows:
            lines.append(f"| {row['primary_attack']} | {'Withheld' if row['withheld_from_new_training'] else 'Retained'} | {row['n']} | "
                         f"{row['historical_baseline_benign_fraction_among_20_neighbors']:.1%} | "
                         f"{row['holdout_lstm_benign_fraction_among_20_neighbors']:.1%} | "
                         f"{100 * row['delta_benign_neighbor_fraction']:+.1f} |")
        lines += ["", "![Independent PCA previews](pca_comparison.png)", "",
                  "PCA panels are fitted independently without sphereization and use the same colors. Crosses identify types withheld from new training. Inspect overlap within each panel; do not interpret cross-panel coordinates as sample displacement.", "",
                  "All delivered TSV pairs were read back and checked. Model/threshold identities, reference file hashes, sample alignment, frozen-logit reconstruction and reload-score agreement were checked. See manifest.json for measured errors and all file identities.", "",
                  "## Reproduce", "", "```powershell",
                  f'.\\.venv-cnn\\Scripts\\python.exe train/export_holdout_lstm_projector.py --run "{args.run.resolve()}"',
                  "```", ""]
        (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(output / "status.json", {"complete": True, "test_opened": False})
        print(f"Saved matched Projector exports to {output}", flush=True)
        print(f"Reload score max difference: {score_delta:.9g}", flush=True)
        return output
    finally:
        validation.close()
        training.close()


if __name__ == "__main__":
    main()
