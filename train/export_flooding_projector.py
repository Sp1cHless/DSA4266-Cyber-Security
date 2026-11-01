"""Inspect flooding test cases against matched real training reference sequences."""
import argparse
from collections import Counter
import csv
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

from train.common.data import Scheme2, check_disjoint
from train.common.experiment import export_evaluation, load_model, score
from train.common.io import DEFAULT_DATASET, PROJECT, ROOT, fresh, read_json, sha256, unique_name, write_json
from train.export_projector import neural_vectors, outcome


def write_rows(path, rows, delimiter="\t"):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_pair(folder, vectors, rows):
    folder = fresh(folder)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(rows) or not np.isfinite(vectors).all():
        raise ValueError("Invalid Projector matrix")
    np.savetxt(folder / "vectors.tsv", vectors, delimiter="\t", fmt="%.9g", encoding="utf-8")
    write_rows(folder / "metadata.tsv", rows)
    restored = np.loadtxt(folder / "vectors.tsv", delimiter="\t", dtype=np.float32, ndmin=2)
    with (folder / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    expected = [{key: str(value) for key, value in row.items()} for row in rows]
    if not np.array_equal(vectors, restored) or metadata != expected:
        raise ValueError("Projector TSV round-trip failed")
    return {"path": str(folder), "rows": len(rows), "dimensions": vectors.shape[1],
            "vectors_sha256": sha256(folder / "vectors.tsv"),
            "metadata_sha256": sha256(folder / "metadata.tsv"), "round_trip_verified": True}


def match_ids(split, ids):
    order = np.argsort(split["ids"])
    sorted_ids = split["ids"][order]
    positions = np.searchsorted(sorted_ids, ids)
    if np.any(positions >= len(split)) or not np.array_equal(sorted_ids[positions], ids):
        raise ValueError("Historical dataset is missing matched sequence IDs")
    return order[positions]


def geometry(vectors, ntrain, rows, heldout):
    train, test = vectors[:ntrain], vectors[ntrain:]
    train_y = np.array([row["label"] == "Attack" for row in rows[:ntrain]])
    test_attacks = np.array([row["primary_attack"] for row in rows[ntrain:]])
    test_y = np.array([row["label"] == "Attack" for row in rows[ntrain:]])
    attack_ids = np.flatnonzero(test_y)
    query = test[attack_ids]
    neighbors = NearestNeighbors(n_neighbors=20, algorithm="brute", n_jobs=4).fit(train)
    distances, indices = neighbors.kneighbors(query)
    benign_fraction = (~train_y[indices]).mean(axis=1)
    nearest = []
    for label in (False, True):
        model = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=4).fit(train[train_y == label])
        nearest.append(model.kneighbors(query)[0][:, 0])
    ratio = nearest[1] / np.maximum(nearest[0], 1e-12)
    result = []
    for attack in sorted(set(test_attacks[test_y])):
        keep = test_attacks[attack_ids] == attack
        result.append({"primary_attack": str(attack), "withheld": attack in heldout,
                       "n_test": int(keep.sum()),
                       "train_benign_fraction_among_20nn": float(benign_fraction[keep].mean()),
                       "median_nearest_train_attack_over_benign_distance": float(np.median(ratio[keep])),
                       "median_nearest_train_distance": float(np.median(distances[keep, 0]))})
    return result


def train_pca(vectors, ntrain):
    model = PCA(n_components=3, svd_solver="full")
    model.fit(vectors[:ntrain])
    return model.transform(vectors).astype(np.float32), {
        "fit_cohort": "selected real training reference only",
        "fit_n": ntrain, "test_used_to_fit_pca": False,
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "sphereized": False, "standardized": False,
    }, model


def preview(output, matrices, projections, rows, ntrain, threshold):
    try:
        import matplotlib
    except ModuleNotFoundError:
        sys.path.append(str(ROOT / "cache/diagnostic_plot_dependencies"))
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = np.array([row["split"] for row in rows])
    labels = np.array([row["label"] == "Attack" for row in rows])
    flood = np.array([row["training_status"] == "Withheld flooding" for row in rows])
    names = np.array([row["primary_attack"] for row in rows])
    outcomes = np.array([row["outcome_1pct"] for row in rows])
    train_benign = (split == "train") & ~labels
    train_attack = (split == "train") & labels
    test_benign = (split == "test") & ~labels
    test_flood = (split == "test") & flood
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(sorted(set(names[test_flood])))}

    def background(ax, points):
        ax.scatter(points[train_benign, 0], points[train_benign, 1], s=4, c="#adb5bd", alpha=.17,
                   rasterized=True, label="Train benign (sample)")
        ax.scatter(points[train_attack, 0], points[train_attack, 1], s=6, c="#159895", alpha=.22,
                   rasterized=True, label="Train retained attacks (all real)")
        ax.scatter(points[test_benign, 0], points[test_benign, 1], s=5, c="#34495e", alpha=.14,
                   rasterized=True, label="Test benign (sample)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, key, title in ((axes[0, 0], "holdout_input", "All-flooding holdout: input (660D)"),
                           (axes[0, 1], "holdout_hidden", "All-flooding holdout: LSTM hidden (64D)"),
                           (axes[1, 0], "historical_hidden", "Historical all-types LSTM: matched sequences (64D)")):
        points = matrices[key]
        background(ax, points)
        for name, color in colors.items():
            keep = test_flood & (names == name)
            if keep.any():
                ax.scatter(points[keep, 0], points[keep, 1], s=45, marker="x", c=[color],
                           linewidths=1.5, label=f"Test {name} (n={keep.sum()})", zorder=4)
        if key == "historical_hidden":
            support = matrices["historical_flooding_train"]
            ax.scatter(support[:, 0], support[:, 1], s=32, marker="o", facecolors="none",
                       edgecolors="#963cbd", linewidths=.9, alpha=.8,
                       label=f"Historical train flooding (n={len(support)})", zorder=3)
            ax.legend(loc="upper left", fontsize=7)
        variance = projections[key]["explained_variance_ratio"]
        ax.set_title(title)
        ax.set_xlabel(f"Train-reference PC1 ({variance[0]:.1%})")
        ax.set_ylabel(f"Train-reference PC2 ({variance[1]:.1%})")
        ax.grid(alpha=.15)
    ax = axes[1, 1]
    points = matrices["holdout_hidden"]
    background(ax, points)
    for result, color, marker in (("FN", "#d62828", "x"), ("TP", "#2463eb", "+")):
        keep = test_flood & (outcomes == result)
        ax.scatter(points[keep, 0], points[keep, 1], c=color, marker=marker, s=52,
                   label=f"Test flooding {result} (n={keep.sum()})", zorder=4)
    variance = projections["holdout_hidden"]["explained_variance_ratio"]
    ax.set_title(f"Holdout flooding decisions: frozen threshold = {threshold:.4g}")
    ax.set_xlabel(f"Train-reference PC1 ({variance[0]:.1%})")
    ax.set_ylabel(f"Train-reference PC2 ({variance[1]:.1%})")
    ax.grid(alpha=.15)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    extra_handles, extra_labels = ax.get_legend_handles_labels()
    fig.legend(handles + extra_handles[-2:], legend_labels + extra_labels[-2:], loc="lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Where do withheld flooding test sequences land?\n"
                 "Each space has its own training-fitted PCA; coordinates across spaces are not aligned", fontsize=14)
    fig.tight_layout(rect=(0, .18, 1, .95))
    fig.savefig(output / "pca_distribution.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, condition in zip(axes, ("historical_baseline", "all_flooding_holdout")):
        metadata = read_json(output / f"{condition}_scores.json")
        scores = np.asarray(metadata["test_scores"])
        local_names = np.asarray(metadata["test_attacks"])
        for name, color in colors.items():
            values = scores[local_names == name]
            if len(values):
                ax.scatter(values, np.full(len(values), list(colors).index(name)), c=[color],
                           s=20, alpha=.7, marker="x")
        ax.axvline(metadata["threshold"], color="black", ls="--", label="Frozen validation 1% FPR threshold")
        ax.set_yticks(range(len(colors)), list(colors))
        ax.set_xlim(-.02, 1.02)
        ax.set_xlabel("Attack score (not a calibrated probability)")
        ax.set_title(condition.replace("_", " ").title())
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle("All flooding test attacks: model scores and frozen thresholds")
    fig.tight_layout()
    fig.savefig(output / "flooding_scores.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "datasets/dataset_holdout_all_flooding")
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc")
    parser.add_argument("--reference-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--benign-count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.benign_count <= 0 or args.seed < 0:
        raise ValueError("Positive benign-count and nonnegative seed required")
    output = fresh(args.out or ROOT / "runs" / unique_name("all_flooding_test_projector"))
    write_json(output / "status.json", {"complete": False, "test_opened": False})
    data = Scheme2(args.dataset)
    heldout = set(data.manifest["train_holdout_attacks"])
    manifest = {"complete": False, "test_opened": True, "seed": args.seed,
                "training_data_reused_for_fit_and_visualization": True,
                "test_used_to_fit_pca_or_model_or_thresholds": False,
                "validation_includes_withheld_attacks_for_selection": True,
                "historical_test_previously_inspected": True, "models": {}, "exports": {},
                "script_sha256": sha256(__file__), "withheld_attacks": sorted(heldout)}
    opened = []
    matrices, projection_info, geometry_rows, table = {}, {}, [], []
    base_ids = base_labels = base_attacks = None
    try:
        training = data.load("train"); opened.append(training)
        testing = data.load("test"); opened.append(testing)
        check_disjoint(training, testing)
        counts = Counter(training["attack"].tolist())
        if any(counts[name] for name in heldout):
            raise ValueError("A withheld primary attack remains in training")
        with np.load(args.dataset / "donor_plan.npz", allow_pickle=False) as donors:
            if np.isin(donors["target_attack"], sorted(heldout)).any():
                raise ValueError("A withheld target attack remains in donor plan")
            donor_audit = dict(Counter(donors["target_attack"].tolist()))
        manifest["training_primary_attack_counts"] = dict(counts)
        manifest["donor_target_counts"] = donor_audit
        manifest["test_primary_attack_counts"] = dict(Counter(testing["attack"].tolist()))
        rng = np.random.default_rng(args.seed)
        indices = {}
        for name, split in (("train", training), ("test", testing)):
            real = np.char.startswith(split["types"], "real_")
            negatives = np.flatnonzero(real & (split["y"] == 0))
            positives = np.flatnonzero(real & (split["y"] == 1))
            if len(negatives) < args.benign_count:
                raise ValueError("Too few real benign examples")
            indices[name] = np.sort(np.concatenate((positives, rng.choice(negatives, args.benign_count, replace=False))))
        ntrain = len(indices["train"])
        manifest["selected_counts"] = {key: len(value) for key, value in indices.items()}
        base_ids = np.concatenate([training["ids"][indices["train"]], testing["ids"][indices["test"]]])
        base_labels = np.concatenate([training["y"][indices["train"]], testing["y"][indices["test"]]])
        base_attacks = np.concatenate([training["attack"][indices["train"]], testing["attack"][indices["test"]]])
        condition_specs = (("all_flooding_holdout", args.run, data, training, testing),
                           ("historical_baseline", args.reference_run, Scheme2(args.reference_dataset), None, None))
        all_rows = {}
        for condition, run, dataset, train_split, test_split in condition_specs:
            print(f"Exporting {condition}...", flush=True)
            summary, config = read_json(run / "summary.json"), read_json(run / "config.json")
            if not summary.get("complete") or summary["model"] != "lstm" or summary["dataset_manifest_sha256"] != dataset.identity:
                raise ValueError("Run/dataset identity mismatch")
            hashes = {"model.pt": summary["model_sha256"], "thresholds.json": summary["thresholds_sha256"]}
            if any(sha256(run / filename) != value for filename, value in hashes.items()):
                raise ValueError("Frozen run changed")
            thresholds = read_json(run / "thresholds.json")["values"]
            if train_split is None:
                train_split = dataset.load("train"); opened.append(train_split)
                test_split = dataset.load("test"); opened.append(test_split)
            check_disjoint(train_split, test_split)
            selected = {"train": match_ids(train_split, base_ids[:ntrain]),
                        "test": match_ids(test_split, base_ids[ntrain:])}
            x = np.concatenate([train_split["x"][selected["train"]], test_split["x"][selected["test"]]])
            mask = np.concatenate([train_split["mask"][selected["train"]], test_split["mask"][selected["test"]]])
            for key, expected in (("y", base_labels), ("attack", base_attacks)):
                actual = np.concatenate([train_split[key][selected["train"]], test_split[key][selected["test"]]])
                if not np.array_equal(actual, expected):
                    raise ValueError(f"Matched cohort differs: {key}")
            vectors, selected_scores, details = neural_vectors("lstm", config, run, x, mask, args.device)
            model = load_model("lstm", config, run, args.device, 4)
            test_scores = score("lstm", model, test_split, args.device, 4)
            reload_delta = float(np.max(np.abs(test_scores[selected["test"]] - selected_scores[ntrain:])))
            if reload_delta > .001:
                raise ValueError("Selected and full-test scores differ")
            # Test metadata uses exactly the full-test evaluation scores and decisions.
            selected_scores[ntrain:] = test_scores[selected["test"]]
            full_folder = fresh(output / "full_test" / condition)
            report = export_evaluation(full_folder, test_split, test_scores, thresholds)
            write_json(full_folder / "summary.json", {"complete": True, "split": "test", "model": "lstm",
                       "thresholds_refitted": False, "metrics": report, "run": str(run.resolve())})
            rows = []
            for pos, (sample_id, label, attack, prediction) in enumerate(zip(base_ids, base_labels, base_attacks, selected_scores)):
                is_train = pos < ntrain
                status = "Benign" if label == 0 else "Withheld flooding" if attack in heldout else "Retained attack"
                rows.append({"sample_id": int(sample_id), "label": "Attack" if label else "Benign",
                             "primary_attack": str(attack), "split": "train" if is_train else "test",
                             "training_status": status, "cohort": f"{'Train' if is_train else 'Test'} {status}",
                             "attack_score": format(float(prediction), ".9g"),
                             "outcome_1pct": outcome(label, prediction >= thresholds["validation_fpr_1pct"]),
                             "outcome_0_1pct": outcome(label, prediction >= thresholds["validation_fpr_0.1pct"]),
                             "active_buckets": int(mask[pos].sum()), "condition": condition})
            all_rows[condition] = rows
            manifest["exports"][condition] = export_pair(output / condition / "hidden64", vectors, rows)
            hidden_key = "holdout_hidden" if condition == "all_flooding_holdout" else "historical_hidden"
            matrices[hidden_key], projection_info[hidden_key], pca = train_pca(vectors, ntrain)
            np.savez_compressed(output / f"{hidden_key}_pca_fit.npz", mean=pca.mean_, components=pca.components_,
                                coordinates=matrices[hidden_key], sample_id=base_ids)
            if condition == "historical_baseline":
                support_ids = np.flatnonzero((train_split["types"] == "real_malicious") &
                                             np.isin(train_split["attack"], sorted(heldout)))
                support_vectors, support_scores, support_details = neural_vectors(
                    "lstm", config, run, np.array(train_split["x"][support_ids]),
                    np.array(train_split["mask"][support_ids]), args.device)
                matrices["historical_flooding_train"] = pca.transform(support_vectors)
                support_rows = []
                for index, prediction in zip(support_ids, support_scores):
                    support_rows.append({"sample_id": int(train_split["ids"][index]), "label": "Attack",
                        "primary_attack": str(train_split["attack"][index]), "split": "train",
                        "training_status": "Historical flooding support", "cohort": "Train Historical flooding",
                        "attack_score": format(float(prediction), ".9g"),
                        "outcome_1pct": outcome(1, prediction >= thresholds["validation_fpr_1pct"]),
                        "outcome_0_1pct": outcome(1, prediction >= thresholds["validation_fpr_0.1pct"]),
                        "active_buckets": int(train_split["mask"][index].sum()), "condition": condition})
                extended_vectors = np.concatenate((vectors[:ntrain], support_vectors, vectors[ntrain:]))
                extended_rows = rows[:ntrain] + support_rows + rows[ntrain:]
                manifest["exports"]["historical_with_training_flooding"] = export_pair(
                    output / condition / "with_training_flooding", extended_vectors, extended_rows)
                manifest["historical_real_flooding_training_counts"] = dict(Counter(
                    train_split["attack"][support_ids].tolist()))
                for result in geometry(extended_vectors, ntrain + len(support_vectors), extended_rows, heldout):
                    geometry_rows.append({"condition": condition, "space": "hidden64_with_flooding_train", **result})
            manifest["exports"][condition + "_train_pca3"] = export_pair(
                output / condition / "train_pca3", matrices[hidden_key], rows)
            flat = np.concatenate((x.reshape(len(x), -1), mask), axis=1)
            for space, matrix in (("hidden64", vectors), ("input660", flat)):
                for result in geometry(matrix, ntrain, rows, heldout):
                    geometry_rows.append({"condition": condition, "space": space, **result})
            if condition == "all_flooding_holdout":
                manifest["exports"]["holdout_input660"] = export_pair(output / condition / "input660", flat, rows)
                matrices["holdout_input"], projection_info["holdout_input"], pca = train_pca(flat, ntrain)
                np.savez_compressed(output / "holdout_input_pca_fit.npz", mean=pca.mean_, components=pca.components_,
                                    coordinates=matrices["holdout_input"], sample_id=base_ids)
                for attack in sorted(heldout):
                    keep = np.array([row["label"] == "Benign" or row["split"] == "train" or row["primary_attack"] == attack for row in rows])
                    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", attack)
                    manifest["exports"]["by_attack_" + attack] = export_pair(
                        output / "by_attack" / safe_name, vectors[keep], [row for row, selected_row in zip(rows, keep) if selected_row])
            write_json(output / f"{condition}_scores.json", {"test_scores": test_scores[test_split["y"] == 1].tolist(),
                       "test_attacks": test_split["attack"][test_split["y"] == 1].tolist(),
                       "test_labels": test_split["y"][test_split["y"] == 1].tolist(),
                       "threshold": thresholds["validation_fpr_1pct"]})
            for attack, metrics in report["per_primary_attack"].items():
                keep = (test_split["y"] == 1) & (test_split["attack"] == attack)
                table.append({"condition": condition, "primary_attack": attack, "withheld": attack in heldout,
                              **metrics, "median_score": float(np.median(test_scores[keep])),
                              "recall_at_validation_fpr_0.1pct": float((test_scores[keep] >= thresholds["validation_fpr_0.1pct"]).mean())})
            manifest["models"][condition] = {"run": str(run.resolve()), "dataset": str(dataset.path),
                    "dataset_manifest_sha256": dataset.identity, **hashes, **details,
                    "selected_vs_full_test_score_max_delta": reload_delta, "thresholds": thresholds,
                    "full_test_metrics": report,
                    "preprocessing": {key: dataset.manifest[key] for key in
                                      ("schema_sha256", "scaler_sha256", "embedding_checkpoint_sha256", "donor_plan_sha256")}}
            if any(sha256(run / filename) != value for filename, value in hashes.items()):
                raise ValueError("Source weights or thresholds changed during analysis")
        write_rows(output / "test_attack_results.csv", table, ",")
        write_rows(output / "native_geometry.csv", geometry_rows, ",")
        manifest["pca"] = projection_info
        preview(output, matrices, projection_info, all_rows["all_flooding_holdout"], ntrain,
                manifest["models"]["all_flooding_holdout"]["thresholds"]["validation_fpr_1pct"])
        lines = ["# Flooding leave-group-out distribution inspection", "",
                 "## Open the same TensorFlow Embedding Projector", "",
                 "Open https://projector.tensorflow.org/ and click Load. Choose all_flooding_holdout/hidden64/vectors.tsv (Step 1) and metadata.tsv (Step 2).", 
                 "Use Color by = cohort first, then primary_attack or outcome_1pct; Label by = sample_id. Turn Sphereize data OFF to preserve the native hidden geometry. Choose PCA. The website fits PCA on the entire uploaded visualization sample; these are original 64D vectors.",
                 "Search for Test Withheld flooding using cohort to inspect all withheld flooding cases. The by_attack folders retain all training references and sampled test benign rows, plus one flooding subtype. Each reload fits a new projection.",
                 "historical_baseline/hidden64 uses the historical all-types LSTM on exactly the same real sequence IDs, with its own original preprocessing. input660 contains position-preserving flattened 10x65 inputs followed by ten mask bits. Labels never enter coordinates.",
                 "historical_baseline/with_training_flooding additionally includes all real historical flooding training rows, showing the support removed from the new training dataset. The historical PCA panel marks them with hollow purple circles, transformed using the same matched-reference PCA fit. training_status describes exclusion from NEW training, even in historical metadata.",
                 "train_pca3 contains coordinates fitted ONLY on the selected real training reference, then used to transform test. PCA screenshots use this same training-only fit. Website PCA may rotate these three coordinates; direct dimensions are available under CUSTOM.", "",
                 "## Cohort and interpretation", "",
                 f"Training reference: all real retained training attacks plus {args.benign_count} real benign rows sampled without replacement. Test: all real attacks plus {args.benign_count} sampled real benign rows. Seed {args.seed}; selected counts {manifest['selected_counts']}.",
                 "The main plot is attack-enriched and excludes synthetic training rows. Native neighbor fractions describe this selected real reference, not the full training support or natural class prevalence. Train rows are reused after fitting and are not validation evidence.",
                 "The seven excluded primary_attack labels occur zero times in actual training shards (including synthetic rows) and donor-plan target_attack. Secondary raw-flow memberships and the upstream bucket-encoder fit cohort cannot be independently established from this portable dataset; complete upstream exclusion is unverified.",
                 "The new LSTM uses the existing recipe: hidden size 64, dropout 0.2, batch 512, AdamW lr 0.001, up to 30 epochs, patience 5, seed 4266, equal weight across four training groups. Training reuses the delivered training cohort. Validation INCLUDING flooding selects the checkpoint and thresholds: this is training-label exclusion, not strict unseen-type model selection.",
                 "Test was evaluated with frozen models and validation thresholds. No test fitting or threshold tuning occurred. Test FPR is reported separately and need not equal its validation target. Historical project work has already inspected the Scheme2 test split.",
                 "Historical vs holdout compares complete pipelines: scaler, bucket encoder and donor plan differ. It does not isolate the effect of removing attack labels. PCA fits across panels are independent; axes, scales and literal point movement across panels are not comparable.",
                 "PCA preserves only the displayed variance directions. Nearby points in 2D may be distant in 64D/660D. native_geometry.csv uses Euclidean distances before PCA and ONLY selected real training references; attack/benign nearest-distance ratios above one mean closer to sampled training benign. No causal failure mechanism is established by these plots alone.", "",
                 "## Full-test attack recall at frozen validation 1% FPR thresholds", "",
                 "| Primary attack | Test N | Historical recall | Holdout recall | Holdout median score |",
                 "|---|---:|---:|---:|---:|"]
        for attack in sorted(heldout):
            matches = {row["condition"]: row for row in table if row["primary_attack"] == attack}
            if not matches:
                lines.append(f"| {attack} | 0 | N/A | N/A | N/A |")
                continue
            old, new = matches["historical_baseline"], matches["all_flooding_holdout"]
            lines.append(f"| {attack} | {new['n']} | {old['recall_at_validation_fpr_1pct']:.1%} | {new['recall_at_validation_fpr_1pct']:.1%} | {new['median_score']:.5g} |")
        for condition, info in manifest["models"].items():
            op = info["full_test_metrics"]["operating_points"]["validation_fpr_1pct"]
            lines += ["", f"{condition}: full-test FPR {op['fpr']:.4%}; AP {info['full_test_metrics']['average_precision']:.6f}; threshold {op['threshold']:.9g}."]
        lines += ["", "![Distribution PCA](pca_distribution.png)", "", "![Flooding scores](flooding_scores.png)", "",
                  "All TSV files were read back and checked for exact float32 values and metadata alignment. Saved weights/threshold hashes and matched IDs/labels were checked; final-layer logit reconstruction and selected/full-test scores were checked. manifest.json records measured errors and provenance.", "",
                  "## Reproduce", "", "```powershell",
                  f'.\\.venv-cnn\\Scripts\\python.exe train/export_flooding_projector.py --run "{args.run.resolve()}"', "```", ""]
        (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
        manifest["complete"] = True
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", {"complete": True, "test_opened": True})
        print(f"Saved Projector analysis to {output}", flush=True)
        return output
    except BaseException as error:
        write_json(output / "status.json", {"complete": False, "error": repr(error), "test_opened": True})
        raise
    finally:
        for split in opened:
            split.close()


if __name__ == "__main__":
    main()
