"""Export frozen model representations for TensorFlow Embedding Projector."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    script_folder = Path(__file__).resolve().parent
    # Direct execution must not shadow the installed xgboost with train/xgboost.
    sys.path = [str(script_folder.parent)] + [
        path for path in sys.path if Path(path or ".").resolve() != script_folder
    ]

from train.common.data import Scheme2
from train.common.io import DEFAULT_DATASET, ROOT, environment, fresh, read_json, sha256, unique_name, write_json


MODELS = {"cnn", "lstm", "hybrid", "xgboost"}


def neural_vectors(name, config, run, x, mask, device):
    import torch
    from train.common.torch_runner import load, setup

    setup(config["seed"], 4, device)
    model = load(name, config, run / "model.pt", device).eval()
    chunks, scores, reconstruction_errors = [], [], []

    def capture(module, inputs):
        chunks.append(inputs[0].detach().cpu().numpy().copy())

    hook = model.head[-1].register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for start in range(0, len(x), 512):
                logits = model(torch.as_tensor(x[start:start + 512], device=device),
                               torch.as_tensor(mask[start:start + 512], device=device))
                scores.append(torch.sigmoid(logits).cpu().numpy())
                head = model.head[-1]
                reconstructed = chunks[-1] @ head.weight.detach().cpu().numpy().T
                reconstructed = reconstructed[:, 0] + head.bias.detach().cpu().numpy()[0]
                reconstruction_errors.append(float(np.max(np.abs(reconstructed - logits.cpu().numpy()))))
    finally:
        hook.remove()
    error = max(reconstruction_errors)
    if error > 1e-4:
        raise ValueError(f"{name}: final-layer reconstruction error {error}")
    return np.concatenate(chunks), np.concatenate(scores), {
        "representation": "Input to the final binary linear layer, after the 64-unit ReLU; dropout disabled.",
        "logit_reconstruction_max_absolute_error": error,
        "projection_fitted_during_export": False,
    }


def tree_vectors(run, x, mask):
    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(run / "model.ubj")
    model.set_param({"device": "cpu", "nthread": 4})
    matrix = xgb.DMatrix(np.concatenate((x.reshape(len(x), -1), mask), axis=1), nthread=4)
    leaves = model.predict(matrix, pred_leaf=True, strict_shape=True).reshape(len(x), -1).astype(np.int64)
    trees = [json.loads(value) for value in model.get_dump(dump_format="json")]
    if leaves.shape[1] != len(trees):
        raise ValueError("Unexpected number of XGBoost leaf columns")
    vectors = np.empty(leaves.shape, dtype=np.float32)
    for column, tree in enumerate(trees):
        values, pending = {}, [tree]
        while pending:
            node = pending.pop()
            if "leaf" in node:
                values[node["nodeid"]] = node["leaf"]
            else:
                pending.extend(node["children"])
        vectors[:, column] = [values[int(node)] for node in leaves[:, column]]
    base = json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"]
    base = np.asarray(json.loads(base), dtype=np.float64).reshape(-1)
    if len(base) != 1 or not 0 < base[0] < 1:
        raise ValueError("Expected one logistic base score")
    bias = float(np.log(base[0] / (1 - base[0])))
    margin = model.predict(matrix, output_margin=True).reshape(-1)
    error = float(np.max(np.abs(vectors.sum(axis=1, dtype=np.float64) + bias - margin)))
    if error > 1e-4:
        raise ValueError(f"XGBoost additive reconstruction error {error}")
    return vectors, model.predict(matrix).reshape(-1), {
        "representation": "One learned leaf response per saved tree; numeric leaf IDs are never used as coordinates.",
        "logit_reconstruction_max_absolute_error": error,
        "logit_bias": bias,
        "projection_fitted_during_export": False,
    }


def outcome(label, prediction):
    return ("TP" if prediction else "FN") if label else ("FP" if prediction else "TN")


def export_files(folder, vectors, split, indices, scores, fresh_scores, thresholds):
    if vectors.ndim != 2 or vectors.shape[0] != len(indices) or not np.isfinite(vectors).all():
        raise ValueError("Invalid representation matrix")
    vector_path, metadata_path = folder / "vectors.tsv", folder / "metadata.tsv"
    np.savetxt(vector_path, vectors, fmt="%.9g", delimiter="\t", encoding="utf-8")
    header = ["sample_id", "label", "primary_attack", "split", "saved_score", "reloaded_score",
              "outcome_1pct", "outcome_0_1pct", "active_buckets", "flow_count"]
    ids = split["ids"][indices]
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for position, index in enumerate(indices):
            label, score = int(split["y"][index]), float(scores[position])
            writer.writerow([int(ids[position]), "Attack" if label else "Benign", str(split["attack"][index]),
                             "validation", format(score, ".9g"), format(float(fresh_scores[position]), ".9g"),
                             outcome(label, score >= thresholds["validation_fpr_1pct"]),
                             outcome(label, score >= thresholds["validation_fpr_0.1pct"]),
                             int(split["mask"][index].sum()), int(split["flows"][index])])
    # Read the actual delivery files back, including row alignment and numeric precision.
    restored = np.loadtxt(vector_path, delimiter="\t", dtype=np.float32, ndmin=2)
    if not np.array_equal(restored, vectors):
        raise ValueError("TSV numeric round-trip failed")
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(ids) or [int(row["sample_id"]) for row in rows] != ids.tolist():
        raise ValueError("Metadata row alignment failed")
    return {"rows": len(ids), "dimensions": vectors.shape[1],
            "vectors_sha256": sha256(vector_path), "metadata_sha256": sha256(metadata_path),
            "tsv_round_trip_verified": True,
            "selected_outcomes_1pct": dict(Counter(row["outcome_1pct"] for row in rows)),
            "selected_outcomes_0_1pct": dict(Counter(row["outcome_0_1pct"] for row in rows))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True, help="Completed four-model comparison.json")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--benign-count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=4266, help="Visualization sampling seed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--score-atol", type=float, default=0.001,
                        help="Maximum absolute reload score difference across backends/versions (recorded, never silently ignored)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.benign_count <= 0 or args.seed < 0 or not 0 <= args.score_atol <= 0.001:
        raise ValueError("benign-count must be positive, seed nonnegative, and score-atol within [0, 0.001]")
    comparison = read_json(args.comparison)
    if not comparison.get("complete_screen") or comparison.get("split") != "validation":
        raise ValueError("A complete validation comparison is required")
    runs = {row["model"]: Path(row["run"]) for row in comparison["rows"]}
    if set(runs) != MODELS or len(comparison["rows"]) != 4:
        raise ValueError("Expected exactly one run for each model")
    summaries = {name: read_json(run / "summary.json") for name, run in runs.items()}
    for key in ("dataset_manifest_sha256", "train_cohort_sha256", "validation_cohort_sha256",
                "evaluation_policy", "weighting", "seed"):
        if len({summary[key] for summary in summaries.values()}) != 1:
            raise ValueError(f"Incompatible runs: {key}")
    data = Scheme2(args.dataset, args.cache)
    if any(summary["dataset_manifest_sha256"] != data.identity for summary in summaries.values()):
        raise ValueError("Dataset differs from the model runs")
    output = fresh(args.out or ROOT / "runs" / unique_name("projector_four_models_validation"))
    write_json(output / "status.json", {"complete": False, "test_opened": False})
    split = data.load("validation")
    try:
        if any(summary["validation_cohort_sha256"] != split.info["cohort_sha256"] for summary in summaries.values()):
            raise ValueError("Validation cohort differs from the model runs")
        positives, negatives = np.flatnonzero(split["y"] == 1), np.flatnonzero(split["y"] == 0)
        if args.benign_count > len(negatives):
            raise ValueError("Requested more benign examples than available")
        rng = np.random.default_rng(args.seed)
        indices = np.sort(np.concatenate((positives, rng.choice(negatives, args.benign_count, replace=False))))
        x, mask = np.array(split["x"][indices]), np.array(split["mask"][indices])
        provenance = {"complete": False, "split": "validation", "test_opened": False,
                      "training_data_reused_for_visualization": False,
                      "validation_reused_after_checkpoint_and_threshold_selection": True,
                      "sampling_seed": args.seed, "selected_benign": args.benign_count,
                      "selected_attacks": len(positives), "population_n": len(split),
                      "population_attack_prevalence": float(split["y"].mean()),
                      "selected_attack_prevalence": float(split["y"][indices].mean()),
                      "dataset_manifest_sha256": data.identity,
                      "validation_cohort_sha256": split.info["cohort_sha256"],
                      "comparison_path": str(args.comparison.resolve()),
                      "comparison_sha256": sha256(args.comparison),
                      "export_script_sha256": sha256(__file__), "environment": environment(),
                      "neural_export_device": args.device, "reload_score_absolute_tolerance": args.score_atol,
                      "models": {}}
        for name in ("lstm", "hybrid", "cnn", "xgboost"):
            run, summary = runs[name], summaries[name]
            if not summary.get("complete") or summary.get("split") != "validation" or summary.get("test_used"):
                raise ValueError(f"{name}: expected a completed validation-only screening run")
            model_path = run / ("model.ubj" if name == "xgboost" else "model.pt")
            hashes = {model_path: summary["model_sha256"], run / "thresholds.json": summary["thresholds_sha256"]}
            if any(sha256(path) != expected for path, expected in hashes.items()):
                raise ValueError(f"{name}: frozen model or thresholds changed")
            with np.load(run / "predictions.npz", allow_pickle=False) as predictions:
                for saved_key, split_key in (("sample_id", "ids"), ("y", "y"), ("primary_attack", "attack")):
                    if not np.array_equal(predictions[saved_key], split[split_key]):
                        raise ValueError(f"{name}: predictions do not match validation rows")
                saved_scores = predictions["score"][indices].copy()
            if name == "xgboost":
                vectors, fresh_scores, details = tree_vectors(run, x, mask)
            else:
                vectors, fresh_scores, details = neural_vectors(name, read_json(run / "config.json"), run, x, mask, args.device)
            delta = float(np.max(np.abs(fresh_scores - saved_scores)))
            if not np.allclose(fresh_scores, saved_scores, atol=args.score_atol, rtol=0):
                raise ValueError(f"{name}: reloaded scores differ from saved predictions (max {delta})")
            folder = fresh(output / name)
            thresholds = read_json(run / "thresholds.json")["values"]
            files = export_files(folder, vectors, split, indices, saved_scores, fresh_scores, thresholds)
            if any(sha256(path) != expected for path, expected in hashes.items()):
                raise ValueError(f"{name}: source files changed during export")
            provenance["models"][name] = {**files, **details, "run": str(run.resolve()),
                                           "model_sha256": summary["model_sha256"],
                                           "thresholds_sha256": summary["thresholds_sha256"],
                                           "reloaded_score_max_absolute_difference": delta,
                                           "reloaded_threshold_decision_disagreements": {
                                               key: int(np.count_nonzero((saved_scores >= value) != (fresh_scores >= value)))
                                               for key, value in thresholds.items()}}
            print(f"Exported {name}: {vectors.shape}; score max difference={delta:.3g}", flush=True)
        provenance["complete"] = True
        write_json(output / "manifest.json", provenance)
        lines = ["# Frozen model representations for TensorFlow Embedding Projector", "",
                 "Open https://projector.tensorflow.org/ and click Load. For each model, choose its vectors.tsv in Step 1 and its metadata.tsv in Step 2.",
                 "Set Label by to sample_id and Color by to label. Also inspect primary_attack, outcome_1pct, and outcome_0_1pct.",
                 "Begin with PCA, then explore UMAP or t-SNE. Record the projection settings and Sphereize data setting when saving figures.", "",
                 "## Exported representations", "", "| Model | Rows | Dimensions | Representation |",
                 "|---|---:|---:|---|"]
        for name, report in provenance["models"].items():
            lines.append(f"| {name} | {report['rows']} | {report['dimensions']} | {report['representation']} |")
        lines += ["", "## Sampling and interpretation", "",
                  f"All {len(positives)} real validation attacks and {args.benign_count} benign validation sequences sampled without replacement with seed {args.seed}. Every model uses the same sample IDs in the same order.",
                  f"The selected attack prevalence is {provenance['selected_attack_prevalence']:.4%}; natural validation prevalence is {provenance['population_attack_prevalence']:.4%}. This attack-enriched sample is for visualization, not performance estimation.",
                  "No training or test examples are exported. These validation examples were previously used for checkpoint and threshold selection; this is exploratory visualization, not independent held-out evidence. Historical project work has already inspected the Scheme2 test split.",
                  "Frozen models are only loaded for inference. No model, scaler, threshold, or representation projection is fitted during export. The website computes visualization projections on the exported sample.",
                  "XGBoost uses learned tree responses, not arbitrary numeric leaf IDs. These are different from neural hidden features. Compare class overlap within each plot; axes, scales, and distances across model plots are not directly comparable.",
                  "Metadata uses the saved validation scores and frozen model-specific thresholds. TP/FP/TN/FN counts refer only to this selected sample. Raw source-host addresses are excluded.",
                  "Reloaded scores may differ slightly across inference backends or library versions. Both saved_score and reloaded_score are exported. The manifest records maximum differences, the explicit tolerance (at most 0.001), and any changed threshold decisions; original run results are never replaced.",
                  "Ground-truth labels and prediction outcomes are metadata only; they do not enter the vector coordinates. Model separation alone does not establish generalization, significance, or calibrated probabilities.",
                  "TSV values were read back exactly at float32 precision. Model/threshold hashes, sample alignment, final-logit reconstruction, and agreement with saved predictions were checked. See manifest.json for measured errors and provenance.",
                  "", "## Reproduce", "", "```powershell",
                  f'.\\.venv-cnn\\Scripts\\python.exe train/export_projector.py --comparison "{args.comparison.resolve()}" --benign-count {args.benign_count} --seed {args.seed} --device {args.device} --score-atol {args.score_atol}',
                  "```", ""]
        (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(output / "status.json", {"complete": True, "test_opened": False})
        print(f"Projector files saved to {output}", flush=True)
    finally:
        split.close()
    return output


if __name__ == "__main__":
    main()
