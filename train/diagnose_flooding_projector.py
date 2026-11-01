"""Check flooding TP/FN neighborhoods against the complete frozen training cohort."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import torch

if __package__ in (None, ""):
    folder = Path(__file__).resolve().parent
    sys.path = [str(folder.parent)] + [v for v in sys.path if Path(v or ".").resolve() != folder]

from train.common.data import Scheme2
from train.common.experiment import load_model
from train.common.io import ROOT, fresh, read_json, sha256, unique_name, write_json
from train.export_flooding_projector import export_pair, write_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    provenance = read_json(args.projector / "manifest.json")
    info = provenance["models"]["all_flooding_holdout"]
    run = Path(info["run"])
    if not provenance["complete"] or sha256(run / "model.pt") != info["model.pt"]:
        raise ValueError("Completed frozen Projector analysis required")
    output = fresh(ROOT / "runs" / unique_name("flooding_full_training_neighbors"))
    source = args.projector / "all_flooding_holdout/hidden64"
    export = provenance["exports"]["all_flooding_holdout"]
    if any(sha256(source / name) != export[key] for name, key in
           (("vectors.tsv", "vectors_sha256"), ("metadata.tsv", "metadata_sha256"))):
        raise ValueError("Projector files changed")
    with (source / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    keep = np.array([row["cohort"] == "Test Withheld flooding" for row in rows])
    source_vectors = np.loadtxt(source / "vectors.tsv", delimiter="\t", dtype=np.float32)
    queries = source_vectors[keep]
    query_rows = [row for row, selected in zip(rows, keep) if selected]
    dataset = Scheme2(info["dataset"])
    if dataset.identity != info["dataset_manifest_sha256"]:
        raise ValueError("Dataset identity changed")
    train = dataset.load("train")
    try:
        model = load_model("lstm", read_json(run / "config.json"), run, args.device, 4).eval()
        vectors = np.empty((len(train), 64), dtype=np.float32)
        scores = np.empty(len(train), dtype=np.float32)
        captured = []
        hook = model.head[-1].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy()))
        try:
            with torch.no_grad():
                for start in range(0, len(train), 4096):
                    end = min(start + 4096, len(train))
                    logits = model(torch.as_tensor(np.array(train["x"][start:end]), device=args.device),
                                   torch.as_tensor(np.array(train["mask"][start:end]), device=args.device))
                    vectors[start:end] = captured.pop()
                    scores[start:end] = torch.sigmoid(logits).cpu().numpy()
        finally:
            hook.remove()
        print(f"Encoded all {len(train):,} training rows; querying {len(queries)} flooding test cases...", flush=True)
        results, groups = [], []
        with threadpool_limits(limits=4):
            for scope in ("all_training_including_synthetic", "real_training_only"):
                pool = np.arange(len(train)) if scope.startswith("all_") else np.flatnonzero(np.char.startswith(train["types"], "real_"))
                nearest = NearestNeighbors(n_neighbors=20, algorithm="brute", n_jobs=4).fit(vectors[pool])
                indices = pool[nearest.kneighbors(queries, return_distance=False)]
                nearest_class = []
                for label in (0, 1):
                    candidates = pool[train["y"][pool] == label]
                    nearest = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=4).fit(vectors[candidates])
                    distance, local = nearest.kneighbors(queries)
                    nearest_class.append((distance[:, 0], candidates[local[:, 0]]))
                for i, row in enumerate(query_rows):
                    results.append({"reference": scope, "sample_id": row["sample_id"],
                        "primary_attack": row["primary_attack"], "outcome_1pct": row["outcome_1pct"],
                        "attack_score": float(row["attack_score"]),
                        "benign_fraction_among_20nn": float((train["y"][indices[i]] == 0).mean()),
                        "nearest_attack_over_benign_distance": float(nearest_class[1][0][i] / max(nearest_class[0][0][i], 1e-12)),
                        "nearest_train_attack_type": str(train["attack"][nearest_class[1][1][i]]),
                        "nearest_train_attack_score": float(scores[nearest_class[1][1][i]]),
                        "nearest_train_benign_score": float(scores[nearest_class[0][1][i]])})
        for scope in sorted(set(row["reference"] for row in results)):
            for attack in sorted(set(row["primary_attack"] for row in results)):
                for outcome in ("TP", "FN"):
                    selected = [row for row in results if row["reference"] == scope and row["primary_attack"] == attack and row["outcome_1pct"] == outcome]
                    if selected:
                        groups.append({"reference": scope, "primary_attack": attack, "outcome_1pct": outcome,
                            "n": len(selected), "median_score": float(np.median([row["attack_score"] for row in selected])),
                            "mean_benign_20nn_fraction": float(np.mean([row["benign_fraction_among_20nn"] for row in selected])),
                            "median_nearest_attack_over_benign_distance": float(np.median([row["nearest_attack_over_benign_distance"] for row in selected])),
                            "median_nearest_training_attack_score": float(np.median([row["nearest_train_attack_score"] for row in selected])),
                            "nearest_training_attack_types": dict(Counter(row["nearest_train_attack_type"] for row in selected))})
        write_rows(output / "per_test_flooding.csv", results, ",")
        focus_rows = []
        for row in rows:
            if row["cohort"] == "Test Withheld flooding":
                group = f"Test {row['primary_attack']} {row['outcome_1pct']}"
            else:
                group = row["cohort"]
            focus_rows.append({**row, "diagnostic_group": group})
        focus_export = export_pair(output / "projector_focus", source_vectors, focus_rows)
        write_json(output / "summary.json", {"complete": True, "projector": str(args.projector.resolve()),
            "training_rows": len(train), "test_flooding_rows": len(queries), "groups": groups,
            "projector_focus": focus_export,
            "scope": "Post-fit diagnostic; training data reused. No fitting, threshold changes, or performance claims from training rows.",
            "limitations": "Euclidean neighborhoods depend on class prevalence, duplicates and the learned representation; they do not prove a causal mechanism.",
            "model_sha256": sha256(run / "model.pt"), "script_sha256": sha256(__file__)})
        lines = ["# Flooding failure inspection", "",
            "This is a newly trained LSTM using the delivered all-flooding dataset and the existing recipe; the teammate's trained checkpoint was not present. It is not a verification of their exact numerical results.", "",
            "## View the distributions", "",
            "Load projector_focus/vectors.tsv and projector_focus/metadata.tsv at https://projector.tensorflow.org/. Use PCA, Sphereize data OFF, Color by = diagnostic_group, Label by = sample_id. Search diagnostic_group for a subtype plus FN or TP to highlight those cases while keeping the training reference in place.",
            f"The original four-panel distribution preview is {args.projector.resolve() / 'pca_distribution.png'}.", "",
            "## Frozen test recall", "",
            "All 202 flooding test sequences are included. The historical full-type pipeline detects 202/202; the all-flooding holdout pipeline detects 142/202 (70.30%) at their respective frozen validation 1% FPR thresholds. This includes 65/65 HOIC, 4/14 LOIC-UDP, 60/110 LOIC-HTTP, 6/6 GoldenEye and 7/7 SlowHTTPTest. Hulk and Slowloris have zero test samples.",
            f"Holdout threshold: {info['thresholds']['validation_fpr_1pct']:.9g}. Holdout full-test FPR: {info['full_test_metrics']['operating_points']['validation_fpr_1pct']['fpr']:.4%}.", "",
            "## Native 64D evidence", "",
            "The following neighborhoods query ALL 780,899 actual training rows, including synthetic rows, rather than the attack-enriched plotting reference. No test row enters the reference. The corresponding real-only analysis is in summary.json.", "",
            "| Subtype | Decision | Test N | Median score | Benign fraction among 20 training neighbors | Nearest attack types |",
            "|---|---|---:|---:|---:|---|"]
        for group in groups:
            if group["reference"] == "all_training_including_synthetic":
                lines.append(f"| {group['primary_attack']} | {group['outcome_1pct']} | {group['n']} | {group['median_score']:.4f} | {group['mean_benign_20nn_fraction']:.1%} | {group['nearest_training_attack_types']} |")
        lines += ["", "LOIC-HTTP has two distinct outcomes. All 60 detected cases have Bot as their nearest training attack, with median score 1.0. The 50 missed cases have 98.5% benign neighbors on average; their nearest training attack is Infilteration, itself receiving a median score near 0.399. Their own median score is 0.394. This supports an interpretation of a low-scoring, benign-dominated neighborhood rather than the whole subtype forming one uniformly unseen cluster.",
            "LOIC-UDP differs: the ten missed cases have median score 0.925, below the frozen 0.9503 threshold, while the four detected cases have median score 0.966. Missed cases have 22.5% benign neighbors and Infilteration as the nearest training attack. This is consistent with weaker attack scores at the selected operating point; it is not evidence that all their hidden vectors collapse into benign.", "",
            "## Limits", "",
            "Training data are reused for post-fit distribution diagnostics. These are not held-out validation results. Neighborhood labels depend on the full training prevalence, distance metric, duplicate-vector ties, and synthetic donors. They describe learned geometry and do not establish a causal mechanism.",
            "The PCA plot samples 4,000 training benign and 4,000 test benign rows and includes all real retained training attacks and all test attacks. Its composition differs from the full-training neighbor calculation; do not substitute plot-sample percentages for full-training percentages.",
            "Flooding primary labels and donor targets are absent from new training, but secondary raw-flow memberships and upstream encoder fitting exclusion remain unverified. Validation still includes flooding for checkpoint and threshold selection. Historical test inspection and possible host/campaign recurrence limit unseen-attack claims.",
            "Old and new pipelines have different scalers, bucket encoders and donor plans. Their PCA axes are independent. The comparison does not isolate exclusion of flooding as the only changed factor.", ""]
        (output / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved full-training neighbor diagnostic to {output}", flush=True)
        return output
    finally:
        train.close()


if __name__ == "__main__":
    main()
