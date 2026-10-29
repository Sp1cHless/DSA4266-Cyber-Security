"""Check probe-seed variability on the same training-only development partitions."""
import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.audit_infilteration_features import split_indices
from train.common.io import fresh, read_json, unique_name, write_json, sha256
from train.run_circular_shift import write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4266, 4267, 4268])
    args = parser.parse_args(argv)
    if read_json(args.audit / "status.json")["status"] != "complete":
        raise ValueError("The initial audit must be complete")
    out = fresh(args.audit / "reports" / unique_name("probe_seed_recheck"))
    write_json(out / "status.json", {"status": "running", "test_used": False})
    with (args.audit / "training_cohort.csv").open() as stream:
        cohort = list(csv.DictReader(stream))
    for row in cohort:
        for name in ("id", "start", "target", "active"):
            row[name] = int(row[name])
    with np.load(args.audit / "training_feature_cache.npz", allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["sample_id"], [r["id"] for r in cohort])
        old = cache["old_aggregates"].astype(np.float32)
        extra = {name: cache[name].reshape(len(cohort), -1).astype(np.float32)
                 for name in ("iat", "distribution", "categories")}
    cases = [("old_aggregates", []), ("old_plus_iat", ["iat"]), ("old_plus_distribution", ["distribution"]),
             ("old_plus_robust_candidates", ["iat", "distribution", "categories"])]
    y = np.array([r["target"] for r in cohort])
    results = []
    for mode in ("time_block", "source_host"):
        fit, hold = split_indices(cohort, mode)
        for name, fields in cases:
            x = np.concatenate([old] + [extra[field] for field in fields], axis=1)
            for seed in args.seeds:
                model = ExtraTreesClassifier(n_estimators=64, max_depth=12, min_samples_leaf=8,
                    max_features="sqrt", class_weight="balanced", n_jobs=4, random_state=seed)
                model.fit(x[fit], y[fit])
                scores = model.predict_proba(x[hold])[:, 1]
                fpr, tpr, thresholds = roc_curve(y[hold], scores)
                eligible = np.flatnonzero(fpr <= .01)
                best = eligible[np.argmax(tpr[eligible])]
                row = {"mode": mode, "features": name, "seed": seed, "roc_auc": roc_auc_score(y[hold], scores),
                    "empirical_holdout_recall_at_1pct": float(tpr[best]), "empirical_holdout_fpr": float(fpr[best]),
                    "holdout_positive_n": int(y[hold].sum()), "holdout_benign_n": int((y[hold] == 0).sum())}
                results.append(row)
                print(f"{mode} {name} seed={seed}: AUC={row['roc_auc']:.4f}, development R@1%={row['empirical_holdout_recall_at_1pct']:.4f}", flush=True)
    aggregate = []
    for mode in ("time_block", "source_host"):
        for name, _ in cases:
            selected = [r for r in results if r["mode"] == mode and r["features"] == name]
            baseline = {r["seed"]: r for r in results if r["mode"] == mode and r["features"] == "old_aggregates"}
            for metric in ("roc_auc", "empirical_holdout_recall_at_1pct"):
                values = np.array([r[metric] for r in selected])
                deltas = np.array([r[metric] - baseline[r["seed"]][metric] for r in selected])
                aggregate.append({"mode": mode, "features": name, "metric": metric, "seeds": len(values),
                    "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "paired_delta_mean": float(deltas.mean()), "paired_delta_min": float(deltas.min()),
                    "paired_delta_max": float(deltas.max()), "positive_seeds": int((deltas > 0).sum())})
    write_csv(out / "per_seed.csv", results)
    write_csv(out / "aggregate.csv", aggregate)
    write_json(out / "provenance.json", {"parent_audit": str(args.audit.resolve()), "seeds": args.seeds,
        "training_cohort_sha256": sha256(args.audit / "training_cohort.csv"), "script_sha256": sha256(Path(__file__)),
        "test_used": False, "validation_used": False, "formal_model_trained": False,
        "limitations": ["Same previously screened train-only development folds are reused; this is not independent confirmation.",
            "Seed variation is conditional on fixed data and group partitions, not uncertainty over campaigns or future data.",
            "Four feature variants were chosen for follow-up after inspecting the initial internal screening; results remain exploratory.",
            "The 1% operating point is read from the internal holdout ROC curve, not a deployable calibrated threshold.",
            "Artificial matched/sampled benign controls and exclusion of other attacks prevent comparison with full-validation LSTM recall.",
            "No 0.1% operating point is reported because the small benign control populations would allow only two or three false alarms.",
            "These internal time-block folds have no additional internal embargo/envelope purge."]})
    write_json(out / "status.json", {"status": "complete", "test_used": False})
    print(str(out), flush=True)
    return out


if __name__ == "__main__":
    main()
