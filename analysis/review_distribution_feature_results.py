"""Review saved validation predictions only; never train or open test features."""
import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.io import ROOT, fresh, read_json, unique_name, write_json
from train.run_distribution_feature_ablation import verified
from train.run_circular_shift import write_csv

ATTACKS = ("Infilteration", "Brute_Force_-Web")
POINTS = ("validation_fpr_1pct", "validation_fpr_0.1pct")


def load_predictions(folder):
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args(argv)
    batch = args.batch.resolve()
    state = read_json(batch / "batch.json")
    if state["status"] != "complete":
        raise ValueError("A completed batch is required")
    old_run = verified(state["settings"]["reference"])
    new_run = verified(state["cases"]["distribution97"]["lstm"])
    old, new = load_predictions(old_run), load_predictions(new_run)
    for key in ("sample_id", "y", "primary_attack", "source_host"):
        np.testing.assert_array_equal(old[key], new[key])
    index_path = old_run.parents[2] / "validation_index.npz"
    with np.load(index_path, allow_pickle=False) as index:
        np.testing.assert_array_equal(index["sample_id"], old["sample_id"])
        active = index["active_buckets"].copy()
    dataset = Path(read_json(new_run / "config.json")["dataset"])
    with np.load(dataset / "flow_metadata_validation.npz", allow_pickle=False) as meta:
        positions = np.searchsorted(meta["sample_id"], old["sample_id"])
        np.testing.assert_array_equal(meta["sample_id"][positions], old["sample_id"])
        pure = meta["total_flows"][positions] == meta["attack_flows"][positions]
    thresholds = [read_json(p / "thresholds.json")["values"] for p in (old_run, new_run)]
    rows, transitions, auc = [], {}, {}
    for attack in ATTACKS:
        target = (old["y"] == 1) & (old["primary_attack"] == attack)
        binary = target | (old["y"] == 0)
        auc[attack] = {label: float(roc_auc_score(target[binary], predictions["score"][binary]))
                      for label, predictions in (("baseline", old), ("new", new))}
        for point in POINTS:
            a, b = old["score"] >= thresholds[0][point], new["score"] >= thresholds[1][point]
            recovered, lost = target & ~a & b, target & a & ~b
            transitions[attack + "/" + point] = {
                "recovered_ids": old["sample_id"][recovered].tolist(),
                "lost_ids": old["sample_id"][lost].tolist(),
                "still_missed": int(np.count_nonzero(target & ~a & ~b)),
                "still_detected": int(np.count_nonzero(target & a & b))}
            for name, group in (("all", target), ("pure", target & pure), ("mixed", target & ~pure),
                                ("single_active_bucket", target & (active == 1)),
                                ("multiple_active_buckets", target & (active > 1))):
                n = int(group.sum())
                rows.append({"attack": attack, "operating_point": point, "group": name, "n": n,
                    "baseline_tp": int((group & a).sum()), "new_tp": int((group & b).sum()),
                    "recovered": int((group & ~a & b).sum()), "lost": int((group & a & ~b).sum()),
                    "baseline_recall": float(a[group].mean()) if n else None,
                    "new_recall": float(b[group].mean()) if n else None})
    seeds = []
    for folder in sorted(old_run.parents[2].glob("seed*/lstm/start0")):
        summary = read_json(folder / "summary.json")
        np.testing.assert_equal(summary["validation_cohort_sha256"], read_json(new_run / "summary.json")["validation_cohort_sha256"])
        pred = load_predictions(folder)
        for key in ("sample_id", "y", "primary_attack", "source_host"):
            np.testing.assert_array_equal(pred[key], old[key])
        limits = read_json(folder / "thresholds.json")["values"]
        for attack in ATTACKS:
            target = (old["y"] == 1) & (old["primary_attack"] == attack)
            for point in POINTS:
                seeds.append({"seed": summary["seed"], "attack": attack, "operating_point": point,
                              "recall": float((pred["score"][target] >= limits[point]).mean())})
    out = fresh(batch / "reports" / unique_name("paired_validation_review"))
    write_csv(out / "subgroups.csv", rows)
    write_csv(out / "existing_baseline_seeds.csv", seeds)
    result = {"baseline": str(old_run), "new": str(new_run), "subgroups": rows,
              "transitions": transitions, "target_vs_benign_auc": auc,
              "existing_baseline_seeds": seeds, "test_used": False, "training_started": False,
              "limitations": ["Saved predictions on previously inspected real validation only.",
                  "Training data were reused by the completed fits; no new fit was performed by this review.",
                  "Global thresholds retain each run's original validation selection policy.",
                  "One new-feature seed and a refitted encoder cannot establish a stable feature effect.",
                  "Pure means all sequence flows have attack labels; it does not establish source-label validity."]}
    write_json(out / "review.json", result)
    lines = ["# Distribution-feature validation review", "", "No training or test-feature access.", "",
             "| Attack | FPR budget | Group | N | Baseline TP | New TP | Recovered | Lost |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['attack']} | {r['operating_point']} | {r['group']} | {r['n']} | {r['baseline_tp']} | {r['new_tp']} | {r['recovered']} | {r['lost']} |" for r in rows)
    lines += ["", "## Limits", ""] + ["- " + s for s in result["limitations"]]
    (out / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for row in rows:
        print(row)
    print("Target-versus-benign AUC:", auc)
    for attack in ATTACKS:
        for point in POINTS:
            values = [r["recall"] for r in seeds if r["attack"] == attack and r["operating_point"] == point]
            print("Existing baseline seeds:", attack, point, "mean", np.mean(values), "min", min(values), "max", max(values))
    return out


if __name__ == "__main__":
    main()
