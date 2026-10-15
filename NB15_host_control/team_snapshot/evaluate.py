"""
evaluate.py — The frozen evaluation harness.

Framework-agnostic: takes anomaly SCORES (higher = more anomalous) plus window
metadata, returns a metrics dict. Every arm of the project calls this and only
this, so numbers are comparable across the AE, the XGBoost baseline and every
ablation. Build it once, do not edit it mid-project.

Metric choices, and why:
  * PR-AUC is PRIMARY. At a ~4% base rate ROC-AUC compresses everything into
    0.95-0.99 and stops discriminating between configurations.
  * Detection rate at FIXED FPR is the operational number. Two configs can
    share a PR-AUC and behave very differently at 0.1% FPR, which is the only
    regime that matters for an IDS alert budget.
  * Per-class RECALL only. A false positive is a benign window and is not
    attributable to an attack class, so per-class precision is undefined.
  * Wilson intervals on per-class recall, because Worms-sized classes cannot
    support a "hardest to detect" claim and the interval makes that visible.
  * Benign score statistics, to separate "learned the normal manifold better"
    from "got worse at reconstruction generally".

Thresholds are ALWAYS derived from benign validation scores, never from test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FPR_TARGETS = (0.001, 0.01, 0.05)


# --------------------------------------------------------------------------
def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def thresholds_from_benign(benign_scores, fpr_targets=FPR_TARGETS):
    """Score threshold giving each target FPR on benign validation data."""
    return {f: float(np.quantile(benign_scores, 1.0 - f)) for f in fpr_targets}


# --------------------------------------------------------------------------
def evaluate(scores, labels, attack_types, benign_val_scores,
             fpr_targets=FPR_TARGETS, min_class_n=30):
    """
    scores            (n,) float   anomaly score, higher = more anomalous
    labels            (n,) int     1 = window contains an attack flow
    attack_types      (n,) str     'Benign' or the attack class name
    benign_val_scores (m,) float   scores on BENIGN VALIDATION windows only
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    attack_types = np.asarray(attack_types)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores: check for exploding reconstruction "
                         "error or unscaled features.")

    out = {
        "n_windows": int(len(scores)),
        "n_malicious": int(labels.sum()),
        "base_rate": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }

    # --- benign score health (the "did it just get worse at everything" check)
    b = scores[labels == 0]
    q1, q3 = np.percentile(b, [25, 75])
    iqr = max(q3 - q1, 1e-12)
    out["benign_median"] = float(np.median(b))
    out["benign_iqr"] = float(iqr)
    out["attack_median"] = float(np.median(scores[labels == 1])) if labels.sum() else float("nan")
    out["separation"] = float((out["attack_median"] - out["benign_median"]) / iqr)

    # --- detection rate at fixed FPR, and per-class recall at each
    thr = thresholds_from_benign(benign_val_scores, fpr_targets)
    out["thresholds"] = thr
    per_class = {}

    for f, t in thr.items():
        flagged = scores >= t
        key = f"{f:g}"
        out[f"dr@fpr{key}"] = float(flagged[labels == 1].mean()) if labels.sum() else float("nan")
        out[f"actual_fpr@fpr{key}"] = float(flagged[labels == 0].mean())

        for cls in np.unique(attack_types[labels == 1]):
            m = (attack_types == cls) & (labels == 1)
            n, k = int(m.sum()), int(flagged[m].sum())
            lo, hi = wilson_interval(k, n)
            per_class.setdefault(cls, {})[key] = {
                "n": n, "recall": k / n if n else float("nan"),
                "ci_lo": lo, "ci_hi": hi,
                "underpowered": n < min_class_n,
            }

    out["per_class"] = per_class
    return out


# --------------------------------------------------------------------------
def per_class_table(metrics, fpr=0.01):
    """Per-class recall with Wilson CIs at one operating point."""
    key = f"{fpr:g}"
    rows = []
    for cls, d in metrics["per_class"].items():
        r = d[key]
        rows.append({
            "attack": cls, "n_windows": r["n"],
            "recall": round(r["recall"], 4),
            "ci_95": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
            "underpowered": r["underpowered"],
        })
    return pd.DataFrame(rows).sort_values("n_windows", ascending=False)


def summary_row(name, metrics, n_params=None):
    row = {"config": name, "params": n_params,
           "pr_auc": round(metrics["pr_auc"], 4),
           "roc_auc": round(metrics["roc_auc"], 4),
           "separation": round(metrics["separation"], 3),
           "benign_median": round(metrics["benign_median"], 4)}
    for f in FPR_TARGETS:
        row[f"dr@{f:g}"] = round(metrics[f"dr@fpr{f:g}"], 4)
    return row


def compare(results, primary="pr_auc"):
    """Aggregate {config: [metrics per seed]} into a mean +/- std table."""
    rows = []
    for name, runs in results.items():
        agg = {"config": name, "n_seeds": len(runs),
               "params": runs[0].get("n_params")}
        for m in [primary, "roc_auc", "separation"] + \
                 [f"dr@fpr{f:g}" for f in FPR_TARGETS]:
            v = np.array([r[m] for r in runs], dtype=float)
            agg[m] = f"{v.mean():.4f} ± {v.std():.4f}"
            agg[f"_{m}_mean"] = float(v.mean())
        rows.append(agg)

    df = pd.DataFrame(rows).sort_values(f"_{primary}_mean", ascending=False)
    best, second = df.iloc[0], df.iloc[1] if len(df) > 1 else None
    note = ""
    if second is not None:
        gap = best[f"_{primary}_mean"] - second[f"_{primary}_mean"]
        sd = float(np.std([r[primary] for r in results[best["config"]]]))
        if gap < sd:
            note = (f"\nNOTE: {best['config']} leads {second['config']} by "
                    f"{gap:.4f}, less than one seed SD ({sd:.4f}). Report these "
                    f"as indistinguishable rather than declaring a winner.")
    return df.drop(columns=[c for c in df.columns if c.startswith("_")]), note


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_b, n_a = 8000, 400
    scores = np.concatenate([rng.normal(0, 1, n_b), rng.normal(2.2, 1.2, n_a)])
    labels = np.concatenate([np.zeros(n_b, int), np.ones(n_a, int)])
    cls = np.concatenate([
        np.full(n_b, "Benign"),
        rng.choice(["Exploits", "Fuzzers", "DoS", "Worms"], n_a,
                   p=[.5, .3, .18, .02])])
    benign_val = rng.normal(0, 1, 3000)

    m = evaluate(scores, labels, cls, benign_val)
    print(pd.DataFrame([summary_row("demo", m, 250_000)]).to_string(index=False))
    print()
    print(per_class_table(m, fpr=0.01).to_string(index=False))

    fake = {"lstm": [evaluate(scores + rng.normal(0, .02, len(scores)),
                              labels, cls, benign_val) for _ in range(3)],
            "cnn": [evaluate(scores * 0.97 + rng.normal(0, .02, len(scores)),
                             labels, cls, benign_val) for _ in range(3)]}
    tbl, note = compare(fake)
    print("\n" + tbl.to_string(index=False))
    print(note)
