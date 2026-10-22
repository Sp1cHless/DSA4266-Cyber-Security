"""One unweighted sequence-level metric contract for every model."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix, matthews_corrcoef)

POLICY = "scheme2_supervised_screening_v1"


def validate_scores(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid scores")
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("Evaluation requires both binary classes")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected positive-class scores in [0,1]")
    return y, scores


def choose_thresholds(y, scores):
    y, scores = validate_scores(y, scores)
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    result = {"default_0.5": 0.5}
    for name, budget in (("validation_fpr_1pct", 0.01), ("validation_fpr_0.1pct", 0.001)):
        eligible = np.flatnonzero(fpr <= budget)
        best_recall = tpr[eligible].max()
        # Highest threshold among maximal-recall eligible points handles score ties conservatively.
        i = eligible[np.flatnonzero(tpr[eligible] == best_recall)[0]]
        value = thresholds[i]
        result[name] = float(value if np.isfinite(value) else np.nextafter(scores.max(), np.inf))
    return result


def at_threshold(y, scores, threshold):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {"threshold": float(threshold), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "fpr": float(1 - specificity),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "accuracy": float((tp + tn) / len(y)), "balanced_accuracy": float((recall + specificity) / 2),
            "mcc": float(matthews_corrcoef(y, predicted))}


def evaluate(split, scores, thresholds):
    y, scores = validate_scores(split["y"], scores)
    operating = {name: at_threshold(y, scores, t) for name, t in thresholds.items()}
    t = thresholds["validation_fpr_1pct"]
    groups = {}
    for name in np.unique(split["attack"][y == 1]):
        keep = (y == 1) & (split["attack"] == name)
        groups[str(name)] = {"n": int(keep.sum()), "hosts": int(len(np.unique(split["host"][keep]))),
                             "recall_at_validation_fpr_1pct": float((scores[keep] >= t).mean())}
    large = [v["recall_at_validation_fpr_1pct"] for v in groups.values() if v["n"] >= 20]
    return {"policy": POLICY, "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)), "operating_points": operating,
            "macro_attack_recall_n_ge_20": float(np.mean(large)) if large else None,
            "per_primary_attack": groups}


def save_curves(path, y, scores):
    fpr, tpr, roc_t = roc_curve(y, scores)
    precision, recall, pr_t = precision_recall_curve(y, scores)
    np.savez_compressed(path, fpr=fpr, tpr=tpr, roc_thresholds=roc_t,
                        precision=precision, recall=recall, pr_thresholds=pr_t)
