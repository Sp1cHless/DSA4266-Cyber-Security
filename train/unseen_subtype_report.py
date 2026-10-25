"""Compare frozen control and training-holdout runs on unseen attack subtypes.

Reads saved predictions only: no refitting, no new thresholds and no dataset
access unless --holdout-dataset is supplied to name the removed labels.
Each run keeps the operating thresholds selected on its own real validation split.
"""
import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.io import ROOT, fresh, read_json, sha256, unique_name, write_json

POLICIES = (("1pct", "validation_fpr_1pct"), ("0_1pct", "validation_fpr_0.1pct"))


def wilson(successes, total, z=1.96):
    if not total:
        return None, None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def load_entry(folder):
    folder = Path(folder).resolve()
    summary = read_json(folder / "summary.json")
    if not summary.get("complete") or "metrics" not in summary:
        raise ValueError(f"Not a completed frozen run or evaluation: {folder}")
    if (folder / "thresholds.json").is_file():
        if sha256(folder / "thresholds.json") != summary.get("thresholds_sha256"):
            raise ValueError(f"Frozen thresholds changed: {folder}")
        thresholds = read_json(folder / "thresholds.json")["values"]
        source = "saved real validation thresholds"
    elif (folder / "plan.json").is_file():
        thresholds = read_json(folder / "plan.json")["thresholds"]
        source = "evaluation plan thresholds copied from the training run"
    else:
        raise ValueError(f"No frozen thresholds found: {folder}")
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        ids, labels, scores, attacks, hosts = [saved[key].copy() for key in
                                              ("sample_id", "y", "score", "primary_attack", "source_host")]
    metrics = summary["metrics"]
    if len(ids) != metrics["n"] or int(labels.sum()) != metrics["positives"]:
        raise ValueError(f"Prediction population differs from the saved metrics: {folder}")
    for alias, policy in POLICIES:
        point = metrics["operating_points"][policy]
        predicted = scores >= thresholds[policy]
        tp = int(np.count_nonzero(predicted & (labels == 1)))
        fp = int(np.count_nonzero(predicted & (labels == 0)))
        if (tp, fp) != (point["tp"], point["fp"]):
            raise ValueError(f"Saved operating point does not match predictions: {folder} {policy}")
    return dict(folder=folder, summary=summary, model=summary["model"], split=summary["split"],
                thresholds=thresholds, threshold_source=source, ids=ids, y=labels, score=scores,
                attacks=attacks, hosts=hosts, metrics=metrics)


def pair_entries(control, holdout):
    control = {entry["model"]: entry for entry in control}
    holdout = {entry["model"]: entry for entry in holdout}
    if len(control) != len(holdout) or set(control) != set(holdout):
        raise ValueError("Control and holdout runs must cover the same model families")
    pairs = []
    for name in sorted(control):
        left, right = control[name], holdout[name]
        if left["split"] != right["split"]:
            raise ValueError("Control and holdout evaluations use different splits")
        if not np.array_equal(left["ids"], right["ids"]) or not np.array_equal(left["y"], right["y"]):
            raise ValueError("Control and holdout prediction populations differ; the split moved")
        if not np.array_equal(left["attacks"], right["attacks"]):
            raise ValueError("Control and holdout attack labels differ")
        pairs.append((name, left, right))
    return pairs


def subtype_rows(left, right, held_out):
    rows = []
    names = sorted(set(np.unique(left["attacks"][left["y"] == 1]).tolist()) | set(held_out))
    for name in names:
        keep = (left["y"] == 1) & (left["attacks"] == name)
        n = int(keep.sum())
        row = {"primary_attack": name, "held_out_from_training": name in held_out,
               "n": n, "source_hosts": len(np.unique(left["hosts"][keep]))}
        for alias, policy in POLICIES:
            for label, entry in (("control", left), ("holdout", right)):
                tp = int(np.count_nonzero(entry["score"][keep] >= entry["thresholds"][policy]))
                row[f"tp_{label}_{alias}"] = tp
                row[f"recall_{label}_{alias}"] = tp / n if n else None
            row[f"delta_{alias}"] = (right["score"][keep] >= right["thresholds"][policy]).mean() - \
                                    (left["score"][keep] >= left["thresholds"][policy]).mean() if n else None
            low, high = wilson(row[f"tp_holdout_{alias}"], n)
            row[f"recall_holdout_{alias}_wilson_low"] = low
            row[f"recall_holdout_{alias}_wilson_high"] = high
        rows.append(row)
    return sorted(rows, key=lambda r: (not r["held_out_from_training"], -r["n"], r["primary_attack"]))


def pooled_row(left, right, held_out):
    keep = (left["y"] == 1) & np.isin(left["attacks"], list(held_out))
    n = int(keep.sum())
    row = {"primary_attack": "ALL HELD OUT", "held_out_from_training": True, "n": n,
           "source_hosts": len(np.unique(left["hosts"][keep]))}
    for alias, policy in POLICIES:
        for label, entry in (("control", left), ("holdout", right)):
            tp = int(np.count_nonzero(entry["score"][keep] >= entry["thresholds"][policy]))
            row[f"tp_{label}_{alias}"] = tp
            row[f"recall_{label}_{alias}"] = tp / n if n else None
        row[f"delta_{alias}"] = (right["score"][keep] >= right["thresholds"][policy]).mean() - \
                                (left["score"][keep] >= left["thresholds"][policy]).mean() if n else None
        low, high = wilson(row[f"tp_holdout_{alias}"], n)
        row[f"recall_holdout_{alias}_wilson_low"] = low
        row[f"recall_holdout_{alias}_wilson_high"] = high
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, nargs="+", required=True,
                        help="Completed training-run or evaluation directories trained with every subtype")
    parser.add_argument("--holdout", type=Path, nargs="+", required=True,
                        help="Completed training-run or evaluation directories trained without the held-out subtypes")
    parser.add_argument("--held-out-attacks", default="",
                        help="Comma-separated primary attack labels removed from training")
    parser.add_argument("--holdout-dataset", type=Path,
                        help="Dataset directory whose manifest records train_holdout_attacks")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    names = sorted({name.strip() for name in args.held_out_attacks.split(",") if name.strip()})
    if args.holdout_dataset is not None:
        recorded = read_json(args.holdout_dataset / "manifest.json").get("train_holdout_attacks") or []
        if names and sorted(recorded) != names:
            raise ValueError("--held-out-attacks differs from the holdout dataset manifest")
        names = sorted(recorded)
    if not names:
        raise ValueError("Supply the held-out labels through --held-out-attacks or --holdout-dataset")

    pairs = pair_entries([load_entry(p) for p in args.control], [load_entry(p) for p in args.holdout])
    output = fresh(args.out or ROOT / "runs" / unique_name("unseen_subtype_report"))
    rows, summaries, names_seen = [], [], set()
    for model, left, right in pairs:
        names_seen |= set(np.unique(left["attacks"][left["y"] == 1]).tolist())
        rows.extend({"model": model, **row} for row in [pooled_row(left, right, names)] +
                    subtype_rows(left, right, names))
        summaries.append({"model": model, "split": left["split"],
                          "ap_control": left["metrics"]["average_precision"],
                          "ap_holdout": right["metrics"]["average_precision"],
                          "n": left["metrics"]["n"], "positives": left["metrics"]["positives"],
                          "control_run": str(left["folder"]), "holdout_run": str(right["folder"]),
                          "control_thresholds": left["thresholds"], "holdout_thresholds": right["thresholds"]})

    fields = list(rows[0])
    with (output / "unseen_subtype.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Unseen attack subtypes: control versus training holdout", "",
             "Both models were scored with their own frozen real-validation thresholds. "
             f"Held out from training only: {', '.join(names)}. "
             "The same real sequences appear in both prediction files, so the split did not move.", "",
             "| Model | Split | N | Attacks | AP control | AP holdout |", "|---|---|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['model']} | {item['split']} | {item['n']:,} | {item['positives']:,} | "
                     f"{item['ap_control']:.4f} | {item['ap_holdout']:.4f} |")
    for model, left, right in pairs:
        subset = [row for row in rows if row["model"] == model]
        lines += ["", f"## {model.upper()}", "",
                  "| Primary attack | Held out | N | Hosts | Control R@1% | Holdout R@1% | Delta | Holdout 95% CI | Control R@0.1% | Holdout R@0.1% | Delta | Holdout 95% CI |",
                  "|---|:--:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
        for row in subset:
            def show(value):
                return "n/a" if value is None else f"{value:.2%}"
            def interval(alias):
                low, high = row[f"recall_holdout_{alias}_wilson_low"], row[f"recall_holdout_{alias}_wilson_high"]
                return "n/a" if low is None else f"{low:.1%}-{high:.1%}"
            lines.append(f"| {row['primary_attack']} | {'yes' if row['held_out_from_training'] else 'no'} | "
                         f"{row['n']} | {row['source_hosts']} | {show(row['recall_control_1pct'])} | "
                         f"{show(row['recall_holdout_1pct'])} | {show(row['delta_1pct'])} | {interval('1pct')} | "
                         f"{show(row['recall_control_0_1pct'])} | {show(row['recall_holdout_0_1pct'])} | "
                         f"{show(row['delta_0_1pct'])} | {interval('0_1pct')} |")
    seen = [row for row in rows if row["model"] == pairs[0][0] and not row["held_out_from_training"]]
    lines += ["", "## Scope", "",
              "Binary attack detection per 100-second sequence, broken down by the primary attack label. "
              "The detector does not predict attack categories, so a held-out subtype can only be flagged through "
              "traffic evidence that generalizes beyond its training siblings.",
              "Held-out labels are absent from every training-fold transform: vocabulary, scaler, frozen bucket "
              "encoder, donor plan and augmentation, because all of them fit on the training fold only. "
              "Validation retained the held-out labels, so they also entered input-encoder checkpoint selection; "
              "this is disclosed and matches the baseline protocol. Per-type counts are small and host-dependent "
              "(for example one source host per DoS subtype). Wilson intervals describe sampling uncertainty only "
              "and do not remove host or campaign dependence. The Scheme2 test split was inspected in earlier "
              "project work; treat test rows as a frozen comparison, not a newly untouched holdout.", ""]
    (output / "UNSEEN_SUBTYPE.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "manifest.json", {
        "complete": True, "held_out_attacks": names, "split": pairs[0][1]["split"],
        "models": [item["model"] for item in summaries], "summaries": summaries,
        "refitting": False, "threshold_source": "each run's saved real validation thresholds",
        "control_and_holdout_share_prediction_population": True,
        "evaluated_attack_subtypes": len(names_seen), "rows": len(rows),
        "test_opened": pairs[0][1]["split"] == "test"})
    print("\n".join(lines), flush=True)
    print(f"Saved to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
