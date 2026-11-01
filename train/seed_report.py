"""Aggregate repeated training seeds across control and training-holdout arms.

Reads saved predictions only: no refitting and no new thresholds. Each run keeps
the operating thresholds selected on its own real validation split, so a seed
changes weight initialization, batch order and dropout, not the split, the frozen
bucket encoder or the threshold policy.

The report pairs every holdout run with the control run of the same seed, so the
delta removes seed-to-seed baseline drift, and it reports mean, standard deviation
and the raw per-seed values. Three seeds bound retraining noise; they do not
support significance testing, and they do not address sampling uncertainty from
the small number of hosts and campaigns.
"""
import argparse
import csv
import statistics
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.io import ROOT, fresh, read_json, sha256, unique_name, write_json

POLICIES = (("1pct", "validation_fpr_1pct"), ("0_1pct", "validation_fpr_0.1pct"))


def resolve(folder):
    """Return (training-run directory, evaluation directory) for either input form."""
    folder = Path(folder).resolve()
    if (folder / "test_eval" / "predictions.npz").is_file():
        return folder, folder / "test_eval"
    if folder.name == "test_eval" and (folder / "predictions.npz").is_file():
        return folder.parent, folder
    raise ValueError(f"Expected a run directory with test_eval/ or a test_eval directory: {folder}")


def load_entry(folder):
    run_dir, eval_dir = resolve(folder)
    summary = read_json(eval_dir / "summary.json")
    if not summary.get("complete") or "metrics" not in summary:
        raise ValueError(f"Not a completed frozen evaluation: {eval_dir}")
    run = read_json(run_dir / "summary.json")
    if not run.get("complete") or "seed" not in run:
        raise ValueError(f"Not a completed training run with a recorded seed: {run_dir}")
    if (eval_dir / "thresholds.json").is_file():
        if sha256(eval_dir / "thresholds.json") != summary.get("thresholds_sha256"):
            raise ValueError(f"Frozen thresholds changed: {eval_dir}")
        thresholds = read_json(eval_dir / "thresholds.json")["values"]
    elif (eval_dir / "plan.json").is_file():
        thresholds = read_json(eval_dir / "plan.json")["thresholds"]
    else:
        raise ValueError(f"No frozen thresholds found: {eval_dir}")
    with np.load(eval_dir / "predictions.npz", allow_pickle=False) as saved:
        ids, labels, scores, attacks, hosts = [saved[key].copy() for key in
                                               ("sample_id", "y", "score", "primary_attack", "source_host")]
    metrics = summary["metrics"]
    if len(ids) != metrics["n"] or int(labels.sum()) != metrics["positives"]:
        raise ValueError(f"Prediction population differs from the saved metrics: {eval_dir}")
    for alias, policy in POLICIES:
        point = metrics["operating_points"][policy]
        predicted = scores >= thresholds[policy]
        tp = int(np.count_nonzero(predicted & (labels == 1)))
        fp = int(np.count_nonzero(predicted & (labels == 0)))
        if (tp, fp) != (point["tp"], point["fp"]):
            raise ValueError(f"Saved operating point does not match predictions: {eval_dir} {policy}")
    validation = run.get("metrics") if run.get("split") == "validation" else None
    return dict(folder=run_dir, eval_dir=eval_dir, model=summary["model"], seed=int(run["seed"]),
                split=summary["split"], dataset_sha=run.get("dataset_manifest_sha256", "unknown"),
                validation_ap=validation["average_precision"] if validation else None,
                thresholds=thresholds, ids=ids, y=labels, score=scores,
                attacks=attacks, hosts=hosts, metrics=metrics)


def parse_arms(values, datasets, labels):
    """Turn repeatable name=dirs / name=path options into ordered arm dictionaries."""
    arms, order = {}, []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Use --arm name=dir1,dir2: {value}")
        name, paths = value.split("=", 1)
        name = name.strip()
        if not name or name in arms:
            raise ValueError(f"Arm names must be unique and non-empty: {name!r}")
        directories = [Path(item) for item in paths.split(",") if item.strip()]
        if not directories:
            raise ValueError(f"Arm {name} has no directories")
        arms[name] = directories
        order.append(name)
    if "control" not in arms:
        raise ValueError("One arm must be named control")
    held_out = {}
    for value in datasets:
        name, path = value.split("=", 1)
        recorded = read_json(Path(path) / "manifest.json").get("train_holdout_attacks") or []
        held_out[name.strip()] = sorted(recorded)
    for value in labels:
        name, names = value.split("=", 1)
        held_out[name.strip()] = sorted({item.strip() for item in names.split(",") if item.strip()})
    for name in order:
        if name == "control":
            continue
        if not held_out.get(name):
            raise ValueError(f"Arm {name} needs --arm-dataset name=path or --held-out-attacks name=labels")
    return order, arms, held_out


def group_by_model(name, entries, expected_seeds):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["model"], []).append(entry)
    for model, items in grouped.items():
        seeds = [entry["seed"] for entry in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{name}/{model}: duplicate seeds {sorted(seeds)}")
        if len(items) != expected_seeds:
            raise ValueError(f"{name}/{model}: expected {expected_seeds} seeds, found {len(items)}")
        base = items[0]
        for entry in items[1:]:
            if not np.array_equal(base["ids"], entry["ids"]) or not np.array_equal(base["y"], entry["y"]):
                raise ValueError(f"{name}/{model}: seeds scored different populations")
            if not np.array_equal(base["attacks"], entry["attacks"]):
                raise ValueError(f"{name}/{model}: seeds scored different attack labels")
        shas = {entry["dataset_sha"] for entry in items}
        if len(shas) != 1:
            raise ValueError(f"{name}/{model}: seeds used different dataset fingerprints {sorted(shas)}")
        grouped[model] = sorted(items, key=lambda entry: entry["seed"])
    return grouped


def mean_sd(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def recall(entry, keep, policy):
    if not keep.any():
        return None
    return float(np.count_nonzero(entry["score"][keep] >= entry["thresholds"][policy]) / int(keep.sum()))


def masked_ap(entry, keep):
    if not np.any(keep & (entry["y"] == 1)):
        return None
    return float(average_precision_score(entry["y"][keep], entry["score"][keep]))


def subtype_ap(entry, name):
    return masked_ap(entry, (entry["y"] == 0) | ((entry["y"] == 1) & (entry["attacks"] == name)))


def series(values):
    return ";".join("n/a" if value is None else f"{value:.6f}" for value in values)


def compare(control_items, holdout_items, labels, arm, model):
    """Paired rows for one model and one holdout arm against the same-seed control."""
    seeds = [entry["seed"] for entry in control_items]
    if seeds != [entry["seed"] for entry in holdout_items]:
        raise ValueError(f"{arm}/{model}: control and holdout arms do not share seeds")
    names = sorted(set(np.unique(control_items[0]["attacks"][control_items[0]["y"] == 1]).tolist()) | set(labels))
    rows = []
    for name in names:
        keep = (control_items[0]["y"] == 1) & (control_items[0]["attacks"] == name)
        n = int(keep.sum())
        row = {"arm": arm, "model": model, "primary_attack": name, "held_out": name in labels,
               "n": n, "hosts": len(np.unique(control_items[0]["hosts"][keep])),
               "seeds": ";".join(str(seed) for seed in seeds)}
        for alias, policy in POLICIES:
            control_values = [recall(entry, keep, policy) for entry in control_items]
            holdout_values = [recall(entry, keep, policy) for entry in holdout_items]
            deltas = [None if (left is None or right is None) else right - left
                      for left, right in zip(control_values, holdout_values)]
            control_mean, control_sd = mean_sd(control_values)
            holdout_mean, holdout_sd = mean_sd(holdout_values)
            delta_mean, delta_sd = mean_sd(deltas)
            row.update({f"recall_control_{alias}_mean": control_mean, f"recall_control_{alias}_sd": control_sd,
                        f"recall_holdout_{alias}_mean": holdout_mean, f"recall_holdout_{alias}_sd": holdout_sd,
                        f"delta_{alias}_mean": delta_mean, f"delta_{alias}_sd": delta_sd,
                        f"seeds_lower_{alias}": sum(1 for delta in deltas if delta is not None and delta < 0),
                        f"per_seed_control_{alias}": series(control_values),
                        f"per_seed_holdout_{alias}": series(holdout_values),
                        f"per_seed_delta_{alias}": series(deltas)})
        ap_control = [subtype_ap(entry, name) for entry in control_items]
        ap_holdout = [subtype_ap(entry, name) for entry in holdout_items]
        ap_deltas = [None if (left is None or right is None) else right - left
                     for left, right in zip(ap_control, ap_holdout)]
        for prefix, values in (("ap_control", ap_control), ("ap_holdout", ap_holdout), ("delta_ap", ap_deltas)):
            mean, sd = mean_sd(values)
            row[f"{prefix}_mean"], row[f"{prefix}_sd"] = mean, sd
        row["per_seed_ap_control"] = series(ap_control)
        row["per_seed_ap_holdout"] = series(ap_holdout)
        row["per_seed_delta_ap"] = series(ap_deltas)
        rows.append(row)
    pooled_keep = (control_items[0]["y"] == 1) & np.isin(control_items[0]["attacks"], list(labels))
    pooled = {"arm": arm, "model": model, "primary_attack": "ALL HELD OUT", "held_out": True,
              "n": int(pooled_keep.sum()),
              "hosts": len(np.unique(control_items[0]["hosts"][pooled_keep])),
              "seeds": ";".join(str(seed) for seed in seeds)}
    for alias, policy in POLICIES:
        control_values = [recall(entry, pooled_keep, policy) for entry in control_items]
        holdout_values = [recall(entry, pooled_keep, policy) for entry in holdout_items]
        deltas = [right - left for left, right in zip(control_values, holdout_values)]
        for prefix, values in ((f"recall_control_{alias}", control_values),
                               (f"recall_holdout_{alias}", holdout_values),
                               (f"delta_{alias}", deltas)):
            mean, sd = mean_sd(values)
            pooled[f"{prefix}_mean"], pooled[f"{prefix}_sd"] = mean, sd
        pooled[f"seeds_lower_{alias}"] = sum(1 for delta in deltas if delta < 0)
        pooled[f"per_seed_control_{alias}"] = series(control_values)
        pooled[f"per_seed_holdout_{alias}"] = series(holdout_values)
        pooled[f"per_seed_delta_{alias}"] = series(deltas)
    ap_control = [masked_ap(entry, (entry["y"] == 0) | pooled_keep) for entry in control_items]
    ap_holdout = [masked_ap(entry, (entry["y"] == 0) | pooled_keep) for entry in holdout_items]
    ap_deltas = [None if (left is None or right is None) else right - left
                 for left, right in zip(ap_control, ap_holdout)]
    for prefix, values in (("ap_control", ap_control), ("ap_holdout", ap_holdout), ("delta_ap", ap_deltas)):
        mean, sd = mean_sd(values)
        pooled[f"{prefix}_mean"], pooled[f"{prefix}_sd"] = mean, sd
    pooled["per_seed_ap_control"] = series(ap_control)
    pooled["per_seed_ap_holdout"] = series(ap_holdout)
    pooled["per_seed_delta_ap"] = series(ap_deltas)
    return pooled, rows


def arm_row(arm, model, items, labels):
    seeds = [entry["seed"] for entry in items]
    row = {"arm": arm, "model": model, "seeds": ";".join(str(seed) for seed in seeds),
           "dataset_manifest_sha256": items[0]["dataset_sha"]}
    for prefix, values in (("validation_ap", [entry["validation_ap"] for entry in items]),
                           ("test_ap", [entry["metrics"]["average_precision"] for entry in items])):
        mean, sd = mean_sd(values)
        row[f"{prefix}_mean"], row[f"{prefix}_sd"] = mean, sd
        row[f"per_seed_{prefix}"] = series(values)
    for alias, policy in POLICIES:
        key = f"pooled_recall_{alias}"
        if not labels:
            row[f"{key}_mean"] = row[f"{key}_sd"] = None
            row[f"per_seed_{key}"] = ""
            continue
        keep = (items[0]["y"] == 1) & np.isin(items[0]["attacks"], list(labels))
        values = [recall(entry, keep, policy) for entry in items]
        mean, sd = mean_sd(values)
        row[f"{key}_mean"], row[f"{key}_sd"] = mean, sd
        row[f"per_seed_{key}"] = series(values)
    return row


def percentage(value):
    return "n/a" if value is None else f"{value:.2%}"


def percent_mean_sd(mean, sd):
    if mean is None:
        return "n/a"
    return f"{mean:.2%} +/- {sd:.2%}"


def number_mean_sd(mean, sd, digits=4):
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIRS",
                        help="Repeatable arm definition; directories are comma-separated run or test_eval paths")
    parser.add_argument("--arm-dataset", action="append", default=[], metavar="NAME=PATH",
                        help="Dataset directory whose manifest records train_holdout_attacks")
    parser.add_argument("--held-out-attacks", action="append", default=[], metavar="NAME=LABELS",
                        help="Comma-separated held-out labels, as an alternative to --arm-dataset")
    parser.add_argument("--expect-seeds", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    order, directories, held_out = parse_arms(args.arm, args.arm_dataset, args.held_out_attacks)
    entries = {name: group_by_model(name, [load_entry(path) for path in directories[name]], args.expect_seeds)
               for name in order}
    control = entries["control"]
    for name in order:
        if set(entries[name]) != set(control):
            raise ValueError(f"Arm {name} and control cover different model families")
        for model in control:
            seeds = [entry["seed"] for entry in entries[name][model]]
            if seeds != [entry["seed"] for entry in control[model]]:
                raise ValueError(f"Arm {name}/{model} does not share the control seed set")
            base, other = control[model][0], entries[name][model][0]
            if not np.array_equal(base["ids"], other["ids"]) or not np.array_equal(base["y"], other["y"]):
                raise ValueError(f"Arm {name}/{model} scored a different prediction population; the split moved")
            if not np.array_equal(base["attacks"], other["attacks"]):
                raise ValueError(f"Arm {name}/{model} has different attack labels")

    output = fresh(args.out or ROOT / "runs" / unique_name("seed_sweep"))
    arm_rows, delta_rows, subtype_rows = [], [], []
    for name in order:
        for model in sorted(entries[name]):
            labels = held_out.get(name, [])
            arm_rows.append(arm_row(name, model, entries[name][model], labels))
            if name == "control":
                continue
            pooled, rows = compare(control[model], entries[name][model], labels, name, model)
            subtype_rows.append(pooled)
            subtype_rows.extend(rows)
            delta_rows.append({"arm": name, "model": model, "seeds": pooled["seeds"],
                               "delta_ap_mean": mean_sd([right["metrics"]["average_precision"] -
                                                         left["metrics"]["average_precision"]
                                                         for left, right in zip(control[model],
                                                                                entries[name][model])])[0],
                               "delta_ap_sd": mean_sd([right["metrics"]["average_precision"] -
                                                       left["metrics"]["average_precision"]
                                                       for left, right in zip(control[model],
                                                                              entries[name][model])])[1],
                               "per_seed_delta_ap": series([right["metrics"]["average_precision"] -
                                                            left["metrics"]["average_precision"]
                                                            for left, right in zip(control[model],
                                                                                   entries[name][model])]),
                               "delta_1pct_mean": pooled["delta_1pct_mean"],
                               "delta_1pct_sd": pooled["delta_1pct_sd"],
                               "per_seed_delta_1pct": pooled["per_seed_delta_1pct"],
                               "delta_0_1pct_mean": pooled["delta_0_1pct_mean"],
                               "delta_0_1pct_sd": pooled["delta_0_1pct_sd"],
                               "per_seed_delta_0_1pct": pooled["per_seed_delta_0_1pct"],
                               "seeds_lower_1pct": pooled["seeds_lower_1pct"],
                               "seeds_lower_0_1pct": pooled["seeds_lower_0_1pct"]})

    write_csv(output / "seed_sweep_arms.csv", arm_rows)
    write_csv(output / "seed_sweep_deltas.csv", delta_rows)
    write_csv(output / "seed_sweep_subtypes.csv", subtype_rows)

    seeds = [entry["seed"] for entry in control[next(iter(control))]]
    lines = ["# Seed sweep: control versus training holdout", "",
             f"Seeds per arm: {', '.join(str(seed) for seed in seeds)}. Each run keeps the operating thresholds "
             "selected on its own real validation split; no refitting and no population change. "
             "Deltas pair each holdout run with the control run of the same seed.", "",
             "Three seeds bound retraining noise only. They do not support significance testing and do not "
             "reduce sampling uncertainty from the small number of hosts and campaigns.", "",
             "## Arm summary", "",
             "| Model | Arm | Seeds | Validation AP | Test AP | Pooled held-out R@1% | Pooled held-out R@0.1% |",
             "|---|---|---:|---|---|---|---|"]
    for row in arm_rows:
        lines.append(f"| {row['model']} | {row['arm']} | {row['seeds']} | "
                     f"{number_mean_sd(row['validation_ap_mean'], row['validation_ap_sd'])} | "
                     f"{number_mean_sd(row['test_ap_mean'], row['test_ap_sd'])} | "
                     f"{percent_mean_sd(row['pooled_recall_1pct_mean'], row['pooled_recall_1pct_sd'])} | "
                     f"{percent_mean_sd(row['pooled_recall_0_1pct_mean'], row['pooled_recall_0_1pct_sd'])} |")
    for name in order:
        if name == "control":
            continue
        lines += ["", f"## Paired delta against control: {name}", "",
                  "| Model | Delta AP mean +/- sd | Delta AP per seed | Delta R@1% mean +/- sd | Delta R@1% per seed | "
                  "Delta R@0.1% mean +/- sd | Seeds lower (R@1%) |",
                  "|---|---|---|---|---|---|---:|"]
        for row in (item for item in delta_rows if item["arm"] == name):
            lines.append(f"| {row['model']} | {number_mean_sd(row['delta_ap_mean'], row['delta_ap_sd'])} | "
                         f"{row['per_seed_delta_ap']} | "
                         f"{percent_mean_sd(row['delta_1pct_mean'], row['delta_1pct_sd'])} | "
                         f"{row['per_seed_delta_1pct']} | "
                         f"{percent_mean_sd(row['delta_0_1pct_mean'], row['delta_0_1pct_sd'])} | "
                         f"{row['seeds_lower_1pct']}/{len(seeds)} |")
        for model in sorted(control):
            subset = [row for row in subtype_rows if row["arm"] == name and row["model"] == model]
            lines += ["", f"### {name} / {model.upper()}", "",
                      "| Primary attack | Held out | N | Hosts | R@1% control | R@1% holdout | Delta R@1% | Seeds lower | "
                      "R@0.1% control | R@0.1% holdout | Delta R@0.1% | AP control | AP holdout |",
                      "|---|:--:|---:|---:|---|---|---|---:|---|---|---|---|---|"]
            for row in subset:
                lines.append(f"| {row['primary_attack']} | {'yes' if row['held_out'] else 'no'} | {row['n']} | "
                             f"{row['hosts']} | {percent_mean_sd(row['recall_control_1pct_mean'], row['recall_control_1pct_sd'])} | "
                             f"{percent_mean_sd(row['recall_holdout_1pct_mean'], row['recall_holdout_1pct_sd'])} | "
                             f"{percent_mean_sd(row['delta_1pct_mean'], row['delta_1pct_sd'])} | "
                             f"{row['seeds_lower_1pct']}/{len(seeds)} | "
                             f"{percent_mean_sd(row['recall_control_0_1pct_mean'], row['recall_control_0_1pct_sd'])} | "
                             f"{percent_mean_sd(row['recall_holdout_0_1pct_mean'], row['recall_holdout_0_1pct_sd'])} | "
                             f"{percent_mean_sd(row['delta_0_1pct_mean'], row['delta_0_1pct_sd'])} | "
                             f"{number_mean_sd(row['ap_control_mean'], row['ap_control_sd'], 3)} | "
                             f"{number_mean_sd(row['ap_holdout_mean'], row['ap_holdout_sd'], 3)} |")
    lines += ["", "## Scope", "",
              "Binary attack detection per 100-second sequence, aggregated over seeds. A seed changes weight "
              "initialization, batch order and dropout only; the split, frozen bucket encoder, features and "
              "class weights are fixed. Mean +/- sd over three seeds describes retraining noise. Per-subtype counts "
              "are small and host-dependent, so the standard deviation is not a confidence interval. Test is "
              "scored once per seed with that seed's own validation thresholds; repeated test use is disclosed.", ""]
    (output / "SEED_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")

    write_json(output / "manifest.json", {
        "complete": True, "arms": order, "seeds": seeds, "models": sorted(control),
        "held_out_attacks": {name: held_out.get(name, []) for name in order},
        "dataset_manifest_sha256": {name: sorted({entry["dataset_sha"] for items in entries[name].values()
                                                  for entry in items}) for name in order},
        "refitting": False, "threshold_source": "each run's saved real validation thresholds",
        "control_and_holdout_share_prediction_population": True,
        "repeated_test_use": True, "split": control[next(iter(control))][0]["split"],
        "statistics": "mean and sample standard deviation over seeds; per-seed values retained",
        "subtype_rows": len(subtype_rows), "delta_rows": len(delta_rows), "arm_rows": len(arm_rows)})
    print("\n".join(lines), flush=True)
    print(f"Saved to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
