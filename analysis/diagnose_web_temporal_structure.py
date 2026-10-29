"""Describe Web temporal support and frozen-model errors on existing real validation data."""
import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from train.common.data import Scheme2
from train.common.io import ROOT, fresh, read_json, sha256, unique_name, write_json
from train.run_circular_shift import write_csv

WEB = "Brute_Force_-Web"
DDOS = ("DDOS_attack-HOIC", "DDOS_attack-LOIC-UDP", "DDoS_attacks-LOIC-HTTP")


def mask_features(mask):
    active = np.asarray(mask, dtype=bool)
    count = active.sum(1)
    adjacent = (active[:, 1:] & active[:, :-1]).sum(1)
    current, longest = np.zeros(len(active), dtype=int), np.zeros(len(active), dtype=int)
    for step in range(active.shape[1]):
        current = np.where(active[:, step], current + 1, 0)
        longest = np.maximum(longest, current)
    position = np.arange(active.shape[1])[None, :]
    previous = np.maximum.accumulate(np.where(active, position, -1), axis=1)
    previous = np.concatenate((np.full((len(active), 1), -1), previous[:, :-1]), axis=1)
    valid_gap = active & (previous >= 0)
    gaps = np.where(valid_gap, position - previous - 1, 0)
    max_gap = gaps.max(1).astype(float)
    max_gap[count < 2] = np.nan
    return {"active_buckets": count, "adjacent_active_pairs": adjacent, "longest_active_run": longest,
            "max_empty_gap_between_active": max_gap}


def embedding_smoothness(x, mask):
    """Compare observed active-neighbor similarity to exact within-sequence permutation expectation."""
    n, steps = mask.shape
    observed = np.full(n, np.nan)
    expected = np.full(n, np.nan)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        active = np.asarray(mask[start:end], dtype=bool)
        vectors = np.array(x[start:end, :, :64], dtype=np.float64, copy=True)
        norm = np.linalg.norm(vectors, axis=2, keepdims=True)
        vectors = np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)
        vectors *= active[..., None]
        k = active.sum(1)
        all_pairs = np.sum(vectors.sum(1) ** 2, axis=1) - np.sum(vectors ** 2, axis=(1, 2))
        expected[start:end] = np.divide(all_pairs, k * (k - 1), out=np.full(end-start, np.nan), where=k >= 3)
        previous = np.maximum.accumulate(np.where(active, np.arange(steps)[None, :], -1), axis=1)
        previous = np.concatenate((np.full((end-start, 1), -1), previous[:, :-1]), axis=1)
        valid = active & (previous >= 0)
        before = vectors[np.arange(end-start)[:, None], np.maximum(previous, 0)]
        dots = (vectors * before).sum(2)
        observed[start:end] = np.divide((dots * valid).sum(1), k - 1,
                                       out=np.full(end-start, np.nan), where=k >= 3)
    return {"active_neighbor_cosine": observed, "within_sequence_permutation_expected_cosine": expected,
            "order_specific_cosine_excess": observed - expected}


def median(values, keep):
    selected = values[keep]
    selected = selected[np.isfinite(selected)]
    return float(np.median(selected)) if len(selected) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "runs/20261024_223102_671768_circular_multiseed_start5_ef34ba")
    parser.add_argument("--context", type=Path, default=ROOT / "runs/20261024_124830_784309_window_context_availability_0ce90b/anchor_window_profiles.csv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    plan = read_json(args.study / "plan.json")
    if read_json(args.study / "status.json")["status"] != "complete":
        raise ValueError("A completed multi-seed study is required")
    if len(plan["seeds"]) != 5 or not {"xgboost", "lstm"}.issubset(plan["models"]):
        raise ValueError("This diagnostic requires five paired seeds with XGBoost and LSTM")
    source = Scheme2(plan["dataset"], plan["cache"])
    if source.identity != plan["dataset_manifest_sha256"] or source.timesteps != 10:
        raise ValueError("Expected the same original ten-bucket dataset")
    split = source.load("validation")
    try:
        features = mask_features(split["mask"])
        features.update(embedding_smoothness(split["x"], split["mask"]))
        attack_mask = np.empty(split["mask"].shape, dtype=bool)
        offset = 0
        for record in source.manifest["shards"]:
            if Path(record["path"]).parts[0] != "validation":
                continue
            with np.load(source.path / record["path"], allow_pickle=False) as shard:
                end = offset + record["count"]
                np.testing.assert_array_equal(shard["sample_id"], split["ids"][offset:end])
                np.testing.assert_array_equal(shard["mask"], split["mask"][offset:end])
                np.testing.assert_array_equal(shard["flow_counts"].sum(1), split["flows"][offset:end])
                attack_mask[offset:end] = shard["y_bucket"] > 0
                offset = end
        if offset != len(split):
            raise ValueError("Validation bucket-label alignment failed")
        np.testing.assert_array_equal(attack_mask.max(1), split["y"] > 0)
        attack_features = mask_features(attack_mask)
        features.update({"attack_" + key: value for key, value in attack_features.items()})
        flow_file = source.path / "flow_metadata_validation.npz"
        if sha256(flow_file) != read_json(source.path / "flow_metadata.json")["splits"]["validation"]["sha256"]:
            raise ValueError("Flow metadata checksum mismatch")
        with np.load(flow_file, allow_pickle=False) as flow:
            np.testing.assert_array_equal(flow["sample_id"], split["ids"])
            np.testing.assert_array_equal(flow["total_flows"], split["flows"])
            attack_fraction = flow["attack_flows"] / flow["total_flows"]
        frequencies = {name: np.zeros(len(split), dtype=np.uint8) for name in plan["models"]}
        per_seed = []
        web = split["attack"] == WEB
        for record in plan["runs"]:
            if record["start_bucket"] != 0:
                continue
            folder = Path(record["path"])
            if sha256(folder / "predictions.npz") != record["hashes"]["predictions.npz"]:
                raise ValueError("Saved predictions changed")
            if sha256(folder / "thresholds.json") != record["hashes"]["thresholds.json"]:
                raise ValueError("Saved thresholds changed")
            summary = read_json(folder / "summary.json")
            threshold = read_json(folder / "thresholds.json")["values"]["validation_fpr_1pct"]
            if summary["validation_cohort_sha256"] != split.info["cohort_sha256"]:
                raise ValueError("Prediction cohort mismatch")
            with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                for key, expected in (("sample_id", "ids"), ("y", "y"), ("primary_attack", "attack"), ("source_host", "host")):
                    np.testing.assert_array_equal(saved[key], split[expected])
                detected = saved["score"] >= threshold
            frequencies[record["model"]] += detected
            for group, condition in (("one", features["active_buckets"] == 1), ("two", features["active_buckets"] == 2),
                                     ("three_or_more", features["active_buckets"] >= 3)):
                keep = web & condition
                per_seed.append({"model": record["model"], "seed": record["seed"], "group": group,
                                 "n": int(keep.sum()), "recall": float(detected[keep].mean())})
        seeds = len(plan["seeds"])
        groups = {"Benign": split["y"] == 0, "Web": web, "Infilteration": split["attack"] == "Infilteration",
                  "DDoS_pooled": np.isin(split["attack"], DDOS),
                  "Web_XGB_ge4_LSTM_le1": web & (frequencies["xgboost"] >= 4) & (frequencies["lstm"] <= 1),
                  "Web_LSTM_ge4_XGB_le1": web & (frequencies["lstm"] >= 4) & (frequencies["xgboost"] <= 1),
                  "Web_both_le1": web & (frequencies["xgboost"] <= 1) & (frequencies["lstm"] <= 1)}
        profiles = []
        for name, keep in groups.items():
            n = int(keep.sum())
            row = {"group": name, "n": n, "hosts": int(len(np.unique(split["host"][keep])))}
            for key, values in features.items():
                row["median_" + key] = median(values, keep)
            row.update({"single_bucket_n": int((keep & (features["active_buckets"] == 1)).sum()),
                        "le2_buckets_n": int((keep & (features["active_buckets"] <= 2)).sum()),
                        "no_adjacent_active_n": int((keep & (features["adjacent_active_pairs"] == 0)).sum()),
                        "single_attack_bucket_n": int((keep & (features["attack_active_buckets"] == 1)).sum()),
                        "median_flows": median(split["flows"], keep), "median_attack_fraction": median(attack_fraction, keep),
                        "smoothness_eligible_n": int((keep & np.isfinite(features["order_specific_cosine_excess"])).sum())})
            profiles.append(row)
        recalls = []
        for name in plan["models"]:
            for group in ("one", "two", "three_or_more"):
                selected = [row for row in per_seed if row["model"] == name and row["group"] == group]
                values = [row["recall"] for row in selected]
                recalls.append({"model": name, "group": group, "n": selected[0]["n"],
                                "mean_recall": float(np.mean(values)), "sd_recall": float(np.std(values, ddof=1))})
        samples = []
        for index in np.flatnonzero(web):
            sample = {"sample_id": int(split["ids"][index]), "source_host": str(split["host"][index]),
                      "flows": int(split["flows"][index]), "attack_fraction": float(attack_fraction[index])}
            sample.update({key: float(value[index]) if np.isfinite(value[index]) else None for key, value in features.items()})
            sample.update({name + "_detections_out_of_5": int(freq[index]) for name, freq in frequencies.items()})
            sample.update({name: bool(keep[index]) for name, keep in groups.items() if name.startswith("Web_")})
            samples.append(sample)
        context_rows = []
        if args.context.exists():
            with args.context.open(encoding="utf-8", newline="") as stream:
                prior = {(int(row["sample_id"]), int(row["window_seconds"])): row for row in csv.DictReader(stream)
                         if row["attack_type"] == WEB and row["bucket_seconds"] == "10" and row["window_seconds"] in ("100", "400")}
            for name, keep in groups.items():
                if not name.startswith("Web"):
                    continue
                selected = [int(sid) for sid in split["ids"][keep] if (int(sid), 100) in prior and (int(sid), 400) in prior]
                if not selected:
                    continue
                original = np.array([int(prior[sid, 100]["active_buckets"]) for sid in selected])
                expanded = np.array([int(prior[sid, 400]["active_buckets"]) for sid in selected])
                context_rows.append({"group": name, "common_eligible_n": len(selected),
                    "median_active_100s": float(np.median(original)), "median_active_400s": float(np.median(expanded)),
                    "unchanged_active_n": int((expanded == original).sum()), "still_single_400s_n": int((expanded == 1).sum())})
        out = fresh(args.out or ROOT / "runs" / unique_name("web_temporal_support_diagnostic"))
        for file, rows in (("cohort_profiles.csv", profiles), ("web_recall_by_occupancy.csv", recalls),
                           ("web_recall_per_seed.csv", per_seed), ("web_samples.csv", samples)):
            write_csv(out / file, rows)
        if context_rows:
            write_csv(out / "past_context_comparison.csv", context_rows)
        limitations = [
            "Descriptive real-validation diagnostic using existing no-shift predictions and thresholds; no model fitting or new test scoring.",
            "The saved classifiers reused the existing training cohort; five seeds are repeated fits on the same observations, not independent validation samples.",
            "Sparse occupancy limits observed cross-bucket support; it does not establish absence of temporal dependence in the underlying attack.",
            "Cosine excess measures local embedding smoothness relative to exact active-vector permutation expectation, not all possible temporal patterns.",
            "Smoothness is undefined for fewer than three active buckets; single-bucket traffic is not assigned zero similarity.",
            "Benign traffic can be equally sparse. Occupancy alone is not an attack discriminator or a causal explanation of model differences.",
            "XGBoost consumes flattened positions and masks, so it is not strictly order-invariant. It can exploit single-bucket feature thresholds without recurrent compression.",
            "Model-specific hidden-space PCA geometry does not directly measure raw temporal continuity; compare full input features before attributing causes.",
            "Past-context rows reuse the earlier validation-only availability audit on common eligible anchors; they are not new detection results.",
            "Stable outcome groups condition on these fitted models and global thresholds; observed associations do not identify causal mechanisms."]
        write_json(out / "summary.json", {"complete": True, "test_used": False, "model_refitting": False,
            "scope": "Existing real-validation descriptive diagnostic", "study": str(args.study.resolve()),
            "dataset_manifest_sha256": source.identity, "profiles": profiles, "recalls": recalls,
            "past_context": context_rows, "context_sha256": sha256(args.context) if args.context.exists() else None,
            "definitions": {"stable_xgb_only": "XGBoost detected >=4/5 seeds, LSTM detected <=1/5, original global 1%-FPR thresholds",
                "order_specific_cosine_excess": "Mean cosine of chronologically successive active vectors minus mean cosine over all distinct within-sequence pairs; only k>=3"},
            "limitations": limitations})
        lines = ["# Web temporal support and classifier disagreement", "",
                 "No retraining, threshold fitting, training-sample evaluation or test scoring. Original validation inputs only.", "",
                 "## Occupancy and consecutive support", "",
                 "| Cohort | n | Single active bucket | No adjacent active bins | Median active bins | Median flows |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in profiles:
            lines.append(f"| {row['group']} | {row['n']} | {row['single_bucket_n']} | {row['no_adjacent_active_n']} | "
                         f"{row['median_active_buckets']} | {row['median_flows']} |")
        lines += ["", "## Web recall at each model's global <=1% validation FPR threshold", "",
                  "| Model | Active-bucket group | n | Mean recall | SD across five seeds |", "|---|---|---:|---:|---:|"]
        for row in recalls:
            lines.append(f"| {row['model']} | {row['group']} | {row['n']} | {row['mean_recall']:.2%} | {row['sd_recall']:.2%} |")
        lines += ["", "## Past-context availability", "", "Matched eligible validation anchors; same 10-second bucket width.", ""]
        for row in context_rows:
            lines.append(f"- {row['group']}: n={row['common_eligible_n']}; median active buckets "
                f"{row['median_active_100s']} -> {row['median_active_400s']} (100s -> 400s); "
                f"unchanged={row['unchanged_active_n']}; still single at 400s={row['still_single_400s_n']}.")
        lines += ["", "Full continuous-feature summaries and sample IDs are retained in CSV/JSON, including exact active-vector permutation expectations.", ""]
        lines += [f"- {item}" for item in limitations]
        (out / "DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Web temporal support diagnostic: {out}", flush=True)
        print(read_json(out / "summary.json")["profiles"], flush=True)
        return out
    finally:
        split.close()


if __name__ == "__main__":
    main()
