#!/usr/bin/env python3
r"""Frozen CNN-AE host-distribution and feature-group reconstruction diagnostics.

NO training, NO threshold recalibration, NO test-based model selection.
Reuses ONLY the local official runner's CNN, data reader, and transformer.
Never imports teammate preprocess.py/window.py/ae_ablation.py/attribute.py.

PowerShell (from repo root):
    python .\analysis\diagnose_cnn_ae_hosts_v2.py --run-dir .\experiments\cnn_ae_temporal_control_seed0

All neural-network inference MUST run on CUDA. CPU is used only for loading,
feature transforms, and aggregate statistics. Outputs do not overwrite the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_cnn_ae_temporal_control import (
    CNNAutoencoder, FeatureTransformer, load_split, read_json, require_cuda,
)

GROUP_ORDER = (
    "flow_volume", "duration", "protocol", "l7", "source_ports",
    "destination_ports", "destination_ips", "tcp_flags", "icmp", "dns", "ftp",
)
PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def assign_group(name: str) -> str:
    if name == "flow_count" or name.endswith("_sum"):
        return "flow_volume"
    if name.startswith("duration_"):
        return "duration"
    if name.startswith("protocol_"):
        return "protocol"
    if name.startswith("l7_"):
        return "l7"
    if name.startswith("src_port_"):
        return "source_ports"
    if name.startswith("dst_port_"):
        return "destination_ports"
    if name.startswith("dst_ip_"):
        return "destination_ips"
    if name.startswith(("tcp_flag_", "client_tcp_flag_", "server_tcp_flag_")):
        return "tcp_flags"
    if name.startswith("icmp_"):
        return "icmp"
    if name.startswith("dns_"):
        return "dns"
    if name.startswith("ftp_"):
        return "ftp"
    raise ValueError(f"Feature has no group: {name}")


def group_indices(names):
    groups = {name: [] for name in GROUP_ORDER}
    for i, name in enumerate(names):
        groups[assign_group(name)].append(i)
    assert sum(map(len, groups.values())) == len(names)
    assert all(groups.values()), "Unexpected empty feature group"
    return groups


def save_json(path, obj):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    path.write_text(json.dumps(convert(obj), indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def load_frozen_model(run_dir, device, expected_names):
    hosts_config = read_json(run_dir / "validation_hosts.json")
    seed = int(hosts_config["seed"])
    checkpoint_path = run_dir / f"cnn_ae_seed{seed}_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint missing: {checkpoint_path}")
    # This checkpoint is created by our own runner and contains plain tensors,
    # numbers, and strings. Do not load untrusted PyTorch pickle files.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt["feature_names"] != expected_names:
        raise ValueError("Checkpoint feature names/order differ from official schema")
    if (int(ckpt["seq_len"]), int(ckpt["n_features"])) != (10, 88):
        raise ValueError("Checkpoint input shape differs from (10,88)")
    if int(ckpt["seed"]) != seed:
        raise ValueError("Checkpoint seed differs from validation_hosts.json")
    model = CNNAutoencoder(
        int(ckpt["n_features"]), int(ckpt["seq_len"]),
        int(ckpt["width"]), int(ckpt["latent"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("Model not on CUDA; refusing CPU inference")
    print(f"[model] loaded {checkpoint_path.name}; best epoch={ckpt['best_epoch']}; "
          f"width={ckpt['width']}, latent={ckpt['latent']}")
    return model, hosts_config


def load_saved_transform(run_dir, names):
    cfg = read_json(run_dir / "feature_transform.json")
    if cfg["feature_names"] != names:
        raise ValueError("Saved feature_transform.json names/order differ from schema")
    trans = FeatureTransformer(names)
    expected_log = [names[i] for i in trans.log_cols]
    expected_bounded = [names[i] for i in trans.bounded_cols]
    if cfg["log1p_and_standard_scale"] != expected_log:
        raise ValueError("Saved log1p/scaled feature set differs from runner definition")
    if cfg["unchanged_bounded_0_1"] != expected_bounded:
        raise ValueError("Saved bounded feature set differs from runner definition")
    trans.scaler = joblib.load(run_dir / "scaler_log_features.joblib")
    if getattr(trans.scaler, "n_features_in_", None) != len(trans.log_cols):
        raise ValueError("Saved StandardScaler geometry mismatch")
    print(f"[scaler] restored saved scaler; log/scaled={len(trans.log_cols)}, "
          f"unscaled/bounded={len(trans.bounded_cols)}")
    return trans


@torch.inference_mode()
def per_window_feature_error(model, X, transformer, device, batch):
    """Feature-wise mean over 10 buckets, shape (N,88), on frozen model."""
    errors = np.empty((len(X), X.shape[2]), dtype=np.float32)
    for start in range(0, len(X), batch):
        raw = X[start:start + batch]
        x_np = transformer.transform(raw)
        x = torch.from_numpy(x_np).to(device)
        if not x.is_cuda:
            raise RuntimeError("CPU inference refused")
        pred = model(x)
        e = (x - pred).square().mean(dim=1)  # (B, 88), all timesteps equally weighted
        errors[start:start + len(raw)] = e.cpu().numpy()
    if not np.isfinite(errors).all():
        raise ValueError("Non-finite reconstruction errors")
    return errors


def stats(scores, mask, role, host, thresholds):
    selected = scores[mask]
    if not len(selected):
        raise ValueError(f"Empty role/host {role} {host}")
    row = {"role": role, "src_ip": host, "n_sequences": int(len(selected)),
           "mean": float(np.mean(selected)), "std": float(np.std(selected)),
           "min": float(np.min(selected)), "max": float(np.max(selected))}
    for q in PERCENTILES:
        row[f"p{round(q * 100):02d}"] = float(np.quantile(selected, q))
    for tag, threshold in thresholds.items():
        row[f"n_flagged_{tag}"] = int(np.count_nonzero(selected >= threshold))
        row[f"flag_rate_{tag}"] = float(np.mean(selected >= threshold))
    return row


def label_group(host):
    if host.startswith("59.166."):
        return "59.166.*"
    if host.startswith("149.171."):
        return "149.171.*"
    return "other"


def score_match_check(run_dir, name, scores, meta):
    path = run_dir / f"{name}_scores.csv"
    if not path.is_file():
        print(f"[check] {path.name} not present; no previous-score comparison")
        return None
    old = pd.read_csv(path, usecols=["sequence_id", "reconstruction_mse"])
    if not np.array_equal(old["sequence_id"].to_numpy(), meta["sequence_id"].to_numpy()):
        raise ValueError(f"{path.name}: sequence ids/order differ; dataset may have changed")
    old_scores = old["reconstruction_mse"].to_numpy(dtype=np.float64)
    abs_err = np.abs(scores - old_scores)
    max_err = float(abs_err.max())
    if not np.allclose(scores, old_scores, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{path.name}: frozen checkpoint inference doesn't reproduce scores; "
                         f"max absolute difference={max_err:.7g}. STOP and inspect checkpoint/data.")
    print(f"[check] {name} frozen scores match earlier results (max diff {max_err:.2g})")
    return max_err


def feature_summary(errors, names, role, groups):
    per_feature = errors.mean(axis=0, dtype=np.float64)
    overall = float(per_feature.mean())
    feat_rows = [
        {"role": role, "feature": n, "group": assign_group(n),
         "mean_feature_mse": float(per_feature[i]),
         "contribution_to_total_mse": float(per_feature[i] / len(names)),
         "share_of_total_mse": float(per_feature[i] / (len(names) * overall))
         if overall > 0 else 0.0}
        for i, n in enumerate(names)
    ]
    group_rows = []
    for group, idx in groups.items():
        group_sum = float(per_feature[idx].sum())
        group_rows.append({
            "role": role, "group": group, "n_features": len(idx),
            "mean_per_feature_mse": group_sum / len(idx),
            "contribution_to_total_mse": group_sum / len(names),
            "share_of_total_mse": group_sum / (len(names) * overall) if overall > 0 else 0.0,
        })
    return feat_rows, group_rows


def compare_errors(base, other, names, groups, comparison):
    a = base.mean(axis=0, dtype=np.float64)
    b = other.mean(axis=0, dtype=np.float64)
    gap = b - a
    positive_mass = float(np.maximum(gap, 0).sum())
    n_features = len(names)
    feat_rows, group_rows = [], []
    for i, name in enumerate(names):
        feat_rows.append({
            "comparison": comparison, "feature": name, "group": assign_group(name),
            "baseline_feature_mse": float(a[i]),
            "target_feature_mse": float(b[i]),
            "difference_target_minus_baseline": float(gap[i]),
            "contribution_to_total_mse_gap": float(gap[i] / n_features),
            "share_of_positive_gap": float(max(gap[i], 0) / positive_mass)
            if positive_mass > 0 else 0.0,
        })
    for group, idx in groups.items():
        group_rows.append({
            "comparison": comparison, "group": group,
            "n_features": len(idx),
            "baseline_group_contribution": float(a[idx].sum() / n_features),
            "target_group_contribution": float(b[idx].sum() / n_features),
            "signed_contribution_to_total_mse_gap": float(gap[idx].sum() / n_features),
            "share_of_positive_feature_gap": float(np.maximum(gap[idx], 0).sum() / positive_mass)
            if positive_mass > 0 else 0.0,
        })
    return feat_rows, group_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("experiments/cnn_ae_temporal_control_seed0"))
    ap.add_argument("--data-dir", type=Path, default=Path("processed_binary_temporal"))
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"Experiment folder missing: {args.run_dir}")
    outdir = args.outdir if args.outdir is not None else args.run_dir / "host_diagnostics"
    if outdir.resolve() == args.run_dir.resolve():
        raise ValueError("--outdir must not be same as frozen training run directory")

    device = require_cuda(args.gpu_index)
    man = read_json(args.data_dir / "split_manifest.json")
    schema = read_json(args.data_dir / "categorical_schema.json")
    names = list(schema["feature_names"])
    if len(names) != 88 or names != man["feature_names"]:
        raise ValueError("Expected 88 features and matching schema/manifest order")
    groups = group_indices(names)
    thresholds_json = read_json(args.run_dir / "thresholds.json")
    thresholds = {k: float(v) for k, v in thresholds_json["thresholds"].items()}
    if thresholds_json["source"] != "benign_validation_only":
        raise ValueError("Frozen thresholds not calibrated on benign validation")
    model, hosts_config = load_frozen_model(args.run_dir, device, names)
    trans = load_saved_transform(args.run_dir, names)
    val_hosts = set(hosts_config["validation_hosts"])
    tr_hosts = set(hosts_config["true_train_hosts"])
    if not val_hosts.isdisjoint(tr_hosts) or len(val_hosts) < 2:
        raise ValueError("Invalid frozen host-disjoint validation split")
    print("[protocol] frozen checkpoint, frozen scaler, frozen thresholds; no training or tuning")

    # Load the original 1st round train NPZ, partition only benign by SAVED hosts.
    X, y, meta = load_split(args.data_dir, "round_1", "train", names, man)
    host = meta["src_ip"].to_numpy()
    idx_tr = np.flatnonzero((y == 0) & np.isin(host, sorted(tr_hosts)))
    idx_val = np.flatnonzero((y == 0) & np.isin(host, sorted(val_hosts)))
    if len(idx_tr) != int(hosts_config["train_sequence_count"]):
        raise ValueError("True training-sequence count doesn't match saved hosts")
    if len(idx_val) != int(hosts_config["val_sequence_count"]):
        raise ValueError("Validation-sequence count doesn't match saved hosts")
    if len(idx_tr) + len(idx_val) != int((y == 0).sum()):
        raise ValueError("Saved host groups do not partition all benign training sequences")

    print(f"[inference] benign train={len(idx_tr):,}; benign validation={len(idx_val):,}")
    err_train = per_window_feature_error(model, X[idx_tr], trans, device, args.batch_size)
    err_val = per_window_feature_error(model, X[idx_val], trans, device, args.batch_size)
    meta_train = meta.iloc[idx_tr].reset_index(drop=True)
    meta_val = meta.iloc[idx_val].reset_index(drop=True)
    del X, y, meta

    print("[inference] round_1/test")
    X1, y1, meta1 = load_split(args.data_dir, "round_1", "test", names, man)
    print("[inference] round_2/test")
    X2, y2, meta2 = load_split(args.data_dir, "round_2", "test", names, man)

    # The two rounds have the SAME benign windows, but different malicious hosts.
    # Check identity on the ORIGINAL NPZ inputs, not on float32 CUDA output.
    # The latter can differ slightly with batch composition/size, even on the
    # same GPU with frozen checkpoint and scaler.
    benign1 = y1 == 0
    benign2 = y2 == 0
    benign_keys = ["src_ip", "start_bucket_ms", "end_bucket_ms", "label"]
    if not np.array_equal(meta1.loc[benign1, benign_keys].to_numpy(),
                          meta2.loc[benign2, benign_keys].to_numpy()):
        raise ValueError("Benign test metadata/windows differ across rounds")
    if not np.array_equal(X1[benign1], X2[benign2]):
        raise ValueError("Benign test NPZ feature values differ across rounds")
    print(f"[check] original benign test inputs match exactly in both rounds "
          f"({int(benign1.sum()):,} sequences, 10x88 features)")

    err_r1 = per_window_feature_error(model, X1, trans, device, args.batch_size)
    err_r2 = per_window_feature_error(model, X2, trans, device, args.batch_size)
    del X1, X2

    scores_val = err_val.mean(axis=1, dtype=np.float64)
    scores_r1 = err_r1.mean(axis=1, dtype=np.float64)
    scores_r2 = err_r2.mean(axis=1, dtype=np.float64)
    maxdiff_r1 = score_match_check(args.run_dir, "round_1", scores_r1, meta1)
    maxdiff_r2 = score_match_check(args.run_dir, "round_2", scores_r2, meta2)
    # No new thresholds; validate saved calibration quantiles for reproducibility.
    calibration_diffs = {}
    for tag, target_fpr in (("0.1pct_fpr", .001), ("1pct_fpr", .01), ("5pct_fpr", .05)):
        threshold = float(np.quantile(scores_val, 1 - target_fpr))
        diff = abs(threshold - thresholds[tag])
        calibration_diffs[tag] = diff
        if not np.isclose(threshold, thresholds[tag], atol=1e-4, rtol=1e-4):
            raise ValueError(f"Saved {tag} threshold {thresholds[tag]:.8g} does not reproduce "
                             f"with frozen validation scores {threshold:.8g}")
    # A numerical comparison of the per-window scores is appropriate here.
    # Earlier saved results already differ by ~1.7e-5 for a small fraction
    # of identical benign inputs, so strict elementwise per-feature equality
    # is not a valid reproducibility requirement for CUDA float32 inference.
    benign_score_delta = np.abs(scores_r1[benign1] - scores_r2[benign2])
    benign_score_max_abs_delta = float(benign_score_delta.max())
    if not np.allclose(scores_r1[benign1], scores_r2[benign2],
                       atol=1e-4, rtol=1e-4):
        raise ValueError("Frozen benign test scores differ materially across rounds: "
                         f"max_abs_diff={benign_score_max_abs_delta:.8g}. "
                         "The raw inputs match, so investigate GPU inference.")
    benign_feature_max_abs_delta = float(
        np.max(np.abs(err_r1[benign1] - err_r2[benign2]))
    )
    print("[check] saved validation thresholds reproduced; frozen benign scores "
          f"agree within CUDA tolerance (max window-score diff="
          f"{benign_score_max_abs_delta:.3g}; "
          f"max individual feature-error diff={benign_feature_max_abs_delta:.3g})")

    role_data = {
        "benign_train": (err_train, meta_train),
        "benign_validation": (err_val, meta_val),
        "round_1_test": (err_r1, meta1),
        "round_2_test": (err_r2, meta2),
    }
    host_rows, cohort_rows, feature_rows, group_rows = [], [], [], []
    role_scores = {}
    for role, (err, mt) in role_data.items():
        s = err.mean(axis=1, dtype=np.float64)
        role_scores[role] = s
        host_arr = mt["src_ip"].to_numpy(dtype=str)
        for ip in sorted(np.unique(host_arr)):
            m = host_arr == ip
            host_rows.append({**stats(s, m, role, ip, thresholds),
                              "host_group": label_group(ip),
                              "class": "attack" if mt.loc[m, "label"].iloc[0] else "benign"})
        for prefix in sorted(set(map(label_group, host_arr))):
            m = np.array([label_group(i) == prefix for i in host_arr])
            cohort_name = f"{role}/{prefix}"
            cohort_rows.append(stats(s, m, cohort_name, "ALL", thresholds))
            fr, gr = feature_summary(err[m], names, cohort_name, groups)
            feature_rows.extend(fr)
            group_rows.extend(gr)
        cohort_rows.append(stats(s, np.ones(len(s), dtype=bool), role, "ALL", thresholds))
        fr, gr = feature_summary(err, names, role, groups)
        feature_rows.extend(fr)
        group_rows.extend(gr)
        for ip in sorted(np.unique(host_arr)):
            _, gr = feature_summary(err[host_arr == ip], names, f"{role}/{ip}", groups)
            group_rows.extend(gr)

    gap_feature_rows, gap_group_rows = [], []
    comparisons = [
        ("validation_149_vs_validation_59",
         err_val[np.array([x.startswith("59.166.") for x in meta_val.src_ip])],
         err_val[np.array([x.startswith("149.171.") for x in meta_val.src_ip])]),
        ("test_benign_149_vs_test_benign_59",
         err_r1[(y1 == 0) & meta1.src_ip.str.startswith("59.166.").to_numpy()],
         err_r1[(y1 == 0) & meta1.src_ip.str.startswith("149.171.").to_numpy()]),
        ("round_1_attack_vs_test_benign_59",
         err_r1[(y1 == 0) & meta1.src_ip.str.startswith("59.166.").to_numpy()], err_r1[y1 == 1]),
        ("round_2_attack_vs_test_benign_59",
         err_r2[(y2 == 0) & meta2.src_ip.str.startswith("59.166.").to_numpy()], err_r2[y2 == 1]),
        ("round_1_attack_vs_test_benign_149",
         err_r1[(y1 == 0) & meta1.src_ip.str.startswith("149.171.").to_numpy()], err_r1[y1 == 1]),
    ]
    for comp, baseline, target in comparisons:
        if not len(baseline) or not len(target):
            raise ValueError(f"No sequences in comparison {comp}")
        f, g = compare_errors(baseline, target, names, groups, comp)
        gap_feature_rows.extend(f)
        gap_group_rows.extend(g)

    outdir.mkdir(parents=True, exist_ok=True)
    val_details = meta_val[["sequence_id", "src_ip", "start_bucket_ms", "end_bucket_ms"]].copy()
    val_details["reconstruction_mse"] = scores_val
    for name, threshold in thresholds.items():
        val_details[f"flag_{name}"] = scores_val >= threshold
    val_details.to_csv(outdir / "validation_scores.csv", index=False)
    pd.DataFrame(host_rows).to_csv(outdir / "host_score_statistics.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(outdir / "cohort_score_statistics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(outdir / "feature_error_by_cohort.csv", index=False)
    pd.DataFrame(group_rows).to_csv(outdir / "group_error_by_host_and_cohort.csv", index=False)
    pd.DataFrame(gap_feature_rows).to_csv(outdir / "feature_error_gaps.csv", index=False)
    pd.DataFrame(gap_group_rows).to_csv(outdir / "group_error_gaps.csv", index=False)
    save_json(outdir / "diagnostic_config.json", {
        "frozen_run": str(args.run_dir), "data_dir": str(args.data_dir),
        "gpu_device": torch.cuda.get_device_name(device),
        "n_features": len(names), "feature_group_members": {
            group: [names[i] for i in idx] for group, idx in groups.items()},
        "n_true_train": len(err_train), "n_validation": len(err_val),
        "n_round_1_test": len(err_r1), "n_round_2_test": len(err_r2),
        "saved_thresholds": thresholds,
        "threshold_reconstruction_max_abs_error": calibration_diffs,
        "earlier_saved_score_max_abs_error": {"round_1": maxdiff_r1, "round_2": maxdiff_r2},
        "cross_round_benign_input_identical": True,
        "cross_round_benign_window_score_max_abs_diff": benign_score_max_abs_delta,
        "cross_round_benign_feature_error_max_abs_diff": benign_feature_max_abs_delta,
        "cross_round_score_check_tolerance": {"atol": 1e-4, "rtol": 1e-4},
        "method": "Frozen CNN AE; squared reconstruction errors averaged over 10 timesteps "
                  "per feature; group contributions are summed then divided by 88 features.",
        "limitations": [
            "Feature/group error gaps are descriptive, not causal feature importance.",
            "Correlated feature proxies may preserve the same host/network-role signal.",
            "Overlapping stride-1 sequences are dependent observations.",
            "No training, recalibration, model selection, or test threshold tuning performed.",
        ],
    })

    df_hosts = pd.DataFrame(host_rows)
    print("\n=== VALIDATION HOSTS (frozen checkpoint) ===")
    cols = ["src_ip", "n_sequences", "mean", "p50", "p95", "p99", "flag_rate_5pct_fpr"]
    print(df_hosts.loc[df_hosts.role == "benign_validation", cols].to_string(index=False,
          float_format=lambda x: f"{x:.5f}"))
    print("\n=== COMPARISON: POSITIVE ERROR-GAP CONTRIBUTORS ===")
    gdf = pd.DataFrame(gap_group_rows)
    for comparison in ("validation_149_vs_validation_59", "test_benign_149_vs_test_benign_59",
                       "round_1_attack_vs_test_benign_59", "round_2_attack_vs_test_benign_59"):
        print(f"\n{comparison}:")
        selected = gdf[gdf.comparison == comparison].sort_values("signed_contribution_to_total_mse_gap", ascending=False)
        print(selected[["group", "signed_contribution_to_total_mse_gap", "share_of_positive_feature_gap"]]
              .head(6).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\n[done] All diagnostic outputs saved to {outdir.resolve()}")
    print("[note] Descriptive decomposition only; no causal attribution or test-based tuning.")


if __name__ == "__main__":
    main()
