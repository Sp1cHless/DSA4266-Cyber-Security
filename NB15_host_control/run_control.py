"""Paired team CNN-AE reproduction: time split versus source-host holdout.

Calls the pinned original feature, window, model, train and evaluation functions.
IndexedWindows only materializes the batches requested by the original trainer.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAPSHOT = HERE / "team_snapshot"
sys.path.insert(0, str(SNAPSHOT))
import preprocess as pre
import window as win
import ae_ablation as ae
from evaluate import evaluate

TEAM_COMMIT = "91f5ecb1d093435e0211277afa7ae09a5157428c"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


class IndexedWindows:
    """Numpy-like lazy batch view; no changes to original training mathematics."""
    def __init__(self, flows, flow_indices, window_ids):
        self.flows = flows
        self.flow_indices = flow_indices
        self.window_ids = np.asarray(window_ids)

    def __len__(self):
        return len(self.window_ids)

    def __getitem__(self, selection):
        return self.flows[self.flow_indices[self.window_ids[selection]]]


def family(host):
    return ".".join(str(host).split(".")[:3])


def host_partition(frame, seed):
    """Predeclared allocation, stratified by label and /24; never by scores.

    Stratification keeps both large benign families represented in all splits.
    With four attack sources, allocate 2/1/1; AE never trains on attack windows.
    Fractions are host fractions, not necessarily exact window fractions.
    """
    hosts = frame.groupby("IPV4_SRC_ADDR")["Label"].agg(["min", "max", "size"])
    if not (hosts["min"] == hosts["max"]).all():
        raise ValueError("This label-stratified protocol requires homogeneous source hosts")
    strata = {}
    for host, row in hosts.iterrows():
        key = (int(row["max"]), family(host))
        strata.setdefault(key, []).append(str(host))
    rng = random.Random(seed)
    mapping = {}
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        n = len(members)
        if n >= 3:
            n_val = max(1, round(n * 0.2))
            n_test = max(1, round(n * 0.2))
            n_train = n - n_val - n_test
        else:
            n_train, n_val = n, 0
        for i, host in enumerate(members):
            mapping[host] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return mapping


def geometry(frame, stride):
    # Float32 row IDs are exact for NB15 (<2**24 rows). Use the actual team
    # window builder and span filter, without allocating a multi-GB tensor.
    if len(frame) >= 2**24:
        raise ValueError("Row-index adapter is intended for this NB15 dataset only")
    small = frame[pre.META_COLS].copy()
    small["__row_index"] = np.arange(len(frame), dtype=np.float32)
    index_seq, masks, meta = win.build_windows(
        small, ["__row_index"], ["IPV4_SRC_ADDR"], 20, stride, False)
    index_seq, masks, meta = win.drop_spanning_windows(index_seq, masks, meta, 3_600_000)
    assert masks.all()
    meta["window_id"] = np.arange(len(meta))
    return index_seq[:, :, 0].astype(np.int32), masks, meta


def evaluate_split(model, X, masks, meta, ids, calibration, device, out, name):
    scores = ae.window_scores(model, IndexedWindows(X.flows, X.flow_indices, ids), masks[ids], device)
    sub = meta.iloc[ids].copy()
    metrics = evaluate(scores, sub.label.to_numpy(), sub.attack.to_numpy(), calibration)
    threshold = metrics["thresholds"][0.01]
    predictions = scores >= threshold
    metrics["confusion_matrix_tn_fp_fn_tp"] = confusion_matrix(sub.label, predictions, labels=[0, 1]).ravel().tolist()
    metrics["precision_at_val_1pct_fpr"] = float(precision_score(sub.label, predictions, zero_division=0))
    metrics["recall_at_val_1pct_fpr"] = float(recall_score(sub.label, predictions, zero_division=0))
    sub["score"] = scores
    sub["pred_at_val_1pct_fpr"] = predictions.astype(int)
    sub.to_csv(out / f"{name}_scores.csv", index=False)
    host_rows = []
    for host, group in sub.groupby("host"):
        benign = group.label == 0
        host_rows.append({"host": host, "n": len(group), "malicious": int(group.label.sum()),
                          "score_median": float(group.score.median()),
                          "flagged_rate": float(group.pred_at_val_1pct_fpr.mean())})
    pd.DataFrame(host_rows).to_csv(out / f"{name}_hosts.csv", index=False)
    # Compare EACH benign family against the identical complete attack set.
    # This prevents dominant 59.* normals from hiding a 149.* failure.
    subfamilies = sub.host.map(family)
    metrics["benign_family_vs_all_attacks"] = {}
    for fam in sorted(subfamilies[sub.label == 0].unique()):
        keep = (sub.label == 1) | ((sub.label == 0) & (subfamilies == fam))
        sm = sub[keep]
        if sm.label.nunique() == 2:
            metrics["benign_family_vs_all_attacks"][fam] = evaluate(
                sm.score.to_numpy(), sm.label.to_numpy(), sm.attack.to_numpy(), calibration)
    save_json(out / f"{name}_metrics.json", metrics)
    return metrics


def run_arm(name, frame, features, binaries, indices, masks, metadata, mapping, args):
    out = args.out / name
    out.mkdir(parents=True, exist_ok=False)
    meta = metadata.copy()
    if name == "time":
        meta = win.time_split(meta)
        fit_mask, cutoff = pre.build_fit_mask(frame)
    else:
        meta["split"] = meta.host.map(mapping)
        fit_mask = (frame.Label == 0) & (frame.IPV4_SRC_ADDR.map(mapping) == "train")
        cutoff = None
        groups = {s: set(meta.loc[meta.split == s, "host"]) for s in ("train", "val", "test")}
        assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
        assert not set(frame.loc[fit_mask, "IPV4_SRC_ADDR"]) & (groups["val"] | groups["test"])
    ids = {s: np.flatnonzero(meta.split.to_numpy() == s) for s in ("train", "val", "test")}
    benign = meta.label.to_numpy() == 0
    tr = ids["train"][benign[ids["train"]]]
    vb = ids["val"][benign[ids["val"]]]
    if not len(tr) or not len(vb) or any(meta.iloc[ids[s]].label.nunique() != 2 for s in ("val", "test")):
        raise ValueError("Train/validation/test lack the required classes")
    print(f"\n[{name}] training={len(tr):,}; benign validation={len(vb):,}", flush=True)
    print(meta[meta.split != ""].groupby("split").agg(windows=("label", "size"), attacks=("label", "sum"), hosts=("host", "nunique")).to_string(), flush=True)
    # Only the fit population changes. Feature formulas and full-data unlabeled
    # vocabulary/dead-bit selection are intentionally the ORIGINAL team policy.
    Xdf, scaler, logged = pre.scale(frame, features, fit_mask, binary_cols=binaries)
    X = np.ascontiguousarray(Xdf.to_numpy(dtype=np.float32))
    del Xdf
    if not np.isfinite(X).all():
        raise ValueError("Non-finite prepared features")
    joblib.dump(scaler, out / "scaler.joblib")
    meta.to_csv(out / "window_metadata.csv", index=False)
    flow_hosts = {s: sorted(h for h, split in mapping.items() if split == s) for s in ("train", "val", "test")}
    used_hosts = {s: sorted(meta.iloc[ids[s]].host.unique().tolist()) for s in ids}
    save_json(out / "split_audit.json", {
        "window_hosts": used_hosts,
        "raw_host_mapping": flow_hosts if name == "host" else None,
        "pairwise_source_host_overlap": {f"{a}_{b}": sorted(set(used_hosts[a]) & set(used_hosts[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "scaler_fit_hosts": sorted(frame.loc[fit_mask, "IPV4_SRC_ADDR"].unique().tolist()),
        "scaler_fit_rows": int(fit_mask.sum()), "time_scaler_cutoff_ms": cutoff,
        "window_attack_counts": pd.crosstab(meta.attack, meta.split).to_dict(),
        "test_absent_attack_types": sorted(set(meta.loc[meta.label == 1, "attack"]) - set(meta.loc[(meta.split == "test") & (meta.label == 1), "attack"]))})
    # Exact original architecture, parameter matching and trainer, no AMP,
    # class weights, window sampling, new clipping or loss changes.
    width, n_params = ae.match_width("cnn", len(features), 20, 32, 250_000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ae.CNNAutoencoder(len(features), 20, width, 32)
    model, val_loss = ae.train(model, IndexedWindows(X, indices, tr), masks[tr],
                              IndexedWindows(X, indices, vb), masks[vb], args.device,
                              epochs=args.epochs, batch=256, lr=1e-3, patience=5)
    torch.save(model.state_dict(), out / f"cnn_seed{args.seed}.pt")
    all_view = IndexedWindows(X, indices, np.arange(len(meta)))
    calibration = ae.window_scores(model, IndexedWindows(X, indices, vb), masks[vb], args.device)
    np.save(out / "benign_validation_scores.npy", calibration)
    results = {s: evaluate_split(model, all_view, masks, meta, ids[s], calibration, args.device, out, s) for s in ("val", "test")}
    save_json(out / "model_manifest.json", {"width": width, "parameters": n_params,
              "features": features, "seed": args.seed, "best_benign_validation_loss": val_loss})
    for split, metrics in results.items():
        print(f"[{name}/{split}] ROC={metrics['roc_auc']:.6f} AP={metrics['pr_auc']:.6f} "
              f"recall={metrics['dr@fpr0.01']:.4%} actual FPR={metrics['actual_fpr@fpr0.01']:.4%}", flush=True)
    del model, X, all_view
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def self_test():
    # Adapter equivalence to original windows and scores, and host isolation.
    frame = pd.DataFrame({"IPV4_SRC_ADDR": ["59.166.0.1"] * 45 + ["59.166.0.2"] * 45,
                          "IPV4_DST_ADDR": ["server"] * 90,
                          "FLOW_START_MILLISECONDS": np.arange(90) * 10,
                          "FLOW_END_MILLISECONDS": np.arange(90) * 10 + 1,
                          "Label": [0] * 90, "Attack": ["Benign"] * 90,
                          "f": np.arange(90, dtype=np.float32)})
    expected, masks, meta = win.build_windows(frame, ["f"], ["IPV4_SRC_ADDR"], 20, 10, False)
    idx, _, _ = geometry(frame, 10)
    actual = IndexedWindows(frame[["f"]].to_numpy(), idx, np.arange(len(idx)))
    np.testing.assert_array_equal(expected, actual[:])
    torch.manual_seed(0)
    model = ae.CNNAutoencoder(1, 20, 4, 3)
    np.testing.assert_allclose(ae.window_scores(model, expected, masks, "cpu"),
                               ae.window_scores(model, actual, masks, "cpu"), rtol=0, atol=0)
    toy = pd.DataFrame({"IPV4_SRC_ADDR": [f"59.166.0.{i}" for i in range(10)] + [f"175.45.176.{i}" for i in range(4)], "Label": [0] * 10 + [1] * 4})
    mapping = host_partition(toy, 0)
    assert len(mapping) == 14 and set(mapping.values()) == {"train", "val", "test"}
    print("SELF-TEST PASSED: exact window/score equivalence and deterministic host allocation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "datasets/NF-UNSW-NB15-v3.csv")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts/seed0_w20s10")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out.exists():
        raise FileExistsError(f"Choose a fresh --out; will not overwrite {args.out}")
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    print(f"Device={args.device}; team commit={TEAM_COMMIT}", flush=True)
    frame = pre.sort_by_start(pd.read_csv(args.csv, low_memory=False))
    mapping = host_partition(frame, args.seed)
    indices, masks, metadata = geometry(frame, args.stride)
    frame = pre.add_relative_time(frame)
    frame, dropped = pre.drop_columns(frame, drop_shortcuts=True)
    frame, flags = pre.decompose_tcp_flags(frame)
    frame, ports, top_ports = pre.bucket_dst_port(frame)
    frame, categories, vocab = pre.onehot_categoricals(frame)
    features = [c for c in frame if c not in pre.META_COLS]
    frame, missing = pre.add_missing_indicators(frame, features)
    features += missing
    binaries = flags + ports + categories + missing
    manifest = {"team_commit": TEAM_COMMIT, "source_csv": str(args.csv), "raw_rows": len(frame),
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SNAPSHOT.glob("*.py")},
                "window_size": 20, "stride": args.stride, "max_span_ms": 3_600_000,
                "pad": False, "seed": args.seed, "epochs_max": args.epochs,
                "budget": 250000, "latent": 32, "batch": 256, "lr": .001, "patience": 5,
                "dropped": dropped, "features": features, "binary_features": binaries,
                "dst_port_top": top_ports, "onehot_values": vocab,
                "host_definition": "source IPv4 only; destination endpoints may overlap",
                "host_partition": "label + source /24 stratification; approximately 60/20/20 hosts; strata <3 assigned train",
                "policy_unchanged_for_faithfulness": "Vocabulary, dead-bit and missing-indicator selection use the complete unlabeled dataset, as in team code. Thus this is model/scaler source-host holdout, not fully inductive preprocessing. Ports retained. Global relative-time construction unchanged.",
                "important": "New independent test metrics are not the team's old validation-selection metrics. Single seed, no tuning. Source label homogeneity and different host/time/attack distributions limit causal attribution."}
    save_json(args.out / "experiment_manifest.json", manifest)
    results = {}
    for name in ("time", "host"):
        results[name] = run_arm(name, frame, features, binaries, indices, masks, metadata, mapping, args)
        save_json(args.out / "results.json", results)
    rows = [{"arm": arm, "split": split, "n": m["n_windows"], "attacks": m["n_malicious"],
             "roc_auc": m["roc_auc"], "average_precision": m["pr_auc"],
             "recall_at_val_1pct_fpr": m["dr@fpr0.01"], "actual_fpr": m["actual_fpr@fpr0.01"],
             "precision": m["precision_at_val_1pct_fpr"]} for arm, splits in results.items() for split, m in splits.items()]
    pd.DataFrame(rows).to_csv(args.out / "comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"DONE: {args.out}", flush=True)


if __name__ == "__main__":
    main()
