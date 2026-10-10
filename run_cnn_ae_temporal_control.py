#!/usr/bin/env python3
"""One-model, CUDA-only CNN Autoencoder control using official 10s temporal data.

Keeps the team's CNN architecture, Adam/MSE training, benign validation early
stopping and reconstruction-error scoring. Does NOT use flow-level preprocess.py
or window.py. No test labels/scores are used for training, selection or thresholds.

From repository root (Windows CMD / PowerShell):
  python run_cnn_ae_temporal_control.py --data-dir processed_binary_temporal

Required: round_1/{train.npz,train_metadata.csv,test.npz,test_metadata.csv},
          round_2/{test.npz,test_metadata.csv}, categorical_schema.json,
          split_manifest.json. round_2/train.npz is deliberately unused.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import platform
import random
import sys
from pathlib import Path

# Must be set before CUDA initialization when strict deterministic operations are used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPECTED_SEQ_LENGTH = 10
EXPECTED_FEATURES = 88
FPR_TARGETS = (0.001, 0.01, 0.05)
REQUIRED_METADATA = {
    "sequence_id", "src_ip", "segment_id", "start_bucket_ms",
    "end_bucket_ms", "label", "num_buckets", "split", "round",
}


# Exact CNN architecture of teammate's ae_ablation.py, with only comments added.
class CNNAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, width: int, latent: int):
        super().__init__()
        self.seq_len, self.width = seq_len, width
        self.t2 = (seq_len + 1) // 2
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.to_latent = nn.Linear(width * self.t2, latent)
        self.from_latent = nn.Linear(latent, width * self.t2)
        self.up = nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)
        self.out = nn.Conv1d(width, n_features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x.transpose(1, 2))
        z = self.to_latent(h.flatten(1))
        d = self.from_latent(z).view(-1, self.width, self.t2)
        d = torch.relu(self.up(d))[:, :, : self.seq_len]
        return self.out(d).transpose(1, 2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_width(n_features: int, seq_len: int, latent: int, budget: int):
    # Same binary search and parameter-count criterion as ae_ablation.py.
    lo, hi = 4, 1024
    best, best_w = None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(CNNAutoencoder(n_features, seq_len, mid, latent))
        if best is None or abs(n - budget) < abs(best - budget):
            best, best_w = n, mid
        if n < budget:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_w, best


def json_dump(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def require_cuda(gpu_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for ALL model training and inference. "
            "torch.cuda.is_available() is False; refusing CPU fallback. "
            "Check the installed PyTorch CUDA build/driver."
        )
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(f"--gpu-index={gpu_index} invalid; available GPUs: {torch.cuda.device_count()}")
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    print(f"[CUDA] GPU={gpu_index}: {torch.cuda.get_device_name(device)}")
    print(f"[CUDA] torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
    return device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split(data_dir: Path, round_name: str, split: str,
               expected_names: list[str], manifest: dict):
    folder = data_dir / round_name
    npz_path = folder / f"{split}.npz"
    csv_path = folder / f"{split}_metadata.csv"
    for p in (npz_path, csv_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required input: {p}")

    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "y", "feature_names"}:
            raise ValueError(f"{npz_path}: expected keys X, y, feature_names; found {archive.files}")
        X = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.int8)
        names = [str(x) for x in archive["feature_names"].tolist()]

    meta = pd.read_csv(csv_path, dtype={"src_ip": str, "sequence_id": str})
    if not REQUIRED_METADATA.issubset(meta.columns):
        raise ValueError(f"{csv_path}: missing {sorted(REQUIRED_METADATA - set(meta.columns))}")
    if X.ndim != 3 or X.shape[1:] != (EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES):
        raise ValueError(f"{npz_path}: need (N,10,88), got {X.shape}")
    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError(f"{npz_path}: NPZ and metadata length mismatch")
    if names != expected_names:
        raise ValueError(f"{npz_path}: feature_names/order differs from schema")
    if not np.isfinite(X).all():
        raise ValueError(f"{npz_path}: non-finite features")
    if not np.isin(y, [0, 1]).all() or not np.array_equal(y, meta.label.to_numpy()):
        raise ValueError(f"{npz_path}: y and metadata binary labels mismatch")
    if not meta["split"].eq(split).all() or not meta["round"].eq(round_name).all():
        raise ValueError(f"{csv_path}: split/round mismatch (possibly uploaded under same basename)")
    if meta["sequence_id"].duplicated().any() or meta.isnull().any().any():
        raise ValueError(f"{csv_path}: duplicate sequence_id or missing metadata")
    if not meta["num_buckets"].eq(EXPECTED_SEQ_LENGTH).all():
        raise ValueError(f"{csv_path}: expected all sequences of 10 buckets")
    delta = meta["end_bucket_ms"] - meta["start_bucket_ms"]
    if not delta.eq((EXPECTED_SEQ_LENGTH - 1) * manifest["bucket_size_ms"]).all():
        raise ValueError(f"{csv_path}: non-consecutive timestamps (or wrong segment handling)")

    count_man = manifest[round_name][f"{split}_sequence_counts"]
    if (len(y), int((y == 0).sum()), int((y == 1).sum())) != (
        count_man["total"], count_man["benign"], count_man["malicious"]
    ):
        raise ValueError(f"{npz_path}: counts differ from split_manifest.json")
    observed = set(meta["src_ip"].unique())
    if not observed.issubset(set(manifest[round_name][f"{split}_ips"])):
        raise ValueError(f"{csv_path}: source IP not included in manifest {round_name} {split}")
    print(f"[data] {round_name}/{split}: {len(y):,} sequences; "
          f"{int((y == 0).sum()):,} benign, {int((y == 1).sum()):,} malicious")
    return X, y, meta


def validate_host_integrity(train_meta: pd.DataFrame, test_meta: dict[str, pd.DataFrame],
                            manifest: dict) -> None:
    benign_train_hosts = set(manifest["benign_train_ips"])
    benign_test_hosts = set(manifest["benign_test_ips"])
    if benign_train_hosts & benign_test_hosts:
        raise ValueError("Benign train/test host leakage in split manifest")
    if set(train_meta.loc[train_meta.label.eq(0), "src_ip"]) - benign_train_hosts:
        raise ValueError("Benign training rows include hosts outside manifest benign train hosts")
    for round_name, mt in test_meta.items():
        if set(mt.src_ip) & set(train_meta.loc[train_meta.label.eq(0), "src_ip"]):
            raise ValueError(f"Train/test benign host overlap in {round_name}")
        attack_ip = manifest[round_name]["test_malicious_ip"]
        if set(mt.loc[mt.label.eq(1), "src_ip"]) != {attack_ip}:
            raise ValueError(f"Unexpected malicious test hosts in {round_name}")
        if set(mt.loc[mt.label.eq(0), "src_ip"]) != benign_test_hosts:
            raise ValueError(f"Unexpected benign test hosts in {round_name}")
    if manifest["round_1"]["test_malicious_ip"] == manifest["round_2"]["test_malicious_ip"]:
        raise ValueError("Round 1 and Round 2 held-out malicious hosts must differ")


def choose_validation_hosts(train_meta: pd.DataFrame, fraction: float, seed: int):
    # Work only with actual benign sequences from hosts generating >=1 sequence.
    benign_meta = train_meta.loc[train_meta.label.eq(0)]
    counts = benign_meta.groupby("src_ip").size().astype(int).to_dict()
    hosts = sorted(counts)
    if len(hosts) < 3:
        raise ValueError("At least three productive benign train hosts needed: >=2 val, >=1 train")
    if len(hosts) > 20:
        raise ValueError("More than 20 productive hosts; exact selection search needs adaptation")
    target = fraction * sum(counts.values())
    rng = np.random.default_rng(seed)
    rank = {host: i for i, host in enumerate(rng.permutation(hosts).tolist())}
    best_key = None
    best_hosts = None
    for k in range(2, len(hosts)):
        for subset in itertools.combinations(hosts, k):
            count = sum(counts[h] for h in subset)
            key = (abs(count - target), k, tuple(sorted(rank[h] for h in subset)))
            if best_key is None or key < best_key:
                best_key, best_hosts = key, subset
    chosen = sorted(best_hosts)
    train_hosts = sorted(set(hosts) - set(chosen))
    n_val = sum(counts[h] for h in chosen)
    n_total = sum(counts.values())
    if n_val <= 0 or n_val == n_total or len(chosen) < 2:
        raise AssertionError("Invalid host-disjoint validation selection")
    print(f"[val] hosts={len(chosen)}; sequences={n_val:,}/{n_total:,} "
          f"({n_val/n_total:.2%}); target={fraction:.2%}")
    print(f"[val] validation hosts: {chosen}")
    return chosen, train_hosts, counts


def feature_groups(names: list[str]):
    log_cols = [i for i, name in enumerate(names) if (
        name == "flow_count" or name.endswith("_sum") or
        name.startswith("duration_") or name.endswith("_nunique")
    )]
    bounded_cols = [i for i, name in enumerate(names) if
                    name.endswith("_ratio") or name.endswith("_entropy")]
    if sorted(log_cols + bounded_cols) != list(range(len(names))):
        unmapped = sorted(set(range(len(names))) - set(log_cols + bounded_cols))
        raise ValueError(f"Unhandled features in log/bounded transform: {[names[i] for i in unmapped]}")
    return log_cols, bounded_cols


class FeatureTransformer:
    """Log1p + z-score on count/volume/duration/nunique; ratios left alone."""
    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.log_cols, self.bounded_cols = feature_groups(feature_names)
        self.scaler = StandardScaler()

    def validate(self, X: np.ndarray):
        if X.shape[1:] != (EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES):
            raise ValueError(f"Unexpected X shape {X.shape}")
        a = X[:, :, self.log_cols]
        if (a < 0).any():
            raise ValueError("Negative count/sum/duration/nunique values make log1p inappropriate")
        b = X[:, :, self.bounded_cols]
        if (b < -1e-6).any() or (b > 1+1e-6).any():
            raise ValueError("Some ratio/entropy features are outside [0, 1]")

    def fit(self, Xtrain: np.ndarray):
        self.validate(Xtrain)
        flattened = np.log1p(Xtrain[:, :, self.log_cols].astype(np.float64))
        self.scaler.fit(flattened.reshape(-1, len(self.log_cols)))
        # The transform can amplify rare log features, but ONLY the log group is z-scored.
        # Strictly bounded ratios/entropies are unscaled and kept in [0, 1].
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.validate(X)
        out = X.copy()
        a = np.log1p(X[:, :, self.log_cols].astype(np.float64))
        out[:, :, self.log_cols] = self.scaler.transform(
            a.reshape(-1, len(self.log_cols))
        ).reshape(X.shape[0], X.shape[1], len(self.log_cols)).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Non-finite values after feature transform")
        return np.ascontiguousarray(out, dtype=np.float32)

    def config(self):
        return {
            "feature_names": self.feature_names,
            "log1p_and_standard_scale": [self.feature_names[i] for i in self.log_cols],
            "unchanged_bounded_0_1": [self.feature_names[i] for i in self.bounded_cols],
            "scaler_fit": "only benign sequences from host-disjoint TRUE training hosts",
            "standard_scaler": "scikit-learn StandardScaler (mean, std), fit on flattened 10 buckets",
            "no_zero_std_ratio_amplification": True,
        }


def masked_mse(x, xhat, mask):
    """Exact loss formula from teammate's ae_ablation.py; mask is all ones."""
    err = ((x - xhat) ** 2).mean(dim=2)
    return (err * mask).sum() / mask.sum().clamp(min=1)


def train(model, Xtr, Xva, device, epochs, batch, lr, patience):
    """Same training loop: Adam, shuffle each epoch, MSE, best val checkpoint."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, best_epoch = float("inf"), None, 0, None
    history = []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            x = torch.from_numpy(Xtr[idx]).to(device)
            m = torch.ones(x.shape[:2], dtype=torch.float32, device=device)
            loss = masked_mse(x, model(x), m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for i in range(0, len(Xva), 512):
                x = torch.from_numpy(Xva[i:i+512]).to(device)
                m = torch.ones(x.shape[:2], dtype=torch.float32, device=device)
                vl += masked_mse(x, model(x), m).item() * len(x)
                n += len(x)
            vl /= n
        t_loss = tot / len(perm)
        print(f"[train] epoch {ep:02d}: train_mse={t_loss:.6f} val_mse={vl:.6f}", flush=True)
        history.append({"epoch": ep, "train_mse": t_loss, "val_benign_mse": vl})
        if vl < best - 1e-6:
            best, bad, best_epoch = vl, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[train] early stopping at epoch {ep}; best epoch={best_epoch}")
                break
    if best_state is None:
        raise RuntimeError("No usable checkpoint: training did not improve finite validation loss")
    model.load_state_dict(best_state)
    return model, float(best), best_epoch, history


@torch.no_grad()
def scores_on_cuda(model: nn.Module, X: np.ndarray, device, batch=512) -> np.ndarray:
    if device.type != "cuda":
        raise RuntimeError("Refusing CPU model inference")
    model.eval()
    results = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(device)
        # For fully observed sequences, teammate's masked timestep-MSE reduces to mean MSE.
        err = ((x - model(x)) ** 2).mean(dim=(1, 2))
        results.append(err.cpu().numpy())
    return np.concatenate(results).astype(np.float64)


def score_stats(scores: np.ndarray, kind: str, round_name: str) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores):
        raise ValueError(f"No {kind} scores to summarize")
    return {
        "round": round_name, "class": kind, "n": len(scores),
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "min": float(np.min(scores)), "p25": float(np.quantile(scores, .25)),
        "median": float(np.median(scores)), "p75": float(np.quantile(scores, .75)),
        "p95": float(np.quantile(scores, .95)), "p99": float(np.quantile(scores, .99)),
        "max": float(np.max(scores)),
    }


def evaluate_round(round_name: str, scores: np.ndarray, y: np.ndarray,
                   meta: pd.DataFrame, thresholds: dict[str, float],
                   malicious_ip: str):
    if not np.isfinite(scores).all():
        raise ValueError(f"{round_name}: nonfinite reconstruction scores")
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError(f"{round_name}: both classes needed for ROC/PR AUC")
    benign, attack = scores[y == 0], scores[y == 1]
    q1, q3 = np.quantile(benign, [.25, .75])
    res = {
        "round": round_name, "malicious_host": malicious_ip,
        "n_sequences": int(len(y)), "n_benign": int((y==0).sum()),
        "n_attack": int((y==1).sum()),
        "base_rate": float(np.mean(y)),
        "pr_auc_average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "benign_median": float(np.median(benign)),
        "attack_median": float(np.median(attack)),
        "separation": float((np.median(attack)-np.median(benign))/max(q3-q1,1e-12)),
    }
    host_rows = []
    for host, frame in meta.groupby("src_ip", sort=True):
        inds = frame.index.to_numpy()
        s, cls = scores[inds], int(y[inds[0]])
        if not np.all(y[inds] == cls):
            raise ValueError(f"Mixed-label host in {round_name}: {host}")
        row = {"round": round_name, "src_ip": host,
               "class": "attack" if cls == 1 else "benign", "n": len(inds),
               "score_median": float(np.median(s)), "score_mean": float(np.mean(s))}
        for name, th in thresholds.items():
            row[f"flag_rate_at_{name}"] = float(np.mean(s >= th))
        host_rows.append(row)
    for name, th in thresholds.items():
        # Exact same >= operating rule as teammate's evaluate.py.
        res[f"dr_at_{name}"] = float(np.mean(attack >= th))
        res[f"actual_test_fpr_at_{name}"] = float(np.mean(benign >= th))
        res[f"threshold_at_{name}"] = float(th)
    stat_rows = [score_stats(benign, "benign", round_name),
                 score_stats(attack, "attack", round_name)]
    output_scores = meta[["sequence_id", "src_ip", "label", "start_bucket_ms", "end_bucket_ms"]].copy()
    output_scores["reconstruction_mse"] = scores
    for name, th in thresholds.items():
        output_scores[f"flag_at_{name}"] = scores >= th
    return res, host_rows, stat_rows, output_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("processed_binary_temporal"))
    ap.add_argument("--outdir", type=Path, default=Path("experiments/cnn_ae_temporal_control"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--budget", type=int, default=250_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()
    if not 0 < args.val_fraction < 1:
        ap.error("--val-fraction must be between zero and one")
    if min(args.batch_size, args.epochs, args.patience, args.latent, args.budget) < 1:
        ap.error("batch-size, epochs, patience, latent and budget must be positive")
    if args.lr <= 0:
        ap.error("--lr must be positive")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Output directory exists and is nonempty: {args.outdir}; choose a NEW --outdir")
    device = require_cuda(args.gpu_index)
    set_seed(args.seed)

    man = read_json(args.data_dir / "split_manifest.json")
    schema = read_json(args.data_dir / "categorical_schema.json")
    names = list(schema["feature_names"])
    if len(names) != EXPECTED_FEATURES or names != man["feature_names"]:
        raise ValueError("Expected same 88 feature names/order in schema and split manifest")
    if man["sequence_length"] != EXPECTED_SEQ_LENGTH or man["bucket_size_ms"] != 10_000:
        raise ValueError("Expected 10 consecutive 10-second buckets")
    if man.get("augmentation_enabled"):
        raise ValueError("Cannot use this runner with augmentation enabled")

    Xtr_all, ytr_all, mtr_all = load_split(args.data_dir, "round_1", "train", names, man)
    Xr1, yr1, mr1 = load_split(args.data_dir, "round_1", "test", names, man)
    Xr2, yr2, mr2 = load_split(args.data_dir, "round_2", "test", names, man)
    validate_host_integrity(mtr_all, {"round_1": mr1, "round_2": mr2}, man)

    # Both rounds share the same benign test hosts: verify the benign test rows
    # and X are really identical, though evaluated once each with SAME model.
    b1 = yr1 == 0
    b2 = yr2 == 0
    common_meta_cols = ["src_ip", "segment_id", "start_bucket_ms", "end_bucket_ms", "label"]
    if not mr1.loc[b1, common_meta_cols].reset_index(drop=True).equals(
        mr2.loc[b2, common_meta_cols].reset_index(drop=True)
    ) or not np.array_equal(Xr1[b1], Xr2[b2]):
        raise ValueError("Benign test sequences differ between rounds; cannot compare on fixed benign pool")

    val_hosts, true_train_hosts, counts = choose_validation_hosts(mtr_all, args.val_fraction, args.seed)
    idx_train = ((ytr_all == 0) & mtr_all["src_ip"].isin(true_train_hosts).to_numpy())
    idx_val = ((ytr_all == 0) & mtr_all["src_ip"].isin(val_hosts).to_numpy())
    if (idx_train & idx_val).any() or idx_train.sum() + idx_val.sum() != (ytr_all == 0).sum():
        raise AssertionError("Benign train/validation partition invalid")
    print(f"[train] true benign train={int(idx_train.sum()):,}, "
          f"benign val={int(idx_val.sum()):,}; malicious train used=0")
    Xtr = Xtr_all[idx_train]
    Xva = Xtr_all[idx_val]
    del Xtr_all, ytr_all

    trans = FeatureTransformer(names).fit(Xtr)
    Xtr = trans.transform(Xtr)
    Xva = trans.transform(Xva)
    Xr1 = trans.transform(Xr1)
    Xr2 = trans.transform(Xr2)

    width, n_params = match_width(EXPECTED_FEATURES, EXPECTED_SEQ_LENGTH, args.latent, args.budget)
    model = CNNAutoencoder(EXPECTED_FEATURES, EXPECTED_SEQ_LENGTH, width, args.latent)
    if n_params != count_params(model):
        raise AssertionError("Parameter count mismatch")
    print(f"[model] CNN AE width={width}, latent={args.latent}, params={n_params:,} "
          f"({(n_params/args.budget - 1):+.2%} vs budget)")
    # Assert real CUDA model forward on correct geometry before fitting.
    with torch.no_grad():
        model = model.to(device)
        tensor = torch.zeros((2, EXPECTED_SEQ_LENGTH, EXPECTED_FEATURES), device=device)
        if model(tensor).shape != tensor.shape:
            raise AssertionError("CNN forward shape mismatch")
        del tensor

    args.outdir.mkdir(parents=True, exist_ok=True)
    trained, val_loss, best_epoch, history = train(
        model, Xtr, Xva, device, args.epochs, args.batch_size, args.lr, args.patience
    )
    pd.DataFrame(history).to_csv(args.outdir / "training_history.csv", index=False)

    val_scores = scores_on_cuda(trained, Xva, device)
    # EXACT team evaluation threshold rule, derived ONLY from benign validation.
    thresholds = {f"{100*t:g}pct_fpr": float(np.quantile(val_scores, 1.0 - t))
                  for t in FPR_TARGETS}
    json_dump(args.outdir / "thresholds.json", {
        "source": "benign_validation_only", "seed": args.seed,
        "n_validation": len(val_scores),
        "thresholds": thresholds,
        "operator": "reconstruction_mse >= threshold",
        "calibration_method": "np.quantile(benign_validation_scores, 1 - target_fpr)",
        "validation_realized_flag_rates": {
            name: float(np.mean(val_scores >= th)) for name, th in thresholds.items()
        },
        "warning": "Host-disjoint validation windows overlap in time; effective sample size is smaller than the sequence count",
    })
    print(f"[thresholds] frozen from benign validation: {thresholds}")

    rows, host_rows, stat_rows = [], [], [score_stats(val_scores, "benign_validation", "validation")]
    for round_name, X, y, meta in [
        ("round_1", Xr1, yr1, mr1), ("round_2", Xr2, yr2, mr2),
    ]:
        scores = scores_on_cuda(trained, X, device)
        result, host, stats, output = evaluate_round(
            round_name, scores, y, meta, thresholds, man[round_name]["test_malicious_ip"]
        )
        json_dump(args.outdir / f"{round_name}_metrics.json", result)
        output.to_csv(args.outdir / f"{round_name}_scores.csv", index=False)
        rows.append(result)
        host_rows.extend(host)
        stat_rows.extend(stats)
        print(f"[{round_name}] PR-AUC(AP)={result['pr_auc_average_precision']:.5f} "
              f"ROC-AUC={result['roc_auc']:.5f}, base_rate={result['base_rate']:.2%}")
        for name in thresholds:
            print(f"    {name}: DR={result[f'dr_at_{name}']:.2%}, "
                  f"actual test FPR={result[f'actual_test_fpr_at_{name}']:.2%}")

    pd.DataFrame(rows).to_csv(args.outdir / "round_comparison.csv", index=False)
    pd.DataFrame(host_rows).to_csv(args.outdir / "host_results.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(args.outdir / "reconstruction_error_statistics.csv", index=False)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in trained.state_dict().items()},
        "n_features": EXPECTED_FEATURES, "seq_len": EXPECTED_SEQ_LENGTH,
        "width": width, "latent": args.latent, "params": n_params,
        "seed": args.seed, "best_epoch": best_epoch, "best_benign_val_loss": val_loss,
        "feature_names": names,
    }, args.outdir / f"cnn_ae_seed{args.seed}_checkpoint.pt")
    joblib.dump(trans.scaler, args.outdir / "scaler_log_features.joblib")
    json_dump(args.outdir / "feature_transform.json", trans.config())
    json_dump(args.outdir / "validation_hosts.json", {
        "validation_hosts": val_hosts, "true_train_hosts": true_train_hosts,
        "benign_sequence_counts_by_productive_host": counts,
        "val_sequence_count": int(idx_val.sum()),
        "train_sequence_count": int(idx_train.sum()),
        "val_fraction_actual": float(idx_val.sum()/(idx_train.sum()+idx_val.sum())),
        "seed": args.seed,
    })
    json_dump(args.outdir / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "pytorch_cuda": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_index": args.gpu_index,
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": True,
    })
    json_dump(args.outdir / "run_config.json", {
        "model": "CNNAutoencoder (exact architecture from ae_ablation.py)",
        "train_loss": "masked_mse with full-one mask (=mean squared error)",
        "optimizer": "Adam", "early_stop": "benign val reconstruction loss only",
        "best_epoch": best_epoch, "best_val_loss": val_loss, "width": width,
        "n_params": n_params, "data_dir": str(args.data_dir),
        **{k: v for k, v in vars(args).items() if k not in {"data_dir", "outdir"}},
    })
    print(f"[done] All outputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
