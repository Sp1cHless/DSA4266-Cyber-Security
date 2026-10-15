"""Train ONE CUDA CNN autoencoder and evaluate the frozen CIC test split.

Only benign train windows fit the scaler/model. Benign validation controls early
stopping. Separate benign calibration freezes thresholds before test inference.
"""
from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from common import (HERE, SEQ_LEN, SPLITS, WindowData, dump_json, empty_output,
                    feature_groups, occupancy, read_json, write_csv)


class MaskedTransformer:
    def __init__(self, names):
        self.log_cols, self.bounded_cols = feature_groups(names)
        self.scaler = StandardScaler()

    def fit(self, data):
        ids = np.unique(data.windows["train"]["rows"])
        ids = ids[ids >= 0]
        if not len(ids):
            raise ValueError("No observed true-training buckets")
        for start in range(0, len(ids), 32_768):
            x = np.asarray(data.buckets[ids[start:start + 32_768]], dtype=np.float64)
            self.scaler.partial_fit(np.log1p(x[:, self.log_cols]))
        self.n_fit_buckets = len(ids)
        return self

    def transform(self, x, mask):
        # Standardize ONLY observations; padding is zero in transformed space.
        observed = mask.astype(bool)
        values = np.array(x[observed], dtype=np.float32, copy=True)
        values[:, self.log_cols] = self.scaler.transform(
            np.log1p(values[:, self.log_cols].astype(np.float64))).astype(np.float32)
        result = np.zeros_like(x)
        result[observed] = values
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transformed feature")
        return result


class MaskedCNNAutoencoder(nn.Module):
    """Same CNN layout as the UNSW control, with one encoder-only mask channel."""

    def __init__(self, n_features=88, seq_len=10, width=64, latent=32):
        super().__init__()
        self.width, self.seq_len = width, seq_len
        self.t2 = (seq_len + 1) // 2
        self.enc = nn.Sequential(
            nn.Conv1d(n_features + 1, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.ReLU())
        self.to_latent = nn.Linear(width * self.t2, latent)
        self.from_latent = nn.Linear(latent, width * self.t2)
        self.up = nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)
        self.out = nn.Conv1d(width, n_features, 3, padding=1)

    def forward(self, x, mask):
        inputs = torch.cat((x * mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=2)
        h = self.enc(inputs.transpose(1, 2))
        z = self.to_latent(h.flatten(1))
        d = self.from_latent(z).view(-1, self.width, self.t2)
        d = torch.relu(self.up(d))[:, :, :self.seq_len]
        return self.out(d).transpose(1, 2)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def match_width(features, latent, budget):
    lo, hi, best = 4, 1024, None
    while lo <= hi:
        mid = (lo + hi) // 2
        count = parameter_count(MaskedCNNAutoencoder(features, SEQ_LEN, mid, latent))
        if best is None or abs(count - budget) < abs(best[1] - budget):
            best = (mid, count)
        if count < budget:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def per_window_mse(x, reconstructed, mask):
    # Each sequence contributes equally, regardless of the number of active bins.
    per_bin = (x - reconstructed).square().mean(dim=2)
    return (per_bin * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def tensor_batch(data, transformer, split, indices, device):
    x, mask = data.batch(split, indices)
    x = transformer.transform(x, mask)
    return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)


@torch.no_grad()
def score(data, transformer, model, split, batch_size, device):
    model.eval()
    result = np.empty(len(data.windows[split]), dtype=np.float64)
    for start in range(0, len(result), batch_size):
        end = min(start + batch_size, len(result))
        x, mask = tensor_batch(data, transformer, split, slice(start, end), device)
        result[start:end] = per_window_mse(x, model(x, mask), mask).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {split} reconstruction scores")
    return result


def fit(data, transformer, model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best_loss, best_epoch, best_state, bad, history = float("inf"), -1, None, 0, []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        order = rng.permutation(len(data.windows["train"]))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            x, mask = tensor_batch(data, transformer, "train", ids, device)
            loss = per_window_mse(x, model(x, mask), mask).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        val_loss = float(score(data, transformer, model, "validation", args.batch_size, device).mean())
        row = {"epoch": epoch, "train_mse": total / len(order), "validation_mse": val_loss,
               "seconds": time.monotonic() - started}
        history.append(row)
        print(f"[epoch {epoch:02d}] train={row['train_mse']:.6f}, val={val_loss:.6f}, "
              f"seconds={row['seconds']:.1f}", flush=True)
        write_csv(args.outdir / "training_history.csv", history)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    return best_epoch, best_loss


def metrics(y, scores, thresholds):
    y = np.asarray(y, dtype=np.int8)
    n_benign, n_attack = int((y == 0).sum()), int((y == 1).sum())
    result = {"n": len(y), "benign": n_benign, "attack": n_attack,
              "attack_prevalence": float(y.mean()) if len(y) else None,
              "roc_auc": None, "pr_auc_average_precision": None}
    if n_benign and n_attack:
        result["roc_auc"] = float(roc_auc_score(y, scores))
        result["pr_auc_average_precision"] = float(average_precision_score(y, scores))
    for name, threshold in thresholds.items():
        flag = scores >= threshold
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        result[f"dr_at_{name}"] = tp / n_attack if n_attack else None
        result[f"actual_fpr_at_{name}"] = fp / n_benign if n_benign else None
        result[f"precision_at_{name}"] = tp / (tp + fp) if tp + fp else 0.0
        result[f"tp_at_{name}"], result[f"fp_at_{name}"] = tp, fp
        result[f"fn_at_{name}"], result[f"tn_at_{name}"] = n_attack - tp, n_benign - fp
    for label, name in ((0, "benign"), (1, "attack")):
        values = scores[y == label]
        result[f"{name}_median_mse"] = float(np.median(values)) if len(values) else None
    return result


def evaluate(data, test_scores, cal_scores, thresholds, outdir):
    windows = data.windows["test"]
    y = windows["label"]
    overall = metrics(y, test_scores, thresholds)
    dump_json(outdir / "test_metrics.json", overall)
    # Attack categories are overlapping subsets, evaluated against ALL benign test windows.
    attack_rows = []
    for i, name in enumerate(data.manifest["attack_types"]):
        positive = (windows["attack_bits"] & (1 << i)) > 0
        subset = positive | (y == 0)
        row = {"attack_type": name, "status": "evaluated" if positive.any() else "not_present_in_test",
               **metrics(positive[subset], test_scores[subset], thresholds)}
        attack_rows.append(row)
    write_csv(outdir / "metrics_by_attack.csv", attack_rows)
    day_ids = windows["start_ms"] // 86_400_000
    day_rows = [{"day_utc": str(np.datetime64(int(day), "D")),
                 **metrics(y[day_ids == day], test_scores[day_ids == day], thresholds)}
                for day in np.unique(day_ids)]
    write_csv(outdir / "metrics_by_day.csv", day_rows)
    group_rows = []
    for split, scores in (("calibration", cal_scores), ("test", test_scores)):
        w = data.windows[split]
        groups = occupancy(w["active"])
        for name in ("1", "2-4", "5-10"):
            ids = groups == name
            group_rows.append({"split": split, "active_buckets": name,
                               **metrics(w["label"][ids], scores[ids], thresholds)})
    write_csv(outdir / "metrics_by_occupancy.csv", group_rows)
    hosts = read_json(data.folder / "hosts.json")
    for split, scores in (("calibration", cal_scores), ("test", test_scores)):
        w = data.windows[split]
        def rows():
            for i, value in enumerate(scores):
                yield {"window_id": i, "src_ip": hosts[int(w[i]["host"])],
                       "start_ms": int(w[i]["start_ms"]), "label": int(w[i]["label"]),
                       "active_buckets": int(w[i]["active"]), "reconstruction_mse": float(value),
                       **{f"flag_at_{name}": int(value >= threshold) for name, threshold in thresholds.items()}}
        write_csv(outdir / f"{split}_scores.csv", rows())
    return overall


def run(args):
    if min(args.epochs, args.batch_size, args.patience, args.latent, args.budget) < 1 or args.lr <= 0:
        raise ValueError("epochs/batch/patience/latent/budget/lr must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: install/select a CUDA-enabled PyTorch environment")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError("Invalid --gpu-index")
    device = torch.device(f"cuda:{args.gpu_index}")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = WindowData(args.data_dir)
    if (data.manifest["sequence_length"], data.manifest["bucket_size_ms"], len(data.names)) != (10, 10_000, 88):
        raise ValueError("Unexpected prepared-data geometry")
    for split in ("train", "validation", "calibration"):
        if not len(data.windows[split]) or np.any(data.windows[split]["label"] != 0):
            raise ValueError(f"{split} must contain only benign windows")
    for split in SPLITS:
        w = data.windows[split]
        if not len(w) or np.any(w["active"] < 1) or not np.array_equal((w["rows"] >= 0).sum(axis=1), w["active"]):
            raise ValueError(f"Invalid mask geometry in {split}")
        if np.any(w["rows"] < -1) or not np.isin(w["label"], [0, 1]).all():
            raise ValueError(f"Invalid saved indices/labels in {split}")
    if set(np.unique(data.windows["test"]["label"])) != {0, 1}:
        raise ValueError("Test split needs both classes")
    # Recheck bucket and time-block isolation on the SAVED arrays, not just a manifest claim.
    bucket_ids, blocks = {}, {}
    for split in SPLITS:
        ids = np.unique(data.windows[split]["rows"])
        bucket_ids[split] = ids[ids >= 0]
        if bucket_ids[split][-1] >= len(data.buckets):
            raise ValueError("Out-of-range bucket index")
        blocks[split] = set((data.windows[split]["start_ms"] // data.manifest["block_size_ms"]).tolist())
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if blocks[left] & blocks[right] or np.intersect1d(bucket_ids[left], bucket_ids[right]).size:
                raise ValueError(f"Cross-split leakage: {left}/{right}")
    empty_output(args.outdir)
    dump_json(args.outdir / "run_config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "completed": False, "dataset_manifest": data.manifest,
    })
    print(f"[CUDA] {torch.cuda.get_device_name(device)}, torch={torch.__version__}", flush=True)
    print("[data] " + "; ".join(f"{s}={len(data.windows[s]):,}" for s in SPLITS), flush=True)
    transformer = MaskedTransformer(data.names).fit(data)
    width, count = match_width(len(data.names), args.latent, args.budget)
    model = MaskedCNNAutoencoder(len(data.names), SEQ_LEN, width, args.latent).to(device)
    print(f"[model] encoder=89, reconstruction=88, width={width}, params={count:,}", flush=True)
    best_epoch, best_loss = fit(data, transformer, model, args, device)
    cal_scores = score(data, transformer, model, "calibration", args.batch_size, device)
    thresholds = {"0.1pct_fpr": float(np.quantile(cal_scores, .999)),
                  "1pct_fpr": float(np.quantile(cal_scores, .99)),
                  "5pct_fpr": float(np.quantile(cal_scores, .95))}
    dump_json(args.outdir / "thresholds.json", {
        "source": "separate benign calibration only", "operator": ">=", "thresholds": thresholds,
        "n_calibration": len(cal_scores),
        "realized_calibration_fpr": {k: float((cal_scores >= v).mean()) for k, v in thresholds.items()},
        "warning": "Overlapping windows are correlated; 0.1% tail estimates have limited support",
    })
    scaler = transformer.scaler
    np.savez(args.outdir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_,
             log_columns=np.asarray(transformer.log_cols), n_fit_buckets=transformer.n_fit_buckets)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "n_features": len(data.names), "seq_len": SEQ_LEN, "width": width,
                "latent": args.latent, "n_params": count, "feature_names": data.names,
                "best_epoch": best_epoch, "best_validation_mse": best_loss,
                "architecture": "MaskedCNNAutoencoder, 89 input / 88 output",
                "score": "per-window mean feature MSE over observed bins only"},
               args.outdir / "cnn_ae_checkpoint.pt")
    dump_json(args.outdir / "feature_transform.json", {
        "feature_names": data.names, "log1p_and_standardize": [data.names[i] for i in transformer.log_cols],
        "unchanged_0_1": [data.names[i] for i in transformer.bounded_cols],
        "fit_source": "unique observed buckets in sampled benign train windows only",
        "unique_fit_buckets": transformer.n_fit_buckets,
        "padding": "zero AFTER transform, mask supplied to encoder; no mask reconstruction loss",
    })
    print(f"[frozen] thresholds={thresholds}; scoring untouched test windows...", flush=True)
    test_scores = score(data, transformer, model, "test", args.batch_size, device)
    result = evaluate(data, test_scores, cal_scores, thresholds, args.outdir)
    dump_json(args.outdir / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "deterministic": True,
    })
    config = read_json(args.outdir / "run_config.json")
    config.update(completed=True, best_epoch=best_epoch, best_validation_mse=best_loss,
                  width=width, n_params=count)
    dump_json(args.outdir / "run_config.json", config)
    print(f"[test] ROC-AUC={result['roc_auc']}, PR-AUC(AP)={result['pr_auc_average_precision']}, "
          f"base rate={result['attack_prevalence']:.3%}", flush=True)
    print(f"[test] DR@1% cal-FPR={result['dr_at_1pct_fpr']:.3%}, "
          f"actual test FPR={result['actual_fpr_at_1pct_fpr']:.3%}", flush=True)
    print(f"[done] {args.outdir.resolve()}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "data_stride5")
    p.add_argument("--outdir", type=Path, default=HERE / "runs/cnn_ae_stride5_seed0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--budget", type=int, default=250_000, help="Model PARAMETER budget, not sample budget")
    p.add_argument("--lr", type=float, default=.001)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
