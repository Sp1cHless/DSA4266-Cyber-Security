"""
ae_ablation.py — Architecture ablation for the sequence autoencoder.

Three variants, PARAMETER-MATCHED, three seeds each, trained on benign windows
only and selected on the VALIDATION split. The test split stays sealed until
architecture AND feature set are both frozen.

  lstm    recurrent encoder/decoder
  cnn     1D convolutional encoder/decoder
  hybrid  both branches in parallel, fused at the bottleneck

Parameter matching is the point. An unmatched hybrid has roughly twice the
capacity of either branch, so "hybrid wins" would mean "hybrid is bigger".
match_width() binary-searches the hidden width of each variant to land within
tolerance of one shared budget.

Selection metric is PR-AUC aggregated over all attack classes on the validation
split, because the model that matters is the one that detects NOVEL attacks --
and since training is benign-only, every attack class is already novel.

Usage:
    python ae_ablation.py --windows windows_w20s10 --outdir ablation_out \
        --budget 250000 --seeds 0 1 2

NOTE: written against PyTorch but not executed in the authoring environment.
Run the shape self-test first:  python ae_ablation.py --self-test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import evaluate, compare, per_class_table, summary_row


# ==========================================================================
# Models
# ==========================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, seq_len, width, latent):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.LSTM(n_features, width, batch_first=True)
        self.to_latent = nn.Linear(width, latent)
        self.from_latent = nn.Linear(latent, width)
        self.dec = nn.LSTM(width, width, batch_first=True)
        self.out = nn.Linear(width, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_latent(h[-1])
        d = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        d, _ = self.dec(d)
        return self.out(d)


class CNNAutoencoder(nn.Module):
    def __init__(self, n_features, seq_len, width, latent):
        super().__init__()
        self.seq_len, self.width = seq_len, width
        self.t2 = (seq_len + 1) // 2
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.ReLU())
        self.to_latent = nn.Linear(width * self.t2, latent)
        self.from_latent = nn.Linear(latent, width * self.t2)
        self.up = nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)
        self.out = nn.Conv1d(width, n_features, 3, padding=1)

    def forward(self, x):
        h = self.enc(x.transpose(1, 2))
        z = self.to_latent(h.flatten(1))
        d = self.from_latent(z).view(-1, self.width, self.t2)
        d = torch.relu(self.up(d))[:, :, :self.seq_len]
        return self.out(d).transpose(1, 2)


class HybridAutoencoder(nn.Module):
    """Parallel LSTM and CNN branches, fused at the bottleneck."""

    def __init__(self, n_features, seq_len, width, latent):
        super().__init__()
        self.seq_len, self.width = seq_len, width
        self.t2 = (seq_len + 1) // 2
        self.lstm_enc = nn.LSTM(n_features, width, batch_first=True)
        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_features, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.ReLU())
        self.to_latent = nn.Linear(width + width * self.t2, latent)

        self.from_latent_l = nn.Linear(latent, width)
        self.lstm_dec = nn.LSTM(width, width, batch_first=True)
        self.from_latent_c = nn.Linear(latent, width * self.t2)
        self.up = nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)
        self.out = nn.Linear(width * 2, n_features)

    def forward(self, x):
        _, (h, _) = self.lstm_enc(x)
        c = self.cnn_enc(x.transpose(1, 2))
        z = self.to_latent(torch.cat([h[-1], c.flatten(1)], dim=1))

        dl = self.from_latent_l(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dl, _ = self.lstm_dec(dl)
        dc = self.from_latent_c(z).view(-1, self.width, self.t2)
        dc = torch.relu(self.up(dc))[:, :, :self.seq_len].transpose(1, 2)
        return self.out(torch.cat([dl, dc], dim=2))


BUILDERS = {"lstm": LSTMAutoencoder, "cnn": CNNAutoencoder,
            "hybrid": HybridAutoencoder}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_width(variant, n_features, seq_len, latent, budget, tol=0.10):
    """Binary-search hidden width so the variant lands near the budget."""
    lo, hi = 4, 1024
    best, best_w = None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(BUILDERS[variant](n_features, seq_len, mid, latent))
        if best is None or abs(n - budget) < abs(best - budget):
            best, best_w = n, mid
        if n < budget:
            lo = mid + 1
        else:
            hi = mid - 1
    off = abs(best - budget) / budget
    if off > tol:
        print(f"  WARNING {variant}: {best:,} params is {off:.1%} off budget. "
              f"Report the counts and note the mismatch.")
    return best_w, best


# ==========================================================================
# Training
# ==========================================================================
def masked_mse(x, xhat, mask):
    """Mean squared error over REAL timesteps only."""
    err = ((x - xhat) ** 2).mean(dim=2)                 # (B, T)
    return (err * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def window_scores(model, X, M, device, batch=512):
    """Per-window anomaly score = mean masked reconstruction error."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(device)
        m = torch.from_numpy(M[i:i + batch]).float().to(device)
        err = ((x - model(x)) ** 2).mean(dim=2)
        out.append(((err * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def train(model, Xtr, Mtr, Xva, Mva, device, epochs=50, batch=256,
          lr=1e-3, patience=5, verbose=True):
    """Train on benign windows; early-stop on benign validation loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            x = torch.from_numpy(Xtr[idx]).to(device)
            m = torch.from_numpy(Mtr[idx]).float().to(device)
            loss = masked_mse(x, model(x), m)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for i in range(0, len(Xva), 512):
                x = torch.from_numpy(Xva[i:i + 512]).to(device)
                m = torch.from_numpy(Mva[i:i + 512]).float().to(device)
                vl += masked_mse(x, model(x), m).item() * len(x); n += len(x)
            vl /= max(n, 1)

        if verbose:
            print(f"    epoch {ep:02d}  train {tot/len(perm):.5f}  val {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best


# ==========================================================================
def load_windows(d):
    X = np.load(os.path.join(d, "X_seq.npy")).astype(np.float32)
    M = np.load(os.path.join(d, "mask.npy"))
    meta = pd.read_parquet(os.path.join(d, "meta.parquet"))
    with open(os.path.join(d, "window_manifest.json")) as f:
        man = json.load(f)
    assert len(X) == len(meta), "X_seq and meta are out of sync"
    return X, M, meta, man


def run(windows_dir, outdir, budget, latent, seeds, epochs, device):
    os.makedirs(outdir, exist_ok=True)
    X, M, meta, man = load_windows(windows_dir)
    n_features, seq_len = X.shape[2], X.shape[1]

    tr = ((meta.split == "train") & (meta.label == 0)).to_numpy()
    va_b = ((meta.split == "val") & (meta.label == 0)).to_numpy()
    va = (meta.split == "val").to_numpy()

    print(f"AE train (benign)      : {tr.sum():,}")
    print(f"AE val benign          : {va_b.sum():,}  (early stop + thresholds)")
    print(f"Selection set (val all): {va.sum():,}  "
          f"({int(meta.label[va].sum()):,} malicious)")
    print("TEST SPLIT SEALED — not touched until features are also frozen.\n")

    widths = {}
    for v in BUILDERS:
        w, n = match_width(v, n_features, seq_len, latent, budget)
        widths[v] = (w, n)
        print(f"  {v:7s} width={w:4d}  params={n:,}")

    results, rows = {}, []
    for v in BUILDERS:
        results[v] = []
        w, n_params = widths[v]
        for s in seeds:
            print(f"\n[{v} seed {s}]")
            torch.manual_seed(s); np.random.seed(s)
            model = BUILDERS[v](n_features, seq_len, w, latent)
            model, vloss = train(model, X[tr], M[tr], X[va_b], M[va_b],
                                 device, epochs=epochs)

            sc_val = window_scores(model, X[va], M[va], device)
            sc_ben = window_scores(model, X[va_b], M[va_b], device)
            m = evaluate(sc_val, meta.label[va].to_numpy(),
                         meta.attack[va].to_numpy(), sc_ben)
            m["n_params"], m["width"], m["val_recon_loss"] = n_params, w, vloss
            results[v].append(m)
            rows.append(summary_row(f"{v}_s{s}", m, n_params))
            print(f"    PR-AUC {m['pr_auc']:.4f}  DR@1%FPR "
                  f"{m['dr@fpr0.01']:.4f}  benign recon {vloss:.5f}")

            torch.save(model.state_dict(),
                       os.path.join(outdir, f"{v}_seed{s}.pt"))

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "runs.csv"), index=False)
    table, note = compare(results, primary="pr_auc")
    print("\n" + "=" * 78 + "\nARCHITECTURE ABLATION\n" + "=" * 78)
    print(table.to_string(index=False))
    if note:
        print(note)
    table.to_csv(os.path.join(outdir, "ablation_summary.csv"), index=False)

    winner = table.iloc[0]["config"]
    print(f"\nPer-class recall at 1% FPR — {winner}, seed {seeds[0]}:")
    pct = per_class_table(results[winner][0], fpr=0.01)
    print(pct.to_string(index=False))
    pct.to_csv(os.path.join(outdir, "per_class_winner.csv"), index=False)

    print(f"""
Check before moving on:
  * Is the ranking class-dependent? If one branch wins Reconnaissance and the
    other wins Exploits, say so -- that is a finding about local vs long-range
    structure, not a tie to be broken.
  * Did benign reconstruction loss move in step with PR-AUC? If a variant
    detects better while reconstructing benign worse, it is not modelling the
    normal manifold better, it is just noisier.
  * Underpowered classes are marked. No detectability claim about them holds.

Next: re-run on the --shuffle-within-window dataset with {winner}. If PR-AUC
is unchanged, the model ignores temporal order and no temporal claim survives.""")
    return results


def self_test():
    """Shape and parameter-matching check; no data needed."""
    F, T, L = 83, 20, 32
    x = torch.randn(8, T, F)
    for v, B in BUILDERS.items():
        w, n = match_width(v, F, T, L, 250_000)
        m = B(F, T, w, L)
        y = m(x)
        assert y.shape == x.shape, f"{v}: {y.shape} != {x.shape}"
        print(f"  {v:7s} width={w:4d} params={n:,} out={tuple(y.shape)} OK")
    mask = torch.ones(8, T); mask[:, 15:] = 0
    print(f"  masked_mse = {masked_mse(x, x * 0.9, mask).item():.5f}")
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows")
    ap.add_argument("--outdir", default="ablation_out")
    ap.add_argument("--budget", type=int, default=250_000)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
    else:
        if not a.windows:
            ap.error("--windows is required unless --self-test")
        run(a.windows, a.outdir, a.budget, a.latent, a.seeds, a.epochs, a.device)
