"""
attribute.py — Where does the reconstruction error actually come from?

Answers three questions about a trained autoencoder:

  1. PER-FEATURE   Which input dimensions drive the anomaly score? If two or
                   three of 141 features account for nearly all of the gap
                   between benign and attack error, the detector has found
                   another shortcut rather than learned temporal behaviour.

  2. PER-CLASS     Does the driving feature set differ by attack type? This is
                   the mechanism behind "which attacks are easier to detect",
                   and it turns a ranking into an explanation.

  3. PER-TIMESTEP  Where inside the window does error concentrate? Error that
                   spikes on the malicious flows means the model localises the
                   attack. Error spread flat across all 20 positions means it
                   is behaving as a set encoder, which corroborates the
                   shuffle control.

Model geometry is inferred from the checkpoint's tensor shapes, so no
hyperparameters need to be passed.

Usage:
    python3 attribute.py --windows win_w20s20 \
        --checkpoint ablation_noshort/cnn_seed0.pt --outdir attribution
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from ae_ablation import (LSTMAutoencoder, CNNAutoencoder, HybridAutoencoder,
                         load_windows)


# --------------------------------------------------------------------------
def infer_geometry(state, seq_len):
    """Work out variant, n_features, width and latent from tensor shapes."""
    t2 = (seq_len + 1) // 2
    keys = set(state.keys())

    if "lstm_enc.weight_ih_l0" in keys:                       # hybrid
        variant = "hybrid"
        gate, n_features = state["lstm_enc.weight_ih_l0"].shape
        width = gate // 4
        latent = state["to_latent.weight"].shape[0]
    elif "enc.weight_ih_l0" in keys:                           # lstm
        variant = "lstm"
        gate, n_features = state["enc.weight_ih_l0"].shape
        width = gate // 4
        latent = state["to_latent.weight"].shape[0]
    elif "enc.0.weight" in keys:                               # cnn
        variant = "cnn"
        width, n_features, _ = state["enc.0.weight"].shape
        latent = state["to_latent.weight"].shape[0]
    else:
        raise ValueError(f"Unrecognised checkpoint. Keys: {sorted(keys)[:8]}")

    builder = {"lstm": LSTMAutoencoder, "cnn": CNNAutoencoder,
               "hybrid": HybridAutoencoder}[variant]
    print(f"[model] {variant}: n_features={n_features} width={width} "
          f"latent={latent} seq_len={seq_len} (t2={t2})")
    return variant, builder, int(n_features), int(width), int(latent)


@torch.no_grad()
def squared_error(model, X, M, device, batch=512):
    """Per-(window, timestep, feature) squared reconstruction error."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(device)
        e = (x - model(x)) ** 2
        e = e * torch.from_numpy(M[i:i + batch]).float().to(device)[:, :, None]
        out.append(e.cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
def per_feature(err, labels, cols, binary, outdir, top=25):
    """Mean squared error per feature, benign vs attack."""
    ben = err[labels == 0].mean(axis=(0, 1))
    atk = err[labels == 1].mean(axis=(0, 1))
    gap = atk - ben

    df = pd.DataFrame({
        "feature": cols, "benign_mse": ben, "attack_mse": atk, "gap": gap,
        "share_of_gap": gap / max(gap.sum(), 1e-12),
        "is_binary": [c in binary for c in cols],
    }).sort_values("gap", ascending=False)
    df.to_csv(os.path.join(outdir, "per_feature.csv"), index=False)

    print("\n" + "=" * 78 + "\nPER-FEATURE ATTRIBUTION\n" + "=" * 78)
    print(df.head(top).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    cum = df["share_of_gap"].cumsum().to_numpy()
    for k in (1, 3, 5, 10):
        if k <= len(cum):
            print(f"\n  top {k:2d} features account for {100*cum[k-1]:5.1f}% "
                  f"of the benign-to-attack error gap")
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    print(f"  {n80} of {len(cols)} features cover 80% of the gap")
    if n80 <= 3:
        print("\n  WARNING: the score is driven by almost nothing. Inspect the")
        print("  named features -- this is the signature of a shortcut, not of")
        print("  a model that has learned normal behaviour.")
    return df


def per_class(err, labels, attacks, cols, outdir, top=5):
    """Top error-driving features for each attack class."""
    ben = err[labels == 0].mean(axis=(0, 1))
    rows = []
    for cls in sorted(set(attacks[labels == 1])):
        m = (attacks == cls) & (labels == 1)
        gap = err[m].mean(axis=(0, 1)) - ben
        order = np.argsort(-gap)[:top]
        rows.append({
            "attack": cls, "n_windows": int(m.sum()),
            "top_features": ", ".join(cols[i] for i in order),
            "top_share": round(float(gap[order].sum() / max(gap.sum(), 1e-12)), 3),
        })
    df = pd.DataFrame(rows).sort_values("n_windows", ascending=False)
    df.to_csv(os.path.join(outdir, "per_class_features.csv"), index=False)

    print("\n" + "=" * 78 + "\nPER-CLASS ERROR DRIVERS\n" + "=" * 78)
    print(df.to_string(index=False))
    print("\n  Classes sharing the same drivers are detected by the same")
    print("  mechanism. Classes with distinct drivers are genuinely different")
    print("  detection problems -- that distinction is your explanation for")
    print("  why some attack types are harder than others.")
    return df


def per_timestep(err, labels, meta, outdir):
    """Where inside the window does error concentrate?"""
    ben = err[labels == 0].mean(axis=(0, 2))
    atk = err[labels == 1].mean(axis=(0, 2))
    df = pd.DataFrame({"timestep": np.arange(len(ben)),
                       "benign_mse": ben, "attack_mse": atk})
    df.to_csv(os.path.join(outdir, "per_timestep.csv"), index=False)

    print("\n" + "=" * 78 + "\nPER-TIMESTEP LOCALISATION\n" + "=" * 78)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    flat = atk.std() / max(atk.mean(), 1e-12)
    print(f"\n  attack error CV across timesteps = {flat:.3f}")
    if flat < 0.15:
        print("  Error is FLAT across the window: the model is not localising")
        print("  anything in time. Combined with the shuffle control, this is")
        print("  evidence it is acting as a set encoder, not a sequence model.")
    else:
        print("  Error varies across positions, so the model is responding to")
        print("  where in the window things happen.")

    dens = meta.loc[labels == 1, "attack_density"].to_numpy()
    if len(dens) and dens.std() > 1e-9:
        bins = pd.cut(dens, [0, .25, .5, .75, 1.0], include_lowest=True)
        w = err[labels == 1].mean(axis=(1, 2))
        print("\n  Mean window error by attack density:")
        print(pd.DataFrame({"density": bins, "mse": w})
              .groupby("density", observed=True)["mse"]
              .agg(["count", "mean"]).to_string())
        print("\n  If error does not rise with density, the detector is not")
        print("  responding to how much attack traffic the window contains.")
    return df


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", default="attribution")
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    X, M, meta, man = load_windows(a.windows)
    cols = np.array(man["feature_cols"])
    binary = set(man.get("binary_cols", []))

    sel = np.ones(len(meta), bool) if a.split == "all" \
        else (meta.split == a.split).to_numpy()
    X, M, meta = X[sel], M[sel], meta[sel].reset_index(drop=True)
    labels = meta.label.to_numpy()
    attacks = meta.attack.to_numpy()
    print(f"[data] split={a.split}: {len(meta):,} windows, "
          f"{int(labels.sum()):,} malicious")

    state = torch.load(a.checkpoint, map_location="cpu")
    variant, builder, n_features, width, latent = infer_geometry(
        state, X.shape[1])
    if n_features != X.shape[2]:
        raise ValueError(f"Checkpoint expects {n_features} features but the "
                         f"windows have {X.shape[2]}. Mismatched datasets.")
    model = builder(n_features, X.shape[1], width, latent)
    model.load_state_dict(state)
    model.to(a.device)

    err = squared_error(model, X, M, a.device)
    print(f"[error] tensor {err.shape}")

    per_feature(err, labels, cols, binary, a.outdir)
    per_class(err, labels, attacks, cols, a.outdir)
    per_timestep(err, labels, meta, a.outdir)
    print(f"\nWrote per_feature.csv, per_class_features.csv, "
          f"per_timestep.csv to {a.outdir}/")


if __name__ == "__main__":
    main()
