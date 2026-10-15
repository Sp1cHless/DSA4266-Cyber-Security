"""
window.py — Convert the flow-level parquet from preprocess.py into fixed-length
per-host sequences.

Emits, from ONE windowing pass so all three arms are comparable:
  X_seq.npy      (n_windows, window_size, n_features)  float32 — the AE arm
  mask.npy       (n_windows, window_size)              bool    — real vs pad
  X_agg.npy      (n_windows, 7 * n_features)           float32 — XGB, order destroyed
  X_flat.npy     (n_windows, window_size * n_features) float32 — XGB, order positional
  meta.parquet   one row per window: host, times, label, attack, density, split

Design decisions implemented here:
  * Time-based splits with a GUARD BAND. Overlapping sliding windows otherwise
    put near-duplicate windows on both sides of the cutoff.
  * Window label = malicious if ANY flow is malicious; attack type = majority
    non-benign class. Attack DENSITY is retained so results can be stratified
    by it later.
  * X_agg is the order-destroyed control: same windows, same flows, summary
    statistics only. The gap between it and the sequence model is the value
    of temporal structure, isolated.
  * --shuffle-within-window produces the shuffle control dataset: identical
    windows with flow order permuted. If the AE performs the same on this,
    it is not using temporal order.

Usage:
    python window.py --parquet prepared.parquet --outdir windows_w20s10 \
        --window-size 20 --stride 10

    python window.py --parquet prepared.parquet --outdir windows_shuffled \
        --window-size 20 --stride 10 --shuffle-within-window
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

META_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS",
             "FLOW_END_MILLISECONDS", "Label", "Attack"]

AGG_NAMES = ["mean", "std", "min", "max", "first", "last", "delta"]


# --------------------------------------------------------------------------
def load(parquet_path):
    df = pd.read_parquet(parquet_path)
    mpath = os.path.splitext(parquet_path)[0] + "_manifest.json"
    manifest = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    feature_cols = manifest.get(
        "feature_cols", [c for c in df.columns if c not in META_COLS])
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Loaded {len(df):,} flows, {len(feature_cols)} features")
    return df, feature_cols, manifest


def estimate_memory(n_flows, window_size, stride, n_features, pad):
    approx_windows = max(1, n_flows // stride)
    seq_gb = approx_windows * window_size * n_features * 4 / 1e9
    print(f"\n[memory] ~{approx_windows:,} windows expected")
    print(f"[memory] sequence tensor ~{seq_gb:.2f} GB (float32)")
    print(f"[memory] flattened tensor ~{seq_gb:.2f} GB")
    if seq_gb > 8:
        print("[memory] WARNING: increase --stride or reduce --window-size.")
        print("         stride == window_size gives non-overlapping windows")
        print("         and removes the leakage concern entirely.")


# --------------------------------------------------------------------------
def window_one_host(values, window_size, stride, pad):
    """Return start indices of each window for a single host's flow block."""
    n = len(values)
    if n >= window_size:
        last = n - window_size
        return np.arange(0, last + 1, stride), False
    if pad:
        return np.array([0]), True
    return np.array([], dtype=int), False


def build_windows(df, feature_cols, group_keys, window_size, stride, pad):
    """Slice each host's contiguous flow block into fixed-length windows."""
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    labels = df["Label"].to_numpy()
    attacks = df["Attack"].to_numpy()
    t_start = df["FLOW_START_MILLISECONDS"].to_numpy()
    t_end = df["FLOW_END_MILLISECONDS"].to_numpy()

    # df is already sorted by (group_key, FLOW_START_MILLISECONDS) upstream.
    codes, uniques = pd.factorize(
        df[group_keys].astype(str).agg("|".join, axis=1), sort=False)
    order = np.argsort(codes, kind="mergesort")
    if not np.array_equal(order, np.arange(len(df))):
        raise ValueError(
            "Flows are not contiguous per group. Re-run preprocess.py with "
            "the same --group-key, or sort before windowing.")

    boundaries = np.flatnonzero(np.diff(codes)) + 1
    blocks = np.split(np.arange(len(df)), boundaries)

    seq_idx, rows = [], []
    n_dropped_hosts = n_padded = 0

    for block in blocks:
        starts, padded = window_one_host(block, window_size, stride, pad)
        if len(starts) == 0:
            n_dropped_hosts += 1
            continue
        if padded:
            n_padded += 1
        host = uniques[codes[block[0]]]
        for s in starts:
            idx = block[s:s + window_size]
            n_real = len(idx)
            if n_real < window_size:                       # pad at the end
                idx = np.concatenate(
                    [idx, np.full(window_size - n_real, -1, dtype=int)])
            seq_idx.append(idx)

            real = idx[idx >= 0]
            lab = labels[real]
            atk = attacks[real]
            mal = atk[lab == 1]
            if len(mal):
                vals, counts = np.unique(mal, return_counts=True)
                attack_type = vals[np.argmax(counts)]
            else:
                attack_type = "Benign"
            rows.append((host, int(t_start[real[0]]), int(t_end[real[-1]]),
                         int(lab.max()), attack_type,
                         float(lab.mean()), int(n_real)))

    if not seq_idx:
        raise ValueError("No windows produced. Lower --window-size or use --pad.")

    seq_idx = np.stack(seq_idx)
    mask = seq_idx >= 0
    safe = np.where(mask, seq_idx, 0)
    X_seq = X[safe]                       # (n_windows, window_size, n_features)
    X_seq[~mask] = 0.0

    meta = pd.DataFrame(rows, columns=[
        "host", "start_ms", "end_ms", "label", "attack",
        "attack_density", "n_real_flows"])
    meta.insert(0, "window_id", np.arange(len(meta)))

    print(f"\n[window] {len(meta):,} windows of shape "
          f"({window_size}, {len(feature_cols)})")
    print(f"[window] hosts dropped (too few flows): {n_dropped_hosts:,}")
    if pad:
        print(f"[window] hosts padded: {n_padded:,}")
    print(f"[window] malicious windows: {meta['label'].sum():,} "
          f"({100*meta['label'].mean():.2f}%)")
    return X_seq, mask, meta


# --------------------------------------------------------------------------
def drop_spanning_windows(X_seq, mask, meta, max_span_ms):
    """Drop windows whose wall-clock span exceeds max_span_ms.

    UNSW-NB15 was captured in two sessions ~624 hours apart. A window built
    from a host's last flows in session 1 and first flows in session 2 spans
    almost a month and is not a behavioural sequence at all.
    """
    span = meta["end_ms"] - meta["start_ms"]
    keep = (span <= max_span_ms).to_numpy()
    n = int((~keep).sum())
    print(f"\n[span] dropped {n:,} windows spanning > {max_span_ms/3.6e6:.1f} h "
          f"(session-gap straddlers)")
    if n:
        print(f"[span] largest kept span: {span[keep].max()/1000:.1f} s")
    return X_seq[keep], mask[keep], meta[keep].reset_index(drop=True)


def time_split(meta, train_frac=0.6, val_frac=0.2):
    """Chronological split with a guard band.

    A window is assigned by its time span. Any window straddling a cutoff is
    DROPPED: with stride < window_size it would otherwise share flows with
    windows on the other side of the boundary.
    """
    c1 = meta["start_ms"].quantile(train_frac)
    c2 = meta["start_ms"].quantile(train_frac + val_frac)

    split = np.full(len(meta), "", dtype=object)
    s, e = meta["start_ms"].to_numpy(), meta["end_ms"].to_numpy()
    split[e <= c1] = "train"
    split[(s > c1) & (e <= c2)] = "val"
    split[s > c2] = "test"

    dropped = int((split == "").sum())
    meta = meta.assign(split=split)
    print(f"\n[split] cutoffs {pd.to_datetime(c1, unit='ms')} | "
          f"{pd.to_datetime(c2, unit='ms')}")
    print(f"[split] guard band dropped {dropped:,} straddling windows")
    print(meta[meta["split"] != ""].groupby("split")
              .agg(windows=("label", "size"),
                   malicious=("label", "sum"),
                   rate=("label", "mean")).to_string())

    print("\n[split] attack classes per split "
          "(check no class is absent from test):")
    kept = meta[meta["split"] != ""]
    print(pd.crosstab(kept["attack"], kept["split"]).to_string())

    print("\n  AE training set = split=='train' AND label==0.")
    print("  Threshold calibration = split=='val' AND label==0.")
    return meta


# --------------------------------------------------------------------------
def aggregate(X_seq, mask):
    """Order-destroyed summary of each window: the XGBoost control."""
    m = mask[:, :, None].astype(np.float32)
    n = np.clip(m.sum(axis=1), 1, None)

    mean = (X_seq * m).sum(axis=1) / n
    var = ((X_seq - mean[:, None, :]) ** 2 * m).sum(axis=1) / n
    std = np.sqrt(var)
    big = np.where(m > 0, X_seq, -np.inf)
    small = np.where(m > 0, X_seq, np.inf)
    mx, mn = big.max(axis=1), small.min(axis=1)

    first = X_seq[:, 0, :]
    last_idx = np.clip(mask.sum(axis=1) - 1, 0, None)
    last = X_seq[np.arange(len(X_seq)), last_idx, :]

    out = np.concatenate([mean, std, mn, mx, first, last, last - first],
                         axis=1).astype(np.float32)
    print(f"[agg] aggregated tensor {out.shape} "
          f"({len(AGG_NAMES)} stats x {X_seq.shape[2]} features)")
    return out


def agg_column_names(feature_cols):
    return [f"{s}__{c}" for s in AGG_NAMES for c in feature_cols]


def flat_column_names(feature_cols, window_size):
    return [f"t{t}__{c}" for t in range(window_size) for c in feature_cols]


# --------------------------------------------------------------------------
def run(parquet, outdir, window_size, stride, group_keys, pad,
        train_frac, val_frac, shuffle_within, emit_agg, emit_flat, seed,
        max_span_ms=3_600_000):
    os.makedirs(outdir, exist_ok=True)
    df, feature_cols, pre_manifest = load(parquet)
    estimate_memory(len(df), window_size, stride, len(feature_cols), pad)

    X_seq, mask, meta = build_windows(
        df, feature_cols, group_keys, window_size, stride, pad)

    X_seq, mask, meta = drop_spanning_windows(X_seq, mask, meta, max_span_ms)
    meta["window_id"] = np.arange(len(meta))

    if shuffle_within:
        rng = np.random.default_rng(seed)
        for i in range(len(X_seq)):
            k = int(mask[i].sum())
            if k > 1:
                X_seq[i, :k] = X_seq[i, rng.permutation(k)]
        print("\n[control] flows permuted WITHIN each window.")
        print("[control] Identical content, order destroyed. If the AE scores")
        print("[control] the same here as on the ordered set, it is not using")
        print("[control] temporal structure and no temporal claim holds.")

    meta = time_split(meta, train_frac, val_frac)

    np.save(os.path.join(outdir, "X_seq.npy"), X_seq)
    np.save(os.path.join(outdir, "mask.npy"), mask)
    meta.to_parquet(os.path.join(outdir, "meta.parquet"), index=False)

    if emit_agg:
        np.save(os.path.join(outdir, "X_agg.npy"), aggregate(X_seq, mask))
    if emit_flat:
        flat = X_seq.reshape(len(X_seq), -1)
        np.save(os.path.join(outdir, "X_flat.npy"), flat)
        print(f"[flat] flattened tensor {flat.shape}")

    manifest = {
        "source_parquet": parquet,
        "window_size": window_size,
        "stride": stride,
        "overlapping": stride < window_size,
        "max_span_ms": max_span_ms,
        "group_keys": group_keys,
        "padded": pad,
        "shuffle_within_window": shuffle_within,
        "seed": seed,
        "n_windows": int(len(meta)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "agg_cols": agg_column_names(feature_cols) if emit_agg else [],
        "flat_cols": flat_column_names(feature_cols, window_size) if emit_flat else [],
        "train_frac": train_frac,
        "val_frac": val_frac,
        "binary_cols": pre_manifest.get("binary_cols", []),
        "feature_groups": pre_manifest.get("feature_groups", {}),
        "ttl_dropped": pre_manifest.get("ttl_dropped", False),
    }
    with open(os.path.join(outdir, "window_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {outdir}/  (X_seq, mask, meta"
          f"{', X_agg' if emit_agg else ''}"
          f"{', X_flat' if emit_flat else ''}, window_manifest.json)")
    print("""
Next:
  AE arm   -> X_seq[train & label==0], validate on X_seq[val & label==0],
              threshold at a percentile of benign val reconstruction error,
              evaluate on the full test split.
  XGB arm  -> X_agg (order destroyed) and X_flat (order positional), same
              windows, same splits. The agg-vs-seq gap is the temporal effect.
  Control  -> re-run with --shuffle-within-window and compare the AE.""")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--window-size", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--group-key", default="IPV4_SRC_ADDR",
                    help="comma-separated, e.g. IPV4_SRC_ADDR,IPV4_DST_ADDR")
    ap.add_argument("--pad", action="store_true",
                    help="pad short hosts instead of dropping them")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--shuffle-within-window", action="store_true")
    ap.add_argument("--no-agg", action="store_true")
    ap.add_argument("--no-flat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-span-ms", type=int, default=3_600_000,
                    help="drop windows spanning longer than this (default 1 h); "
                         "guards against the 624-hour capture-session gap")
    a = ap.parse_args()

    run(a.parquet, a.outdir, a.window_size, a.stride,
        [k.strip() for k in a.group_key.split(",")], a.pad,
        a.train_frac, a.val_frac, a.shuffle_within_window,
        not a.no_agg, not a.no_flat, a.seed, a.max_span_ms)
