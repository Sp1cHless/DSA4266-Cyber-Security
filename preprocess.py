"""
preprocess.py — NF-UNSW-NB15-v2 flow-level cleaning and encoding.

Runs BEFORE windowing. Output is a flow-level table, sorted per host by start
time, with model features encoded and scaled, plus the metadata columns the
windowing step needs (source IP, timestamps, Label, Attack).

Design rules baked in:
  * Identifiers, absolute timestamps and targets never become features.
  * Bitmask and categorical columns are decomposed / one-hot encoded, never
    fed as integers.
  * Heavy-tailed counts get log1p before standardisation, so that the AE's
    reconstruction error is not dominated by whichever column has the largest
    raw magnitude.
  * The scaler is fitted on BENIGN TRAINING ROWS ONLY and reused everywhere.

Usage:
    python preprocess.py --csv NF-UNSW-NB15-v2.csv --out prepared.parquet
    python preprocess.py --csv ... --drop-ttl --out prepared_nottl.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================================================================
# COLUMN DECISIONS
# Edit these after reading audit.py output. Every entry carries its reason.
# ==========================================================================

# Kept alongside the features, but never fed to a model.
META_COLS = [
    "IPV4_SRC_ADDR",              # grouping key for windowing
    "IPV4_DST_ADDR",              # grouping key if using (src,dst) pairs
    "FLOW_START_MILLISECONDS",    # sort key; source of relative time features
    "FLOW_END_MILLISECONDS",
    "Label",
    "Attack",
]

DROP_IDENTIFIERS = {
    "IPV4_SRC_ADDR": "host identity; in this testbed it is close to the label",
    "IPV4_DST_ADDR": "victim identity; encodes fixed testbed topology",
    "L4_SRC_PORT": "ephemeral, OS-assigned per connection — noise",
    "FLOW_START_MILLISECONDS": "absolute epoch; lets the model memorise sessions",
    "FLOW_END_MILLISECONDS": "absolute epoch",
}

DROP_NOISE = {
    "DNS_QUERY_ID": "random DNS transaction id; pure noise, wastes AE capacity",
}

# Confirm against audit_out/redundant_pairs.csv before trusting these.
DROP_REDUNDANT = {
    "MIN_IP_PKT_LEN": "identical to SHORTEST_FLOW_PKT",
    "MAX_IP_PKT_LEN": "identical to LONGEST_FLOW_PKT",
}

# Populate from audit_out/column_profile.csv (top_freq > 0.99).
DROP_NEAR_CONSTANT = {}

TTL_COLS = ["MIN_TTL", "MAX_TTL"]

# --- columns needing non-numeric treatment -------------------------------
TCP_FLAG_COLS = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
TCP_FLAG_BITS = [("FIN", 0x01), ("SYN", 0x02), ("RST", 0x04), ("PSH", 0x08),
                 ("ACK", 0x10), ("URG", 0x20), ("ECE", 0x40), ("CWR", 0x80)]

ONEHOT_COLS = {          # column -> number of top categories to keep
    "PROTOCOL": 6,
    "L7_PROTO": 20,
    "ICMP_TYPE": 6,
    "ICMP_IPV4_TYPE": 6,
    "DNS_QUERY_TYPE": 6,
    "FTP_COMMAND_RET_CODE": 6,
}

DST_PORT_TOP_N = 20      # one-hot the N most frequent destination ports

# --- scaling -------------------------------------------------------------
# Heavy-tailed, non-negative: log1p before standardising.
LOG1P_COLS = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
    # derived below
    "GAP_FROM_PREV_MS", "ELAPSED_SINCE_HOST_START_MS",
]

# Feature groups for the later feature ablation.
FEATURE_GROUPS = {
    "ttl": TTL_COLS,
    "temporal": ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
                 "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
                 "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
                 "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
                 "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
                 "GAP_FROM_PREV_MS", "ELAPSED_SINCE_HOST_START_MS"],
    "volumetric": ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                   "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
                   "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"],
    "packet_size": ["LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
                    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
                    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
                    "NUM_PKTS_1024_TO_1514_BYTES"],
    "tcp_flags": TCP_FLAG_COLS,
    "retransmission": ["RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
                       "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS"],
}


# ==========================================================================
# Steps
# ==========================================================================
def sort_by_start(df, group_key="IPV4_SRC_ADDR"):
    """Sort per host by flow START time.

    The raw file is END-sorted, because NetFlow exporters emit a record when
    the flow terminates. Start-time order reflects the order the host actually
    initiated activity, which is what we want to model. It is NOT the order a
    live detector would observe — note this as a limitation.
    """
    before_end_sorted = df["FLOW_END_MILLISECONDS"].is_monotonic_increasing
    df = df.sort_values([group_key, "FLOW_START_MILLISECONDS"],
                        kind="mergesort").reset_index(drop=True)
    print(f"[sort] input was end-sorted: {before_end_sorted}")
    print(f"[sort] sorted by ({group_key}, FLOW_START_MILLISECONDS)")
    return df


def add_relative_time(df, group_key="IPV4_SRC_ADDR"):
    """Relative time features. These replace the absolute timestamps.

    Per-flow IAT columns describe packet timing INSIDE a flow. These describe
    spacing BETWEEN flows, which is the sequential signal the model needs.
    """
    g = df.groupby(group_key)["FLOW_START_MILLISECONDS"]
    df["GAP_FROM_PREV_MS"] = g.diff().fillna(0).clip(lower=0)
    df["ELAPSED_SINCE_HOST_START_MS"] = (
        df["FLOW_START_MILLISECONDS"] - g.transform("min")).clip(lower=0)
    print("[time] added GAP_FROM_PREV_MS, ELAPSED_SINCE_HOST_START_MS")
    return df


def decompose_tcp_flags(df):
    """Split flag bitmasks into binary indicators.

    TCP_FLAGS=27 is FIN|SYN|PSH|ACK and 18 is SYN|ACK; 27 is not 'more' than
    18. As an integer this structure is destroyed, and for an autoencoder the
    reconstruction target becomes meaningless.
    """
    new = []
    for col in TCP_FLAG_COLS:
        if col not in df.columns:
            continue
        v = df[col].fillna(0).astype(int).values
        for name, bit in TCP_FLAG_BITS:
            out = f"{col}_{name}"
            df[out] = ((v & bit) > 0).astype(np.int8)
            new.append(out)
        df = df.drop(columns=[col])
    # drop flag bits that never fire
    dead = [c for c in new if df[c].nunique() < 2]
    if dead:
        df = df.drop(columns=dead)
        new = [c for c in new if c not in dead]
    print(f"[flags] {len(new)} flag indicators kept, {len(dead)} never set")
    return df, new


def onehot_categoricals(df, spec=None, top_values=None):
    """One-hot encode categorical IDs, keeping the top-N values plus 'other'.

    PROTOCOL and L7_PROTO are numeric-typed identifiers. Averaging protocol 6
    and 17 gives 11.5, which decodes to an unrelated protocol.
    """
    spec = spec or ONEHOT_COLS
    learned = {} if top_values is None else top_values
    new = []
    for col, n in spec.items():
        if col not in df.columns:
            continue
        if col not in learned:
            learned[col] = df[col].value_counts().head(n).index.tolist()
        keep = learned[col]
        v = df[col].where(df[col].isin(keep), other="__other__")
        d = pd.get_dummies(v, prefix=col, dtype=np.int8)
        for k in keep:
            c = f"{col}_{k}"
            if c not in d.columns:
                d[c] = np.int8(0)
        df = pd.concat([df.drop(columns=[col]), d], axis=1)
        new.extend(d.columns.tolist())
    print(f"[onehot] {len(new)} indicator columns from {len(spec)} categoricals")
    return df, new, learned


def bucket_dst_port(df, top_ports=None):
    """Destination port: service class + one-hot of the top-N ports.

    Port 53 vs 21 vs 8088 carries meaning, but the integer ordering does not.
    """
    col = "L4_DST_PORT"
    if col not in df.columns:
        return df, [], top_ports
    p = df[col].fillna(-1).astype(int)
    if top_ports is None:
        top_ports = p.value_counts().head(DST_PORT_TOP_N).index.tolist()

    new = []
    for name, mask in [
        ("wellknown", p < 1024),
        ("registered", (p >= 1024) & (p < 49152)),
        ("ephemeral", p >= 49152),
    ]:
        c = f"DSTPORT_{name}"
        df[c] = mask.astype(np.int8)
        new.append(c)
    for port in top_ports:
        c = f"DSTPORT_is_{port}"
        df[c] = (p == port).astype(np.int8)
        new.append(c)
    df = df.drop(columns=[col])
    print(f"[dstport] {len(new)} indicators (3 classes + top {len(top_ports)})")
    return df, new, top_ports


def drop_columns(df, drop_ttl=False):
    dropped = {}
    for table in (DROP_IDENTIFIERS, DROP_NOISE, DROP_REDUNDANT,
                  DROP_NEAR_CONSTANT):
        for col, reason in table.items():
            if col in df.columns and col not in META_COLS:
                df = df.drop(columns=[col])
                dropped[col] = reason
    if drop_ttl:
        for col in TTL_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped[col] = "TTL: IXIA PerfectStorm topology artifact"
    print(f"[drop] removed {len(dropped)} feature columns")
    for c, r in dropped.items():
        print(f"        {c:32s} {r}")
    return df, dropped


def scale(df, feature_cols, fit_mask, scaler=None):
    """log1p the heavy-tailed columns, then standardise.

    fit_mask MUST select benign training rows only. Reconstruction error is a
    sum over features, so scaling decides what the anomaly score measures.
    """
    X = df[feature_cols].astype(np.float32).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[fit_mask].median())

    logged = [c for c in LOG1P_COLS if c in X.columns]
    X[logged] = np.log1p(X[logged].clip(lower=0))
    print(f"[scale] log1p applied to {len(logged)} columns")

    if scaler is None:
        scaler = StandardScaler().fit(X.loc[fit_mask])
        print(f"[scale] StandardScaler fitted on {int(fit_mask.sum()):,} "
              f"benign training rows")
    X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X, scaler, logged


def build_fit_mask(df, train_end_ms=None, train_frac=0.6):
    """Benign rows inside the training time range.

    Time-based, not random: overlapping sliding windows put near-duplicates
    on both sides of a random split.
    """
    if train_end_ms is None:
        train_end_ms = df["FLOW_START_MILLISECONDS"].quantile(train_frac)
    mask = (df["Label"] == 0) & (df["FLOW_START_MILLISECONDS"] <= train_end_ms)
    print(f"[split] train cutoff {pd.to_datetime(train_end_ms, unit='ms')}; "
          f"{int(mask.sum()):,} benign training flows")
    return mask, float(train_end_ms)


# ==========================================================================
def run(csv_path, out_path, nrows=None, drop_ttl=False,
        group_key="IPV4_SRC_ADDR", train_frac=0.6):
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    print(f"Loaded {len(df):,} x {df.shape[1]}")

    df = sort_by_start(df, group_key)
    df = add_relative_time(df, group_key)
    df, dropped = drop_columns(df, drop_ttl=drop_ttl)
    df, flag_cols = decompose_tcp_flags(df)
    df, dstport_cols, top_ports = bucket_dst_port(df)
    df, onehot_cols, onehot_vals = onehot_categoricals(df)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    fit_mask, cutoff = build_fit_mask(df, train_frac=train_frac)
    X, scaler, logged = scale(df, feature_cols, fit_mask)

    out = pd.concat([df[META_COLS].reset_index(drop=True),
                     X.reset_index(drop=True)], axis=1)
    out.to_parquet(out_path, index=False)

    manifest = {
        "source_csv": csv_path,
        "rows": int(len(out)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "dropped": dropped,
        "log1p_cols": logged,
        "binary_cols": flag_cols + dstport_cols + onehot_cols,
        "onehot_values": {k: [str(x) for x in v]
                          for k, v in onehot_vals.items()},
        "dst_port_top": [int(p) for p in top_ports],
        "train_cutoff_ms": cutoff,
        "group_key": group_key,
        "ttl_dropped": drop_ttl,
        "feature_groups": {k: [c for c in v if c in feature_cols]
                           for k, v in FEATURE_GROUPS.items()},
        "sort_order": "per-host by FLOW_START_MILLISECONDS",
    }
    mpath = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {out_path}  ({len(out):,} rows, {len(feature_cols)} features)")
    print(f"Wrote {mpath}")
    print("""
NOTE: binary indicator columns (flags, one-hots) are standardised along with
everything else. For an autoencoder you may prefer to leave them in {0,1} and
reconstruct them with BCE rather than MSE. The manifest lists them under
'binary_cols' so you can split the loss if you go that way.

This parquet is the FROZEN artifact. One owner, one version. Windowing, the
AE arm and the XGBoost arm all read this file and nothing else.""")
    return out, manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="prepared.parquet")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--drop-ttl", action="store_true",
                    help="produce the without-TTL variant for the ablation")
    ap.add_argument("--group-key", default="IPV4_SRC_ADDR")
    ap.add_argument("--train-frac", type=float, default=0.6)
    a = ap.parse_args()
    run(a.csv, a.out, a.nrows, a.drop_ttl, a.group_key, a.train_frac)
