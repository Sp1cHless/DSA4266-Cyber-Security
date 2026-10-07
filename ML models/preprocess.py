"""
preprocess.py — NF-UNSW-NB15-v3 flow-level cleaning and encoding.

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
    python preprocess.py --csv NF-UNSW-NB15-v3.csv --out prepared.parquet
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
# Confirmed against audit output on NF-UNSW-NB15-v3.
DROP_REDUNDANT = {
    "MAX_IP_PKT_LEN": "identical to LONGEST_FLOW_PKT (corr 1.000, exact)",
    # NOTE: MIN_IP_PKT_LEN is NOT identical to SHORTEST_FLOW_PKT in v3 (it was
    # in v2). It is kept as a feature and treated as a SHORTCUT instead --
    # univariate AUC 0.986.
}

# From audit_out/column_profile.csv (top_freq > 0.99).
DROP_NEAR_CONSTANT = {
    "SRC_TO_DST_IAT_MIN": "zero in 99.94% of flows",
    "DST_TO_SRC_IAT_MIN": "zero in 99.99% of flows",
}

# --- shortcut features ----------------------------------------------------
# Every column below scores >= 0.90 AUC on its own (audit section 6). A model
# reaching 0.98 while one of these reaches 0.99 alone has learned nothing.
# Run the pipeline in tiers and report all three.
TTL_COLS = ["MIN_TTL", "MAX_TTL"]
# Documented and independently reproduced: benign and attack traffic originate
# from different VMs behind fixed routers, so TTL encodes originating-VM initial
# TTL minus a constant hop count (benign 31, attack 254 in this data).

SHORTCUT_COLS = TTL_COLS + [
    # excluded on empirical grounds after the univariate screen
    "MIN_IP_PKT_LEN",              # AUC 0.986
    "DST_TO_SRC_AVG_THROUGHPUT",   # AUC 0.955 -- weakest case; throughput is
                                   # plausibly real behaviour. Say so.
    # found by attribution on the trained AE, not by the univariate screen
    "SHORTEST_FLOW_PKT",           # 32% of the benign-to-attack error gap
                                   # alone, 1400x ratio; twin of MIN_IP_PKT_LEN
    "DNS_TTL_ANSWER",              # TTL of a DNS answer record: a property of
                                   # the nameserver queried, zero for non-DNS
                                   # flows; no cross-network mechanism
]
# Tiers are strict subsets, so reporting several turns the exclusion question
# into a measurement:
#   no flag          -> 0 removed   ("full")
#   --drop-ttl       -> 2 removed   ("ttlonly")
#   --drop-shortcuts -> 6 removed   ("noshort2")
# For the 4-feature tier, comment out the last two entries above and rebuild.

# --- columns needing non-numeric treatment -------------------------------
TCP_FLAG_COLS = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
TCP_FLAG_BITS = [("FIN", 0x01), ("SYN", 0x02), ("RST", 0x04), ("PSH", 0x08),
                 ("ACK", 0x10), ("URG", 0x20), ("ECE", 0x40), ("CWR", 0x80)]

ONEHOT_COLS = {          # column -> number of top categories to keep
    "PROTOCOL": 12,      # 255 distinct values in v3; top value covers 76%
    "L7_PROTO": 20,
    "ICMP_TYPE": 10,     # 1188 distinct (nProbe encodes type*256+code)
    "ICMP_IPV4_TYPE": 10,
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
    "shortcut": SHORTCUT_COLS,
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


def drop_columns(df, drop_ttl=False, drop_shortcuts=False):
    dropped = {}
    for table in (DROP_IDENTIFIERS, DROP_NOISE, DROP_REDUNDANT,
                  DROP_NEAR_CONSTANT):
        for col, reason in table.items():
            if col in df.columns and col not in META_COLS:
                df = df.drop(columns=[col])
                dropped[col] = reason
    shortcut_set = SHORTCUT_COLS if drop_shortcuts else (TTL_COLS if drop_ttl else [])
    for col in shortcut_set:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped[col] = "shortcut: testbed artifact / no cross-network mechanism"
    print(f"[shortcuts] {len(shortcut_set)} shortcut columns excluded")
    print(f"[drop] removed {len(dropped)} feature columns")
    for c, r in dropped.items():
        print(f"        {c:32s} {r}")
    return df, dropped


def add_missing_indicators(df, feature_cols):
    """Flag columns with nulls. Missingness itself can be informative, and
    silently median-filling hides it."""
    new = []
    for c in feature_cols:
        n = int(df[c].isna().sum())
        if n:
            ind = f"{c}__isna"
            df[ind] = df[c].isna().astype(np.int8)
            new.append(ind)
            print(f"[nulls] {c}: {n:,} missing -> added {ind}")
    return df, new


def scale(df, feature_cols, fit_mask, binary_cols=(), scaler=None,
          warn_z=50.0):
    """log1p the heavy-tailed columns, then standardise the CONTINUOUS ones.

    Binary indicators (flag bits, one-hots, missingness flags) are left in
    {0, 1} and NOT standardised. Standardising them is actively harmful for a
    reconstruction-error detector: a category present in 0.001% of benign
    training rows has std ~0.003, so a value of 1 becomes ~300 SD and its
    squared error alone reaches ~90,000. The detector then fires on "this
    window contains a rare category" rather than on temporal structure.

    fit_mask MUST select benign training rows only. Reconstruction error is a
    sum over features, so scaling decides what the anomaly score measures.
    """
    X = df[feature_cols].astype(np.float32).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[fit_mask].median())

    logged = [c for c in LOG1P_COLS if c in X.columns]
    X[logged] = np.log1p(X[logged].clip(lower=0))
    print(f"[scale] log1p applied to {len(logged)} columns")

    binary_cols = [c for c in binary_cols if c in X.columns]
    cont = [c for c in X.columns if c not in set(binary_cols)]
    print(f"[scale] {len(cont)} continuous columns standardised, "
          f"{len(binary_cols)} binary columns left in {{0,1}}")

    if scaler is None:
        scaler = StandardScaler().fit(X.loc[fit_mask, cont])
        print(f"[scale] StandardScaler fitted on {int(fit_mask.sum()):,} "
              f"benign training rows")
    X[cont] = scaler.transform(X[cont])

    # Scale sanity check: anything enormous will dominate reconstruction error.
    mx = X.abs().max()
    bad = mx[mx > warn_z].sort_values(ascending=False)
    if len(bad):
        print(f"\n[scale] WARNING: {len(bad)} columns exceed |z| > {warn_z:g}. "
              f"These will dominate the AE's reconstruction error:")
        for c, v in bad.head(10).items():
            print(f"          {c:45s} max|z| = {v:10.1f}")
        print("        Consider dropping them, or adding them to LOG1P_COLS.")
    else:
        print(f"[scale] max |z| across all columns = {mx.max():.1f}  (OK)")
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
def run(csv_path, out_path, nrows=None, drop_ttl=False, drop_shortcuts=False,
        group_key="IPV4_SRC_ADDR", train_frac=0.6, dedupe=False):
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    print(f"Loaded {len(df):,} x {df.shape[1]}")

    if dedupe:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[dedupe] removed {before - len(df):,} exact duplicate flows")

    df = sort_by_start(df, group_key)
    df = add_relative_time(df, group_key)
    df, dropped = drop_columns(df, drop_ttl=drop_ttl,
                               drop_shortcuts=drop_shortcuts)
    df, flag_cols = decompose_tcp_flags(df)
    df, dstport_cols, top_ports = bucket_dst_port(df)
    df, onehot_cols, onehot_vals = onehot_categoricals(df)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    df, na_cols = add_missing_indicators(df, feature_cols)
    binary_cols = flag_cols + dstport_cols + onehot_cols + na_cols
    feature_cols = feature_cols + na_cols
    fit_mask, cutoff = build_fit_mask(df, train_frac=train_frac)
    X, scaler, logged = scale(df, feature_cols, fit_mask,
                              binary_cols=binary_cols)

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
        "binary_cols": binary_cols,
        "onehot_values": {k: [str(x) for x in v]
                          for k, v in onehot_vals.items()},
        "dst_port_top": [int(p) for p in top_ports],
        "train_cutoff_ms": cutoff,
        "group_key": group_key,
        "ttl_dropped": drop_ttl,
        "shortcuts_dropped": drop_shortcuts,
        "shortcut_cols_excluded": (SHORTCUT_COLS if drop_shortcuts
                                   else (TTL_COLS if drop_ttl else [])),
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
                    help="drop MIN_TTL/MAX_TTL only")
    ap.add_argument("--drop-shortcuts", action="store_true",
                    help="drop every column in SHORTCUT_COLS")
    ap.add_argument("--dedupe", action="store_true",
                    help="remove exact duplicate flow records")
    ap.add_argument("--group-key", default="IPV4_SRC_ADDR")
    ap.add_argument("--train-frac", type=float, default=0.6)
    a = ap.parse_args()
    run(a.csv, a.out, a.nrows, a.drop_ttl, a.drop_shortcuts,
        a.group_key, a.train_frac, a.dedupe)
