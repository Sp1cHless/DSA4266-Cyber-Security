"""
audit.py — Diagnostic pass over NF-UNSW-NB15-v2.

Produces the evidence needed to justify every column-drop decision, plus the
data-forensics checks that should gate the rest of the project.

This script DECIDES NOTHING. It prints and saves evidence; you read it and
record the decisions in preprocess.py.

Usage:
    python audit.py --csv NF-UNSW-NB15-v2.csv --outdir audit_out
    python audit.py --csv NF-UNSW-NB15-v2.csv --nrows 500000   # quick pass
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

# --------------------------------------------------------------------------
# Column roles (see preprocess.py for the authoritative version)
# --------------------------------------------------------------------------
ID_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
TIME_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
TARGET_COLS = ["Label", "Attack"]

CATEGORICAL_CANDIDATES = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]

TTL_COLS = ["MIN_TTL", "MAX_TTL"]

SECTION = "\n" + "=" * 78 + "\n{}\n" + "=" * 78


def load(csv_path, nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {csv_path}")
    return df


# --------------------------------------------------------------------------
# 1. Basic integrity
# --------------------------------------------------------------------------
def basic_integrity(df):
    print(SECTION.format("1. BASIC INTEGRITY"))
    print(f"Shape: {df.shape}")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print(f"\nColumns with nulls: {len(nulls)}")
    if len(nulls):
        print(nulls)

    dup_rows = df.duplicated().sum()
    print(f"\nFully duplicated rows: {dup_rows:,} ({100*dup_rows/len(df):.2f}%)")
    if dup_rows:
        print("  -> Decide whether to dedupe. Duplicated flows inflate any")
        print("     random split and can put identical records in train+test.")

    print("\nLabel distribution:")
    print(df["Label"].value_counts().to_string())
    print(f"  attack rate = {df['Label'].mean():.4f}")

    print("\nAttack class distribution:")
    ac = df["Attack"].value_counts()
    print(ac.to_string())
    print("\n  Classes with <5000 flows will yield very few windows after")
    print("  grouping. Per-class recall for these needs Wilson CIs, and any")
    print("  'hardest to detect' claim about them is likely noise.")
    return ac


# --------------------------------------------------------------------------
# 2. Host / subnet structure  (the leakage check)
# --------------------------------------------------------------------------
def host_structure(df):
    print(SECTION.format("2. HOST & SUBNET STRUCTURE (leakage check)"))

    def subnet24(ip):
        return ip.rsplit(".", 1)[0] + ".x"

    df = df.copy()
    df["_src24"] = df["IPV4_SRC_ADDR"].map(subnet24)

    g = df.groupby("_src24").agg(
        flows=("Label", "size"),
        attack_flows=("Label", "sum"),
    )
    g["attack_rate"] = g["attack_flows"] / g["flows"]
    g["share_of_all_attacks"] = g["attack_flows"] / df["Label"].sum()
    g = g.sort_values("attack_flows", ascending=False)
    print("\nTop source /24 subnets by attack volume:")
    print(g.head(10).to_string())

    # Per-host purity: what fraction of hosts are all-benign or all-attack?
    h = df.groupby("IPV4_SRC_ADDR")["Label"].agg(["size", "mean"])
    pure = ((h["mean"] == 0) | (h["mean"] == 1)).mean()
    print(f"\nDistinct source hosts: {len(h):,}")
    print(f"Fraction of hosts that are 100% benign or 100% attack: {pure:.3f}")
    print("\n  If this is near 1.0, the source IP is effectively the label.")
    print("  Grouping sequences by IPV4_SRC_ADDR then makes every window")
    print("  near-pure by construction. Report this number in limitations.")

    # The source-IP-only baseline: how well does host identity alone predict?
    host_rate = df.groupby("IPV4_SRC_ADDR")["Label"].transform("mean")
    try:
        auc = roc_auc_score(df["Label"], host_rate)
        print(f"\nSOURCE-IP-ONLY BASELINE AUC: {auc:.4f}")
        print("  This is the ceiling attributable to testbed topology alone.")
        print("  Every model score should be read against it.")
    except ValueError:
        pass

    # Attacker-IP flows labelled Benign (PerfectStorm artifact)
    top_attack_subnet = g.index[0]
    mask = df["_src24"] == top_attack_subnet
    n_benign_from_attacker = int((mask & (df["Label"] == 0)).sum())
    print(f"\nFlows from {top_attack_subnet} labelled Benign: "
          f"{n_benign_from_attacker:,}")
    print("  These interact badly with 'window is malicious if any flow is'.")
    return g


# --------------------------------------------------------------------------
# 3. Timestamp sanity
# --------------------------------------------------------------------------
def timestamp_checks(df):
    print(SECTION.format("3. TIMESTAMP SANITY"))

    start, end = df["FLOW_START_MILLISECONDS"], df["FLOW_END_MILLISECONDS"]
    print(f"Start range: {pd.to_datetime(start.min(), unit='ms')} .. "
          f"{pd.to_datetime(start.max(), unit='ms')}")

    print(f"\nFile sorted by START? {start.is_monotonic_increasing}")
    print(f"File sorted by END?   {end.is_monotonic_increasing}")
    print("  NetFlow exporters emit on flow termination, so files are usually")
    print("  end-sorted. Sorting by start changes flow order materially.")

    bad = int((end < start).sum())
    print(f"\nRows with end < start: {bad:,}")

    # Capture sessions: look for large gaps in the global timeline
    s = np.sort(start.values)
    gaps = np.diff(s)
    big = np.where(gaps > 3_600_000)[0]   # >1 hour
    print(f"\nGaps > 1 hour in the timeline: {len(big)}")
    for i in big[:10]:
        print(f"  {pd.to_datetime(s[i], unit='ms')} -> "
              f"{pd.to_datetime(s[i+1], unit='ms')}  "
              f"({gaps[i]/3.6e6:.1f} h)")
    print("\n  UNSW-NB15 was captured in two sessions. A naive chronological")
    print("  split may hand you one whole session as test, with a different")
    print("  attack mix. Check the per-session class balance before splitting.")

    # Per-host sequence viability
    print("\nPer-host flow counts:")
    counts = df.groupby("IPV4_SRC_ADDR").size()
    print(counts.describe(percentiles=[.1, .25, .5, .75, .9]).to_string())

    for w in (10, 20, 50):
        keep_hosts = (counts >= w)
        kept_flows = counts[keep_hosts].sum()
        print(f"\n  window_size={w}: hosts kept {keep_hosts.sum():,}/"
              f"{len(counts):,}, flows kept {kept_flows:,}/{len(df):,} "
              f"({100*kept_flows/len(df):.1f}%)")

    # Per-class impact of the drop rule (critical for small classes)
    print("\nFlows retained per attack class at window_size=20:")
    ok_hosts = set(counts[counts >= 20].index)
    kept = df["IPV4_SRC_ADDR"].isin(ok_hosts)
    tab = pd.DataFrame({
        "total": df.groupby("Attack").size(),
        "kept": df[kept].groupby("Attack").size(),
    }).fillna(0).astype(int)
    tab["pct_kept"] = (100 * tab["kept"] / tab["total"]).round(1)
    print(tab.to_string())
    print("\n  If a class loses most of its flows here, the drop rule is")
    print("  deleting the very classes you want to study -> pad + mask instead.")

    # Wall-clock span of a 20-flow window
    print("\nWall-clock span of consecutive 20-flow blocks (seconds):")
    spans = []
    for _, grp in df.sort_values(
            ["IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS"]
    ).groupby("IPV4_SRC_ADDR")["FLOW_START_MILLISECONDS"]:
        v = grp.values
        if len(v) >= 20:
            spans.append((v[19::20] - v[0:-19:20]) / 1000.0)
    if spans:
        spans = np.concatenate(spans)
        print(pd.Series(spans).describe(
            percentiles=[.1, .5, .9]).to_string())
        print("\n  If the median span is milliseconds, inter-flow timing may")
        print("  carry little signal and a time-binned window may suit better.")


# --------------------------------------------------------------------------
# 4. Near-constant and high-cardinality columns
# --------------------------------------------------------------------------
def column_profile(df, outdir):
    print(SECTION.format("4. COLUMN PROFILE"))

    rows = []
    for c in df.columns:
        if c in TARGET_COLS:
            continue
        s = df[c]
        vc = s.value_counts(dropna=False)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "nunique": s.nunique(dropna=False),
            "top_value": vc.index[0],
            "top_freq": vc.iloc[0] / len(s),
            "uniqueness": s.nunique(dropna=False) / len(s),
        })
    prof = pd.DataFrame(rows).sort_values("top_freq", ascending=False)
    prof.to_csv(os.path.join(outdir, "column_profile.csv"), index=False)

    print("\nNEAR-CONSTANT columns (single value covers >99% of rows):")
    nc = prof[prof["top_freq"] > 0.99]
    print(nc[["column", "nunique", "top_value", "top_freq"]].to_string(index=False)
          if len(nc) else "  (none)")
    print("\n  These contribute nothing but still enter the AE's reconstruction")
    print("  error sum. Drop them.")

    print("\nHIGH-UNIQUENESS columns (>50% distinct values) — identifier-like:")
    hu = prof[prof["uniqueness"] > 0.5]
    print(hu[["column", "nunique", "uniqueness"]].to_string(index=False)
          if len(hu) else "  (none)")

    print("\nCardinality of categorical candidates:")
    print(prof[prof["column"].isin(CATEGORICAL_CANDIDATES)]
          [["column", "nunique", "top_value", "top_freq"]]
          .to_string(index=False))
    print("\n  These must be one-hot / bit-decomposed, not fed as integers.")
    return prof


# --------------------------------------------------------------------------
# 5. Duplicate & redundant column pairs
# --------------------------------------------------------------------------
def duplicate_pairs(df, outdir, corr_threshold=0.999):
    print(SECTION.format("5. DUPLICATE / REDUNDANT COLUMN PAIRS"))

    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in TARGET_COLS + TIME_COLS if c in df.columns],
        errors="ignore")
    num = num.loc[:, num.nunique() > 1]

    corr = num.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if r >= corr_threshold:
                a, b = cols[i], cols[j]
                identical = bool(num[a].equals(num[b]))
                pairs.append({"col_a": a, "col_b": b,
                              "abs_corr": round(float(r), 6),
                              "identical": identical})

    out = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)
    out.to_csv(os.path.join(outdir, "redundant_pairs.csv"), index=False)

    if len(out):
        print(f"\nPairs with |corr| >= {corr_threshold}:")
        print(out.to_string(index=False))
        ident = out[out["identical"]]
        if len(ident):
            print("\nEXACTLY IDENTICAL — drop one of each pair:")
            for _, r in ident.iterrows():
                print(f"  {r.col_a} == {r.col_b}")
    else:
        print("  (none found)")

    print("\n  Redundant features silently upweight whatever signal they carry")
    print("  in the AE's reconstruction error. Keep one per pair.")
    return out


# --------------------------------------------------------------------------
# 6. Univariate discriminative power  (the shortcut screen)
# --------------------------------------------------------------------------
def univariate_auc(df, outdir):
    print(SECTION.format("6. UNIVARIATE AUC SCREEN (shortcut detection)"))

    y = df["Label"].values
    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in TARGET_COLS + TIME_COLS if c in df.columns],
        errors="ignore")

    rows = []
    for c in num.columns:
        x = num[c].values.astype(float)
        if np.nanstd(x) == 0:
            continue
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        try:
            a = roc_auc_score(y, x)
        except ValueError:
            continue
        rows.append({"feature": c, "auc": max(a, 1 - a)})

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    res.to_csv(os.path.join(outdir, "univariate_auc.csv"), index=False)

    print("\nTop 15 single features by AUC (attack vs benign):")
    print(res.head(15).to_string(index=False))

    strong = res[res["auc"] >= 0.90]
    print(f"\nFeatures with AUC >= 0.90 on their own: {len(strong)}")
    if len(strong):
        print(strong.to_string(index=False))
        print("\n  Any feature here is a shortcut candidate. A model scoring")
        print("  0.98 while one column scores 0.97 alone has learned nothing.")

    print("\nTTL columns specifically:")
    print(res[res["feature"].isin(TTL_COLS)].to_string(index=False))

    # TTL value distributions by class — the testbed artifact, made visible
    for c in TTL_COLS:
        if c in df.columns:
            print(f"\n{c} — most common values by class:")
            t = (df.groupby("Label")[c]
                   .apply(lambda s: s.value_counts().head(4).to_dict()))
            print(t.to_string())
    print("\n  Expect benign near 31/62 and attack near 254: this is the")
    print("  IXIA PerfectStorm topology artifact, not attack behaviour.")
    return res


# --------------------------------------------------------------------------
# 7. Justify dropping L4_SRC_PORT and DNS_QUERY_ID
# --------------------------------------------------------------------------
def justify_noise_columns(df, auc_table):
    print(SECTION.format("7. JUSTIFYING SPECIFIC DROPS"))

    def auc_of(col):
        r = auc_table[auc_table["feature"] == col]
        return float(r["auc"].iloc[0]) if len(r) else float("nan")

    for col, note in [
        ("L4_SRC_PORT", "ephemeral, assigned per-connection by the OS"),
        ("DNS_QUERY_ID", "random DNS transaction identifier"),
        ("L4_DST_PORT", "semantically meaningful but not ordinal"),
    ]:
        if col not in df.columns:
            continue
        s = df[col]
        print(f"\n{col}  ({note})")
        print(f"  distinct values : {s.nunique():,}")
        print(f"  uniqueness      : {s.nunique()/len(s):.4f}")
        print(f"  univariate AUC  : {auc_of(col):.4f}")
        if col.endswith("PORT"):
            print(f"  share >= 1024   : {(s >= 1024).mean():.3f}")
            print(f"  share < 1024    : {(s < 1024).mean():.3f}")
            top = s.value_counts().head(8)
            print(f"  top values      : {dict(top)}")

    print("""
  Reading the evidence:
    - High uniqueness + AUC near 0.5  -> noise. Drop (L4_SRC_PORT, DNS_QUERY_ID).
    - High uniqueness + AUC well above 0.5 -> it is an identifier that
      encodes testbed topology. Drop, and say so; it will not transfer.
    - L4_DST_PORT: if a handful of ports dominate, bucket or one-hot the
      top-N rather than feeding the raw integer.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--outdir", default="audit_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.csv, args.nrows)

    basic_integrity(df)
    host_structure(df)
    timestamp_checks(df)
    column_profile(df, args.outdir)
    duplicate_pairs(df, args.outdir)
    auc_table = univariate_auc(df, args.outdir)
    justify_noise_columns(df, auc_table)

    print(SECTION.format("DONE"))
    print(f"CSV reports written to {args.outdir}/")
    print("""
Decisions to record in preprocess.py after reading the above:
  1. Confirm the exactly-identical pairs; keep one of each.
  2. Add near-constant columns to DROP_NEAR_CONSTANT.
  3. Note the source-IP-only AUC in your limitations section.
  4. Note the TTL AUC; run the architecture ablation with AND without TTL
     if it is >= 0.90.
  5. Decide window_size from the per-host flow-count table, and decide
     pad-vs-drop from the per-class retention table.""")


if __name__ == "__main__":
    main()
