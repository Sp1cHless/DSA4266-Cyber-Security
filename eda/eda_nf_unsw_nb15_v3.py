"""Direction-neutral temporal EDA for NF-UNSW-NB15-v3.

This script intentionally avoids model fitting and preprocessing decisions. It
validates the real timestamps, describes attacks over time, identifies capture
gaps, and audits whether a chronological train/validation/test split is
feasible.

The CSV is processed in chunks so the complete dataset can be analysed on a
laptop without loading every feature into memory at once.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd


START_COL = "FLOW_START_MILLISECONDS"
END_COL = "FLOW_END_MILLISECONDS"
DURATION_COL = "FLOW_DURATION_MILLISECONDS"
LABEL_COL = "Label"
ATTACK_COL = "Attack"

TEMPORAL_SAMPLE_COLS = [
    DURATION_COL,
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
    LABEL_COL,
    ATTACK_COL,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal EDA on the official NF-UNSW-NB15-v3 CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--feature-dictionary",
        type=Path,
        default=None,
        help="Optional NetFlow_v3_Features.csv path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/temporal_eda"),
        help="Directory for tables, plots, JSON summary, and Markdown report",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument(
        "--sample-per-chunk",
        type=int,
        default=5_000,
        help="Rows sampled per chunk for distribution plots only",
    )
    parser.add_argument(
        "--bucket",
        default="auto",
        help="Pandas time frequency such as 5min or 1h; default chooses from span",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=float,
        default=60.0,
        help="A larger gap starts a new capture session",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def utc_string(milliseconds: int | float | None) -> str | None:
    if milliseconds is None or pd.isna(milliseconds):
        return None
    return pd.to_datetime(int(milliseconds), unit="ms", utc=True).isoformat()


def choose_bucket(span_seconds: float) -> str:
    if span_seconds <= 24 * 3600:
        return "1min"
    if span_seconds <= 3 * 24 * 3600:
        return "5min"
    if span_seconds <= 14 * 24 * 3600:
        return "30min"
    if span_seconds <= 60 * 24 * 3600:
        return "1h"
    return "1D"


def safe_percentage(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def combine_counter_series(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype="int64")
    return pd.concat(parts, axis=1).fillna(0).sum(axis=1).astype("int64")


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna().clip(lower=0).to_numpy(dtype="float64")
    if len(clean) == 0:
        return np.array([]), np.array([])
    x = np.sort(np.log1p(clean))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def jensen_shannon_divergence(
    left: np.ndarray | pd.Series, right: np.ndarray | pd.Series
) -> float:
    """Return base-2 Jensen-Shannon divergence for two count vectors.

    The result is bounded by zero and one. Zero means identical composition;
    one means the distributions have disjoint support.
    """
    left_array = np.asarray(left, dtype="float64")
    right_array = np.asarray(right, dtype="float64")
    if left_array.shape != right_array.shape:
        raise ValueError("Distribution vectors must have the same shape")
    if (left_array < 0).any() or (right_array < 0).any():
        raise ValueError("Distribution vectors cannot contain negative values")
    if left_array.sum() == 0 or right_array.sum() == 0:
        return float("nan")

    left_probability = left_array / left_array.sum()
    right_probability = right_array / right_array.sum()
    midpoint = 0.5 * (left_probability + right_probability)

    def kl_divergence(probability: np.ndarray) -> float:
        nonzero = probability > 0
        return float(
            np.sum(
                probability[nonzero]
                * np.log2(probability[nonzero] / midpoint[nonzero])
            )
        )

    return 0.5 * (
        kl_divergence(left_probability) + kl_divergence(right_probability)
    )


def build_split_drift_table(split_counts: pd.DataFrame) -> pd.DataFrame:
    """Quantify composition shift between chronological data splits."""
    split_order = ["train", "validation", "test"]
    class_counts = split_counts.pivot_table(
        index=ATTACK_COL,
        columns="split",
        values="flow_count",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=split_order, fill_value=0)

    rows: list[dict[str, str | float]] = []
    for scope, scoped_counts in [
        ("all_traffic", class_counts),
        ("attacks_only", class_counts.drop(index="Benign", errors="ignore")),
    ]:
        for left_index, left_split in enumerate(split_order):
            for right_split in split_order[left_index + 1 :]:
                left = scoped_counts[left_split].to_numpy(dtype="float64")
                right = scoped_counts[right_split].to_numpy(dtype="float64")
                left_probability = left / left.sum()
                right_probability = right / right.sum()
                absolute_change = np.abs(left_probability - right_probability)
                largest_index = int(np.argmax(absolute_change))
                rows.append(
                    {
                        "scope": scope,
                        "left_split": left_split,
                        "right_split": right_split,
                        "jensen_shannon_divergence": jensen_shannon_divergence(
                            left, right
                        ),
                        "total_variation_distance": float(
                            0.5 * absolute_change.sum()
                        ),
                        "largest_shift_class": str(scoped_counts.index[largest_index]),
                        "largest_absolute_shift_percentage_points": float(
                            100 * absolute_change[largest_index]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def audit_first_pass(args: argparse.Namespace) -> dict:
    required = {START_COL, END_COL, DURATION_COL, LABEL_COL, ATTACK_COL}
    header = pd.read_csv(args.input, nrows=0)
    missing_required = sorted(required - set(header.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    row_count = 0
    invalid_start = 0
    invalid_end = 0
    end_before_start = 0
    order_inversions = 0
    previous_start: int | None = None
    exact_duration_matches = 0
    within_one_ms = 0
    duration_comparisons = 0
    duration_abs_sum = 0.0
    duration_abs_max = 0.0
    duplicate_hash_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    duration_diff_parts: list[np.ndarray] = []
    sample_parts: list[pd.DataFrame] = []
    missing_parts: list[pd.Series] = []
    inf_parts: list[pd.Series] = []
    binary_counts: Counter = Counter()
    attack_counts: Counter = Counter()
    minute_total_parts: list[pd.DataFrame] = []
    minute_attack_parts: list[pd.DataFrame] = []
    dtypes: pd.Series | None = None

    reader = pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False)
    for chunk_index, chunk in enumerate(reader):
        if dtypes is None:
            dtypes = chunk.dtypes.astype(str)
        row_count += len(chunk)
        missing_parts.append(chunk.isna().sum())

        numeric = chunk.select_dtypes(include=[np.number])
        if not numeric.empty:
            inf_counts = pd.Series(
                np.isinf(numeric.to_numpy(dtype="float64", copy=False)).sum(axis=0),
                index=numeric.columns,
            )
            inf_parts.append(inf_counts)

        binary_counts.update(chunk[LABEL_COL].value_counts(dropna=False).to_dict())
        attack_counts.update(chunk[ATTACK_COL].fillna("<MISSING>").value_counts().to_dict())
        duplicate_hash_parts.append(
            pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
        )

        start = pd.to_numeric(chunk[START_COL], errors="coerce")
        end = pd.to_numeric(chunk[END_COL], errors="coerce")
        duration = pd.to_numeric(chunk[DURATION_COL], errors="coerce")
        invalid_start += int(start.isna().sum())
        invalid_end += int(end.isna().sum())

        valid_start = start.dropna().to_numpy(dtype="int64")
        if len(valid_start):
            if previous_start is not None and valid_start[0] < previous_start:
                order_inversions += 1
            order_inversions += int((np.diff(valid_start) < 0).sum())
            previous_start = int(valid_start[-1])
            start_parts.append(valid_start)

        valid_times = start.notna() & end.notna()
        end_before_start += int((end[valid_times] < start[valid_times]).sum())
        comparable = valid_times & duration.notna()
        if comparable.any():
            observed_delta = end[comparable].to_numpy(dtype="float64") - start[
                comparable
            ].to_numpy(dtype="float64")
            provided_duration = duration[comparable].to_numpy(dtype="float64")
            absolute_difference = np.abs(observed_delta - provided_duration)
            duration_diff_parts.append(absolute_difference)
            duration_comparisons += len(absolute_difference)
            exact_duration_matches += int((absolute_difference == 0).sum())
            within_one_ms += int((absolute_difference <= 1).sum())
            duration_abs_sum += float(absolute_difference.sum())
            duration_abs_max = max(duration_abs_max, float(absolute_difference.max()))

        temporal = pd.DataFrame(
            {
                "minute_ms": (start // 60_000) * 60_000,
                "is_attack": pd.to_numeric(chunk[LABEL_COL], errors="coerce").fillna(0),
                ATTACK_COL: chunk[ATTACK_COL].fillna("<MISSING>"),
            }
        ).dropna(subset=["minute_ms"])
        temporal["minute_ms"] = temporal["minute_ms"].astype("int64")
        minute_total = temporal.groupby("minute_ms", observed=True).agg(
            flow_count=("is_attack", "size"), attack_count=("is_attack", "sum")
        )
        minute_total_parts.append(minute_total.reset_index())
        minute_attack = (
            temporal.groupby(["minute_ms", ATTACK_COL], observed=True)
            .size()
            .rename("flow_count")
            .reset_index()
        )
        minute_attack_parts.append(minute_attack)

        available_sample_cols = [c for c in TEMPORAL_SAMPLE_COLS if c in chunk.columns]
        sample_n = min(args.sample_per_chunk, len(chunk))
        if sample_n > 0:
            sample_parts.append(
                chunk[available_sample_cols].sample(
                    n=sample_n, random_state=args.seed + chunk_index
                )
            )

        print(f"Processed {row_count:,} rows", flush=True)

    if not start_parts:
        raise ValueError("No valid flow start timestamps were found")

    starts = np.concatenate(start_parts)
    sorted_starts = np.sort(starts)
    gaps = np.diff(sorted_starts)
    duration_diffs = (
        np.concatenate(duration_diff_parts) if duration_diff_parts else np.array([])
    )
    row_hashes = np.concatenate(duplicate_hash_parts)
    _, hash_counts = np.unique(row_hashes, return_counts=True)
    exact_duplicate_rows = int(np.maximum(hash_counts - 1, 0).sum())

    minute_total = (
        pd.concat(minute_total_parts, ignore_index=True)
        .groupby("minute_ms", as_index=False)
        .agg(flow_count=("flow_count", "sum"), attack_count=("attack_count", "sum"))
        .sort_values("minute_ms")
    )
    minute_attack = (
        pd.concat(minute_attack_parts, ignore_index=True)
        .groupby(["minute_ms", ATTACK_COL], as_index=False)["flow_count"]
        .sum()
        .sort_values(["minute_ms", ATTACK_COL])
    )

    return {
        "columns": list(header.columns),
        "dtypes": dtypes if dtypes is not None else pd.Series(dtype="object"),
        "row_count": row_count,
        "missing_counts": combine_counter_series(missing_parts),
        "inf_counts": combine_counter_series(inf_parts),
        "binary_counts": binary_counts,
        "attack_counts": attack_counts,
        "starts": starts,
        "sorted_starts": sorted_starts,
        "gaps": gaps,
        "minute_total": minute_total,
        "minute_attack": minute_attack,
        "sample": pd.concat(sample_parts, ignore_index=True),
        "invalid_start": invalid_start,
        "invalid_end": invalid_end,
        "end_before_start": end_before_start,
        "order_inversions": order_inversions,
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_start_timestamps": int((gaps == 0).sum()),
        "duration_comparisons": duration_comparisons,
        "exact_duration_matches": exact_duration_matches,
        "within_one_ms": within_one_ms,
        "duration_abs_mean": (
            duration_abs_sum / duration_comparisons
            if duration_comparisons
            else float("nan")
        ),
        "duration_abs_max": duration_abs_max,
        "duration_diff_quantiles": (
            np.quantile(duration_diffs, [0, 0.5, 0.9, 0.99, 1]).tolist()
            if len(duration_diffs)
            else []
        ),
    }


def build_sessions_and_splits(audit: dict, args: argparse.Namespace) -> dict:
    starts = audit["sorted_starts"]
    gaps = audit["gaps"]
    session_gap_ms = int(args.session_gap_minutes * 60_000)
    break_positions = np.flatnonzero(gaps > session_gap_ms) + 1
    session_first_positions = np.r_[0, break_positions]
    session_last_positions = np.r_[break_positions - 1, len(starts) - 1]
    session_starts = starts[session_first_positions]
    session_ends = starts[session_last_positions]

    validation_boundary = int(starts[math.floor(0.70 * (len(starts) - 1))])
    test_boundary = int(starts[math.floor(0.85 * (len(starts) - 1))])

    split_parts: list[pd.DataFrame] = []
    session_parts: list[pd.DataFrame] = []
    usecols = [START_COL, LABEL_COL, ATTACK_COL]
    reader = pd.read_csv(
        args.input,
        usecols=usecols,
        chunksize=args.chunk_size,
        low_memory=False,
    )
    for chunk in reader:
        start = pd.to_numeric(chunk[START_COL], errors="coerce")
        label = pd.to_numeric(chunk[LABEL_COL], errors="coerce")
        valid = start.notna()
        frame = pd.DataFrame(
            {
                "start_ms": start[valid].astype("int64"),
                LABEL_COL: label[valid],
                ATTACK_COL: chunk.loc[valid, ATTACK_COL].fillna("<MISSING>"),
            }
        )
        frame["split"] = np.select(
            [
                frame["start_ms"] < validation_boundary,
                frame["start_ms"] < test_boundary,
            ],
            ["train", "validation"],
            default="test",
        )
        split_parts.append(
            frame.groupby(["split", ATTACK_COL], observed=True)
            .agg(flow_count=(LABEL_COL, "size"), attack_count=(LABEL_COL, "sum"))
            .reset_index()
        )

        frame["session_id"] = np.searchsorted(
            session_starts[1:], frame["start_ms"].to_numpy(), side="right"
        ) + 1
        session_parts.append(
            frame.groupby("session_id", observed=True)
            .agg(flow_count=(LABEL_COL, "size"), attack_count=(LABEL_COL, "sum"))
            .reset_index()
        )

    split_counts = (
        pd.concat(split_parts, ignore_index=True)
        .groupby(["split", ATTACK_COL], as_index=False)
        .agg(flow_count=("flow_count", "sum"), attack_count=("attack_count", "sum"))
    )
    session_counts = (
        pd.concat(session_parts, ignore_index=True)
        .groupby("session_id", as_index=False)
        .agg(flow_count=("flow_count", "sum"), attack_count=("attack_count", "sum"))
    )
    sessions = pd.DataFrame(
        {
            "session_id": np.arange(1, len(session_starts) + 1),
            "start_ms": session_starts,
            "end_ms": session_ends,
            "start_utc": [utc_string(x) for x in session_starts],
            "end_utc": [utc_string(x) for x in session_ends],
            "span_minutes": (session_ends - session_starts) / 60_000,
        }
    ).merge(session_counts, on="session_id", how="left")
    sessions["attack_percentage"] = (
        100 * sessions["attack_count"] / sessions["flow_count"]
    )

    return {
        "validation_boundary": validation_boundary,
        "test_boundary": test_boundary,
        "split_counts": split_counts,
        "sessions": sessions,
        "session_gap_ms": session_gap_ms,
    }


def build_gap_table(sorted_starts: np.ndarray, gaps: np.ndarray) -> pd.DataFrame:
    if len(gaps) == 0:
        return pd.DataFrame(
            columns=["previous_utc", "next_utc", "gap_seconds", "gap_minutes"]
        )
    top_indices = np.argsort(gaps)[-20:][::-1]
    return pd.DataFrame(
        {
            "previous_utc": [utc_string(sorted_starts[i]) for i in top_indices],
            "next_utc": [utc_string(sorted_starts[i + 1]) for i in top_indices],
            "gap_seconds": gaps[top_indices] / 1_000,
            "gap_minutes": gaps[top_indices] / 60_000,
        }
    )


def prepare_time_tables(audit: dict, bucket: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    minute_total = audit["minute_total"].copy()
    minute_total["timestamp_utc"] = pd.to_datetime(
        minute_total["minute_ms"], unit="ms", utc=True
    )
    time_summary = (
        minute_total.set_index("timestamp_utc")[["flow_count", "attack_count"]]
        .resample(bucket)
        .sum()
    )
    time_summary["attack_percentage"] = np.where(
        time_summary["flow_count"] > 0,
        100 * time_summary["attack_count"] / time_summary["flow_count"],
        np.nan,
    )
    time_summary = time_summary.reset_index()

    minute_attack = audit["minute_attack"].copy()
    minute_attack["timestamp_utc"] = pd.to_datetime(
        minute_attack["minute_ms"], unit="ms", utc=True
    )
    attack_pivot = minute_attack.pivot_table(
        index="timestamp_utc",
        columns=ATTACK_COL,
        values="flow_count",
        aggfunc="sum",
        fill_value=0,
    )
    attack_time = attack_pivot.resample(bucket).sum().reset_index()
    return time_summary, attack_time


def save_tables(
    audit: dict,
    split_audit: dict,
    args: argparse.Namespace,
) -> None:
    tables = args.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    descriptions = pd.DataFrame(columns=["Feature", "Description"])
    if args.feature_dictionary and args.feature_dictionary.exists():
        descriptions = pd.read_csv(args.feature_dictionary, encoding_errors="replace")
        descriptions["Feature"] = descriptions["Feature"].astype(str).str.strip()

    schema = pd.DataFrame(
        {
            "Feature": audit["columns"],
            "dtype": [audit["dtypes"].get(c, "unknown") for c in audit["columns"]],
            "missing_count": [audit["missing_counts"].get(c, 0) for c in audit["columns"]],
            "infinite_count": [audit["inf_counts"].get(c, 0) for c in audit["columns"]],
        }
    ).merge(descriptions, on="Feature", how="left")

    schema.to_csv(tables / "schema_and_quality.csv", index=False)
    split_audit["split_counts"].to_csv(
        tables / "chronological_split_class_counts.csv", index=False
    )
    split_audit["sessions"].to_csv(tables / "capture_sessions.csv", index=False)
    build_split_drift_table(split_audit["split_counts"]).to_csv(
        tables / "chronological_split_distribution_shift.csv", index=False
    )


def save_plots(
    audit: dict,
    split_audit: dict,
    time_summary: pd.DataFrame,
    attack_time: pd.DataFrame,
    bucket: str,
    args: argparse.Namespace,
) -> None:
    import os

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    attack_distribution = pd.DataFrame(
        list(audit["attack_counts"].items()), columns=[ATTACK_COL, "flow_count"]
    ).sort_values("flow_count")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C78A8" if x == "Benign" else "#E45756" for x in attack_distribution[ATTACK_COL]]
    ax.barh(attack_distribution[ATTACK_COL], attack_distribution["flow_count"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Flow count (log scale)")
    ax.set_ylabel("")
    ax.set_title("NF-UNSW-NB15-v3 class distribution")
    for index, value in enumerate(attack_distribution["flow_count"]):
        ax.text(value * 1.05, index, f"{value:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sessions = split_audit["sessions"]
    session_count = len(sessions)
    fig, axes = plt.subplots(
        2,
        session_count,
        figsize=(7 * session_count, 8),
        squeeze=False,
        sharey="row",
    )
    for column_index, session in sessions.iterrows():
        session_start = pd.to_datetime(session["start_ms"], unit="ms", utc=True)
        session_end = pd.to_datetime(session["end_ms"], unit="ms", utc=True)
        in_session = (
            (time_summary["timestamp_utc"] >= session_start)
            & (time_summary["timestamp_utc"] <= session_end)
            & (time_summary["flow_count"] > 0)
        )
        session_time = time_summary.loc[in_session]
        axes[0, column_index].plot(
            session_time["timestamp_utc"],
            session_time["flow_count"],
            color="#4C78A8",
            linewidth=1.1,
        )
        axes[0, column_index].set_title(
            f"Session {int(session['session_id'])}: "
            f"{session_start.strftime('%Y-%m-%d')}\n"
            f"overall attack rate {session['attack_percentage']:.2f}%"
        )
        axes[0, column_index].set_xlabel("UTC time")
        axes[0, column_index].tick_params(axis="x", rotation=25)
        axes[1, column_index].plot(
            session_time["timestamp_utc"],
            session_time["attack_percentage"],
            color="#E45756",
            linewidth=1.1,
        )
        axes[1, column_index].set_xlabel("UTC time")
        axes[1, column_index].set_ylim(-2, 102)
        axes[1, column_index].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel(f"Flows per {bucket}")
    axes[1, 0].set_ylabel("Attack flows (%)")
    fig.suptitle(
        "Traffic volume and attack prevalence within each capture session",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(plots / "traffic_and_attack_rate_over_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    attack_columns = [c for c in attack_time.columns if c not in {"timestamp_utc", "Benign"}]
    if attack_columns:
        heat = attack_time.set_index("timestamp_utc")[attack_columns].T
        nonempty_columns = heat.sum(axis=0) > 0
        heat = heat.loc[:, nonempty_columns]
        fig_width = max(12, min(22, heat.shape[1] / 30))
        fig, ax = plt.subplots(figsize=(fig_width, 7))
        sns.heatmap(
            np.log1p(heat),
            cmap="mako",
            ax=ax,
            xticklabels=False,
            cbar_kws={"label": "log(1 + flows)"},
        )
        tick_count = min(10, heat.shape[1])
        if tick_count:
            tick_indices = np.linspace(0, heat.shape[1] - 1, tick_count).astype(int)
            ax.set_xticks(tick_indices + 0.5)
            ax.set_xticklabels(
                [heat.columns[i].strftime("%m-%d %H:%M") for i in tick_indices],
                rotation=30,
                ha="right",
            )
        ax.set_title(f"Attack-category activity by {bucket} bucket")
        ax.set_xlabel("Non-empty chronological time buckets")
        ax.set_ylabel("Attack category")
        fig.tight_layout()
        fig.savefig(plots / "attack_categories_over_time.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    sample = audit["sample"].copy()
    temporal_features = [
        DURATION_COL,
        "SRC_TO_DST_IAT_AVG",
        "DST_TO_SRC_IAT_AVG",
    ]
    temporal_features = [c for c in temporal_features if c in sample.columns]
    if temporal_features:
        fig, axes = plt.subplots(1, len(temporal_features), figsize=(6 * len(temporal_features), 5))
        if len(temporal_features) == 1:
            axes = [axes]
        for ax, feature in zip(axes, temporal_features):
            for label_value, label_name, color in [(0, "Benign", "#4C78A8"), (1, "Attack", "#E45756")]:
                x, y = ecdf(sample.loc[sample[LABEL_COL] == label_value, feature])
                if len(x):
                    ax.plot(x, y, label=label_name, color=color)
            ax.set_title(feature)
            ax.set_xlabel("log(1 + value)")
            ax.set_ylabel("Empirical cumulative probability")
            ax.legend()
        fig.suptitle("Temporal feature distributions (reproducible sample)", y=1.02)
        fig.tight_layout()
        fig.savefig(plots / "duration_and_iat_ecdf.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    split_counts = split_audit["split_counts"].pivot_table(
        index=ATTACK_COL,
        columns="split",
        values="flow_count",
        aggfunc="sum",
        fill_value=0,
    )
    split_counts = split_counts.reindex(columns=["train", "validation", "test"], fill_value=0)
    proportions = split_counts.div(split_counts.sum(axis=0), axis=1) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(proportions, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"label": "% within split"})
    ax.set_title("Class composition under chronological 70/15/15 split")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(plots / "chronological_split_class_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    drift = build_split_drift_table(split_audit["split_counts"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    pair_order = ["train → validation", "train → test", "validation → test"]
    for ax, (scope, title) in zip(
        axes,
        [
            ("all_traffic", "All traffic"),
            ("attacks_only", "Attack classes only"),
        ],
    ):
        scoped = drift.loc[drift["scope"] == scope].copy()
        scoped["comparison"] = (
            scoped["left_split"] + " → " + scoped["right_split"]
        )
        scoped["comparison"] = pd.Categorical(
            scoped["comparison"], categories=pair_order, ordered=True
        )
        scoped = scoped.sort_values("comparison")
        bars = ax.barh(
            scoped["comparison"],
            scoped["jensen_shannon_divergence"],
            color="#F58518" if scope == "all_traffic" else "#54A24B",
        )
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(title)
        ax.set_xlabel("Jensen–Shannon divergence (base 2)")
        ax.set_xlim(0, max(0.02, drift["jensen_shannon_divergence"].max() * 1.25))
    axes[0].set_ylabel("")
    fig.suptitle("Composition shift between chronological splits")
    fig.tight_layout()
    fig.savefig(
        plots / "chronological_split_distribution_shift.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary_and_report(
    audit: dict,
    split_audit: dict,
    gap_table: pd.DataFrame,
    bucket: str,
    args: argparse.Namespace,
) -> None:
    starts = audit["sorted_starts"]
    span_seconds = (int(starts[-1]) - int(starts[0])) / 1_000
    attack_total = int(sum(v for k, v in audit["binary_counts"].items() if str(k) in {"1", "1.0"} or k == 1))
    gap_thresholds = {
        "over_1_minute": int((audit["gaps"] > 60_000).sum()),
        "over_10_minutes": int((audit["gaps"] > 600_000).sum()),
        "over_1_hour": int((audit["gaps"] > 3_600_000).sum()),
    }
    missing_feature_counts = {
        str(feature): int(count)
        for feature, count in audit["missing_counts"].items()
        if count > 0
    }
    infinite_feature_counts = {
        str(feature): int(count)
        for feature, count in audit["inf_counts"].items()
        if count > 0
    }
    missing_feature_text = ", ".join(
        f"`{feature}` ({count:,})"
        for feature, count in missing_feature_counts.items()
    ) or "none"
    infinite_feature_text = ", ".join(
        f"`{feature}` ({count:,})"
        for feature, count in infinite_feature_counts.items()
    ) or "none"

    split_counts = split_audit["split_counts"]
    split_totals = split_counts.groupby("split")["flow_count"].sum().to_dict()
    split_attack_totals = split_counts.groupby("split")["attack_count"].sum().to_dict()
    split_attack_rates = {
        split: safe_percentage(split_attack_totals.get(split, 0), total)
        for split, total in split_totals.items()
    }
    attacks_by_split = {
        split: group.set_index(ATTACK_COL)["flow_count"].astype(int).to_dict()
        for split, group in split_counts.groupby("split")
    }
    all_attack_classes = sorted(audit["attack_counts"].keys())
    missing_classes = {
        split: [c for c in all_attack_classes if attacks_by_split.get(split, {}).get(c, 0) == 0]
        for split in ["train", "validation", "test"]
    }

    sample_temporal_medians: dict[str, dict[str, float | None]] = {}
    sample = audit["sample"]
    for feature in [DURATION_COL, "SRC_TO_DST_IAT_AVG", "DST_TO_SRC_IAT_AVG"]:
        if feature not in sample.columns:
            continue
        sample_temporal_medians[feature] = {}
        for label_value, label_name in [(0, "benign"), (1, "attack")]:
            values = pd.to_numeric(
                sample.loc[sample[LABEL_COL] == label_value, feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()
            sample_temporal_medians[feature][label_name] = (
                float(values.median()) if len(values) else None
            )

    split_drift = build_split_drift_table(split_counts)
    train_test_drift = split_drift.loc[
        (split_drift["left_split"] == "train")
        & (split_drift["right_split"] == "test")
    ].set_index("scope")
    validation_test_drift = split_drift.loc[
        (split_drift["left_split"] == "validation")
        & (split_drift["right_split"] == "test")
    ].set_index("scope")

    summary = {
        # Store only the filename so generated artifacts never expose a contributor's
        # local username or checkout location.
        "input_file": args.input.name,
        "row_count": audit["row_count"],
        "column_count": len(audit["columns"]),
        "capture_start_utc": utc_string(starts[0]),
        "capture_end_utc": utc_string(starts[-1]),
        "calendar_span_days": span_seconds / 86_400,
        "chosen_time_bucket": bucket,
        "binary_class_counts": {str(k): int(v) for k, v in audit["binary_counts"].items()},
        "attack_class_counts": {str(k): int(v) for k, v in audit["attack_counts"].items()},
        "attack_percentage": safe_percentage(attack_total, audit["row_count"]),
        "invalid_start_timestamps": audit["invalid_start"],
        "invalid_end_timestamps": audit["invalid_end"],
        "flows_with_end_before_start": audit["end_before_start"],
        "total_missing_cells": int(sum(missing_feature_counts.values())),
        "features_with_missing_values": missing_feature_counts,
        "total_infinite_cells": int(sum(infinite_feature_counts.values())),
        "features_with_infinite_values": infinite_feature_counts,
        "original_row_order_inversions": audit["order_inversions"],
        "duplicate_start_timestamps": audit["duplicate_start_timestamps"],
        "exact_duplicate_rows_by_64bit_hash": audit["exact_duplicate_rows"],
        "duration_comparisons": audit["duration_comparisons"],
        "duration_exact_match_percentage": safe_percentage(
            audit["exact_duration_matches"], audit["duration_comparisons"]
        ),
        "duration_within_one_ms_percentage": safe_percentage(
            audit["within_one_ms"], audit["duration_comparisons"]
        ),
        "duration_absolute_difference_mean_ms": audit["duration_abs_mean"],
        "duration_absolute_difference_max_ms": audit["duration_abs_max"],
        "duration_absolute_difference_quantiles_ms": audit["duration_diff_quantiles"],
        "timestamp_gap_counts": gap_thresholds,
        "capture_session_count": int(len(split_audit["sessions"])),
        "session_gap_definition_minutes": args.session_gap_minutes,
        "split_boundaries_utc": {
            "validation_starts": utc_string(split_audit["validation_boundary"]),
            "test_starts": utc_string(split_audit["test_boundary"]),
        },
        "split_row_counts": {str(k): int(v) for k, v in split_totals.items()},
        "split_attack_percentages": {
            str(k): float(v) for k, v in split_attack_rates.items()
        },
        "classes_missing_from_splits": missing_classes,
        "capture_session_attack_percentages": {
            str(int(row.session_id)): float(row.attack_percentage)
            for row in split_audit["sessions"].itertuples()
        },
        "largest_gap_minutes": (
            float(gap_table.iloc[0]["gap_minutes"]) if len(gap_table) else 0.0
        ),
        "sample_temporal_feature_medians": sample_temporal_medians,
        "chronological_split_train_test_drift": {
            scope: {
                "jensen_shannon_divergence": float(
                    train_test_drift.loc[scope, "jensen_shannon_divergence"]
                ),
                "total_variation_distance": float(
                    train_test_drift.loc[scope, "total_variation_distance"]
                ),
                "largest_shift_class": str(
                    train_test_drift.loc[scope, "largest_shift_class"]
                ),
                "largest_absolute_shift_percentage_points": float(
                    train_test_drift.loc[
                        scope, "largest_absolute_shift_percentage_points"
                    ]
                ),
            }
            for scope in train_test_drift.index
        },
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    missing_lines = []
    for split, classes in missing_classes.items():
        missing_lines.append(
            f"- **{split}:** {', '.join(classes) if classes else 'none'}"
        )
    plot_insights = f"""# Temporal EDA plot insights

## `class_distribution.png`

Benign traffic accounts for {safe_percentage(audit['attack_counts'].get('Benign', 0), audit['row_count']):.2f}% of all flows. Exploits and Fuzzers are the largest attack classes, while Worms has only {audit['attack_counts'].get('Worms', 0):,} observations. The extreme imbalance means accuracy alone would be misleading; per-class recall, macro-F1, PR-AUC, and the confusion matrix should be reported.

## `traffic_and_attack_rate_over_time.png`

Traffic comes from two roughly 12-hour capture sessions separated by a {summary['largest_gap_minutes'] / 1_440:.2f}-day gap. Attack prevalence rises from {summary['capture_session_attack_percentages'].get('1', float('nan')):.2f}% in session 1 to {summary['capture_session_attack_percentages'].get('2', float('nan')):.2f}% in session 2. The gap must not be interpolated as zero traffic, and this session shift makes a purely random split liable to overstate generalization.

## `attack_categories_over_time.png`

Attack activity is concentrated in temporal bursts rather than being uniformly distributed. Several attack categories are active in nearby time buckets, so randomly placing neighboring flows into both training and testing can leak capture-specific patterns. A chronological or session-aware evaluation is therefore more defensible.

## `duration_and_iat_ecdf.png`

In the reproducible sample, the median flow duration is {sample_temporal_medians.get(DURATION_COL, {}).get('benign', float('nan')):.1f} ms for benign traffic and {sample_temporal_medians.get(DURATION_COL, {}).get('attack', float('nan')):.1f} ms for attacks. Median source-to-destination IAT is {sample_temporal_medians.get('SRC_TO_DST_IAT_AVG', {}).get('benign', float('nan')):.1f} versus {sample_temporal_medians.get('SRC_TO_DST_IAT_AVG', {}).get('attack', float('nan')):.1f}, and destination-to-source IAT is {sample_temporal_medians.get('DST_TO_SRC_IAT_AVG', {}).get('benign', float('nan')):.1f} versus {sample_temporal_medians.get('DST_TO_SRC_IAT_AVG', {}).get('attack', float('nan')):.1f}. These temporal features contain useful separation, but the overlapping ECDFs show that none should be treated as a standalone attack rule.

## `chronological_split_class_composition.png`

All attack classes appear in the candidate train, validation, and test periods, but prevalence shifts from {split_attack_rates.get('train', float('nan')):.2f}% in training to {split_attack_rates.get('validation', float('nan')):.2f}% and {split_attack_rates.get('test', float('nan')):.2f}% in validation and test. Worms remains especially sparse ({attacks_by_split.get('train', {}).get('Worms', 0):,}/{attacks_by_split.get('validation', {}).get('Worms', 0):,}/{attacks_by_split.get('test', {}).get('Worms', 0):,} flows), so its class-specific metric will be unstable and should be interpreted with counts or confidence intervals.

## `chronological_split_distribution_shift.png`

The train-to-test Jensen–Shannon divergence is {train_test_drift.loc['all_traffic', 'jensen_shannon_divergence']:.4f} across all traffic and {train_test_drift.loc['attacks_only', 'jensen_shannon_divergence']:.4f} among attack classes. Reporting both prevents the dominant benign class from hiding changes in the attack mixture. Validation and test are very similar overall ({validation_test_drift.loc['all_traffic', 'jensen_shannon_divergence']:.4f}), although their attack mixtures differ more ({validation_test_drift.loc['attacks_only', 'jensen_shannon_divergence']:.4f}), mainly because the {validation_test_drift.loc['attacks_only', 'largest_shift_class']} share changes by {validation_test_drift.loc['attacks_only', 'largest_absolute_shift_percentage_points']:.2f} percentage points. These values quantify dataset shift; they are descriptive diagnostics rather than universal pass/fail thresholds.
"""
    report = f"""# NF-UNSW-NB15-v3 temporal EDA

## Dataset and timestamp integrity

- **Rows / columns:** {audit['row_count']:,} / {len(audit['columns'])}
- **Capture-time range:** {summary['capture_start_utc']} to {summary['capture_end_utc']}
- **Calendar span:** {summary['calendar_span_days']:.2f} days
- **Attack prevalence:** {summary['attack_percentage']:.2f}%
- **Invalid start/end timestamps:** {audit['invalid_start']:,} / {audit['invalid_end']:,}
- **Flows ending before they start:** {audit['end_before_start']:,}
- **Missing cells:** {summary['total_missing_cells']:,} across {len(missing_feature_counts):,} feature(s)
- **Infinite numeric cells:** {summary['total_infinite_cells']:,} across {len(infinite_feature_counts):,} feature(s)
- **Original-order timestamp inversions:** {audit['order_inversions']:,}
- **Exact duplicate rows (64-bit row hash):** {audit['exact_duplicate_rows']:,}
- **Duration agrees within 1 ms:** {summary['duration_within_one_ms_percentage']:.2f}%

The original CSV row order should{' not' if audit['order_inversions'] else ''} be treated as chronological. All temporal analysis in this workflow sorts or groups by the real flow-start timestamp.

Missing values occur in {missing_feature_text}; infinite values occur in {infinite_feature_text}. These non-finite values are confined to the two per-second byte-rate features in this dataset, but they must be handled using a train-fitted preprocessing rule before modeling. They should not be silently passed into a scaler or estimator.

## Capture continuity

- **Time bucket used for overview plots:** `{bucket}`
- **Gaps exceeding one minute:** {gap_thresholds['over_1_minute']:,}
- **Gaps exceeding one hour:** {gap_thresholds['over_1_hour']:,}
- **Largest observed gap:** {summary['largest_gap_minutes']:.2f} minutes
- **Capture sessions using a {args.session_gap_minutes:g}-minute gap rule:** {summary['capture_session_count']:,}
- **Session 1 attack prevalence:** {summary['capture_session_attack_percentages'].get('1', float('nan')):.2f}%
- **Session 2 attack prevalence:** {summary['capture_session_attack_percentages'].get('2', float('nan')):.2f}%

Large gaps mean the calendar range is not one continuous monitoring period. The substantial change in attack prevalence between sessions is direct evidence of temporal distribution shift. Refer to `tables/capture_sessions.csv` before defining any temporal split or interpreting uncaptured time as benign traffic.

## Candidate chronological split

- **Train:** before {summary['split_boundaries_utc']['validation_starts']} ({split_totals.get('train', 0):,} flows; {split_attack_rates.get('train', float('nan')):.2f}% attacks)
- **Validation:** until {summary['split_boundaries_utc']['test_starts']} ({split_totals.get('validation', 0):,} flows; {split_attack_rates.get('validation', float('nan')):.2f}% attacks)
- **Test:** thereafter ({split_totals.get('test', 0):,} flows; {split_attack_rates.get('test', float('nan')):.2f}% attacks)

Classes absent from each split:

{chr(10).join(missing_lines)}

This split is a feasibility audit, not a finalized modeling decision. If rare attack classes are absent or concentrated in only one period, the team must decide whether to evaluate binary detection, redesign boundaries around capture sessions, or explicitly study unseen-class generalization.

## Plot-by-plot findings

{plot_insights.removeprefix('# Temporal EDA plot insights').strip()}

## Files to review before Wednesday

1. `plots/class_distribution.png`
2. `plots/traffic_and_attack_rate_over_time.png`
3. `plots/attack_categories_over_time.png`
4. `plots/duration_and_iat_ecdf.png`
5. `plots/chronological_split_class_composition.png`
6. `plots/chronological_split_distribution_shift.png`
7. `tables/capture_sessions.csv`
8. `tables/chronological_split_class_counts.csv`
9. `tables/chronological_split_distribution_shift.csv`
10. `dataset_summary.json`

No scaling, balancing, feature selection, model fitting, or synthetic timestamp construction is performed here.
"""
    (args.output_dir / "temporal_eda_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.feature_dictionary:
        args.feature_dictionary = args.feature_dictionary.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_first_pass(args)
    split_audit = build_sessions_and_splits(audit, args)
    gap_table = build_gap_table(audit["sorted_starts"], audit["gaps"])
    longest_session_seconds = (
        float(split_audit["sessions"]["span_minutes"].max()) * 60
    )
    bucket = (
        choose_bucket(longest_session_seconds) if args.bucket == "auto" else args.bucket
    )
    time_summary, attack_time = prepare_time_tables(audit, bucket)

    save_tables(audit, split_audit, args)
    save_plots(audit, split_audit, time_summary, attack_time, bucket, args)
    write_summary_and_report(audit, split_audit, gap_table, bucket, args)
    print(f"Temporal EDA complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
