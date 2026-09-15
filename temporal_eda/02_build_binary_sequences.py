"""Build reproducible source-host-disjoint binary temporal datasets."""

import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "datasets" / "network_traffic.db"
BUCKET_TABLE = "temporal_buckets_10s"
OUTPUT_DIR = REPO_ROOT / "processed_binary_temporal"

BIN_MS = 10_000
SEQUENCE_LENGTH = 10
STRIDE = 1

BENIGN_TEST_FRACTION = 0.20
MIN_BENIGN_TEST_HOSTS = 2
RANDOM_SEED = 42

ROUND_1_TEST_MALICIOUS_IP = "175.45.176.0"
ROUND_2_TEST_MALICIOUS_IP = "175.45.176.1"

ENABLE_AUGMENTATION = False

SOURCE_IP_COL = "ipv4_src_addr"
TIME_COL = "bucket_start_ms"
LABEL_COL = "label"

FEATURE_COLS = [
    "flow_count",
    "dst_ip_count",
    "dst_port_count",
    "src_port_count",
    "in_bytes_sum",
    "out_bytes_sum",
    "in_pkts_sum",
    "out_pkts_sum",
    "duration_mean",
    "duration_min",
    "duration_max",
]

METADATA_COLS = [
    "sequence_id",
    "src_ip",
    "segment_id",
    "start_bucket_ms",
    "end_bucket_ms",
    "label",
    "num_buckets",
    "split",
    "round",
]


def validate_configuration() -> None:
    if BIN_MS <= 0:
        raise ValueError("BIN_MS must be positive.")
    if SEQUENCE_LENGTH <= 0:
        raise ValueError("SEQUENCE_LENGTH must be positive.")
    if STRIDE <= 0:
        raise ValueError("STRIDE must be positive.")
    if not 0 < BENIGN_TEST_FRACTION < 1:
        raise ValueError("BENIGN_TEST_FRACTION must be between 0 and 1.")
    if MIN_BENIGN_TEST_HOSTS < 2:
        raise ValueError("MIN_BENIGN_TEST_HOSTS must be at least 2.")

    forbidden = {SOURCE_IP_COL, TIME_COL, LABEL_COL, "attack"}
    included = forbidden.intersection(FEATURE_COLS)
    if included:
        raise ValueError(
            "FEATURE_COLS contains metadata or target columns: "
            + ", ".join(sorted(included))
        )


def load_buckets() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}\n"
            "Run eda/upload_to_sqlite.py first."
        )

    with sqlite3.connect(DB_PATH) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (BUCKET_TABLE,),
        ).fetchone()
        if not table_exists:
            raise ValueError(
                f"Missing table '{BUCKET_TABLE}'. Run "
                "temporal_eda/01_build_temporal_buckets.sql first."
            )

        buckets = pd.read_sql_query(
            f"SELECT * FROM {BUCKET_TABLE} "
            f"ORDER BY {SOURCE_IP_COL}, {TIME_COL}",
            conn,
        )

    required = {SOURCE_IP_COL, TIME_COL, LABEL_COL, *FEATURE_COLS}
    missing = sorted(required - set(buckets.columns))
    if missing:
        raise ValueError(
            f"Table '{BUCKET_TABLE}' is missing columns: {', '.join(missing)}"
        )
    if buckets.empty:
        raise ValueError(f"Table '{BUCKET_TABLE}' is empty.")
    if buckets[list(required)].isnull().any().any():
        null_columns = buckets[list(required)].columns[
            buckets[list(required)].isnull().any()
        ].tolist()
        raise ValueError(
            "Bucket table contains missing values in: " + ", ".join(null_columns)
        )
    if not buckets[LABEL_COL].isin([0, 1]).all():
        raise ValueError("Bucket labels must contain only 0 and 1.")
    if (buckets[TIME_COL] % BIN_MS != 0).any():
        raise ValueError(f"Some bucket timestamps are not aligned to {BIN_MS} ms.")
    if buckets.duplicated([SOURCE_IP_COL, TIME_COL]).any():
        raise ValueError("Duplicate source-IP/time buckets were found.")

    return buckets


def discover_host_classes(
    buckets: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    label_ranges = buckets.groupby(SOURCE_IP_COL)[LABEL_COL].agg(["min", "max"])
    mixed_mask = label_ranges["min"] != label_ranges["max"]
    if mixed_mask.any():
        mixed_ips = label_ranges.index[mixed_mask].astype(str).tolist()
        raise ValueError(
            "Mixed-label source IPs violate this pipeline's host split assumption: "
            + ", ".join(mixed_ips)
        )

    benign_ips = sorted(
        label_ranges.index[label_ranges["max"] == 0].astype(str).tolist()
    )
    malicious_ips = sorted(
        label_ranges.index[label_ranges["min"] == 1].astype(str).tolist()
    )
    if not benign_ips or not malicious_ips:
        raise ValueError("Both benign-only and malicious-only source IPs are required.")
    return benign_ips, malicious_ips


def iter_segments(host_buckets: pd.DataFrame):
    """Yield truly consecutive bucket segments for one source IP."""
    gap_starts = host_buckets[TIME_COL].diff().ne(BIN_MS)
    segment_ids = gap_starts.cumsum().astype(int)
    for segment_id, segment in host_buckets.groupby(segment_ids, sort=True):
        yield int(segment_id), segment


def count_host_sequences(
    buckets: pd.DataFrame, host_ips: list[str]
) -> dict[str, int]:
    counts = {host_ip: 0 for host_ip in host_ips}
    selected = buckets[buckets[SOURCE_IP_COL].isin(host_ips)]
    for host_ip, host_buckets in selected.groupby(SOURCE_IP_COL, sort=True):
        count = 0
        for _, segment in iter_segments(host_buckets):
            if len(segment) >= SEQUENCE_LENGTH:
                count += 1 + (len(segment) - SEQUENCE_LENGTH) // STRIDE
        counts[str(host_ip)] = count
    return counts


def select_benign_test_hosts(
    sequence_counts: dict[str, int],
) -> tuple[list[str], list[str], int, int]:
    """Find the host subset closest to the target benign sequence fraction."""
    productive_hosts = sorted(
        host for host, count in sequence_counts.items() if count > 0
    )
    if len(productive_hosts) < MIN_BENIGN_TEST_HOSTS:
        raise ValueError(
            f"Only {len(productive_hosts)} benign hosts produce valid sequences; "
            f"at least {MIN_BENIGN_TEST_HOSTS} are required for testing."
        )

    total_sequences = sum(sequence_counts.values())
    target_sequences = total_sequences * BENIGN_TEST_FRACTION

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(productive_hosts).tolist()
    tie_rank = {host: rank for rank, host in enumerate(shuffled)}

    best_key = None
    best_hosts = None
    for host_count in range(MIN_BENIGN_TEST_HOSTS, len(productive_hosts) + 1):
        for combination in itertools.combinations(productive_hosts, host_count):
            selected_sequences = sum(sequence_counts[host] for host in combination)
            seeded_tie_break = tuple(sorted(tie_rank[host] for host in combination))
            key = (
                abs(selected_sequences - target_sequences),
                host_count,
                seeded_tie_break,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_hosts = combination

    test_hosts = sorted(best_hosts)
    train_hosts = sorted(set(sequence_counts) - set(test_hosts))
    test_sequences = sum(sequence_counts[host] for host in test_hosts)
    return train_hosts, test_hosts, total_sequences, test_sequences


def build_sequences(
    split_buckets: pd.DataFrame,
    split_name: str,
    round_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequence_arrays = []
    sequence_labels = []
    metadata_rows = []
    sequence_number = 0

    for host_ip, host_buckets in split_buckets.groupby(SOURCE_IP_COL, sort=True):
        for segment_id, segment in iter_segments(host_buckets):
            if len(segment) < SEQUENCE_LENGTH:
                continue

            feature_values = segment[FEATURE_COLS].to_numpy(dtype=np.float32)
            labels = segment[LABEL_COL].to_numpy(dtype=np.int8)
            times = segment[TIME_COL].to_numpy(dtype=np.int64)

            for start in range(0, len(segment) - SEQUENCE_LENGTH + 1, STRIDE):
                end = start + SEQUENCE_LENGTH
                sequence_number += 1
                sequence_label = int(labels[start:end].max())
                sequence_arrays.append(feature_values[start:end])
                sequence_labels.append(sequence_label)
                metadata_rows.append(
                    {
                        "sequence_id": (
                            f"{round_name}_{split_name}_{sequence_number:08d}"
                        ),
                        "src_ip": str(host_ip),
                        "segment_id": segment_id,
                        "start_bucket_ms": int(times[start]),
                        "end_bucket_ms": int(times[end - 1]),
                        "label": sequence_label,
                        "num_buckets": SEQUENCE_LENGTH,
                        "split": split_name,
                        "round": round_name,
                    }
                )

    if not sequence_arrays:
        raise ValueError(f"No sequences were produced for {round_name} {split_name}.")

    X = np.stack(sequence_arrays).astype(np.float32, copy=False)
    y = np.asarray(sequence_labels, dtype=np.int8)
    metadata = pd.DataFrame(metadata_rows, columns=METADATA_COLS)

    if X.shape[2] != len(FEATURE_COLS):
        raise AssertionError("Unexpected number of model features.")
    if SOURCE_IP_COL in FEATURE_COLS or "src_ip" in FEATURE_COLS:
        raise AssertionError("Source IP must not be present in model features.")
    return X, y, metadata


def augmentation_hook(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not ENABLE_AUGMENTATION:
        return X, y
    raise NotImplementedError(
        "Augmentation must be performed on training raw-flow windows before "
        "bucket aggregation; aggregated sequence vectors cannot be added safely."
    )


def save_split(
    round_dir: Path,
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        round_dir / f"{split_name}.npz",
        X=X,
        y=y,
        feature_names=np.asarray(FEATURE_COLS, dtype=str),
    )
    metadata.to_csv(round_dir / f"{split_name}_metadata.csv", index=False)


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {
        "total": int(len(y)),
        "benign": int((y == 0).sum()),
        "malicious": int((y == 1).sum()),
    }


def build_round(
    buckets: pd.DataFrame,
    round_number: int,
    benign_train_ips: list[str],
    benign_test_ips: list[str],
    malicious_ips: list[str],
    test_malicious_ip: str,
) -> dict:
    round_name = f"round_{round_number}"
    if test_malicious_ip not in malicious_ips:
        raise ValueError(
            f"Configured test malicious IP is absent: {test_malicious_ip}"
        )

    train_malicious_ips = sorted(set(malicious_ips) - {test_malicious_ip})
    train_ips = sorted(set(benign_train_ips) | set(train_malicious_ips))
    test_ips = sorted(set(benign_test_ips) | {test_malicious_ip})
    overlap = set(train_ips).intersection(test_ips)
    if overlap:
        raise AssertionError(
            f"Host overlap in {round_name}: {', '.join(sorted(overlap))}"
        )
    if test_malicious_ip in train_ips:
        raise AssertionError(f"{test_malicious_ip} was included in training.")

    train_buckets = buckets[buckets[SOURCE_IP_COL].isin(train_ips)].copy()
    test_buckets = buckets[buckets[SOURCE_IP_COL].isin(test_ips)].copy()

    X_train, y_train, train_metadata = build_sequences(
        train_buckets, "train", round_name
    )
    X_test, y_test, test_metadata = build_sequences(
        test_buckets, "test", round_name
    )
    X_train, y_train = augmentation_hook(X_train, y_train)

    if len(X_train) != len(train_metadata):
        raise AssertionError("Training metadata no longer matches training arrays.")

    round_dir = OUTPUT_DIR / round_name
    save_split(round_dir, "train", X_train, y_train, train_metadata)
    save_split(round_dir, "test", X_test, y_test, test_metadata)

    train_counts = class_counts(y_train)
    test_counts = class_counts(y_test)
    print(f"\n{round_name.upper()}")
    print(f"Malicious test host: {test_malicious_ip}")
    print(
        f"Train: {len(train_ips)} hosts, {len(train_buckets):,} buckets, "
        f"{train_counts['total']:,} sequences "
        f"({train_counts['benign']:,} benign, "
        f"{train_counts['malicious']:,} malicious)"
    )
    print(
        f"Test:  {len(test_ips)} hosts, {len(test_buckets):,} buckets, "
        f"{test_counts['total']:,} sequences "
        f"({test_counts['benign']:,} benign, "
        f"{test_counts['malicious']:,} malicious)"
    )
    print(f"Host overlap: {len(overlap)}")

    return {
        "test_malicious_ip": test_malicious_ip,
        "train_malicious_ips": train_malicious_ips,
        "train_ips": train_ips,
        "test_ips": test_ips,
        "train_bucket_count": int(len(train_buckets)),
        "test_bucket_count": int(len(test_buckets)),
        "train_sequence_counts": train_counts,
        "test_sequence_counts": test_counts,
        "host_overlap": len(overlap),
    }


def main() -> None:
    validate_configuration()
    buckets = load_buckets()
    benign_ips, malicious_ips = discover_host_classes(buckets)

    for required_ip in [
        ROUND_1_TEST_MALICIOUS_IP,
        ROUND_2_TEST_MALICIOUS_IP,
    ]:
        if required_ip not in malicious_ips:
            raise ValueError(f"Configured malicious test IP is absent: {required_ip}")

    benign_sequence_counts = count_host_sequences(buckets, benign_ips)
    (
        benign_train_ips,
        benign_test_ips,
        total_benign_sequences,
        test_benign_sequences,
    ) = select_benign_test_hosts(benign_sequence_counts)

    actual_test_fraction = test_benign_sequences / total_benign_sequences
    print(f"SQLite DB: {DB_PATH}")
    print(f"Raw bucket table: {BUCKET_TABLE}")
    print(f"Feature count: {len(FEATURE_COLS)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Bucket size: {BIN_MS // 1000} sec")
    print(f"Stride: {STRIDE}")
    print(f"Benign train hosts: {len(benign_train_ips)}")
    print(f"Benign test hosts: {len(benign_test_ips)}")
    print(
        "Benign test sequence target/actual: "
        f"{BENIGN_TEST_FRACTION:.2%} / {actual_test_fraction:.2%}"
    )
    print("Benign test host IPs: " + ", ".join(benign_test_ips))

    round_1 = build_round(
        buckets,
        1,
        benign_train_ips,
        benign_test_ips,
        malicious_ips,
        ROUND_1_TEST_MALICIOUS_IP,
    )
    round_2 = build_round(
        buckets,
        2,
        benign_train_ips,
        benign_test_ips,
        malicious_ips,
        ROUND_2_TEST_MALICIOUS_IP,
    )

    manifest = {
        "database": str(DB_PATH),
        "bucket_table": BUCKET_TABLE,
        "bucket_size_ms": BIN_MS,
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "random_seed": RANDOM_SEED,
        "benign_test_fraction": BENIGN_TEST_FRACTION,
        "benign_test_fraction_target": BENIGN_TEST_FRACTION,
        "benign_test_fraction_actual_by_sequence": actual_test_fraction,
        "benign_split_strategy": (
            "source-IP-disjoint subset closest to target valid-sequence count"
        ),
        "minimum_benign_test_hosts": MIN_BENIGN_TEST_HOSTS,
        "benign_train_ips": benign_train_ips,
        "benign_test_ips": benign_test_ips,
        "benign_sequence_counts_by_ip": benign_sequence_counts,
        "round_1": round_1,
        "round_2": round_2,
        "feature_names": FEATURE_COLS,
        "augmentation_enabled": ENABLE_AUGMENTATION,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
