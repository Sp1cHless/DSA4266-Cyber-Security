"""Build 10-second source-IP buckets with categorical feature engineering."""

import json
import math
import re
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "datasets" / "network_traffic.db"
OUTPUT_DIR = REPO_ROOT / "processed_binary_temporal"
SCHEMA_PATH = OUTPUT_DIR / "categorical_schema.json"

RAW_TABLE = "network_flows"
BUCKET_TABLE = "temporal_buckets_10s_full"
BIN_MS = 10_000
L7_MIN_GLOBAL_COUNT = 1_000

NUMERICAL_FEATURES = [
    "flow_count",
    "in_bytes_sum",
    "out_bytes_sum",
    "in_pkts_sum",
    "out_pkts_sum",
    "duration_mean",
    "duration_min",
    "duration_max",
]

TCP_FLAG_BITS = {
    "fin": 0x01,
    "syn": 0x02,
    "rst": 0x04,
    "psh": 0x08,
    "ack": 0x10,
    "urg": 0x20,
    "ece": 0x40,
    "cwr": 0x80,
}


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def canonical_category(value: float) -> str:
    return format(value, ".15g")


def safe_category_name(category: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_")
    if not safe:
        raise ValueError(f"Cannot create a safe column name for L7 category {category!r}.")
    return safe


def ensure_raw_table(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (RAW_TABLE,),
    ).fetchone()
    if not exists:
        raise ValueError(
            f"Missing table '{RAW_TABLE}'. Run eda/upload_to_sqlite.py first."
        )


def discover_l7_categories(conn: sqlite3.Connection) -> list[tuple[float, str, str]]:
    rows = conn.execute(
        f"""
        SELECT l7_proto, COUNT(*) AS global_count
        FROM {quote_identifier(RAW_TABLE)}
        WHERE l7_proto IS NOT NULL
        GROUP BY l7_proto
        HAVING COUNT(*) >= ?
        ORDER BY l7_proto
        """,
        (L7_MIN_GLOBAL_COUNT,),
    ).fetchall()

    categories = []
    used_names = set()
    for raw_value, _ in rows:
        category = canonical_category(raw_value)
        safe_name = safe_category_name(category)
        if safe_name in used_names:
            raise ValueError(
                "L7 categories produce duplicate safe column names: " + safe_name
            )
        used_names.add(safe_name)
        categories.append((raw_value, category, safe_name))
    return categories


def distribution_table_sql(
    table_name: str,
    value_column: str,
    feature_prefix: str,
    where_clause: str = "1 = 1",
    top1_uses_total_flow_count: bool = True,
) -> str:
    """Return SQL for nunique, top-1 ratio, and normalized entropy."""
    top1_denominator = (
        "MAX(b.flow_count)"
        if top1_uses_total_flow_count
        else "MAX(d.relevant_count)"
    )
    return f"""
    DROP TABLE IF EXISTS temp.{quote_identifier(table_name)};
    CREATE TEMP TABLE {quote_identifier(table_name)} AS
    WITH category_counts AS (
        SELECT
            ipv4_src_addr,
            bucket_start_ms,
            {quote_identifier(value_column)} AS category,
            COUNT(*) AS category_count
        FROM flow_rows
        WHERE ({where_clause}) AND {quote_identifier(value_column)} IS NOT NULL
        GROUP BY ipv4_src_addr, bucket_start_ms, {quote_identifier(value_column)}
    ),
    distributions AS (
        SELECT
            ipv4_src_addr,
            bucket_start_ms,
            category_count,
            SUM(category_count) OVER (
                PARTITION BY ipv4_src_addr, bucket_start_ms
            ) AS relevant_count,
            COUNT(*) OVER (
                PARTITION BY ipv4_src_addr, bucket_start_ms
            ) AS category_nunique
        FROM category_counts
    )
    SELECT
        d.ipv4_src_addr,
        d.bucket_start_ms,
        MAX(d.category_nunique) AS {feature_prefix}_nunique,
        1.0 * MAX(d.category_count) / {top1_denominator}
            AS {feature_prefix}_top1_ratio,
        CASE
            WHEN MAX(d.category_nunique) <= 1 THEN 0.0
            ELSE -SUM(
                (1.0 * d.category_count / d.relevant_count)
                * ln(1.0 * d.category_count / d.relevant_count)
            ) / ln(MAX(d.category_nunique))
        END AS {feature_prefix}_entropy
    FROM distributions AS d
    JOIN base_aggregates AS b
        USING (ipv4_src_addr, bucket_start_ms)
    GROUP BY d.ipv4_src_addr, d.bucket_start_ms;

    CREATE INDEX {quote_identifier('idx_' + table_name + '_key')}
    ON {quote_identifier(table_name)} (ipv4_src_addr, bucket_start_ms);
    """


def build_flow_rows(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        DROP TABLE IF EXISTS temp.flow_rows;
        CREATE TEMP TABLE flow_rows AS
        SELECT
            *,
            (flow_start_milliseconds / {BIN_MS}) * {BIN_MS} AS bucket_start_ms
        FROM {quote_identifier(RAW_TABLE)};

        CREATE INDEX idx_flow_rows_key
        ON flow_rows (ipv4_src_addr, bucket_start_ms);
        """
    )


def build_distribution_helpers(conn: sqlite3.Connection) -> None:
    helpers = [
        distribution_table_sql(
            "l7_distribution", "l7_proto", "l7_proto"
        ),
        distribution_table_sql(
            "src_port_distribution", "l4_src_port", "src_port"
        ),
        distribution_table_sql(
            "dst_port_distribution", "l4_dst_port", "dst_port"
        ),
        distribution_table_sql(
            "dst_ip_distribution", "ipv4_dst_addr", "dst_ip"
        ),
        distribution_table_sql(
            "icmp_distribution",
            "icmp_ipv4_type",
            "icmp_type",
            "protocol = 1",
            top1_uses_total_flow_count=False,
        ),
        distribution_table_sql(
            "dns_distribution",
            "dns_query_type",
            "dns_query_type",
            "COALESCE(dns_query_id, 0) != 0 "
            "OR COALESCE(dns_query_type, 0) != 0 "
            "OR COALESCE(dns_ttl_answer, 0) != 0",
            top1_uses_total_flow_count=False,
        ),
    ]
    for sql in helpers:
        conn.executescript(sql)


def l7_ratio_expressions(
    categories: list[tuple[float, str, str]],
) -> tuple[list[str], list[str]]:
    expressions = []
    feature_names = []
    for raw_value, _, safe_name in categories:
        feature_name = f"l7_{safe_name}_ratio"
        expressions.append(
            "1.0 * SUM(CASE WHEN l7_proto = "
            f"{raw_value!r} THEN 1 ELSE 0 END) / COUNT(*) "
            f"AS {quote_identifier(feature_name)}"
        )
        feature_names.append(feature_name)

    if categories:
        placeholders = ", ".join(repr(value) for value, _, _ in categories)
        other_condition = f"l7_proto NOT IN ({placeholders})"
    else:
        other_condition = "l7_proto IS NOT NULL"
    expressions.append(
        "1.0 * SUM(CASE WHEN l7_proto IS NOT NULL AND "
        f"{other_condition} THEN 1 ELSE 0 END) / COUNT(*) AS l7_other_ratio"
    )
    feature_names.append("l7_other_ratio")
    return expressions, feature_names


def tcp_flag_expressions() -> tuple[list[str], list[str]]:
    expressions = []
    feature_names = []
    for raw_column, prefix in [
        ("tcp_flags", "tcp_flag"),
        ("client_tcp_flags", "client_tcp_flag"),
        ("server_tcp_flags", "server_tcp_flag"),
    ]:
        for flag_name, bit in TCP_FLAG_BITS.items():
            feature_name = f"{prefix}_{flag_name}_ratio"
            expressions.append(
                "CASE WHEN SUM(CASE WHEN protocol = 6 THEN 1 ELSE 0 END) = 0 "
                "THEN 0.0 ELSE 1.0 * SUM(CASE WHEN protocol = 6 AND "
                f"(COALESCE({raw_column}, 0) & {bit}) != 0 THEN 1 ELSE 0 END) "
                "/ SUM(CASE WHEN protocol = 6 THEN 1 ELSE 0 END) END "
                f"AS {quote_identifier(feature_name)}"
            )
            feature_names.append(feature_name)
    return expressions, feature_names


def build_base_aggregates(
    conn: sqlite3.Connection,
    l7_categories: list[tuple[float, str, str]],
) -> tuple[list[str], list[str]]:
    l7_expressions, l7_features = l7_ratio_expressions(l7_categories)
    tcp_expressions, tcp_features = tcp_flag_expressions()
    extra_expressions = l7_expressions + tcp_expressions

    sql = f"""
    DROP TABLE IF EXISTS temp.base_aggregates;
    CREATE TEMP TABLE base_aggregates AS
    SELECT
        ipv4_src_addr,
        bucket_start_ms,
        COUNT(*) AS flow_count,
        SUM(in_bytes) AS in_bytes_sum,
        SUM(out_bytes) AS out_bytes_sum,
        SUM(in_pkts) AS in_pkts_sum,
        SUM(out_pkts) AS out_pkts_sum,
        AVG(flow_duration_milliseconds) AS duration_mean,
        MIN(flow_duration_milliseconds) AS duration_min,
        MAX(flow_duration_milliseconds) AS duration_max,
        1.0 * SUM(CASE WHEN protocol = 6 THEN 1 ELSE 0 END) / COUNT(*)
            AS protocol_tcp_ratio,
        1.0 * SUM(CASE WHEN protocol = 17 THEN 1 ELSE 0 END) / COUNT(*)
            AS protocol_udp_ratio,
        1.0 * SUM(CASE WHEN protocol NOT IN (6, 17) THEN 1 ELSE 0 END) / COUNT(*)
            AS protocol_other_ratio,
        COUNT(DISTINCT protocol) AS protocol_nunique,
        {', '.join(extra_expressions)},
        1.0 * SUM(CASE WHEN protocol = 1 THEN 1 ELSE 0 END) / COUNT(*)
            AS icmp_flow_ratio,
        1.0 * SUM(CASE WHEN
            COALESCE(dns_query_id, 0) != 0
            OR COALESCE(dns_query_type, 0) != 0
            OR COALESCE(dns_ttl_answer, 0) != 0
            THEN 1 ELSE 0 END) / COUNT(*) AS dns_flow_ratio,
        1.0 * SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) / COUNT(*) AS ftp_flow_ratio,
        CASE WHEN SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) = 0 THEN 0.0 ELSE
            1.0 * SUM(CASE WHEN ftp_command_ret_code BETWEEN 100 AND 199
                THEN 1 ELSE 0 END)
            / SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                THEN 1 ELSE 0 END) END AS ftp_1xx_ratio,
        CASE WHEN SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) = 0 THEN 0.0 ELSE
            1.0 * SUM(CASE WHEN ftp_command_ret_code BETWEEN 200 AND 299
                THEN 1 ELSE 0 END)
            / SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                THEN 1 ELSE 0 END) END AS ftp_2xx_ratio,
        CASE WHEN SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) = 0 THEN 0.0 ELSE
            1.0 * SUM(CASE WHEN ftp_command_ret_code BETWEEN 300 AND 399
                THEN 1 ELSE 0 END)
            / SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                THEN 1 ELSE 0 END) END AS ftp_3xx_ratio,
        CASE WHEN SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) = 0 THEN 0.0 ELSE
            1.0 * SUM(CASE WHEN ftp_command_ret_code BETWEEN 400 AND 499
                THEN 1 ELSE 0 END)
            / SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                THEN 1 ELSE 0 END) END AS ftp_4xx_ratio,
        CASE WHEN SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) = 0 THEN 0.0 ELSE
            1.0 * SUM(CASE WHEN ftp_command_ret_code BETWEEN 500 AND 599
                THEN 1 ELSE 0 END)
            / SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                THEN 1 ELSE 0 END) END AS ftp_5xx_ratio,
        CASE WHEN SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN 1 ELSE 0 END) = 0 THEN 0.0 ELSE
            1.0 * SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                AND NOT (ftp_command_ret_code BETWEEN 100 AND 599)
                THEN 1 ELSE 0 END)
            / SUM(CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
                THEN 1 ELSE 0 END) END AS ftp_other_code_ratio,
        COUNT(DISTINCT CASE WHEN COALESCE(ftp_command_ret_code, 0) != 0
            THEN ftp_command_ret_code END) AS ftp_code_nunique,
        MAX(label) AS label
    FROM flow_rows
    GROUP BY ipv4_src_addr, bucket_start_ms;

    CREATE INDEX idx_base_aggregates_key
    ON base_aggregates (ipv4_src_addr, bucket_start_ms);
    """
    conn.executescript(sql)
    return l7_features, tcp_features


def build_final_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        DROP TABLE IF EXISTS {quote_identifier(BUCKET_TABLE)};
        CREATE TABLE {quote_identifier(BUCKET_TABLE)} AS
        SELECT
            b.*,
            COALESCE(l7.l7_proto_nunique, 0) AS l7_proto_nunique,
            COALESCE(l7.l7_proto_top1_ratio, 0.0) AS l7_top1_ratio,
            COALESCE(l7.l7_proto_entropy, 0.0) AS l7_entropy,
            COALESCE(sp.src_port_nunique, 0) AS src_port_nunique,
            CASE WHEN b.flow_count <= 1 THEN 0.0 ELSE
                MAX(0.0, MIN(1.0,
                    1.0 - 1.0 * COALESCE(sp.src_port_nunique, 0) / b.flow_count
                )) END AS src_port_reuse_ratio,
            COALESCE(sp.src_port_top1_ratio, 0.0) AS src_port_top1_ratio,
            COALESCE(sp.src_port_entropy, 0.0) AS src_port_entropy,
            COALESCE(dp.dst_port_nunique, 0) AS dst_port_nunique,
            CASE WHEN b.flow_count <= 1 THEN 0.0 ELSE
                MAX(0.0, MIN(1.0,
                    1.0 - 1.0 * COALESCE(dp.dst_port_nunique, 0) / b.flow_count
                )) END AS dst_port_reuse_ratio,
            COALESCE(dp.dst_port_top1_ratio, 0.0) AS dst_port_top1_ratio,
            COALESCE(dp.dst_port_entropy, 0.0) AS dst_port_entropy,
            COALESCE(di.dst_ip_nunique, 0) AS dst_ip_nunique,
            CASE WHEN b.flow_count <= 1 THEN 0.0 ELSE
                MAX(0.0, MIN(1.0,
                    1.0 - 1.0 * COALESCE(di.dst_ip_nunique, 0) / b.flow_count
                )) END AS dst_ip_reuse_ratio,
            COALESCE(di.dst_ip_top1_ratio, 0.0) AS dst_ip_top1_ratio,
            COALESCE(di.dst_ip_entropy, 0.0) AS dst_ip_entropy,
            COALESCE(ic.icmp_type_nunique, 0) AS icmp_type_nunique,
            COALESCE(ic.icmp_type_top1_ratio, 0.0) AS icmp_type_top1_ratio,
            COALESCE(ic.icmp_type_entropy, 0.0) AS icmp_type_entropy,
            COALESCE(dn.dns_query_type_nunique, 0) AS dns_query_type_nunique,
            COALESCE(dn.dns_query_type_top1_ratio, 0.0)
                AS dns_query_type_top1_ratio,
            COALESCE(dn.dns_query_type_entropy, 0.0)
                AS dns_query_type_entropy
        FROM base_aggregates AS b
        LEFT JOIN l7_distribution AS l7 USING (ipv4_src_addr, bucket_start_ms)
        LEFT JOIN src_port_distribution AS sp USING (ipv4_src_addr, bucket_start_ms)
        LEFT JOIN dst_port_distribution AS dp USING (ipv4_src_addr, bucket_start_ms)
        LEFT JOIN dst_ip_distribution AS di USING (ipv4_src_addr, bucket_start_ms)
        LEFT JOIN icmp_distribution AS ic USING (ipv4_src_addr, bucket_start_ms)
        LEFT JOIN dns_distribution AS dn USING (ipv4_src_addr, bucket_start_ms);

        CREATE UNIQUE INDEX idx_temporal_buckets_10s_full_key
        ON {quote_identifier(BUCKET_TABLE)} (ipv4_src_addr, bucket_start_ms);
        """
    )


def feature_names(
    l7_features: list[str], tcp_features: list[str]
) -> list[str]:
    return [
        *NUMERICAL_FEATURES,
        "protocol_tcp_ratio",
        "protocol_udp_ratio",
        "protocol_other_ratio",
        "protocol_nunique",
        *l7_features,
        "l7_proto_nunique",
        "l7_top1_ratio",
        "l7_entropy",
        "src_port_nunique",
        "src_port_reuse_ratio",
        "src_port_top1_ratio",
        "src_port_entropy",
        "dst_port_nunique",
        "dst_port_reuse_ratio",
        "dst_port_top1_ratio",
        "dst_port_entropy",
        *tcp_features,
        "icmp_flow_ratio",
        "icmp_type_nunique",
        "icmp_type_top1_ratio",
        "icmp_type_entropy",
        "dns_flow_ratio",
        "dns_query_type_nunique",
        "dns_query_type_top1_ratio",
        "dns_query_type_entropy",
        "ftp_flow_ratio",
        "ftp_1xx_ratio",
        "ftp_2xx_ratio",
        "ftp_3xx_ratio",
        "ftp_4xx_ratio",
        "ftp_5xx_ratio",
        "ftp_other_code_ratio",
        "ftp_code_nunique",
        "dst_ip_nunique",
        "dst_ip_reuse_ratio",
        "dst_ip_top1_ratio",
        "dst_ip_entropy",
    ]


def validate_table(conn: sqlite3.Connection, features: list[str]) -> int:
    columns = {
        row[1] for row in conn.execute(
            f"PRAGMA table_info({quote_identifier(BUCKET_TABLE)})"
        )
    }
    missing = sorted(set(features) - columns)
    if missing:
        raise AssertionError("Missing generated features: " + ", ".join(missing))

    forbidden = {
        "ipv4_src_addr",
        "ipv4_dst_addr",
        "l4_src_port",
        "l4_dst_port",
        "protocol",
        "l7_proto",
        "dns_query_id",
        "tcp_flags",
        "client_tcp_flags",
        "server_tcp_flags",
    }
    included = sorted(forbidden.intersection(features))
    if included:
        raise AssertionError("Raw categorical fields leaked into features: " + ", ".join(included))

    unique_rows = conn.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT ipv4_src_addr, bucket_start_ms
            FROM {quote_identifier(BUCKET_TABLE)}
            GROUP BY ipv4_src_addr, bucket_start_ms
        )
        """
    ).fetchone()[0]
    actual_rows = conn.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(BUCKET_TABLE)}"
    ).fetchone()[0]
    if unique_rows != actual_rows:
        raise AssertionError("Duplicate bucket keys were generated.")

    for feature in features:
        values = conn.execute(
            f"SELECT MIN({quote_identifier(feature)}), MAX({quote_identifier(feature)}), "
            f"SUM(CASE WHEN {quote_identifier(feature)} IS NULL THEN 1 ELSE 0 END) "
            f"FROM {quote_identifier(BUCKET_TABLE)}"
        ).fetchone()
        minimum, maximum, null_count = values
        if null_count:
            raise AssertionError(f"{feature} contains NULL values.")
        if not math.isfinite(float(minimum)) or not math.isfinite(float(maximum)):
            raise AssertionError(f"{feature} contains a non-finite value.")
        if feature.endswith("_ratio") or feature.endswith("_entropy"):
            if minimum < -1e-9 or maximum > 1.0 + 1e-9:
                raise AssertionError(f"{feature} falls outside [0, 1].")
    return actual_rows


def save_schema(
    categories: list[tuple[float, str, str]], features: list[str]
) -> None:
    schema = {
        "bucket_table": BUCKET_TABLE,
        "bucket_size_ms": BIN_MS,
        "l7_min_global_count": L7_MIN_GLOBAL_COUNT,
        "kept_l7_categories": [category for _, category, _ in categories],
        "protocol_special_categories": ["6", "17"],
        "ports_use_exact_values": False,
        "dns_query_id_used": False,
        "dns_flow_definition": (
            "dns_query_id != 0 OR dns_query_type != 0 OR dns_ttl_answer != 0"
        ),
        "ftp_flow_definition": "ftp_command_ret_code != 0",
        "feature_names": features,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SCHEMA_PATH.open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)
        file.write("\n")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}\n"
            "Run eda/upload_to_sqlite.py first."
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA temp_store = FILE")
        ensure_raw_table(conn)
        categories = discover_l7_categories(conn)
        print(f"Retained L7 categories: {len(categories)}")
        print("Building temporary flow and distribution tables...")
        build_flow_rows(conn)
        l7_features, tcp_features = build_base_aggregates(conn, categories)
        build_distribution_helpers(conn)
        print(f"Building {BUCKET_TABLE}...")
        build_final_table(conn)
        features = feature_names(l7_features, tcp_features)
        row_count = validate_table(conn, features)

    save_schema(categories, features)
    print(f"Created {BUCKET_TABLE}: {row_count:,} rows, {len(features)} features")
    print(f"Saved categorical schema: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
