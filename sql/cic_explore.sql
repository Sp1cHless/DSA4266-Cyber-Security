-- ===========================================================================
-- CIC-IDS2018 SQLite explorer
-- File:  datasets/cic_2018_traffic.db   (6.57 GB, source NF-CICIDS2018-v3.csv)
--
-- How to run in VS Code (extension: SQLTools + SQLTools SQLite driver)
--   Ctrl+E Ctrl+E  -> run the statement under the cursor (or the selection)
--   Command Palette -> "SQLTools: Run This File"  -> runs every statement below
--   Command Palette -> "SQLTools: Run Query"      -> type one-off SQL
-- Keep the cursor INSIDE a statement before pressing the shortcut.
-- Running the whole file takes about two minutes; statement 8 dominates.
--
-- The driver opens the file READ-WRITE. Only use SELECT here; the database is
-- a derived artifact and can be rebuilt with temporal_eda/01_build_temporal_buckets.py.
--
-- ---------------------------------------------------------------------------
-- Tables
--   network_flows          20,115,529 rows  raw L7 flow records (one row per flow)
--   cic_temporal_buckets    3,657,936 rows  10-second per-source-IP buckets, 88 features
--   bucket_progress           181,876 rows  per-host build bookkeeping (n_buckets)
--   import_status                   1 row   provenance of the imported CSV
--
-- Key columns
--   network_flows
--     flow_start_milliseconds / flow_end_milliseconds  epoch milliseconds
--     ipv4_src_addr, ipv4_dst_addr, l4_src_port, l4_dst_port, protocol
--     in_bytes, out_bytes, in_pkts, out_pkts, flow_duration_milliseconds
--     label      0 = benign, 1 = attack
--     attack     raw attack name (e.g. 'FTP-BruteForce'); NULL when label = 0
--   cic_temporal_buckets
--     ipv4_src_addr    source host
--     bucket_start_ms  epoch milliseconds, snapped to the 10-second grid
--     flow_count       flows attributed to this host in this 10 s bin
--     ... 87 more engineered features (ratios, entropies, distinct counts)
--     label            1 if any flow in this bin is an attack
--
-- Most columns are declared without a type (NONE affinity), so values keep
-- their stored integer/real form. Timestamps are epoch MILLISECONDS, hence the
-- /1000 in every datetime() call.
--
-- Indexes: UNIQUE(ipv4_src_addr, bucket_start_ms) on cic_temporal_buckets and
--          (ipv4_src_addr) on network_flows. Both tables also have a rowid.
--          Queries that filter on a host are fast; whole-table aggregates scan
--          a few seconds each.
-- ===========================================================================


-- === 1. Inventory: what is in the file =====================================
-- rowid max equals the row count for these append-only tables (rowid is faster
-- than COUNT(*) on the 20 M-row flow table).
SELECT 'network_flows' AS tbl, MAX(rowid) AS n_rows FROM network_flows
UNION ALL SELECT 'cic_temporal_buckets', MAX(rowid) FROM cic_temporal_buckets
UNION ALL SELECT 'bucket_progress', MAX(rowid) FROM bucket_progress
UNION ALL SELECT 'import_status', MAX(rowid) FROM import_status;


-- === 2. Objects (tables, indexes, views) and their DDL =====================
SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index', 'view') ORDER BY type, name;


-- === 3. Provenance of the raw flow import ==================================
SELECT csv_path, csv_size, n_rows, completed FROM import_status;


-- === 4. Bucket label balance (attack prevalence) ===========================
SELECT label, COUNT(*) AS buckets,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM cic_temporal_buckets), 4) AS pct
FROM cic_temporal_buckets
GROUP BY label;


-- === 5. Capture span (UTC) =================================================
SELECT datetime(MIN(bucket_start_ms) / 1000, 'unixepoch') AS first_bucket_utc,
       datetime(MAX(bucket_start_ms) / 1000, 'unixepoch') AS last_bucket_utc,
       COUNT(DISTINCT bucket_start_ms) AS distinct_10s_bins
FROM cic_temporal_buckets;


-- === 6. Per captured day: volume, attack bins, active hosts ================
SELECT date(bucket_start_ms / 1000, 'unixepoch') AS day_utc,
       COUNT(*)                       AS buckets,
       SUM(label)                     AS attack_buckets,
       COUNT(DISTINCT ipv4_src_addr)  AS src_hosts
FROM cic_temporal_buckets
GROUP BY 1
ORDER BY 1;


-- === 7. Raw attack types (ground truth in network_flows) ===================
SELECT attack,
       COUNT(*)                       AS flows,
       COUNT(DISTINCT ipv4_src_addr)  AS attack_hosts
FROM network_flows
WHERE label = 1
GROUP BY attack
ORDER BY flows DESC;


-- === 8. Same attack types, counted as 10-second buckets ====================
-- Mirrors the annotation step in CIC_test_train/prepare_data.py (attack_lookup),
-- collapsing each flow to its (host, 10 s bin). This is the slow statement in
-- the file (~70 s) because it scans all 20 M flows; everything else is seconds.
SELECT attack, COUNT(*) AS attack_buckets
FROM (
  SELECT ipv4_src_addr, (flow_start_milliseconds / 10000) * 10000 AS bucket_start_ms, attack
  FROM network_flows
  WHERE label = 1
  GROUP BY 1, 2, 3
)
GROUP BY attack
ORDER BY attack_buckets DESC;


-- === 9. Reconcile buckets against raw flows (must match) ===================
SELECT (SELECT SUM(flow_count) FROM cic_temporal_buckets) AS flows_in_buckets,
       (SELECT COUNT(*) FROM network_flows)              AS raw_flows;


-- === 10. Busiest source hosts, split by attack activity ====================
SELECT ipv4_src_addr,
       COUNT(*)         AS buckets,
       SUM(flow_count)  AS flows,
       SUM(label)       AS attack_buckets
FROM cic_temporal_buckets
GROUP BY ipv4_src_addr
ORDER BY attack_buckets DESC, flows DESC
LIMIT 25;


-- === 11. Sanity checks on bucket geometry =================================
-- All three columns should be 0 / 0: bins must sit on the 10 s grid
-- (1518611280000 = 2018-02-14 12:28:00 UTC is divisible by 10000) and the
-- unique index must guarantee one row per (host, bin).
SELECT SUM(bucket_start_ms % 10000 != 0) AS off_grid_bins,
       COUNT(*) - COUNT(DISTINCT ipv4_src_addr || '|' || bucket_start_ms) AS duplicate_keys,
       SUM(flow_count <= 0) AS empty_bins
FROM cic_temporal_buckets;


-- === 12. Drill-down: one host's bucket series =============================
-- Replace the IP with a value from query 10. Uses the unique index, so it is
-- instant.
SELECT datetime(bucket_start_ms / 1000, 'unixepoch') AS bucket_utc,
       flow_count, in_bytes_sum, out_bytes_sum, in_pkts_sum, out_pkts_sum,
       ROUND(duration_mean, 3)  AS duration_mean,
       ROUND(protocol_tcp_ratio, 3) AS tcp_ratio,
       ROUND(l7_entropy, 3)     AS l7_entropy,
       dst_ip_nunique, label
FROM cic_temporal_buckets
WHERE ipv4_src_addr = '18.221.219.4'
ORDER BY bucket_start_ms
LIMIT 200;


-- === 13. Drill-down: the raw flows behind one host's attack bins ==========
SELECT datetime(flow_start_milliseconds / 1000, 'unixepoch') AS flow_start_utc,
       ipv4_dst_addr, l4_dst_port, protocol,
       in_bytes, out_bytes, in_pkts, out_pkts,
       flow_duration_milliseconds, label, attack
FROM network_flows
WHERE ipv4_src_addr = '18.221.219.4'
ORDER BY flow_start_milliseconds
LIMIT 100;


-- === 14. Gap analysis: how often does a host go silent? ===================
-- Consecutive bins differ by exactly 10000 ms; a larger jump means skipped bins.
WITH ordered AS (
  SELECT ipv4_src_addr, bucket_start_ms,
         bucket_start_ms - LAG(bucket_start_ms) OVER (
           PARTITION BY ipv4_src_addr ORDER BY bucket_start_ms) AS gap_ms
  FROM cic_temporal_buckets
)
SELECT CASE WHEN gap_ms IS NULL THEN 'first bin'
            WHEN gap_ms = 10000 THEN 'contiguous (10 s)'
            WHEN gap_ms < 1800000 THEN 'gap within 30 min'
            ELSE 'gap, 30+ min' END AS kind,
       COUNT(*) AS bins
FROM ordered
WHERE ipv4_src_addr = '18.221.219.4'
GROUP BY kind;
