-- Build the minimal 10-second temporal bucket table.
-- To use a different raw table, change only the table name on the next line.
DROP VIEW IF EXISTS raw_flows_input;
CREATE TEMP VIEW raw_flows_input AS SELECT * FROM network_flows;

DROP TABLE IF EXISTS temporal_buckets_10s;

CREATE TABLE temporal_buckets_10s AS
SELECT
    ipv4_src_addr,
    (flow_start_milliseconds / 10000) * 10000 AS bucket_start_ms,
    COUNT(*) AS flow_count,
    COUNT(DISTINCT ipv4_dst_addr) AS dst_ip_count,
    COUNT(DISTINCT l4_dst_port) AS dst_port_count,
    COUNT(DISTINCT l4_src_port) AS src_port_count,
    SUM(in_bytes) AS in_bytes_sum,
    SUM(out_bytes) AS out_bytes_sum,
    SUM(in_pkts) AS in_pkts_sum,
    SUM(out_pkts) AS out_pkts_sum,
    AVG(flow_duration_milliseconds) AS duration_mean,
    MIN(flow_duration_milliseconds) AS duration_min,
    MAX(flow_duration_milliseconds) AS duration_max,
    MAX(label) AS label
FROM raw_flows_input
GROUP BY ipv4_src_addr, bucket_start_ms;

CREATE INDEX IF NOT EXISTS idx_temporal_buckets_ip_time
ON temporal_buckets_10s (ipv4_src_addr, bucket_start_ms);

-- These results are informational sanity checks, not fixed requirements.
SELECT
    COUNT(*) AS total_buckets,
    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS benign_buckets,
    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS malicious_buckets,
    ROUND(AVG(flow_count), 2) AS average_flows_per_bucket
FROM temporal_buckets_10s;

DROP VIEW raw_flows_input;
