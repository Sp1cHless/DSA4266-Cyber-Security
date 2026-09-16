# DSA4266-Cyber-Security

## Local Dataset Setup

The full `NF-UNSW-NB15-v3.csv` dataset is too large to commit to GitHub, so each teammate should download/copy it locally and generate their own SQLite database.

Expected local files:

```text
datasets/NF-UNSW-NB15-v3.csv      # raw dataset, not committed
datasets/network_traffic.db       # generated SQLite DB, not committed
```

Both files are ignored by Git. This keeps the repository lightweight while still allowing everyone to run the same analysis code.

### 1. Place the Raw CSV

Put the full dataset CSV here:

```bash
datasets/NF-UNSW-NB15-v3.csv
```

Check that it exists:

```bash
ls -lh datasets/NF-UNSW-NB15-v3.csv
```

### 2. Install Python Dependencies

Create and activate a virtual environment if you have not already:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the packages used by the SQLite upload/query scripts:

```bash
pip install pandas
```

### 3. Generate the SQLite Database

From the repository root, run:

```bash
python eda/upload_to_sqlite.py
```

This reads `datasets/NF-UNSW-NB15-v3.csv` and creates:

```bash
datasets/network_traffic.db
```

The script creates a table called `network_flows`.

### 4. Verify the Database

Run:

```bash
sqlite3 datasets/network_traffic.db "SELECT COUNT(*) FROM network_flows;"
```

You can also inspect the table:

```bash
sqlite3 datasets/network_traffic.db ".schema network_flows"
```

## Temporal EDA and Data Processing

Temporal EDA and feature-engineering scripts are kept in `temporal_eda/`. The
local data pipeline has three levels:

- `network_flows` contains the original raw flow rows and imported columns.
- `temporal_buckets_10s_full` contains one engineered row per source IP and
  10-second bucket.
- `processed_binary_temporal/` contains model-ready fixed-length train/test
  sequences and their metadata.

Generated database tables and processed files remain local and are ignored by
Git.

### 1. Build 10-Second Source-IP Buckets

From the repository root, run:

```bash
python temporal_eda/01_build_temporal_buckets.py
```

This creates `temporal_buckets_10s_full` without changing `network_flows`.
Each bucket retains the current numerical flow aggregates and transforms raw
categorical codes into behavioural features such as protocol ratios,
distribution diversity/concentration, decoded TCP-flag ratios, and grouped
ICMP, DNS, and FTP activity. Exact IP addresses, ports, identifiers, and raw
categorical codes are not used as model inputs. The generated feature order and
retained frequent L7 categories are recorded in
`processed_binary_temporal/categorical_schema.json`. Additional continuous
NetFlow fields are intentionally deferred until their numerical aggregation
rules are defined.

### 2. Build Host-Disjoint Binary Sequences

```bash
python temporal_eda/02_build_binary_sequences.py
```

The script creates two reproducible evaluation rounds under:

```text
processed_binary_temporal/
```

Each round keeps source IPs disjoint between training and testing. The benign
test hosts are selected so their valid sequence count is as close as possible
to 20% of all benign sequences, with at least two contributing test hosts.

## Temporal Feature Definitions

Raw flows are grouped by `ipv4_src_addr` and
`floor(flow_start_milliseconds / 10000)`, producing one row for each source IP
and 10-second interval. Source IP and `bucket_start_ms` are retained only as
grouping metadata and are not included in the model input.

For the distribution features below, let `N` be the relevant number of flows,
`K` the number of distinct observed categories, and `c_max` the count of the
most common category. The shared definitions are:

```text
nunique     = K
reuse_ratio = 0 when N <= 1, otherwise 1 - K/N
top1_ratio  = c_max/N
entropy     = 0 when K <= 1, otherwise -sum(p_i * ln(p_i)) / ln(K)
```

All ratio and entropy features are between 0 and 1. A relevant-flow group that
does not occur in a bucket produces zeros rather than missing or infinite
values.

| Feature(s) | How the feature is built within each 10-second bucket |
|---|---|
| `flow_count` | Number of raw flows. |
| `in_bytes_sum`, `out_bytes_sum` | Sum of incoming and outgoing bytes. |
| `in_pkts_sum`, `out_pkts_sum` | Sum of incoming and outgoing packet counts. |
| `duration_mean`, `duration_min`, `duration_max` | Mean, minimum, and maximum `flow_duration_milliseconds`. |
| `protocol_tcp_ratio` | Proportion of all bucket flows where `protocol = 6`. |
| `protocol_udp_ratio` | Proportion of all bucket flows where `protocol = 17`. |
| `protocol_other_ratio` | Proportion of all bucket flows using another protocol. |
| `protocol_nunique` | Number of distinct protocol identifiers. Raw protocol numbers are not model features. |
| `l7_<category>_ratio` | Proportion of all bucket flows belonging to a retained `l7_proto` category. A category is retained when it occurs in at least 1,000 raw flows globally; selection uses frequency only, not the label. |
| `l7_other_ratio` | Proportion belonging to non-null L7 categories below the global threshold. |
| `l7_proto_nunique`, `l7_top1_ratio`, `l7_entropy` | Diversity, concentration, and normalized entropy of the L7 category distribution. `l7_top1_ratio` uses all bucket flows as its denominator. |
| `src_port_nunique`, `src_port_reuse_ratio`, `src_port_top1_ratio`, `src_port_entropy` | Distribution of source ports. Exact port values are not retained. |
| `dst_port_nunique`, `dst_port_reuse_ratio`, `dst_port_top1_ratio`, `dst_port_entropy` | Distribution of destination ports. Exact port values are not retained. |
| `dst_ip_nunique`, `dst_ip_reuse_ratio`, `dst_ip_top1_ratio`, `dst_ip_entropy` | Distribution of contacted destination IPs. Exact destination addresses are not retained. |
| `tcp_flag_<flag>_ratio` | Proportion of TCP flows whose `tcp_flags` bitmask contains `<flag>`. Separate features are created for FIN, SYN, RST, PSH, ACK, URG, ECE, and CWR. |
| `client_tcp_flag_<flag>_ratio` | The same eight decoded flag ratios using `client_tcp_flags`. |
| `server_tcp_flag_<flag>_ratio` | The same eight decoded flag ratios using `server_tcp_flags`. All TCP flag ratios use only TCP flows as the denominator. |
| `icmp_flow_ratio` | Proportion of all flows where `protocol = 1`. |
| `icmp_type_nunique`, `icmp_type_top1_ratio`, `icmp_type_entropy` | Distribution of `icmp_ipv4_type` among ICMP flows only. The combined raw `icmp_type` code is dropped. |
| `dns_flow_ratio` | Proportion of flows with DNS information, defined as any of `dns_query_id`, `dns_query_type`, or `dns_ttl_answer` being nonzero. |
| `dns_query_type_nunique`, `dns_query_type_top1_ratio`, `dns_query_type_entropy` | Distribution of `dns_query_type` among DNS flows only. `dns_query_id` is never used as a model feature. |
| `ftp_flow_ratio` | Proportion of flows where `ftp_command_ret_code` is nonzero. |
| `ftp_1xx_ratio`, `ftp_2xx_ratio`, `ftp_3xx_ratio`, `ftp_4xx_ratio`, `ftp_5xx_ratio`, `ftp_other_code_ratio` | Proportions of FTP flows in each response-code class. Exact FTP response codes are not retained. |
| `ftp_code_nunique` | Number of distinct nonzero FTP response codes in the bucket. |

For the current dataset, the retained L7 categories are `0`, `1`, `2`, `3`,
`4`, `5`, `7`, `7.37`, `10.16`, `11`, `13`, `17`, `36`, `37`, `41`, `85`,
`92`, `115`, `175`, and `370`. Decimal points are replaced by underscores in
column names, for example `l7_7_37_ratio`.

The bucket target is `label = MAX(label)`, so a bucket is malicious if it
contains at least one malicious flow. Consecutive buckets from the same source
IP are then arranged into sequences of 10 buckets. With 88 bucket features, the
saved model input has shape `(number_of_sequences, 10, 88)`; labels and source
IP metadata are stored separately from the feature tensor.
