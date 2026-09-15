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
