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

Temporal bucketing and sequence-generation scripts are kept in `temporal_eda/`.
The generated bucket table remains inside the local SQLite database, and the
processed NPZ/CSV files are ignored by Git.

### 1. Build 10-Second Source-IP Buckets

From the repository root, run:

```bash
sqlite3 datasets/network_traffic.db < temporal_eda/01_build_temporal_buckets.sql
```

This creates the `temporal_buckets_10s` table without changing
`network_flows`.

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
