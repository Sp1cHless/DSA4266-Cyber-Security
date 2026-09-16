import pandas as pd
import sqlite3
from pathlib import Path

# === CONNECT TO DATABASE ===
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / 'datasets' / 'network_traffic.db'
conn = sqlite3.connect(DB_PATH)

# === BASIC QUERIES ===

# 1. Count all rows without loading the full table into memory
df_count = pd.read_sql_query("SELECT COUNT(*) AS total_rows FROM network_flows", conn)
print(df_count)

# 2. View first 10 rows
df_head = pd.read_sql_query("SELECT * FROM network_flows LIMIT 10", conn)
print(df_head)

# 3. Get all attacks
df_attacks = pd.read_sql_query("SELECT * FROM network_flows WHERE attack != 'Benign'", conn)
print(df_attacks)

# 4. Get by attack type
df_fuzzers = pd.read_sql_query("SELECT * FROM network_flows WHERE attack = 'Fuzzers'", conn)
print(df_fuzzers)

# 5. Get by IP address
df_ip = pd.read_sql_query("SELECT * FROM network_flows WHERE ipv4_src_addr = '175.45.176.0'", conn)
print(df_ip)

# 6. Get attack distribution
df_dist = pd.read_sql_query("""
    SELECT attack, COUNT(*) as count 
    FROM network_flows 
    GROUP BY attack 
    ORDER BY count DESC
""", conn)
print(df_dist)

# 7. Get protocol distribution
df_proto = pd.read_sql_query("""
    SELECT protocol, COUNT(*) as count 
    FROM network_flows 
    GROUP BY protocol 
    ORDER BY count DESC
""", conn)
print(df_proto)

# 8. Get flows with high throughput
df_high = pd.read_sql_query("""
    SELECT * FROM network_flows 
    WHERE src_to_dst_avg_throughput > 100000
""", conn)
print(df_high)

# 9. Get flows by port
df_port = pd.read_sql_query("""
    SELECT * FROM network_flows 
    WHERE l4_src_port = 8088 OR l4_dst_port = 8088
""", conn)
print(df_port)

# 10. Get time-ordered flows
df_time = pd.read_sql_query("""
    SELECT * FROM network_flows 
    ORDER BY flow_start_milliseconds
""", conn)
print(df_time)

# === CLOSE CONNECTION ===
conn.close()
