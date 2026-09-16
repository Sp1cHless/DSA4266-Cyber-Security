"""
query_helper.py
Helper functions for querying the SQLite database
"""

import pandas as pd
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / 'datasets' / 'network_traffic.db'

class NetworkTrafficDB:
    """Wrapper class for querying the network traffic database."""
    
    def __init__(self, db_path=None):
        """Initialize with database path."""
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path
        self.conn = None
        self.table_name = 'network_flows'
    
    def connect(self):
        """Establish connection to database."""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def query(self, sql, params=None):
        """Execute SQL query and return results as DataFrame."""
        self.connect()
        if params:
            return pd.read_sql_query(sql, self.conn, params=params)
        else:
            return pd.read_sql_query(sql, self.conn)
    
    def get_all_attacks(self):
        """Get all attack flows."""
        sql = f"SELECT * FROM {self.table_name} WHERE attack != 'Benign'"
        return self.query(sql)
    
    def get_benign_flows(self):
        """Get all benign flows."""
        sql = f"SELECT * FROM {self.table_name} WHERE attack = 'Benign'"
        return self.query(sql)
    
    def get_by_attack_type(self, attack_type):
        """Get flows by specific attack type."""
        sql = f"SELECT * FROM {self.table_name} WHERE attack = ?"
        return self.query(sql, params=(attack_type,))
    
    def get_by_protocol(self, protocol):
        """Get flows by protocol number."""
        sql = f"SELECT * FROM {self.table_name} WHERE protocol = ?"
        return self.query(sql, params=(protocol,))
    
    def get_by_ip(self, ip):
        """Get flows by source or destination IP."""
        sql = f"""
        SELECT * FROM {self.table_name} 
        WHERE ipv4_src_addr = ? OR ipv4_dst_addr = ?
        """
        return self.query(sql, params=(ip, ip))
    
    def get_by_src_ip(self, ip):
        """Get flows by source IP only."""
        sql = f"SELECT * FROM {self.table_name} WHERE ipv4_src_addr = ?"
        return self.query(sql, params=(ip,))
    
    def get_by_dst_ip(self, ip):
        """Get flows by destination IP only."""
        sql = f"SELECT * FROM {self.table_name} WHERE ipv4_dst_addr = ?"
        return self.query(sql, params=(ip,))
    
    def get_by_port(self, port):
        """Get flows by source or destination port."""
        sql = f"""
        SELECT * FROM {self.table_name} 
        WHERE l4_src_port = ? OR l4_dst_port = ?
        """
        return self.query(sql, params=(port, port))
    
    def get_attack_distribution(self):
        """Get count of flows by attack type."""
        sql = f"""
        SELECT attack, COUNT(*) as count 
        FROM {self.table_name} 
        GROUP BY attack 
        ORDER BY count DESC
        """
        return self.query(sql)
    
    def get_protocol_distribution(self):
        """Get count of flows by protocol."""
        sql = f"""
        SELECT protocol, COUNT(*) as count 
        FROM {self.table_name} 
        GROUP BY protocol 
        ORDER BY count DESC
        """
        return self.query(sql)
    
    def get_temporal_patterns(self, ip=None):
        """Get time-ordered flows for temporal analysis."""
        if ip:
            sql = f"""
            SELECT * FROM {self.table_name} 
            WHERE ipv4_src_addr = ? OR ipv4_dst_addr = ?
            ORDER BY flow_start_milliseconds
            """
            return self.query(sql, params=(ip, ip))
        else:
            sql = f"SELECT * FROM {self.table_name} ORDER BY flow_start_milliseconds"
            return self.query(sql)
    
    def get_high_throughput_flows(self, threshold=100000):
        """Get flows with high throughput."""
        sql = f"""
        SELECT * FROM {self.table_name} 
        WHERE src_to_dst_avg_throughput > ? 
           OR dst_to_src_avg_throughput > ?
        """
        return self.query(sql, params=(threshold, threshold))
    
    def get_flows_by_ip_and_time(self, ip, start_time, end_time):
        """Get flows for an IP within a time range."""
        sql = f"""
        SELECT * FROM {self.table_name} 
        WHERE (ipv4_src_addr = ? OR ipv4_dst_addr = ?)
          AND flow_start_milliseconds >= ?
          AND flow_end_milliseconds <= ?
        ORDER BY flow_start_milliseconds
        """
        return self.query(sql, params=(ip, ip, start_time, end_time))
    
    def get_host_summary(self):
        """Get summary statistics per source IP."""
        sql = f"""
        SELECT 
            ipv4_src_addr as host_ip,
            COUNT(*) as total_flows,
            SUM(CASE WHEN attack != 'Benign' THEN 1 ELSE 0 END) as attack_flows,
            SUM(in_bytes + out_bytes) as total_bytes,
            MIN(flow_start_milliseconds) as first_seen,
            MAX(flow_start_milliseconds) as last_seen,
            COUNT(DISTINCT ipv4_dst_addr) as unique_destinations
        FROM {self.table_name}
        GROUP BY ipv4_src_addr
        ORDER BY total_flows DESC
        """
        return self.query(sql)
    
    def export_to_csv(self, sql, output_file):
        """Export query results to CSV."""
        df = self.query(sql)
        df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(df)} rows to: {output_file}")
        return df


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    # Initialize database connection
    db = NetworkTrafficDB()
    
    print("="*60)
    print("🔍 QUERYING THE DATABASE")
    print("="*60)
    
    # 1. Get attack distribution
    print("\n📊 Attack Distribution:")
    dist = db.get_attack_distribution()
    print(dist.to_string(index=False))
    
    # 2. Get all attacks
    print("\n🔍 Attack Flows:")
    attacks = db.get_all_attacks()
    if len(attacks) > 0:
        print(attacks[['ipv4_src_addr', 'ipv4_dst_addr', 'protocol', 'attack']].to_string(index=False))
    else:
        print("  No attack flows found")
    
    # 3. Get by specific attack type
    print("\n🔍 Fuzzers Attacks:")
    fuzzers = db.get_by_attack_type('Fuzzers')
    if len(fuzzers) > 0:
        print(fuzzers[['ipv4_src_addr', 'ipv4_dst_addr', 'attack']].to_string(index=False))
    
    # 4. Get by IP
    print("\n🔍 Flows involving IP 175.45.176.0:")
    ip_flows = db.get_by_ip('175.45.176.0')
    if len(ip_flows) > 0:
        print(ip_flows[['ipv4_src_addr', 'ipv4_dst_addr', 'attack']].to_string(index=False))
    
    # 5. Get host summary
    print("\n📊 Host Summary:")
    summary = db.get_host_summary()
    print(summary.to_string(index=False))
    
    # 6. Get temporal patterns
    print("\n🕐 Temporal Patterns:")
    temporal = db.get_temporal_patterns()
    print(temporal[['ipv4_src_addr', 'flow_start_milliseconds', 'attack']].head(10).to_string(index=False))
    
    # Close connection
    db.close()
    print("\n✅ Done!")
