"""
upload_to_sqlite.py
Upload CSV to SQLite database for easy querying
Full production-ready version with error handling and logging
"""

import pandas as pd
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

# CSV file location (input)
CSV_FILE = REPO_ROOT / 'datasets' / 'NF-UNSW-NB15-v3.csv'

# SQLite database location (output)
DB_FILE = REPO_ROOT / 'datasets' / 'network_traffic.db'

# Table name in the database
TABLE_NAME = 'network_flows'

# Number of CSV rows to load at a time
CHUNK_SIZE = 100_000

# ============================================================
# FUNCTIONS
# ============================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_success(text):
    """Print a success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print an error message."""
    print(f"❌ {text}")

def print_info(text):
    """Print an info message."""
    print(f"📌 {text}")

def clean_column_names(df):
    """Clean column names for SQL compatibility."""
    df.columns = [
        str(col).lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '_')
        .replace('.', '_')
        .replace('%', 'percent')
        .replace('&', 'and')
        .replace('+', 'plus')
        for col in df.columns
    ]
    return df

def infer_sql_type(dtype, column_name, sample_value):
    """
    Infer SQLite type from pandas dtype and sample value.
    SQLite uses dynamic typing but we define types for clarity.
    """
    # Check for timestamp columns
    if 'timestamp' in column_name.lower() or 'time' in column_name.lower():
        return 'INTEGER'  # Store timestamps as milliseconds
    
    # Check for IP address columns
    if 'ip' in column_name.lower() or 'addr' in column_name.lower():
        return 'TEXT'
    
    # Check for string columns
    if dtype == 'object':
        if isinstance(sample_value, str):
            # Check if it's a string that should be TEXT
            return 'TEXT'
        return 'TEXT'
    
    # Numeric types
    if 'int' in str(dtype):
        return 'INTEGER'
    if 'float' in str(dtype):
        return 'REAL'
    
    # Default fallback
    return 'TEXT'

def generate_create_table_sql(df, table_name):
    """Generate CREATE TABLE SQL with inferred data types."""
    columns = []
    for col in df.columns:
        sample_value = df[col].iloc[0] if len(df) > 0 else None
        sql_type = infer_sql_type(df[col].dtype, col, sample_value)
        columns.append(f'    "{col}" {sql_type}')
    
    create_sql = f"CREATE TABLE {table_name} (\n"
    create_sql += ",\n".join(columns)
    create_sql += "\n)"
    return create_sql

def get_table_info(conn, table_name):
    """Get table schema information."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def get_row_count(conn, table_name):
    """Get total row count from table."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]

# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    print_header("🚀 UPLOAD CSV TO SQLITE DATABASE")
    print_info(f"Input CSV:  {CSV_FILE}")
    print_info(f"Output DB:  {DB_FILE}")
    print_info(f"Table name: {TABLE_NAME}")
    
    # ------------------------------------------------------------------
    # STEP 1: Check if CSV file exists
    # ------------------------------------------------------------------
    if not CSV_FILE.exists():
        print_error(f"CSV file not found: {CSV_FILE}")
        print_info("Please check the file path and try again.")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # STEP 2: Connect to SQLite
    # ------------------------------------------------------------------
    print_header("🗄️ CONNECTING TO DATABASE")
    
    # Ensure directory exists
    db_dir = DB_FILE.parent
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Created directory: {db_dir}")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        print_success(f"Connected to database: {DB_FILE}")
    except Exception as e:
        print_error(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # STEP 3: Import CSV in chunks
    # ------------------------------------------------------------------
    print_header("📥 IMPORTING CSV IN CHUNKS")
    print_info(f"Chunk size: {CHUNK_SIZE:,} rows")

    try:
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        print_info(f"Dropped existing table (if any): {TABLE_NAME}")

        total_rows = 0
        column_count = 0

        for chunk_number, chunk in enumerate(pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE), start=1):
            chunk = clean_column_names(chunk)
            column_count = len(chunk.columns)
            if chunk_number == 1:
                print_success(f"Detected {column_count} columns")
                print_info(f"Columns: {', '.join(chunk.columns[:10])}...")

            chunk.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
            total_rows += len(chunk)
            print(f"  ✓ Imported chunk {chunk_number}: {len(chunk):,} rows ({total_rows:,} total)")

        if total_rows == 0:
            print_error("CSV file has no data rows.")
            conn.close()
            sys.exit(1)

        conn.commit()
        print_success(f"Inserted {total_rows:,} rows successfully!")
        print_success(f"Columns: {column_count}")
    except pd.errors.EmptyDataError:
        print_error("CSV file is empty.")
        conn.close()
        sys.exit(1)
    except Exception as e:
        print_error(f"Error importing CSV: {e}")
        conn.close()
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # STEP 8: Verify and display results
    # ------------------------------------------------------------------
    print_header("🔍 VERIFICATION")
    
    # Get row count
    final_count = get_row_count(conn, TABLE_NAME)
    print_info(f"Total rows in table: {final_count}")
    
    # Get table schema
    schema = get_table_info(conn, TABLE_NAME)
    print_info("Table schema:")
    for col in schema:
        print(f"  • {col[1]} ({col[2]}){'  PRIMARY KEY' if col[5] else ''}")
    
    # Query and display sample data
    print_info("Sample data (first 5 rows):")
    try:
        sample_df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} LIMIT 5", conn)
        print(sample_df.to_string())
    except Exception as e:
        print_error(f"Error querying sample data: {e}")
    
    # Show attack distribution
    try:
        print_info("Attack distribution:")
        dist_df = pd.read_sql_query(
            f"SELECT attack, COUNT(*) as count FROM {TABLE_NAME} GROUP BY attack",
            conn
        )
        print(dist_df.to_string(index=False))
    except Exception as e:
        print_error(f"Error getting distribution: {e}")
    
    # ------------------------------------------------------------------
    # STEP 9: Close connection
    # ------------------------------------------------------------------
    conn.close()
    
    # ------------------------------------------------------------------
    # STEP 10: Summary
    # ------------------------------------------------------------------
    print_header("✅ COMPLETE")
    print_success(f"Database created: {DB_FILE}")
    print_success(f"Table: {TABLE_NAME}")
    print_success(f"Rows inserted: {final_count}")
    print_success(f"Columns: {len(schema)}")
    
    print("\n" + "="*70)
    print("📝 HOW TO USE:")
    print("="*70)
    print("""
    # Python:
    import pandas as pd
    import sqlite3
    
    conn = sqlite3.connect('{}')
    df = pd.read_sql_query('SELECT * FROM {} LIMIT 10', conn)
    
    # Or use the helper functions in query_helper.py
    """.format(DB_FILE, TABLE_NAME))
    
    print("\n📝 QUERY EXAMPLES:")
    print("="*70)
    print("""
    # Get all attacks
    SELECT * FROM {} WHERE attack != 'Benign';
    
    # Count by attack type
    SELECT attack, COUNT(*) FROM {} GROUP BY attack;
    
    # Get flows by IP
    SELECT * FROM {} WHERE ipv4_src_addr = '175.45.176.0';
    
    # Get flows with high throughput
    SELECT * FROM {} WHERE src_to_dst_avg_throughput > 100000;
    """.format(TABLE_NAME, TABLE_NAME, TABLE_NAME, TABLE_NAME))

# ============================================================
# SCRIPT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
