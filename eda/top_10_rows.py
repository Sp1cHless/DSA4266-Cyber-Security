import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / 'datasets' / 'NF-UNSW-NB15-v3.csv'
OUTPUT_FILE = REPO_ROOT / 'datasets' / 'head_10_rows.csv'
N_ROWS = 10

# === READ INPUT CSV ===
try:
    df = pd.read_csv(INPUT_FILE, nrows=N_ROWS)
    
    print(f"✓ Successfully read: {INPUT_FILE}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    
    # === SAVE TO NEW CSV ===
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved first {N_ROWS} rows to: {OUTPUT_FILE}")
    
    # === DISPLAY THE DATA ===
    print(f"\n{'='*60}")
    print(f"First {N_ROWS} rows:")
    print(f"{'='*60}")
    print(df.to_string())
    
except FileNotFoundError:
    print(f"✗ Error: Input file not found at: {INPUT_FILE}")
    print("  Please check the file path and try again.")
except Exception as e:
    print(f"✗ Error: {e}")
