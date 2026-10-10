import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ------------------------------
# TIME SERIES BUCKETING FUNCTION
# ------------------------------

def create_time_buckets(df, time_column='dur', bucket_size='10s'):
    """
    Create time-based buckets from duration data
    
    Parameters:
    - df: DataFrame with duration column
    - time_column: Column name containing duration/timestamp
    - bucket_size: Size of time buckets (e.g., '10s', '1min', '1h')
    
    Returns:
    - DataFrame with time buckets and attack counts
    """
    # Create a copy to avoid modifying original
    df_buckets = df.copy()
    
    # Convert duration to datetime (starting from a reference point)
    # Since we don't have actual timestamps, we'll create relative time
    # by cumulatively summing durations
    df_buckets['relative_time'] = df_buckets[time_column].cumsum()
    
    # Create time buckets
    df_buckets['time_bucket'] = pd.cut(
        df_buckets['relative_time'],
        bins=range(0, int(df_buckets['relative_time'].max()) + 1, 10),
        labels=False
    )
    
    return df_buckets

# ------------------------------
# BUCKET ATTACKS BY TIME PERIODS
# ------------------------------

def bucket_attacks_by_time(df, bucket_size='10s'):
    """
    Organize attacks into time buckets and analyze patterns
    
    Parameters:
    - df: DataFrame with 'dur' and 'attack_cat' columns
    - bucket_size: Size of time buckets
    
    Returns:
    - DataFrame with time buckets and attack statistics
    """
    
    # 1. Create time buckets
    df['cumulative_time'] = df['dur'].cumsum()
    
    # 2. Create bucket labels based on time intervals
    # Let's create buckets of equal time intervals (e.g., every 10 seconds)
    max_time = df['cumulative_time'].max()
    num_buckets = int(max_time / 10) + 1  # 10-second buckets
    
    df['time_bucket'] = pd.cut(
        df['cumulative_time'], 
        bins=num_buckets, 
        labels=False
    )
    
    # 3. Group by time bucket and attack category
    bucket_counts = df.groupby(['time_bucket', 'attack_cat']).size().reset_index(name='count')
    
    # 4. Pivot to get attacks per bucket
    pivot_table = bucket_counts.pivot(
        index='time_bucket', 
        columns='attack_cat', 
        values='count'
    ).fillna(0)
    
    # 5. Add total attacks per bucket
    pivot_table['total_attacks'] = pivot_table.sum(axis=1)
    
    # 6. Add bucket time range
    bucket_edges = pd.cut(df['cumulative_time'], bins=num_buckets, retbins=True)[1]
    pivot_table['time_start'] = bucket_edges[:-1]
    pivot_table['time_end'] = bucket_edges[1:]
    
    return pivot_table

# ------------------------------
# VISUALIZE TIME-BASED ATTACKS
# ------------------------------

def visualize_attack_buckets(df_buckets):
    """Create visualizations of attack patterns over time"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total attacks over time
    axes[0, 0].plot(df_buckets.index, df_buckets['total_attacks'], linewidth=2)
    axes[0, 0].set_title('Total Attacks Over Time')
    axes[0, 0].set_xlabel('Time Bucket')
    axes[0, 0].set_ylabel('Number of Attacks')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top attack categories over time
    attack_cols = [col for col in df_buckets.columns if col not in ['total_attacks', 'time_start', 'time_end']]
    top_attacks = df_buckets[attack_cols].sum().sort_values(ascending=False).head(5).index
    
    for attack in top_attacks:
        axes[0, 1].plot(df_buckets.index, df_buckets[attack], label=attack, linewidth=2)
    axes[0, 1].set_title('Top 5 Attack Categories Over Time')
    axes[0, 1].set_xlabel('Time Bucket')
    axes[0, 1].set_ylabel('Number of Attacks')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Heatmap of attack patterns
    heatmap_data = df_buckets[top_attacks].T
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Number of Attacks'})
    axes[1, 0].set_title('Attack Heatmap Over Time')
    axes[1, 0].set_xlabel('Time Bucket')
    axes[1, 0].set_ylabel('Attack Category')
    
    # 4. Stacked area chart
    axes[1, 1].stackplot(df_buckets.index, 
                         [df_buckets[attack] for attack in top_attacks],
                         labels=top_attacks, alpha=0.7)
    axes[1, 1].set_title('Stacked Attack Distribution Over Time')
    axes[1, 1].set_xlabel('Time Bucket')
    axes[1, 1].set_ylabel('Number of Attacks')
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# MAIN FUNCTION TO RUN TIME SERIES ANALYSIS
# ------------------------------

def analyze_attack_patterns(df):
    """
    Main function to organize and analyze attacks by time buckets
    """
    print("\n" + "="*60)
    print("TIME SERIES ATTACK ANALYSIS")
    print("="*60)
    
    # 1. Create time buckets
    print("\n--- Creating Time Buckets ---")
    df_buckets = bucket_attacks_by_time(df, bucket_size='10s')
    print(f"Created {len(df_buckets)} time buckets")
    
    # 2. Show summary statistics
    print("\n--- Attack Statistics by Time Bucket ---")
    print(f"Average attacks per bucket: {df_buckets['total_attacks'].mean():.2f}")
    print(f"Max attacks in a bucket: {df_buckets['total_attacks'].max()}")
    print(f"Min attacks in a bucket: {df_buckets['total_attacks'].min()}")
    
    # 3. Find peak attack times
    peak_bucket = df_buckets['total_attacks'].idxmax()
    print(f"\nPeak attack time: Bucket {peak_bucket}")
    print(f"Attacks during peak: {df_buckets.loc[peak_bucket, 'total_attacks']}")
    print(f"Time range: {df_buckets.loc[peak_bucket, 'time_start']:.2f}s - {df_buckets.loc[peak_bucket, 'time_end']:.2f}s")
    
    # 4. Top attack categories overall
    attack_cols = [col for col in df_buckets.columns if col not in ['total_attacks', 'time_start', 'time_end']]
    print("\n--- Top Attack Categories ---")
    totals = df_buckets[attack_cols].sum().sort_values(ascending=False)
    for attack, count in totals.head(10).items():
        print(f"  {attack}: {count}")
    
    # 5. Visualize
    visualize_attack_buckets(df_buckets)
    
    return df_buckets

# ------------------------------
# RUN THE ANALYSIS
# ------------------------------

# Load your cleaned data (assuming you've run the EDA code)
# df is already loaded in your script

# Run time series analysis
bucketed_data = analyze_attack_patterns(df)

# ------------------------------
# ADDITIONAL: Bucket by Larger Time Windows
# ------------------------------

def create_larger_time_buckets(df, bucket_seconds=60):
    """
    Create larger time buckets for macro-level analysis
    
    Parameters:
    - df: DataFrame with duration data
    - bucket_seconds: Size of each bucket in seconds
    """
    
    df_large = df.copy()
    df_large['cumulative_time'] = df_large['dur'].cumsum()
    
    # Create buckets of specified size
    max_time = df_large['cumulative_time'].max()
    num_buckets = int(max_time / bucket_seconds) + 1
    
    df_large['time_bucket'] = pd.cut(
        df_large['cumulative_time'],
        bins=num_buckets,
        labels=False
    )
    
    # Group and analyze
    bucket_summary = df_large.groupby('time_bucket').agg({
        'attack_cat': lambda x: x.value_counts().to_dict(),
        'label': 'sum',
        'dur': 'sum'
    }).reset_index()
    
    print(f"\n--- {bucket_60s}-Second Bucket Summary ---")
    print(f"Number of buckets: {len(bucket_summary)}")
    print(f"Average attacks per bucket: {bucket_summary['label'].mean():.2f}")
    
    return bucket_summary

# Try different bucket sizes
large_buckets = create_larger_time_buckets(df, bucket_seconds=60)