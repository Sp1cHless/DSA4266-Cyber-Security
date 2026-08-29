import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# set display options to see more columns
pd.set_option('display.max_columns', None)

# ------------------------------
# 0. Setup Paths for New Folder Structure
# ------------------------------
# Get the current directory (where this script is running from)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current script directory: {current_dir}")

# Go up one level to the root folder (DSA4266-Cyber-Security)
root_dir = os.path.dirname(current_dir)
print(f"Root directory: {root_dir}")

# Define paths for datasets and output
datasets_dir = os.path.join(root_dir, 'datasets')
output_dir = os.path.join(root_dir, 'output')
plots_dir = os.path.join(output_dir, 'plots')
processed_dir = os.path.join(output_dir, 'processed_data')

# Create output directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Define the dataset file path - NOTE: Now using .xlsx file
dataset_path = os.path.join(datasets_dir, 'UNSW_NB15_training-set.xlsx')

# Check if the dataset exists
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at: {dataset_path}")
    print(f"Please make sure the file is in the 'datasets' folder.")
    print(f"Files in datasets folder: {os.listdir(datasets_dir) if os.path.exists(datasets_dir) else 'Folder not found'}")
    exit()

print(f"Dataset path: {dataset_path}")

# ------------------------------
# 1. Data Loading & Initial Inspection
# ------------------------------
print("\n" + "="*60)
print("1. DATA LOADING & INITIAL INSPECTION")
print("="*60)

# Load the dataset
try:
    df = pd.read_excel(dataset_path)
    print("✓ Successfully loaded Excel file")
except Exception as e:
    print(f"Error loading Excel file: {e}")
    print("Trying alternative loading method...")
    df = pd.read_excel(dataset_path, engine='openpyxl')
    print("✓ Successfully loaded Excel file with openpyxl engine")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 rows ---")
print(df.head())

# Check the shape
print(f"\nDataset shape: {df.shape}")

# ------------------------------
# 2. Data Cleaning
# ------------------------------
print("\n" + "="*60)
print("2. DATA CLEANING")
print("="*60)

# 2a. Handle Missing Values
missing_values = df.isnull().sum()
print(f"Missing values per column:\n{missing_values[missing_values > 0]}")
if missing_values.sum() == 0:
    print("✓ No missing values found!")

# 2b. Handle Infinite Values
print("\n--- Handling Infinite Values ---")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Fill numeric columns with median
for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print(f"✓ Filled NaN values with median for {len(numeric_cols)} numeric columns")

# 2c. Check and fix data types
if 'rate' in df.columns and df['rate'].dtype == 'object':
    df['rate'] = df['rate'].astype(float)
    print("✓ Converted 'rate' column to float")

# The 'id' column is not useful for modeling; drop it
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("✓ Dropped 'id' column")

# Check data types after cleaning
print("\n--- Data types after cleaning ---")
print(df.dtypes)

# ------------------------------
# 3. Exploratory Data Analysis (EDA)
# ------------------------------
print("\n" + "="*60)
print("3. EXPLORATORY DATA ANALYSIS")
print("="*60)

# 3a. Summary Statistics
print("\n--- Summary Statistics (Numeric) ---")
print(df.describe())

# 3b. Check for Class Imbalance
if 'attack_cat' in df.columns:
    print("\n--- Class Distribution (attack_cat) ---")
    print(df['attack_cat'].value_counts())

if 'label' in df.columns:
    print("\n--- Class Distribution (label - Binary) ---")
    print(df['label'].value_counts(normalize=True) * 100)

# 3c. Visualizations
sns.set_style("whitegrid")

# i. Distribution of a key numeric feature
if 'sbytes' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sbytes'], bins=50, kde=True)
    plt.title('Distribution of Source Bytes (sbytes)')
    plt.xlabel('Source Bytes')
    plt.savefig(os.path.join(plots_dir, 'sbytes_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ii. Boxplot for outliers
if 'rate' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['rate'])
    plt.title('Boxplot of Rate')
    plt.xlim(0, 5000)
    plt.savefig(os.path.join(plots_dir, 'rate_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.show()

# iii. Correlation Matrix
plt.figure(figsize=(15, 10))
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

# iv. Feature vs Target
if 'label' in df.columns and 'sbytes' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='sbytes', data=df)
    plt.title('Source Bytes by Label (0: Normal, 1: Attack)')
    plt.yscale('log')
    plt.savefig(os.path.join(plots_dir, 'sbytes_by_label.png'), dpi=300, bbox_inches='tight')
    plt.show()

# v. Attack Category vs Protocol
if 'attack_cat' in df.columns and 'proto' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(y='attack_cat', hue='proto', data=df)
    plt.title('Attack Categories by Protocol')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'attack_by_protocol.png'), dpi=300, bbox_inches='tight')
    plt.show()

print(f"✓ Basic plots saved to: {plots_dir}")

# ------------------------------
# 3d. ADVANCED EDA
# ------------------------------
print("\n" + "="*60)
print("3d. ADVANCED EDA - ATTACK PATTERN ANALYSIS")
print("="*60)

def analyze_attack_patterns_advanced(df):
    """Advanced attack pattern analysis"""
    print("\n--- Attack Co-occurrence Analysis ---")
    # Create binary matrix of attacks
    attack_matrix = pd.get_dummies(df['attack_cat'])
    attack_correlation = attack_matrix.corr()
    
    # Find highly correlated attacks
    high_corr = attack_correlation.where(
        (attack_correlation > 0.3) & (attack_correlation < 1.0)
    ).stack().sort_values(ascending=False)
    print("Highly correlated attack pairs:")
    print(high_corr.head(10))
    
    # Protocol analysis
    print("\n--- Protocol Usage by Attack Type ---")
    protocol_attack = pd.crosstab(df['attack_cat'], df['proto'])
    protocol_attack['total'] = protocol_attack.sum(axis=1)
    protocol_attack_pct = protocol_attack.div(protocol_attack['total'], axis=0) * 100
    print(protocol_attack_pct.round(2))
    
    return attack_correlation, protocol_attack

attack_corr, protocol_attack = analyze_attack_patterns_advanced(df)

# ------------------------------
# 3e. FEATURE IMPORTANCE ANALYSIS
# ------------------------------
print("\n" + "="*60)
print("3e. FEATURE IMPORTANCE ANALYSIS")
print("="*60)

def analyze_feature_importance(df):
    """Analyze which features are most important for detecting attacks"""
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'label' in numeric_cols:
        X = df[numeric_cols].drop('label', axis=1)
    else:
        X = df[numeric_cols]
    y = df['label']
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance['feature'].head(20), importance['importance'].head(20))
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features for Attack Detection')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    return importance

feature_importance = analyze_feature_importance(df)
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# ------------------------------
# 3f. TEMPORAL ATTACK PATTERNS
# ------------------------------
print("\n" + "="*60)
print("3f. TEMPORAL ATTACK PATTERNS")
print("="*60)

def analyze_temporal_patterns(df):
    """Analyze attack patterns over time (duration-based)"""
    
    df_temp = df.copy()
    df_temp['cumulative_time'] = df_temp['dur'].cumsum()
    
    # Rolling window analysis
    window = 1000
    rolling_attacks = df_temp['label'].rolling(window).mean() * 100
    
    plt.figure(figsize=(15, 5))
    plt.plot(df_temp['cumulative_time'][window:], rolling_attacks[window:])
    plt.title(f'Attack Percentage (Rolling Window: {window} records)')
    plt.xlabel('Cumulative Time (seconds)')
    plt.ylabel('Attack Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'rolling_attack_percentage.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Attack duration statistics by type
    print("\n--- Attack Duration Statistics ---")
    duration_stats = df.groupby('attack_cat')['dur'].agg(['mean', 'median', 'max', 'min']).round(4)
    print(duration_stats.sort_values('mean', ascending=False))

analyze_temporal_patterns(df)

# ------------------------------
# 3g. NETWORK TRAFFIC ANALYSIS
# ------------------------------
print("\n" + "="*60)
print("3g. NETWORK TRAFFIC ANALYSIS")
print("="*60)

def analyze_network_traffic(df):
    """Analyze network traffic patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Source bytes distribution
    axes[0, 0].boxplot([df[df['label']==0]['sbytes'], df[df['label']==1]['sbytes']], 
                        labels=['Normal', 'Attack'])
    axes[0, 0].set_title('Source Bytes by Attack Status')
    axes[0, 0].set_yscale('log')
    
    # Destination bytes distribution
    axes[0, 1].boxplot([df[df['label']==0]['dbytes'], df[df['label']==1]['dbytes']], 
                        labels=['Normal', 'Attack'])
    axes[0, 1].set_title('Destination Bytes by Attack Status')
    axes[0, 1].set_yscale('log')
    
    # Packet count analysis
    axes[1, 0].scatter(df['spkts'], df['dpkts'], c=df['label'], alpha=0.5, cmap='coolwarm')
    axes[1, 0].set_xlabel('Source Packets')
    axes[1, 0].set_ylabel('Destination Packets')
    axes[1, 0].set_title('Packet Count Distribution')
    
    # Rate analysis by attack
    axes[1, 1].boxplot([df[df['attack_cat']=='Normal']['rate'],
                        df[df['attack_cat']=='Exploits']['rate'],
                        df[df['attack_cat']=='DoS']['rate']],
                        labels=['Normal', 'Exploits', 'DoS'])
    axes[1, 1].set_title('Rate Distribution by Attack Type')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'network_traffic_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Traffic correlation
    traffic_features = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'rate']
    traffic_corr = df[traffic_features].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(traffic_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Traffic Feature Correlation')
    plt.savefig(os.path.join(plots_dir, 'traffic_correlation.png'), dpi=300, bbox_inches='tight')
    plt.show()

analyze_network_traffic(df)

# ------------------------------
# 3h. OUTLIER ANALYSIS
# ------------------------------
print("\n" + "="*60)
print("3h. OUTLIER ANALYSIS")
print("="*60)

def analyze_outliers(df):
    """Identify and analyze outliers in the data"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['label']]
    
    # Z-score analysis
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outliers = (z_scores > 3).sum()
    
    print(f"\nColumns with outliers (Z-score > 3):")
    for col, count in outliers.items():
        if count > 0:
            print(f"  {col}: {count} outliers ({count/len(df)*100:.2f}%)")
    
    # IQR analysis
    print("\n--- IQR-based Outlier Detection ---")
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_summary.append({
            'feature': col,
            'outliers': outlier_count,
            'percentage': outlier_count/len(df)*100
        })
    
    outlier_df = pd.DataFrame(outlier_summary).sort_values('outliers', ascending=False)
    print(outlier_df.head(10))
    
    return outlier_df

outlier_df = analyze_outliers(df)
outlier_df.to_csv(os.path.join(output_dir, 'outlier_analysis.csv'), index=False)

# ------------------------------
# 4. Data Transformation
# ------------------------------
print("\n" + "="*60)
print("4. DATA TRANSFORMATION")
print("="*60)

# 4a. Identify Feature Types
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target variables from transformation
if 'label' in num_cols:
    num_cols.remove('label')
if 'attack_cat' in cat_cols:
    cat_cols.remove('attack_cat')

print(f"Categorical columns to transform: {cat_cols}")
print(f"Numerical columns to transform: {num_cols}")

# 4b. Handle Outliers (IQR method)
print("\n--- Handling Outliers ---")
df_transformed = df.copy()

if num_cols:
    for col in num_cols:
        Q1 = df_transformed[col].quantile(0.25)
        Q3 = df_transformed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_transformed[col] = df_transformed[col].clip(lower=lower_bound, upper=upper_bound)
    print("✓ Outliers capped using IQR method")
else:
    print("⚠ No numeric columns to process for outliers")

# 4c. Feature Scaling (Standardization)
if num_cols:
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    df_transformed[num_cols] = scaler.fit_transform(df_transformed[num_cols])
    print("✓ Numerical features standardized")
else:
    print("⚠ No numeric columns to standardize")

# 4d. Encoding Categorical Variables
if cat_cols:
    print("\n--- Encoding Categorical Variables ---")
    df_encoded = pd.get_dummies(df_transformed, columns=cat_cols, drop_first=True)
    print(f"✓ One-hot encoding complete. New shape: {df_encoded.shape}")
else:
    print("⚠ No categorical columns to encode")
    df_encoded = df_transformed.copy()

# 4e. Separate Features and Target
if 'label' in df_encoded.columns and 'attack_cat' in df_encoded.columns:
    X = df_encoded.drop(['label', 'attack_cat'], axis=1)
    y_binary = df_encoded['label']
    y_multi = df_encoded['attack_cat']

    print(f"Features shape: {X.shape}")
    print(f"Binary target shape: {y_binary.shape}")
    print(f"Multi-class target shape: {y_multi.shape}")

    # 4f. Split the Data
    print("\n--- Data Splitting ---")
    X_train, X_temp, y_train_bin, y_temp_bin = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    X_val, X_test, y_val_bin, y_test_bin = train_test_split(
        X_temp, y_temp_bin, test_size=0.5, random_state=42, stratify=y_temp_bin
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Save processed data
    print("\n--- Saving Processed Data ---")
    df_encoded.to_csv(os.path.join(processed_dir, 'processed_data.csv'), index=False)
    print(f"✓ Processed data saved to: {processed_dir}")

    # Display final feature names
    print(f"\nFinal number of features: {len(X.columns)}")
    print(f"First few features: {X.columns[:5].tolist()}...")
else:
    print("⚠ Required columns 'label' and/or 'attack_cat' not found in dataset")

# ------------------------------
# 5. TIME SERIES ANALYSIS
# ------------------------------
print("\n" + "="*60)
print("5. TIME SERIES ATTACK ANALYSIS")
print("="*60)

def bucket_attacks_by_time(df, bucket_size=10):
    """Organize attacks into time buckets and analyze patterns"""
    
    # Create time buckets
    df_buckets = df.copy()
    df_buckets['cumulative_time'] = df_buckets['dur'].cumsum()
    
    # Create bucket labels
    max_time = df_buckets['cumulative_time'].max()
    num_buckets = int(max_time / bucket_size) + 1
    
    df_buckets['time_bucket'] = pd.cut(
        df_buckets['cumulative_time'], 
        bins=num_buckets, 
        labels=False
    )
    
    # Group by time bucket and attack category
    bucket_counts = df_buckets.groupby(['time_bucket', 'attack_cat']).size().reset_index(name='count')
    
    # Pivot to get attacks per bucket
    pivot_table = bucket_counts.pivot(
        index='time_bucket', 
        columns='attack_cat', 
        values='count'
    ).fillna(0)
    
    pivot_table['total_attacks'] = pivot_table.sum(axis=1)
    
    return pivot_table, df_buckets

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
    attack_cols = [col for col in df_buckets.columns if col != 'total_attacks']
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
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=axes[1, 0], 
                cbar_kws={'label': 'Number of Attacks'})
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
    plt.savefig(os.path.join(plots_dir, 'time_series_attack_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Run time series analysis
print("\n--- Creating Time Buckets (10-second intervals) ---")
pivot_table, df_bucketed = bucket_attacks_by_time(df, bucket_size=10)
print(f"Created {len(pivot_table)} time buckets")

print("\n--- Attack Statistics by Time Bucket ---")
print(f"Average attacks per bucket: {pivot_table['total_attacks'].mean():.2f}")
print(f"Max attacks in a bucket: {pivot_table['total_attacks'].max()}")
print(f"Min attacks in a bucket: {pivot_table['total_attacks'].min()}")

# Find peak attack times
peak_bucket = pivot_table['total_attacks'].idxmax()
print(f"\nPeak attack time: Bucket {peak_bucket}")
print(f"Attacks during peak: {pivot_table.loc[peak_bucket, 'total_attacks']}")

# Visualize
visualize_attack_buckets(pivot_table)

# Save time series results
pivot_table.to_csv(os.path.join(output_dir, 'attack_buckets.csv'))

print(f"✓ Time series analysis saved to: {output_dir}")

# ------------------------------
# 6. COMPLETE SUMMARY
# ------------------------------
print("\n" + "="*60)
print("COMPLETE ANALYSIS SUMMARY")
print("="*60)
print(f"✓ Dataset processed: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Plots saved to: {plots_dir}")
print(f"✓ Processed data saved to: {processed_dir}")
print(f"✓ Feature importance saved to: {output_dir}")
print(f"✓ Outlier analysis saved to: {output_dir}")
print(f"✓ Time series analysis saved to: {output_dir}")

print("\n" + "="*60)
print("DATA PROCESSING COMPLETE!")
print("="*60)