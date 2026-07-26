"""
Exploratory Data Analysis (EDA) Runner.

This script loads the cleaned transactions dataset, prints a data quality report,
and generates/saves the required exploratory plots in reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def run_analysis_and_plots(data_path: str, figures_dir: str):
    """
    Run data profiling and generate exploratory visualizations.
    
    Args:
        data_path (str): Path to the cleaned transactions CSV dataset.
        figures_dir (str): Directory where visualization figures will be saved.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure directory for figures exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Feature Classification
    print("\n=== 1. Feature Classification ===")
    identifiers = ["transaction_id", "customer_id", "timestamp"]
    numerical = ["amount"]
    categorical = ["merchant_category", "location", "device_type"]
    target = "is_anomaly"
    
    print(f"Identifiers: {identifiers}")
    print(f"Numerical: {numerical}")
    print(f"Categorical: {categorical}")
    print(f"Target: {target}")
    
    # 2. Data Quality Report
    print("\n=== 2. Data Quality Report ===")
    
    # Missing values
    missing = df.isnull().sum()
    print("\nMissing Values per column:")
    for col, val in missing.items():
        print(f"  {col}: {val}")
        
    # Unique values
    print("\nUnique Values per column:")
    unique_counts = df.nunique()
    for col, val in unique_counts.items():
        print(f"  {col}: {val}")
        
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows in dataset: {duplicates}")
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"\nConstant columns (nunique <= 1): {constant_cols}")
    
    # High-cardinality categorical columns (e.g. unique values > 10% of total rows)
    threshold = int(len(df) * 0.1)
    high_cardinality = [col for col in categorical if df[col].nunique() > 20]
    print(f"High-cardinality categorical columns (> 20 unique values): {high_cardinality}")
    
    # Set visual theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    # 3. Create Visualizations
    
    # Plot 1: Target Distribution & Class Imbalance
    print("\nGenerating target distribution plot...")
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=target, hue=target, data=df, palette={0: "#4F46E5", 1: "#EF4444"}, legend=False)
    plt.title("Transaction Class Distribution (Imbalance Check)", pad=15)
    plt.xlabel("Is Anomaly (0 = Normal, 1 = Anomalous)")
    plt.ylabel("Count")
    
    # Add percentage labels on top of bars
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height) or height == 0:
            continue
        percentage = 100 * height / total
        ax.annotate(f'{int(height)}\n({percentage:.2f}%)', (p.get_x() + p.get_width() / 2., height - (height * 0.2 if height > 1000 else -20)),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                    color='white' if height > 1000 else 'black', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "target_distribution.png"), dpi=200)
    plt.close()
    
    # Plot 2: Transaction Amount Distribution (Log Scale and Split)
    print("Generating transaction amount distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution (Log amount)
    df_log = df.copy()
    df_log['log_amount'] = np.log10(df_log['amount'] + 1)
    
    sns.histplot(x='log_amount', data=df_log, kde=True, color="#4F46E5", ax=axes[0])
    axes[0].set_title("Overall Log-Amount Distribution", pad=10)
    axes[0].set_xlabel("Log10(Amount + 1)")
    axes[0].set_ylabel("Density/Frequency")
    
    # Amount distribution by Target
    sns.boxplot(x=target, y='amount', hue=target, data=df, palette={0: "#4F46E5", 1: "#EF4444"}, ax=axes[1], legend=False)
    axes[1].set_title("Transaction Amount by Anomaly Class (Boxplot)", pad=10)
    axes[1].set_xlabel("Is Anomaly")
    axes[1].set_ylabel("Amount ($)")
    axes[1].set_yscale('log')  # Log scale for y-axis to handle outliers visually
    
    plt.suptitle("Transaction Amount Distribution Analysis", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "amount_distribution.png"), dpi=200)
    plt.close()
    
    # Plot 3: Top Transaction Categories
    print("Generating transaction categories distribution plot...")
    plt.figure(figsize=(9, 5))
    # Order by category count
    order = df['merchant_category'].value_counts().index
    # Set custom color palette
    colors = ["#4F46E5" if c not in ["transfer", "cash_withdrawal", "travel"] else "#818CF8" for c in order]
    sns.countplot(y='merchant_category', hue='merchant_category', data=df, order=order, palette=colors, legend=False)
    plt.title("Transaction Frequency by Merchant Category", pad=15)
    plt.xlabel("Count")
    plt.ylabel("Merchant Category")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "categories_distribution.png"), dpi=200)
    plt.close()
    
    # Plot 4: Correlation Heatmap for Numerical and Encoded Features
    print("Generating correlation heatmap...")
    # One-hot encode the categorical variables to show in the correlation heatmap
    df_encoded = df.copy()
    
    # We will compute the correlation of amount, target, and top categories
    # Drop identifier columns for correlation
    df_corr_input = pd.get_dummies(df_encoded.drop(columns=identifiers), columns=categorical, drop_first=False)
    
    # Convert bool columns to numeric (0 or 1) so corr() executes correctly
    for col in df_corr_input.select_dtypes(include=['bool']).columns:
        df_corr_input[col] = df_corr_input[col].astype(int)
        
    # Calculate correlations
    corr_matrix = df_corr_input.corr()
    
    # Filter to show correlation with target and amount to prevent massive unreadable heatmap
    top_corr_features = ['amount', 'is_anomaly'] + [c for c in corr_matrix.columns if 'merchant_category' in c or 'device_type' in c]
    subset_corr = corr_matrix.loc[top_corr_features, top_corr_features]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(subset_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1.0, vmax=1.0)
    plt.title("Correlation Heatmap (Numerical & Selected Encoded Features)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()
    
    # Plot 5: Device Type and Location Anomaly Analysis
    print("Generating device and location anomaly analysis plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Device Type Anomaly Rate
    device_anomaly = df.groupby('device_type')[target].mean().reset_index()
    device_anomaly[target] = device_anomaly[target] * 100  # Convert to %
    sns.barplot(x='device_type', y=target, hue='device_type', data=device_anomaly, palette="Purples_r", ax=axes[0], legend=False)
    axes[0].set_title("Anomaly Rate (%) by Device Type", pad=10)
    axes[0].set_xlabel("Device Type")
    axes[0].set_ylabel("Anomaly Rate (%)")
    for p in axes[0].patches:
        height = p.get_height()
        if pd.isna(height) or height == 0:
            continue
        axes[0].annotate(f'{height:.2f}%', (p.get_x() + p.get_width() / 2., height + 0.1),
                         ha='center', va='bottom', fontweight='bold')
                         
    # Location Anomaly Rate (top 10 locations by count)
    loc_counts = df['location'].value_counts().index[:10]
    df_top_loc = df[df['location'].isin(loc_counts)]
    loc_anomaly = df_top_loc.groupby('location')[target].mean().reset_index()
    loc_anomaly[target] = loc_anomaly[target] * 100  # Convert to %
    loc_anomaly = loc_anomaly.sort_values(by=target, ascending=False)
    
    sns.barplot(x=target, y='location', hue='location', data=loc_anomaly, palette="Reds_r", ax=axes[1], legend=False)
    axes[1].set_title("Anomaly Rate (%) by Location (Top 10 Locations)", pad=10)
    axes[1].set_xlabel("Anomaly Rate (%)")
    axes[1].set_ylabel("Location")
    for p in axes[1].patches:
        width = p.get_width()
        if pd.isna(width) or width == 0:
            continue
        axes[1].annotate(f'{width:.2f}%', (width + 0.1, p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', fontweight='bold')
                         
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "device_location_analysis.png"), dpi=200)
    plt.close()
    
    print("\nVisualizations successfully generated and saved to reports/figures/.")

if __name__ == "__main__":
    run_analysis_and_plots(config.PROCESSED_DATA_PATH, config.FIGURES_DIR)
