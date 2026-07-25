import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess(raw_filepath, processed_filepath):
    """
    Load raw data, perform basic exploration, clean, and save to processed directory.
    """
    print("--- 1. Loading Dataset ---")
    df = pd.read_csv(raw_filepath)
    
    print("\n--- 2. Dataset Shape ---")
    print(df.shape)
    
    print("\n--- 3. Column Names ---")
    print(df.columns.tolist())
    
    print("\n--- 4. Data Types ---")
    print(df.dtypes)
    
    print("\n--- 5. Missing Values ---")
    # TotalCharges is often read as object because of blank spaces
    # We will convert it to numeric to properly check missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(df.isnull().sum())
    
    print("\n--- 6. Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    print("\n--- 7. Basic Statistics ---")
    print(df.describe(include='all'))
    
    # Handle the missing values (drop them for now as they are very few)
    df_clean = df.dropna().copy()
    
    # Save cleaned output
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df_clean.to_csv(processed_filepath, index=False)
    print(f"\nSaved cleaned dataset to {processed_filepath}")
    
    return df_clean

def create_visualizations(df, figures_dir):
    """
    Generate required visualizations and save them.
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Churn distribution count plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution')
    plt.savefig(os.path.join(figures_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Gender vs Churn
    plt.figure(figsize=(8, 6))
    sns.countplot(x='gender', hue='Churn', data=df, palette='Set2')
    plt.title('Gender vs Churn')
    plt.savefig(os.path.join(figures_dir, 'gender_vs_churn.png'))
    plt.close()
    
    # 3. Contract Type vs Churn
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
    plt.title('Contract Type vs Churn')
    plt.savefig(os.path.join(figures_dir, 'contract_vs_churn.png'))
    plt.close()
    
    # 4. Monthly Charges distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['MonthlyCharges'], kde=True, bins=30, color='purple')
    plt.title('Monthly Charges Distribution')
    plt.savefig(os.path.join(figures_dir, 'monthly_charges_dist.png'))
    plt.close()
    
    # 5. Tenure distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['tenure'], kde=True, bins=30, color='teal')
    plt.title('Tenure Distribution')
    plt.savefig(os.path.join(figures_dir, 'tenure_dist.png'))
    plt.close()
    print(f"Saved 5 visualizations to {figures_dir}")

if __name__ == "__main__":
    # Define paths based on project structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'Telco-Customer-Churn.csv')
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'churn_clean.csv')
    figures_path = os.path.join(base_dir, 'reports', 'figures')
    
    # Run the pipeline
    if os.path.exists(raw_data_path):
        clean_df = load_and_preprocess(raw_data_path, processed_data_path)
        create_visualizations(clean_df, figures_path)
    else:
        print(f"Error: Dataset not found at {raw_data_path}. Please download it first.")
