"""
Data Preprocessing Module.

This module handles loading the raw transaction data, performing diagnostic
checks (missing values, duplicates, memory usage), and saving a cleaned copy
without applying any transformations.
"""

import os
import pandas as pd
from src import config
from src.utils import setup_logger, load_dataset, print_dataset_summary

def preprocess_pipeline(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Run the data loading, summarization, and cleaning pipeline.
    
    This function reads the raw transaction dataset, performs basic profiling,
    drops duplicate rows (if any), drops rows with missing values (if any),
    and saves the cleaned copy into the processed data directory.
    
    Args:
        raw_path (str): Path to the raw transactions CSV file.
        processed_path (str): Path to save the cleaned transactions CSV file.
        
    Returns:
        pd.DataFrame: Cleaned pandas DataFrame.
    """
    logger = setup_logger("data_preprocessing")
    logger.info("Initializing Data Preprocessing Pipeline...")
    
    # 1. Load the dataset
    df = load_dataset(raw_path, logger)
    
    # 2. Print and log dataset summary
    logger.info("Generating dataset diagnostic summary...")
    print_dataset_summary(df, logger)
    
    # 3. Perform basic cleaning (duplicates & missing values)
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in the dataset. Dropping missing values...")
        df_clean = df.dropna().copy()
    else:
        logger.info("No missing values found in the dataset.")
        df_clean = df.copy()
        
    # Check for duplicate rows
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows. Dropping duplicates...")
        df_clean = df_clean.drop_duplicates().copy()
    else:
        logger.info("No duplicate rows found in the dataset.")
        
    # 4. Save cleaned copy (without feature transformations)
    logger.info(f"Saving cleaned dataset (Shape: {df_clean.shape}) to: {processed_path}")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    logger.info("Data Preprocessing Pipeline completed successfully.")
    
    return df_clean

if __name__ == "__main__":
    # If run as a standalone script, use path configurations from config.py
    preprocess_pipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
