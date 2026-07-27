"""
Data Preprocessing Module.

This module provides a production-ready, modular preprocessing pipeline for
the Financial Transaction Risk & Anomaly Engine. Each preprocessing step is
implemented as a separate, reusable function.
"""

import os
import logging
import pandas as pd
import numpy as np
from src import config
from src.utils import setup_logger

# Initialize logger
logger = setup_logger("data_preprocessing")

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw transaction data from a CSV file.

    Args:
        filepath (str): Path to the raw dataset.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at filepath.
    """
    logger.info(f"Loading raw dataset from {filepath}...")
    if not os.path.exists(filepath):
        error_msg = f"Raw dataset not found at path: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file. Error: {e}")
        raise e

def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove duplicate rows from the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    logger.info("Checking for duplicate rows...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows. Removing duplicates...")
        df_cleaned = df.drop_duplicates().copy()
        logger.info(f"Duplicates removed. New shape: {df_cleaned.shape}")
        return df_cleaned
    else:
        logger.info("No duplicate rows found.")
        return df.copy()

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and handle missing values appropriately.
    - Numerical features filled with median.
    - Categorical features filled with mode.
    - Key columns/Target columns drop any missing rows.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with handled missing values.
    """
    logger.info("Checking for missing values...")
    df_cleaned = df.copy()
    missing_counts = df_cleaned.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        logger.info("No missing values found.")
        return df_cleaned

    logger.warning(f"Found {total_missing} total missing values in the dataset.")
    
    # Handle numeric columns
    for col in config.NUMERIC_COLUMNS:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            median_val = df_cleaned[col].median()
            logger.info(f"Imputing missing values in numeric column '{col}' with median: {median_val}")
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            
    # Handle categorical columns
    for col in config.CATEGORICAL_COLUMNS:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            mode_val = df_cleaned[col].mode()[0]
            logger.info(f"Imputing missing values in categorical column '{col}' with mode: '{mode_val}'")
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            
    # Drop rows if missing in key columns or target column
    critical_cols = config.KEY_COLUMNS + [config.TARGET_COLUMN]
    for col in critical_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            logger.warning(f"Found missing values in critical column '{col}'. Dropping affected rows...")
            df_cleaned = df_cleaned.dropna(subset=[col]).copy()
            
    logger.info(f"Missing values handling completed. Current shape: {df_cleaned.shape}")
    return df_cleaned

def validate_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns have correct data types.
    - timestamp converted to datetime
    - amount converted to float
    - categoricals and IDs formatted as strings/objects
    - is_anomaly target column converted to integer

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with corrected data types.
    """
    logger.info("Validating and converting data types...")
    df_cleaned = df.copy()

    # Convert timestamp
    if "timestamp" in df_cleaned.columns:
        logger.info("Converting 'timestamp' to datetime...")
        df_cleaned["timestamp"] = pd.to_datetime(df_cleaned["timestamp"])

    # Convert amount
    if "amount" in df_cleaned.columns:
        logger.info("Ensuring 'amount' is float...")
        df_cleaned["amount"] = pd.to_numeric(df_cleaned["amount"], errors="coerce")

    # Convert target
    if config.TARGET_COLUMN in df_cleaned.columns:
        logger.info(f"Ensuring target '{config.TARGET_COLUMN}' is integer...")
        df_cleaned[config.TARGET_COLUMN] = pd.to_numeric(df_cleaned[config.TARGET_COLUMN], errors="coerce").astype(int)

    # Convert categorical and key string columns
    str_cols = config.CATEGORICAL_COLUMNS + config.KEY_COLUMNS
    for col in str_cols:
        if col in df_cleaned.columns and col != "timestamp":
            df_cleaned[col] = df_cleaned[col].astype(str)

    logger.info("Data type validation and conversion completed.")
    return df_cleaned

def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize text in categorical columns by stripping spaces and converting to lowercase.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardized categorical values.
    """
    logger.info("Standardizing categorical values (lowercasing & stripping whitespace)...")
    df_cleaned = df.copy()

    for col in config.CATEGORICAL_COLUMNS:
        if col in df_cleaned.columns:
            logger.info(f"Standardizing column: {col}")
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()

    logger.info("Categorical standardization completed.")
    return df_cleaned

def detect_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect invalid or unrealistic data:
    - Amount <= 0 is considered invalid/unrealistic for this financial transaction context.
    - Logs warnings and removes such records.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with invalid values handled.
    """
    logger.info("Detecting invalid/unrealistic values...")
    df_cleaned = df.copy()

    if "amount" in df_cleaned.columns:
        invalid_mask = df_cleaned["amount"] <= 0
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"Detected {invalid_count} transactions with amount <= 0. Removing invalid records...")
            df_cleaned = df_cleaned[~invalid_mask].copy()
            logger.info(f"Invalid transactions removed. New shape: {df_cleaned.shape}")
        else:
            logger.info("No invalid transaction amounts detected (all amounts > 0).")

    return df_cleaned

def preprocess_pipeline(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Orchestrate the preprocessing pipeline steps and save the cleaned dataset.

    Args:
        raw_path (str): Path to the raw transactions CSV file.
        processed_path (str): Path to save the cleaned transactions CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # 1. Load Raw Data
    df_raw = load_raw_data(raw_path)
    original_shape = df_raw.shape
    
    # Track statistics for reporting
    duplicates_before = df_raw.duplicated().sum()
    missing_before = df_raw.isnull().sum().sum()
    invalid_before = (df_raw["amount"] <= 0).sum() if "amount" in df_raw.columns else 0

    # 2. Execute Preprocessing Steps
    df_step = clean_duplicates(df_raw)
    df_step = handle_missing_values(df_step)
    df_step = validate_and_convert_types(df_step)
    df_step = standardize_categorical_values(df_step)
    df_cleaned = detect_invalid_values(df_step)
    
    final_shape = df_cleaned.shape

    # 3. Save clean data
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_cleaned.to_csv(processed_path, index=False)
    logger.info(f"Saved processed dataset to: {processed_path}")

    # Log summary of decisions/actions
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE SUMMARY")
    logger.info("-" * 60)
    logger.info(f"Original shape:       {original_shape}")
    logger.info(f"Final shape:          {final_shape}")
    logger.info(f"Duplicates removed:   {duplicates_before}")
    logger.info(f"Missing values resolved: {missing_before}")
    logger.info(f"Invalid values removed: {invalid_before}")
    logger.info("=" * 60)

    return df_cleaned

if __name__ == "__main__":
    # If run as standalone, load configurations from config
    preprocess_pipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
