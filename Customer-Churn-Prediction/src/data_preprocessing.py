"""
Data Preprocessing Module.

This module provides a production-ready, modular preprocessing pipeline for
the Financial Transaction Risk & Anomaly Engine. Each preprocessing step is
implemented as a separate, reusable function.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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

def prepare_ml_dataset(df: pd.DataFrame) -> None:
    """
    Split the dataset into stratified train/test sets, build and apply a
    preprocessing ColumnTransformer pipeline, and save the split datasets, dimensions,
    and preprocessor.

    Args:
        df (pd.DataFrame): DataFrame containing engineered features and target.
    """
    logger.info("=" * 60)
    logger.info("PREPARING MACHINE LEARNING DATASET & PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # 1. Separate Features (X) and Target (y)
    y = df[config.TARGET_COLUMN]
    feature_cols = config.ENGINEERED_NUMERIC_FEATURES + config.ENGINEERED_CATEGORICAL_FEATURES
    X = df[feature_cols].copy()

    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
    logger.info(f"Features shape: {X.shape}")

    # 2. Split the dataset: 80% train, 20% test (stratified, fixed random state)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    logger.info(f"Train split shape: {X_train_raw.shape}, Test split shape: {X_test_raw.shape}")

    # 3. Build scikit-learn preprocessing pipeline
    # StandardScaler for numerical, OneHotEncoder for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.ENGINEERED_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.ENGINEERED_CATEGORICAL_FEATURES)
        ]
    )

    # 4. Fit and transform
    logger.info("Fitting ColumnTransformer on training split and transforming both splits...")
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    # Reconstruct DataFrame with feature names for readability
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(config.ENGINEERED_CATEGORICAL_FEATURES).tolist()
    all_feature_names = config.ENGINEERED_NUMERIC_FEATURES + cat_feature_names

    X_train_processed = pd.DataFrame(X_train_arr, columns=all_feature_names)
    X_test_processed = pd.DataFrame(X_test_arr, columns=all_feature_names)

    # 5. Save artifacts
    logger.info("Saving split datasets to disk...")
    os.makedirs(os.path.dirname(config.X_TRAIN_PATH), exist_ok=True)
    X_train_processed.to_csv(config.X_TRAIN_PATH, index=False)
    X_test_processed.to_csv(config.X_TEST_PATH, index=False)
    y_train.to_csv(config.y_TRAIN_PATH, index=False)
    y_test.to_csv(config.y_TEST_PATH, index=False)

    # Save preprocessing object
    logger.info(f"Saving preprocessor object to: {config.PREPROCESSOR_PATH}")
    os.makedirs(os.path.dirname(config.PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    # Save train/test dimensions
    dimensions = {
        "X_train_shape": X_train_processed.shape,
        "X_test_shape": X_test_processed.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "y_train_anomaly_rate": float(y_train.mean()),
        "y_test_anomaly_rate": float(y_test.mean())
    }

    logger.info(f"Saving split dimensions to: {config.SPLIT_DIMENSIONS_PATH}")
    with open(config.SPLIT_DIMENSIONS_PATH, "w") as f:
        json.dump(dimensions, f, indent=4)

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION PIPELINE SUMMARY")
    logger.info("-" * 60)
    logger.info(f"X_train shape: {X_train_processed.shape}")
    logger.info(f"X_test shape:  {X_test_processed.shape}")
    logger.info(f"Train anomaly rate: {dimensions['y_train_anomaly_rate']:.4%}")
    logger.info(f"Test anomaly rate:  {dimensions['y_test_anomaly_rate']:.4%}")
    logger.info("=" * 60)

if __name__ == "__main__":
    from src.feature_engineering import engineer_features
    # Run full sequence locally for standalone execution
    df_clean = preprocess_pipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
    df_feat = engineer_features(df_clean)
    prepare_ml_dataset(df_feat)

