"""
Inference Pipeline Module.

This module provides a reusable prediction pipeline to make predictions on new unseen
transaction data using a saved preprocessing pipeline and a trained model.
It validates input data, automatically constructs customer velocity features
aligned with historical context, runs inference, and scores risk.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.utils import setup_logger, load_dataset
from src.risk_scoring import load_risk_config, categorize_risk, calculate_risk_scores

logger = setup_logger("inference")

class ValidationError(Exception):
    """Custom exception class for input data validation errors."""
    pass

def validate_input_data(df: pd.DataFrame) -> None:
    """
    Perform thorough validation on the input transactions DataFrame.
    
    Checks for:
    - Empty files/DataFrames
    - Missing required columns
    - Incorrect data types (non-numeric amounts or invalid timestamps)
    - Invalid values (nulls in key fields or non-positive transaction amounts)
    
    Args:
        df (pd.DataFrame): Input transaction dataset to validate.
        
    Raises:
        ValidationError: If any validation checks fail.
    """
    logger.info("Starting input data validation...")
    
    # 1. Check if DataFrame is empty
    if df.empty:
        raise ValidationError("The input dataset is empty and contains no records.")
        
    # 2. Check for missing required columns
    required_cols = [
        "transaction_id", "customer_id", "timestamp", "amount",
        "merchant_category", "location", "device_type"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns in input CSV: {missing_cols}")
        
    # 3. Check for null values in key columns
    critical_cols = ["transaction_id", "customer_id", "timestamp", "amount"]
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_rows = df[df[col].isnull()].index.tolist()
            # Show up to 5 row indices
            rows_snippet = null_rows[:5]
            raise ValidationError(
                f"Column '{col}' contains {null_count} null/missing value(s). "
                f"Affected row indices (0-indexed): {rows_snippet}..."
            )
            
    # 4. Check data types and parseability
    # Check amount data type
    try:
        pd.to_numeric(df["amount"], errors="raise")
    except (ValueError, TypeError) as e:
        # Find which values are non-numeric
        invalid_amounts = []
        for i, val in enumerate(df["amount"]):
            try:
                float(val)
            except (ValueError, TypeError):
                invalid_amounts.append((i, val))
                if len(invalid_amounts) >= 5:
                    break
        raise ValidationError(
            f"Column 'amount' contains non-numeric values. "
            f"First few invalid values: {invalid_amounts}"
        )
        
    # Check timestamp parseability
    try:
        pd.to_datetime(df["timestamp"], errors="raise")
    except (ValueError, TypeError) as e:
        invalid_timestamps = []
        for i, val in enumerate(df["timestamp"]):
            try:
                pd.to_datetime(val)
            except (ValueError, TypeError):
                invalid_timestamps.append((i, val))
                if len(invalid_timestamps) >= 5:
                    break
        raise ValidationError(
            f"Column 'timestamp' contains invalid datetime formats. "
            f"First few unparseable values: {invalid_timestamps}"
        )
        
    # 5. Check for invalid values
    # Verify amounts are positive
    numeric_amounts = pd.to_numeric(df["amount"], errors="coerce")
    invalid_amount_mask = numeric_amounts <= 0
    invalid_count = invalid_amount_mask.sum()
    if invalid_count > 0:
        invalid_rows = df[invalid_amount_mask]["transaction_id"].head(5).tolist()
        raise ValidationError(
            f"Detected {invalid_count} transactions with invalid amounts (<= 0). "
            f"Affected transaction IDs: {invalid_rows}..."
        )
        
    logger.info("Input data validation completed successfully. All constraints satisfied.")

def preprocess_and_engineer_features(df_new: pd.DataFrame, history_path: str = None) -> pd.DataFrame:
    """
    Apply cleaning and feature engineering automatically.
    Merges input transactions with clean historical transactions to ensure rolling metrics
    (e.g., 30-day customer velocity counts) are computed accurately without lookahead bias.
    
    Args:
        df_new (pd.DataFrame): New transaction records.
        history_path (str): Path to clean historical transactions database.
        
    Returns:
        pd.DataFrame: Engineered features dataset matching the rows and order of df_new.
    """
    logger.info("Initializing preprocessing and feature engineering...")
    
    # Work on copies
    new_data = df_new.copy()
    
    # Store original row order index to map predictions back exactly
    new_data["_original_order"] = range(len(new_data))
    new_data["_is_new_txn"] = True
    
    # Load history for velocity features if available
    hist_path = history_path or config.PROCESSED_DATA_PATH
    history_data = pd.DataFrame()
    if os.path.exists(hist_path):
        try:
            logger.info(f"Loading historical transaction context from: {hist_path}")
            history_data = pd.read_csv(hist_path)
            # Retain only required columns and target if present, mark as old
            req_cols_with_target = [col for col in new_data.columns if col in history_data.columns]
            history_data = history_data[req_cols_with_target].copy()
            history_data["_is_new_txn"] = False
            history_data["_original_order"] = -1
            logger.info(f"Loaded {len(history_data)} historical transactions for context.")
        except Exception as e:
            logger.warning(f"Failed to load history file. Proceeding with cold start velocity calculation. Error: {e}")
    else:
        logger.warning(f"Historical transactions file not found at {hist_path}. Proceeding with cold start velocity calculation.")
        
    # Combine sets
    if not history_data.empty:
        df_combined = pd.concat([history_data, new_data], ignore_index=True)
    else:
        df_combined = new_data
        
    # Data Cleaning and Standardization on combined dataset
    df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
    df_combined["amount"] = pd.to_numeric(df_combined["amount"], errors="coerce")
    
    # Fill missing categoricals if any
    for col in config.CATEGORICAL_COLUMNS:
        if col in df_combined.columns and df_combined[col].isnull().sum() > 0:
            # Impute using mode or fallback
            mode_series = df_combined[col].mode()
            mode_val = mode_series[0] if not mode_series.empty else "unknown"
            df_combined[col] = df_combined[col].fillna(mode_val)
            
    # Standardize categoricals (strip & lowercase)
    for col in config.CATEGORICAL_COLUMNS:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].astype(str).str.strip().str.lower()
            
    # Sort chronologically to prevent lookahead bias in rolling window calculations
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    
    # 1. Temporal Features
    df_combined["hour_of_day"] = df_combined["timestamp"].dt.hour
    df_combined["day_of_week"] = df_combined["timestamp"].dt.dayofweek
    df_combined["is_weekend"] = df_combined["day_of_week"].isin([5, 6]).astype(int)
    
    # 2. Geographical & Channel Features
    df_combined["is_international"] = (~df_combined["location"].str.lower().str.endswith("us")).astype(int)
    
    # 3. Customer Transaction Velocity & History (30-day rolling metrics)
    df_combined["customer_txn_count_30d"] = 1.0
    df_combined["customer_avg_amount_30d"] = df_combined["amount"]
    
    for cust_id, group in df_combined.groupby("customer_id"):
        # group is sorted because df_combined is sorted
        temp = group.set_index("timestamp")
        r_count = temp["transaction_id"].rolling("30D").count().values
        r_mean = temp["amount"].rolling("30D").mean().values
        
        df_combined.loc[group.index, "customer_txn_count_30d"] = r_count
        df_combined.loc[group.index, "customer_avg_amount_30d"] = r_mean
        
    # 4. Ratio features
    avg_amt_safe = df_combined["customer_avg_amount_30d"].replace(0, np.nan)
    df_combined["amount_ratio_to_avg"] = df_combined["amount"] / avg_amt_safe
    df_combined["amount_ratio_to_avg"] = df_combined["amount_ratio_to_avg"].fillna(1.0)
    
    # Extract only the new transactions
    df_features = df_combined[df_combined["_is_new_txn"] == True].copy()
    
    # Sort back to match original user row order
    df_features = df_features.sort_values("_original_order").reset_index(drop=True)
    
    # Clean temporary internal tracking columns
    df_features = df_features.drop(columns=["_is_new_txn", "_original_order"])
    
    logger.info(f"Features engineered successfully. Row count matches input: {len(df_features)}")
    return df_features

def run_predictions(input_csv: str, output_csv: str, model_path: str = None, 
                    preprocessor_path: str = None, history_path: str = None, 
                    risk_config_path: str = None) -> None:
    """
    Load preprocessor and model, load and validate input CSV, run feature engineering,
    generate classification predictions, estimate probabilities, run risk scoring, and save to CSV.
    
    Args:
        input_csv (str): Path to input dataset of transactions.
        output_csv (str): Path to save prediction results.
        model_path (str): Optional path to joblib file of the trained model (default: config.BEST_MODEL_PATH).
        preprocessor_path (str): Optional path to joblib file of the fitted preprocessor (default: config.PREPROCESSOR_PATH).
        history_path (str): Optional path to historical database.
        risk_config_path (str): Optional path to risk config JSON file.
    """
    logger.info("=" * 60)
    logger.info("STARTING BATCH INFERENCE PIPELINE")
    logger.info("=" * 60)
    
    # Determine paths
    model_file = model_path or config.BEST_MODEL_PATH
    if not os.path.exists(model_file) and not model_path:
        # Fallback to tuned RF model if best model path is not trained yet
        model_file = config.RF_TUNED_MODEL_PATH
        if not os.path.exists(model_file):
            model_file = config.RF_MODEL_PATH
            
    prep_file = preprocessor_path or config.PREPROCESSOR_PATH
    
    logger.info(f"Loading Model from:        {model_file}")
    logger.info(f"Loading Preprocessor from: {prep_file}")
    
    # Validate artifacts exist
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model not found at {model_file}. Please run training first.")
    if not os.path.exists(prep_file):
        raise FileNotFoundError(f"Preprocessor pipeline not found at {prep_file}. Please run training first.")
        
    # Load model and preprocessor
    model = joblib.load(model_file)
    preprocessor = joblib.load(prep_file)
    logger.info("Successfully loaded ML model and preprocessor artifacts.")
    
    # Load input data
    logger.info(f"Loading input transaction file: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input transaction file not found at: {input_csv}")
    df_input = pd.read_csv(input_csv)
    logger.info(f"Loaded input dataset with shape {df_input.shape}")
    
    # Validate input data
    validate_input_data(df_input)
    
    # Preprocess & Feature Engineering
    df_engineered = preprocess_and_engineer_features(df_input, history_path=history_path)
    
    # Build inputs for the preprocessor
    feature_cols = config.ENGINEERED_NUMERIC_FEATURES + config.ENGINEERED_CATEGORICAL_FEATURES
    X_untransformed = df_engineered[feature_cols].copy()
    
    # Apply ColumnTransformer
    logger.info("Applying fitted preprocessing pipeline (scaling and encoding)...")
    X_processed = preprocessor.transform(X_untransformed)
    
    # Predict probabilities and classes
    logger.info("Generating predictions and probabilities...")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_processed)[:, 1]
    else:
        logger.warning("Model does not support predict_proba; generating fallback probabilities.")
        probabilities = model.predict(X_processed).astype(float)
        
    predictions = model.predict(X_processed)
    
    # Risk scoring configuration
    risk_cfg = load_risk_config(risk_config_path)
    low_thresh = risk_cfg["low_risk_threshold"]
    high_thresh = risk_cfg["high_risk_threshold"]
    
    # Map to risk scores and levels
    risk_scores = calculate_risk_scores(probabilities)
    risk_levels = categorize_risk(probabilities, low_thresh, high_thresh)
    
    # Assemble predictions CSV
    df_output = df_input.copy()
    df_output["prediction"] = predictions.astype(int)
    df_output["probability_score"] = np.round(probabilities, 4)
    df_output["risk_score"] = risk_scores
    df_output["risk_level"] = risk_levels
    
    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    logger.info(f"Prediction results saved successfully to: {output_csv}")
    
    # Compute summary statistics
    total = len(df_output)
    anomaly_count = int(predictions.sum())
    risk_counts = df_output["risk_level"].value_counts()
    low_count = int(risk_counts.get("Low Risk", 0))
    medium_count = int(risk_counts.get("Medium Risk", 0))
    high_count = int(risk_counts.get("High Risk", 0))
    
    # Log summary statistics
    logger.info("=" * 60)
    logger.info("BATCH INFERENCE PIPELINE SUMMARY")
    logger.info("-" * 60)
    logger.info(f"Total Transactions Processed: {total}")
    logger.info(f"Anomaly Predictions (Class 1): {anomaly_count} ({anomaly_count/total:.2%})")
    logger.info(f"Normal Predictions (Class 0):  {total - anomaly_count} ({(total - anomaly_count)/total:.2%})")
    logger.info(f"High Risk Level Cases:         {high_count} ({high_count/total:.2%})")
    logger.info(f"Medium Risk Level Cases:       {medium_count} ({medium_count/total:.2%})")
    logger.info(f"Low Risk Level Cases:          {low_count} ({low_count/total:.2%})")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch predictions and risk scoring on a new transaction dataset."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file containing transactions."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save the output predictions CSV file."
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Optional path to joblib file of the trained model (default: config.BEST_MODEL_PATH)."
    )
    parser.add_argument(
        "--preprocessor", "-p",
        default=None,
        help="Optional path to joblib file of the fitted preprocessor (default: config.PREPROCESSOR_PATH)."
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional path to the historical transactions CSV for rolling features context."
    )
    parser.add_argument(
        "--risk-config",
        default=None,
        help="Optional path to the risk scoring configuration JSON."
    )
    
    args = parser.parse_args()
    
    try:
        run_predictions(
            input_csv=args.input,
            output_csv=args.output,
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            history_path=args.history,
            risk_config_path=args.risk_config
        )
    except ValidationError as ve:
        logger.error(f"DATA VALIDATION FAILED: {ve}")
        sys.exit(1)
    except FileNotFoundError as fnfe:
        logger.error(f"FILE NOT FOUND ERROR: {fnfe}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"UNEXPECTED ERROR RUNNING INFERENCE PIPELINE: {e}", exc_info=True)
        sys.exit(1)
