"""
Feature Engineering Module.

This module implements the feature engineering logic for the Financial Transaction
Risk & Anomaly Engine. It generates temporal, geographical, and customer velocity/history features.
"""

import logging
import numpy as np
import pandas as pd
from src import config
from src.utils import setup_logger

logger = setup_logger("feature_engineering")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the preprocessed transactions DataFrame.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame. Must contain 'timestamp', 'amount',
                           'location', and 'customer_id'.
                           
    Returns:
        pd.DataFrame: DataFrame with engineered features added.
    """
    logger.info("Starting feature engineering process...")
    df_feat = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat["timestamp"]):
        logger.info("Converting 'timestamp' to datetime objects...")
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
        
    # Sort chronologically to prevent lookahead bias in rolling window calculations
    logger.info("Sorting transactions chronologically...")
    df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)
    
    # 1. Temporal features
    logger.info("Engineering temporal features...")
    df_feat["hour_of_day"] = df_feat["timestamp"].dt.hour
    df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
    df_feat["is_weekend"] = df_feat["day_of_week"].isin([5, 6]).astype(int)
    
    # 2. Geographical & Channel features
    logger.info("Engineering geographical features...")
    # Clean/lowercase location is already done in preprocessing, check if it ends with 'us'
    df_feat["is_international"] = (~df_feat["location"].str.lower().str.endswith("us")).astype(int)
    
    # 3. Customer Transaction Velocity & History (30-day rolling metrics)
    logger.info("Engineering customer transaction velocity and historical features...")
    
    # Create rolling group by customer_id over a 30-day lookback window
    rolling_group = df_feat.groupby("customer_id").rolling("30D", on="timestamp")
    
    # Count of transactions in the last 30 days (including the current transaction)
    df_feat["customer_txn_count_30d"] = (
        rolling_group["transaction_id"]
        .count()
        .reset_index(level=0, drop=True)
        .astype(float)
    )
    
    # Average transaction amount in the last 30 days (including the current transaction)
    df_feat["customer_avg_amount_30d"] = (
        rolling_group["amount"]
        .mean()
        .reset_index(level=0, drop=True)
        .astype(float)
    )
    
    # 4. Ratio-based features
    # Ratio of current amount to 30-day rolling average
    logger.info("Engineering ratio-based features...")
    # Avoid division by zero by replacing 0 with NaN
    avg_amt_safe = df_feat["customer_avg_amount_30d"].replace(0, np.nan)
    df_feat["amount_ratio_to_avg"] = df_feat["amount"] / avg_amt_safe
    df_feat["amount_ratio_to_avg"] = df_feat["amount_ratio_to_avg"].fillna(1.0)
    
    logger.info(f"Feature engineering completed. Shape: {df_feat.shape}")
    return df_feat

if __name__ == "__main__":
    from src.utils import load_dataset
    # Test execution
    df_clean = load_dataset(config.PROCESSED_DATA_PATH)
    df_feat = engineer_features(df_clean)
    print(df_feat[["customer_id", "timestamp", "amount", "customer_txn_count_30d", "customer_avg_amount_30d", "amount_ratio_to_avg"]].head(10))
