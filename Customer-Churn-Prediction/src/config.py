"""
Central Configuration Module.

This module defines dataset paths, common project constants, and logger settings
for the Financial Transaction Risk & Anomaly Engine.
"""

import os

# Project Directories & Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

# Dataset Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "transactions.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_clean.csv")

# Train/Test Split Paths
X_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "X_test.csv")
y_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "y_train.csv")
y_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "y_test.csv")
SPLIT_DIMENSIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "split_dimensions.json")

# Preprocessor Saving Path
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

# Reports & Figure Directories
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Logging Configuration
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "pipeline.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Common Constants
RANDOM_STATE = 42
TARGET_COLUMN = "is_anomaly"

# Feature Columns definitions
NUMERIC_COLUMNS = [
    "amount"
]

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "location",
    "device_type"
]

KEY_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp"
]

# Feature sets after engineering
ENGINEERED_NUMERIC_FEATURES = [
    "amount",
    "customer_txn_count_30d",
    "customer_avg_amount_30d",
    "amount_ratio_to_avg",
    "hour_of_day",
    "day_of_week"
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "merchant_category",
    "location",
    "device_type",
    "is_weekend",
    "is_international"
]

