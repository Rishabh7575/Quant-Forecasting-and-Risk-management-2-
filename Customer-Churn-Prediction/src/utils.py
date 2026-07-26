"""
Utility Functions Module.

This module contains helper functions used across the project,
such as logging configuration, dataset loading, and dataset profiling summaries.
"""

import logging
import os
from typing import Optional
import pandas as pd
from src import config

def setup_logger(name: str = "pipeline") -> logging.Logger:
    """
    Configure and return a standard logger for the project.
    
    This function sets up logging to both the console and a file defined
    in the configuration (logs/pipeline.log).
    
    Args:
        name (str): Name of the logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # Avoid duplicate handlers if the logger was already configured
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(config.LOG_FORMAT)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8")
        file_handler.setLevel(config.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def load_dataset(filepath: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    Args:
        filepath (str): Absolute path to the CSV file.
        logger (Optional[logging.Logger]): Logger instance for logging messages.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """
    if logger:
        logger.info(f"Attempting to load dataset from: {filepath}")
        
    if not os.path.exists(filepath):
        error_msg = f"Dataset file not found at {filepath}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        df = pd.read_csv(filepath)
        if logger:
            logger.info(f"Successfully loaded dataset. Shape: {df.shape}")
        return df
    except Exception as e:
        error_msg = f"Failed to load dataset from {filepath}. Error: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise e

def print_dataset_summary(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    """
    Display/log a comprehensive summary of the dataset.
    
    Includes shape, column names, data types, missing values, duplicate rows,
    memory usage, and the first five rows.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        logger (Optional[logging.Logger]): Logger instance to output the summary.
    """
    def log_or_print(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
    log_or_print("=" * 60)
    log_or_print("DATASET SUMMARY")
    log_or_print("=" * 60)
    
    # 1. Dataset Shape
    log_or_print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    log_or_print("-" * 40)
    
    # 2. Columns, Data Types, and Missing Values
    log_or_print("Columns, Data Types, and Missing Values:")
    null_counts = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    
    # Formatting column info as a table-like structure for the logs
    log_or_print(f"{'Column Name':<25} | {'Data Type':<15} | {'Missing Values':<15}")
    log_or_print("-" * 60)
    for col in df.columns:
        log_or_print(f"{col:<25} | {str(df[col].dtype):<15} | {null_counts[col]:<15}")
    log_or_print("-" * 40)
    
    # 3. Duplicate Rows
    log_or_print(f"Number of duplicate rows: {duplicate_count}")
    log_or_print("-" * 40)
    
    # 4. Memory Usage
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    log_or_print(f"Memory Usage: {memory_bytes:,} bytes (~{memory_mb:.2f} MB)")
    log_or_print("-" * 40)
    
    # 5. First 5 Rows
    log_or_print("First 5 Rows:")
    log_or_print(f"\n{df.head().to_string()}\n")
    log_or_print("=" * 60)
