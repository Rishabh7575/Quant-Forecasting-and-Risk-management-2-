"""
Risk Scoring Engine Module.

This module converts raw model prediction probabilities into business-friendly
risk scores (0-100) and categorizes transactions into Low, Medium, and High Risk levels.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src import config
from src.utils import setup_logger, load_dataset
from src.feature_engineering import engineer_features

# Initialize logger
logger = setup_logger("risk_scoring")

def load_risk_config(config_path: str = None) -> dict:
    """
    Load risk threshold and model settings from configuration file.
    Falls back to default settings if file doesn't exist or is invalid.
    """
    path = config_path or config.RISK_CONFIG_PATH
    defaults = {
        "model_path": "models/random_forest_tuned.joblib",
        "low_risk_threshold": 0.20,
        "high_risk_threshold": 0.60
    }
    
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            logger.info(f"Loaded risk scoring configuration from {path}")
            # Ensure required keys exist, else merge with defaults
            for key, val in defaults.items():
                if key not in cfg:
                    cfg[key] = val
            return cfg
        except Exception as e:
            logger.error(f"Failed to load risk configuration from {path}. Error: {e}. Using default thresholds.")
    else:
        logger.info(f"Risk configuration file not found at {path}. Using default thresholds.")
        
    return defaults

def get_test_metadata() -> pd.DataFrame:
    """
    Reconstruct the test split and return transaction metadata (IDs, timestamp, target).
    Uses the exact same splitting parameters as the modeling pipeline to ensure row alignment.
    """
    logger.info("Reconstructing test split to retrieve transaction/customer IDs...")
    df_clean = load_dataset(config.PROCESSED_DATA_PATH)
    df_feat = engineer_features(df_clean)
    
    # Stratified split using config settings
    _, df_test = train_test_split(
        df_feat,
        test_size=0.2,
        stratify=df_feat[config.TARGET_COLUMN],
        random_state=config.RANDOM_STATE
    )
    
    # Reset index to match X_test index alignment
    df_test = df_test.reset_index(drop=True)
    
    metadata_cols = config.KEY_COLUMNS + [config.TARGET_COLUMN]
    return df_test[metadata_cols].copy()

def calculate_risk_scores(probabilities: np.ndarray) -> np.ndarray:
    """
    Map raw model probabilities (0.0 to 1.0) to a business-friendly scale of 0 to 100.
    """
    return np.round(probabilities * 100, 2)

def categorize_risk(probabilities: np.ndarray, low_threshold: float, high_threshold: float) -> list:
    """
    Categorize predictions into Low Risk, Medium Risk, and High Risk.
    """
    categories = []
    for prob in probabilities:
        if prob < low_threshold:
            categories.append("Low Risk")
        elif prob < high_threshold:
            categories.append("Medium Risk")
        else:
            categories.append("High Risk")
    return categories

def generate_prediction_table(df_meta: pd.DataFrame, predictions: np.ndarray, 
                              probabilities: np.ndarray, low_threshold: float, 
                              high_threshold: float) -> pd.DataFrame:
    """
    Assemble the final predictions table containing identifiers, model predictions,
    probabilities, risk scores, and risk levels.
    """
    logger.info("Generating prediction table with risk scores and levels...")
    df_preds = df_meta.copy()
    
    # Add model output columns
    df_preds["prediction"] = predictions
    df_preds["probability_score"] = probabilities
    df_preds["risk_score"] = calculate_risk_scores(probabilities)
    df_preds["risk_level"] = categorize_risk(probabilities, low_threshold, high_threshold)
    
    # Rename columns for clarity and reorder
    df_preds = df_preds.rename(columns={
        "transaction_id": "Transaction ID",
        "customer_id": "Customer ID",
        "timestamp": "Timestamp",
        "is_anomaly": "Actual Class"
    })
    
    column_order = [
        "Transaction ID", "Customer ID", "Timestamp", 
        "prediction", "probability_score", "risk_score", 
        "risk_level", "Actual Class"
    ]
    
    return df_preds[column_order]

def calculate_summary_statistics(df_preds: pd.DataFrame) -> dict:
    """
    Calculate summary statistics of predictions and risk levels.
    """
    total = len(df_preds)
    value_counts = df_preds["risk_level"].value_counts()
    
    low_count = int(value_counts.get("Low Risk", 0))
    medium_count = int(value_counts.get("Medium Risk", 0))
    high_count = int(value_counts.get("High Risk", 0))
    
    summary = {
        "total_predictions": total,
        "high_risk_cases": high_count,
        "medium_risk_cases": medium_count,
        "low_risk_cases": low_count,
        "percentage_high_risk": round((high_count / total) * 100, 2),
        "percentage_medium_risk": round((medium_count / total) * 100, 2),
        "percentage_low_risk": round((low_count / total) * 100, 2)
    }
    
    logger.info(f"Summary Statistics: {json.dumps(summary, indent=2)}")
    return summary

def generate_risk_visualizations(df_preds: pd.DataFrame, low_threshold: float, 
                                 high_threshold: float, figures_dir: str):
    """
    Create and save high-quality visualizations of risk distribution, scores, and probabilities.
    """
    logger.info(f"Generating risk visualizations in {figures_dir}...")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Palette definition
    colors = {"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}
    
    # 1. Risk Distribution (Bar Chart)
    plt.figure(figsize=(8, 5))
    counts = df_preds["risk_level"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    total = len(df_preds)
    percentages = (counts / total) * 100
    
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False)
    plt.title("Transaction Risk Level Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Risk Level", fontsize=12, labelpad=10)
    plt.ylabel("Transaction Count", fontsize=12, labelpad=10)
    
    # Annotate bars with counts and percentages
    for i, p in enumerate(ax.patches):
        count = int(counts.values[i])
        pct = percentages.values[i]
        ax.annotate(f"{count}\n({pct:.2f}%)", 
                    (p.get_x() + p.get_width() / 2., p.get_height() - (p.get_height() * 0.15 if p.get_height() > 100 else -10)),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points",
                    color="white" if p.get_height() > 100 else "black", fontweight="bold")
                    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "risk_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Risk Score Histogram
    plt.figure(figsize=(9, 5.5))
    # Bin size of 5 for score 0-100
    sns.histplot(data=df_preds, x="risk_score", bins=20, kde=True, color="#4F46E5", edgecolor="white", alpha=0.7)
    
    # Add vertical lines and shade regions for thresholds
    low_boundary = low_threshold * 100
    high_boundary = high_threshold * 100
    
    plt.axvline(low_boundary, color="#F59E0B", linestyle="--", linewidth=1.5, label=f"Low Risk Limit ({low_boundary})")
    plt.axvline(high_boundary, color="#EF4444", linestyle="--", linewidth=1.5, label=f"High Risk Limit ({high_boundary})")
    
    # Shade regions
    plt.axvspan(0, low_boundary, color="#10B981", alpha=0.1, label="Low Risk Zone")
    plt.axvspan(low_boundary, high_boundary, color="#F59E0B", alpha=0.1, label="Medium Risk Zone")
    plt.axvspan(high_boundary, 100, color="#EF4444", alpha=0.1, label="High Risk Zone")
    
    plt.title("Transaction Risk Score Distribution (0-100 Scale)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Business Risk Score", fontsize=12, labelpad=10)
    plt.ylabel("Frequency", fontsize=12, labelpad=10)
    plt.xlim(-2, 102)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "risk_score_histogram.png"), dpi=300)
    plt.close()
    
    # 3. Probability Distribution Plot
    plt.figure(figsize=(9, 5.5))
    # Plotting probability distribution with detailed KDE representation
    sns.kdeplot(data=df_preds, x="probability_score", fill=True, color="#818CF8", alpha=0.5, bw_adjust=0.5)
    sns.rugplot(data=df_preds, x="probability_score", color="#4F46E5", alpha=0.3)
    
    plt.title("Model Prediction Probability Density (KDE)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Raw Prediction Probability (Anomaly)", fontsize=12, labelpad=10)
    plt.ylabel("Density", fontsize=12, labelpad=10)
    plt.xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "probability_distribution.png"), dpi=300)
    plt.close()
    
    logger.info("Risk visualizations generated and saved successfully.")

def run_risk_pipeline():
    """
    Execute the end-to-end Risk Scoring Engine pipeline.
    Loads configurations, retrieves test data details, calls model to predict risk probabilities,
    saves the prediction table, compiles summary statistics, and generates visualizations.
    """
    logger.info("=========================================")
    logger.info("Initializing Risk Scoring Engine Pipeline")
    logger.info("=========================================")
    
    # 1. Load Configurations
    cfg = load_risk_config()
    model_path = cfg["model_path"]
    low_thresh = cfg["low_risk_threshold"]
    high_thresh = cfg["high_risk_threshold"]
    
    # Resolve relative model path
    if not os.path.isabs(model_path):
        model_path = os.path.join(config.BASE_DIR, model_path)
        
    logger.info(f"Model Path:         {model_path}")
    logger.info(f"Low Risk Threshold:  {low_thresh}")
    logger.info(f"High Risk Threshold: {high_thresh}")
    
    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train/tune the model first.")
        
    # 2. Reconstruct test metadata & load processed X_test
    df_meta = get_test_metadata()
    X_test = load_dataset(config.X_TEST_PATH)
    
    if len(df_meta) != len(X_test):
        raise ValueError(f"Metadata length ({len(df_meta)}) and feature dataset length ({len(X_test)}) mismatch!")
        
    # 3. Load Trained Model & Predict Probabilities
    logger.info(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    
    logger.info("Calculating prediction probabilities on test set...")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        logger.warning("Model does not support predict_proba. Generating binary indicator probabilities.")
        probabilities = model.predict(X_test).astype(float)
        
    predictions = model.predict(X_test)
    
    # 4. Generate prediction table and save
    df_preds = generate_prediction_table(df_meta, predictions, probabilities, low_thresh, high_thresh)
    os.makedirs(os.path.dirname(config.RISK_PREDICTIONS_PATH), exist_ok=True)
    df_preds.to_csv(config.RISK_PREDICTIONS_PATH, index=False)
    logger.info(f"Saved risk predictions table to: {config.RISK_PREDICTIONS_PATH}")
    
    # 5. Compile summary statistics and save
    summary = calculate_summary_statistics(df_preds)
    # Inject threshold metadata into summary
    summary["configured_low_risk_threshold"] = low_thresh
    summary["configured_high_risk_threshold"] = high_thresh
    
    os.makedirs(os.path.dirname(config.RISK_SUMMARY_PATH), exist_ok=True)
    with open(config.RISK_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved risk summary report to: {config.RISK_SUMMARY_PATH}")
    
    # 6. Generate visualizations
    generate_risk_visualizations(df_preds, low_thresh, high_thresh, config.FIGURES_DIR)
    
    logger.info("=========================================")
    logger.info("Risk Scoring Engine Execution Completed")
    logger.info("=========================================")

if __name__ == "__main__":
    run_risk_pipeline()
