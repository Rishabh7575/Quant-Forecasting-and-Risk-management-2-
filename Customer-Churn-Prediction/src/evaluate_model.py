"""
Model Evaluation Module.

This module provides functions to assess the performance of trained machine
learning models for the Financial Transaction Risk & Anomaly Engine.
"""

import os
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from src import config
from src.utils import setup_logger

logger = setup_logger("evaluate_model")

def evaluate(model, X_test, y_test):
    """
    Evaluate the model on test data and compute key performance metrics.

    Args:
        model: The trained model instance.
        X_test: Testing features.
        y_test: Testing labels (ground truth).

    Returns:
        tuple: (metrics_dict, y_pred, y_prob)
    """
    logger.info("Starting baseline model evaluation on test set...")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    
    logger.info("=========================================")
    logger.info("BASELINE EVALUATION REPORT")
    logger.info("-----------------------------------------")
    logger.info(f"Accuracy:         {accuracy:.4%}")
    logger.info(f"Precision:        {precision:.4%}")
    logger.info(f"Recall:           {recall:.4%}")
    logger.info(f"F1-score:         {f1:.4%}")
    if roc_auc is not None:
        logger.info(f"ROC-AUC:          {roc_auc:.4%}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report_str}")
    logger.info("=========================================")
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
    
    return metrics, y_pred, y_prob

def save_predictions(y_true, y_pred, y_prob, filepath: str):
    """
    Save target ground truth, predictions, and prediction probabilities to disk.

    Args:
        y_true: Ground truth target series.
        y_pred: Predicted class labels.
        y_prob: Predicted anomaly probabilities.
        filepath (str): Destination path for CSV.
    """
    logger.info(f"Saving predictions to: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_preds = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "probability": y_prob
    })
    df_preds.to_csv(filepath, index=False)
    logger.info("Predictions saved successfully.")

def save_metrics(metrics: dict, filepath: str):
    """
    Save evaluation metrics dictionary in JSON format to disk.

    Args:
        metrics (dict): Dictionary of model metrics.
        filepath (str): Destination path for JSON.
    """
    logger.info(f"Saving metrics report to: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics report saved successfully.")

def compare_models(metrics_lr: dict, metrics_rf: dict, filepath: str) -> pd.DataFrame:
    """
    Compare performance metrics of Logistic Regression and Random Forest models,
    save the comparison as a CSV file, and log a formatted comparison table.

    Args:
        metrics_lr (dict): Evaluation metrics dictionary for Logistic Regression.
        metrics_rf (dict): Evaluation metrics dictionary for Random Forest.
        filepath (str): Destination path for CSV.

    Returns:
        pd.DataFrame: Comparison DataFrame.
    """
    logger.info("Comparing model performance metrics...")
    
    comparison_data = [
        {
            "Model": "Logistic Regression",
            "Accuracy": metrics_lr["accuracy"],
            "Precision": metrics_lr["precision"],
            "Recall": metrics_lr["recall"],
            "F1-Score": metrics_lr["f1_score"],
            "ROC-AUC": metrics_lr["roc_auc"]
        },
        {
            "Model": "Random Forest",
            "Accuracy": metrics_rf["accuracy"],
            "Precision": metrics_rf["precision"],
            "Recall": metrics_rf["recall"],
            "F1-Score": metrics_rf["f1_score"],
            "ROC-AUC": metrics_rf["roc_auc"]
        }
    ]
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save to disk
    logger.info(f"Saving comparison metrics CSV to: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_comparison.to_csv(filepath, index=False)
    
    # Log formatted table
    logger.info("=========================================")
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("-----------------------------------------")
    logger.info("\n" + df_comparison.to_string(index=False))
    logger.info("=========================================")
    
    return df_comparison

def compare_three_models(metrics_lr: dict, metrics_rf_base: dict, metrics_rf_tuned: dict, filepath: str) -> pd.DataFrame:
    """
    Compare performance metrics of three models (Logistic Regression, Baseline Random Forest,
    and Tuned Random Forest), compute improvement percentages, save as CSV, and log summary.

    Args:
        metrics_lr (dict): Evaluation metrics dictionary for Logistic Regression.
        metrics_rf_base (dict): Evaluation metrics dictionary for Baseline Random Forest.
        metrics_rf_tuned (dict): Evaluation metrics dictionary for Tuned Random Forest.
        filepath (str): Destination path for comparison CSV.

    Returns:
        pd.DataFrame: Comparison DataFrame.
    """
    logger.info("Comparing three models and computing improvement percentages...")
    
    comparison_data = [
        {
            "Model": "Logistic Regression",
            "Accuracy": metrics_lr["accuracy"],
            "Precision": metrics_lr["precision"],
            "Recall": metrics_lr["recall"],
            "F1-Score": metrics_lr["f1_score"],
            "ROC-AUC": metrics_lr["roc_auc"]
        },
        {
            "Model": "Random Forest (Baseline)",
            "Accuracy": metrics_rf_base["accuracy"],
            "Precision": metrics_rf_base["precision"],
            "Recall": metrics_rf_base["recall"],
            "F1-Score": metrics_rf_base["f1_score"],
            "ROC-AUC": metrics_rf_base["roc_auc"]
        },
        {
            "Model": "Random Forest (Tuned)",
            "Accuracy": metrics_rf_tuned["accuracy"],
            "Precision": metrics_rf_tuned["precision"],
            "Recall": metrics_rf_tuned["recall"],
            "F1-Score": metrics_rf_tuned["f1_score"],
            "ROC-AUC": metrics_rf_tuned["roc_auc"]
        }
    ]
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Calculate improvement percentages of Tuned RF relative to Baseline RF
    base_f1 = metrics_rf_base["f1_score"]
    tuned_f1 = metrics_rf_tuned["f1_score"]
    f1_improvement = ((tuned_f1 - base_f1) / base_f1) if base_f1 != 0 else 0
    
    base_recall = metrics_rf_base["recall"]
    tuned_recall = metrics_rf_tuned["recall"]
    recall_improvement = ((tuned_recall - base_recall) / base_recall) if base_recall != 0 else 0
    
    # Save comparison metrics CSV
    logger.info(f"Saving three-way comparison CSV to: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_comparison.to_csv(filepath, index=False)
    
    logger.info("=========================================")
    logger.info("THREE-WAY MODEL COMPARISON SUMMARY")
    logger.info("-----------------------------------------")
    logger.info("\n" + df_comparison.to_string(index=False))
    logger.info(f"\nF1 Improvement (Tuned vs. Baseline RF): {f1_improvement:+.2%}")
    logger.info(f"Recall Improvement (Tuned vs. Baseline RF): {recall_improvement:+.2%}")
    logger.info("=========================================")
    
    return df_comparison


