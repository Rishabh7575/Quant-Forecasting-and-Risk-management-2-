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
