"""
Visualization Module.

This module provides functions for generating and saving beautifully formatted evaluation plots,
including Confusion Matrices, ROC Curves, and Precision-Recall Curves.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
from src.utils import setup_logger

logger = setup_logger("visualization")

# Apply modern aesthetic settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "grid.alpha": 0.4
})

def plot_confusion_matrix(y_true, y_pred, filepath: str):
    """
    Generate and save a beautifully styled Confusion Matrix heatmap.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        filepath (str): Destination file path to save the figure.
    """
    logger.info(f"Generating Confusion Matrix heatmap at {filepath}...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    # Draw heatmap with integer formatting and soft blue colormap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=True,
        xticklabels=["Normal (0)", "Anomaly (1)"],
        yticklabels=["Normal (0)", "Anomaly (1)"],
        annot_kws={"size": 12, "weight": "bold"}
    )
    
    plt.title("Confusion Matrix (Baseline Model)", pad=15, fontweight="bold")
    plt.xlabel("Predicted Class", labelpad=10)
    plt.ylabel("Actual Class", labelpad=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info("Confusion Matrix heatmap saved successfully.")

def plot_roc_curve(y_true, y_prob, filepath: str):
    """
    Generate and save a Receiver Operating Characteristic (ROC) Curve.

    Args:
        y_true: Ground truth target labels.
        y_prob: Predicted anomaly probabilities.
        filepath (str): Destination file path to save the figure.
    """
    logger.info(f"Generating ROC Curve plot at {filepath}...")
    if y_prob is None:
        logger.warning("Probabilities are None; skipping ROC Curve visualization.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4F46E5", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#EF4444", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.5000)")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", labelpad=10)
    plt.title("Receiver Operating Characteristic (ROC) Curve", pad=15, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info("ROC Curve plot saved successfully.")

def plot_precision_recall_curve(y_true, y_prob, filepath: str):
    """
    Generate and save a Precision-Recall Curve.

    Args:
        y_true: Ground truth target labels.
        y_prob: Predicted anomaly probabilities.
        filepath (str): Destination file path to save the figure.
    """
    logger.info(f"Generating Precision-Recall Curve plot at {filepath}...")
    if y_prob is None:
        logger.warning("Probabilities are None; skipping Precision-Recall Curve visualization.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    # Use clean green shades for precision-recall focus
    plt.step(recall, precision, color="#10B981", where="post", alpha=0.8, lw=2.5, label=f"PR Curve (AP = {avg_precision:.4f})")
    plt.fill_between(recall, precision, alpha=0.15, color="#10B981", step="post")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (TPR)", labelpad=10)
    plt.ylabel("Precision", labelpad=10)
    plt.title("Precision-Recall (PR) Curve", pad=15, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info("Precision-Recall Curve plot saved successfully.")
