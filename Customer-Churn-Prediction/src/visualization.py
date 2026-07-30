"""
Visualization Module.

This module provides functions for generating and saving beautifully formatted evaluation plots,
including Confusion Matrices, ROC Curves, and Precision-Recall Curves.
"""

import os
import pandas as pd
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

def plot_comparison_metrics(df_comparison: pd.DataFrame, filepath: str):
    """
    Generate and save a grouped bar chart comparing multiple classification metrics
    for Logistic Regression vs. Random Forest.

    Args:
        df_comparison (pd.DataFrame): Comparison metrics DataFrame.
        filepath (str): Destination file path to save the bar chart.
    """
    logger.info(f"Generating metrics comparison bar chart at {filepath}...")
    
    # Melt wide DataFrame to long-form for seaborn
    df_melted = df_comparison.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )
    
    plt.figure(figsize=(8, 6))
    
    # Plot using a cohesive color palette
    ax = sns.barplot(
        x="Metric", 
        y="Score", 
        hue="Model", 
        data=df_melted, 
        palette={"Logistic Regression": "#818CF8", "Random Forest": "#4F46E5"}
    )
    
    # Style details
    plt.title("Baseline Model Performance Comparison", pad=20, fontweight="bold")
    plt.xlabel("Evaluation Metric", labelpad=10)
    plt.ylabel("Score Value", labelpad=10)
    plt.ylim([0, 1.1])  # Keep space for labels
    plt.legend(title="Classifier", loc="lower right", frameon=True)
    
    # Annotate bar heights
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height) or height == 0:
            continue
        ax.annotate(
            f"{height:.2%}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center', 
            va='bottom', 
            xytext=(0, 3), 
            textcoords='offset points',
            fontsize=9,
            fontweight="bold"
        )
        
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info("Metrics comparison bar chart saved successfully.")

def plot_comparison_roc_curves(y_true, y_prob_lr, y_prob_rf, filepath: str):
    """
    Generate and save overlaid Receiver Operating Characteristic (ROC) Curves
    comparing both models on a single figure.

    Args:
        y_true: Ground truth target labels.
        y_prob_lr: Predicted anomaly probabilities for Logistic Regression.
        y_prob_rf: Predicted anomaly probabilities for Random Forest.
        filepath (str): Destination file path to save the ROC curve figure.
    """
    logger.info(f"Generating overlaid ROC curves at {filepath}...")
    
    plt.figure(figsize=(6, 5))
    
    # Plot Logistic Regression
    if y_prob_lr is not None:
        fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)
        auc_lr = auc(fpr_lr, tpr_lr)
        plt.plot(fpr_lr, tpr_lr, color="#818CF8", lw=2, label=f"Logistic Regression (AUC = {auc_lr:.4f})")
        
    # Plot Random Forest
    if y_prob_rf is not None:
        fpr_rf, tpr_rf, _ = roc_curve(y_true, y_prob_rf)
        auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, color="#4F46E5", lw=2.5, label=f"Random Forest (AUC = {auc_rf:.4f})")
        
    # Random guess baseline
    plt.plot([0, 1], [0, 1], color="#EF4444", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.5000)")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", labelpad=10)
    plt.title("Receiver Operating Characteristic (ROC) Curve Comparison", pad=15, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info("Overlaid ROC curves saved successfully.")

