"""
Model Training Module.

This module contains logic for training a baseline Logistic Regression model
for the Financial Transaction Risk & Anomaly Engine.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src import config
from src.utils import setup_logger

logger = setup_logger("train_model")

def train(X_train, y_train, random_state: int = config.RANDOM_STATE) -> LogisticRegression:
    """
    Train a baseline Logistic Regression classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state (int): Seed for reproducibility.

    Returns:
        LogisticRegression: The trained model instance.
    """
    logger.info("Initializing and training baseline Logistic Regression model...")
    # Use max_iter=1000 to ensure convergence on normalized features
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training completed successfully.")
    return model

def train_random_forest(X_train, y_train, random_state: int = config.RANDOM_STATE) -> RandomForestClassifier:
    """
    Train a baseline Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state (int): Seed for reproducibility.

    Returns:
        RandomForestClassifier: The trained model instance.
    """
    logger.info("Initializing and training baseline Random Forest model...")
    # Use standard default parameters (n_estimators=100) and fixed random_state
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Random Forest training completed successfully.")
    return model


def save_model(model, filepath: str):
    """
    Serialize and save the trained model to disk.

    Args:
        model: The trained model instance.
        filepath (str): The destination path to save the model.
    """
    logger.info(f"Saving model object to: {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model saved successfully.")
