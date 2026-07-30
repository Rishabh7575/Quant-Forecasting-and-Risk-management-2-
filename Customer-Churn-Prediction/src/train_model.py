"""
Model Training Module.

This module contains logic for training a baseline Logistic Regression model
for the Financial Transaction Risk & Anomaly Engine.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

def tune_random_forest(X_train, y_train, random_state: int = config.RANDOM_STATE):
    """
    Perform systematic hyperparameter tuning of the Random Forest model
    using 5-fold cross-validation over key parameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (best_estimator, best_params, best_score, total_combinations, cv_results_df)
    """
    logger.info("Setting up hyperparameter tuning for Random Forest Classifier...")
    
    # Define hyperparameter search space
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    
    # Base classifier
    rf = RandomForestClassifier(random_state=random_state)
    
    # We choose RandomizedSearchCV for efficiency and coverage
    n_iter = 15
    cv = 5
    
    logger.info(f"Running RandomizedSearchCV (cv={cv}, n_iter={n_iter}, scoring='f1')...")
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",  # Optimize for minority class f1-score due to severe class imbalance
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    best_params = search.best_params_
    best_score = search.best_score_
    
    logger.info("Hyperparameter tuning completed successfully.")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best CV F1-score: {best_score:.4%}")
    
    # Convert cross-validation results to DataFrame
    cv_results = pd.DataFrame(search.cv_results_)
    
    return search.best_estimator_, best_params, best_score, n_iter, cv_results

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
