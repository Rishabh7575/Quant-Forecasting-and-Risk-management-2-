"""
Main Entry Point for the Financial Transaction Risk & Anomaly Engine ML Pipeline.

This module orchestrates the end-to-end machine learning pipeline,
beginning with initialization of logging, loading configurations,
and running the data preprocessing pipeline.
"""

from src import config
from src.utils import setup_logger, load_dataset
from src.data_preprocessing import preprocess_pipeline, prepare_ml_dataset
from src.feature_engineering import engineer_features
from src.train_model import train, save_model, train_random_forest, tune_random_forest
from src.evaluate_model import evaluate, save_predictions, save_metrics, compare_models, compare_three_models
from src.visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve,
    plot_comparison_metrics,
    plot_comparison_roc_curves,
    plot_three_way_comparison
)

def main():
    """
    Execute the machine learning pipeline.
    
    This function coordinates the execution of:
    1. Initializing logging.
    2. Executing data loading, checks, and cleaning.
    3. Running feature engineering.
    4. Splitting, preprocessing, and saving ML dataset splits.
    5. Training the baseline Logistic Regression model.
    6. Evaluating the model on the test split.
    7. Generating and saving evaluation metrics, predictions, and plots.
    """
    # 1. Initialize Logger
    logger = setup_logger("main_pipeline")
    logger.info("=========================================")
    logger.info("Starting Financial Transaction Risk Engine")
    logger.info("=========================================")
    
    try:
        # 2. Run Preprocessing Pipeline
        logger.info("Starting preprocessing step (data cleaning)...")
        df_clean = preprocess_pipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
        logger.info(f"Preprocessing completed. Cleaned dataset has {df_clean.shape[0]} rows.")
        
        # 3. Run Feature Engineering
        logger.info("Starting feature engineering step...")
        df_feat = engineer_features(df_clean)
        
        # 4. Prepare ML Dataset (Splitting, Preprocessing Pipeline, Saving)
        logger.info("Starting ML dataset preparation (scaling, encoding, splitting, saving)...")
        prepare_ml_dataset(df_feat)
        
        # 5. Train Baseline Logistic Regression Model
        logger.info("Loading prepared train/test splits for modeling...")
        X_train = load_dataset(config.X_TRAIN_PATH)
        X_test = load_dataset(config.X_TEST_PATH)
        y_train = load_dataset(config.y_TRAIN_PATH).iloc[:, 0]  # squeeze to 1D Series
        y_test = load_dataset(config.y_TEST_PATH).iloc[:, 0]
        
        logger.info("Starting baseline Logistic Regression model training...")
        lr_model = train(X_train, y_train)
        save_model(lr_model, config.BASELINE_MODEL_PATH)
        
        # Evaluate Logistic Regression
        logger.info("Evaluating baseline Logistic Regression model...")
        metrics_lr, y_pred_lr, y_prob_lr = evaluate(lr_model, X_test, y_test)
        save_predictions(y_test, y_pred_lr, y_prob_lr, config.BASELINE_PREDICTIONS_PATH)
        save_metrics(metrics_lr, config.BASELINE_METRICS_PATH)
        plot_confusion_matrix(y_test, y_pred_lr, config.BASELINE_CONF_MATRIX_PATH)
        plot_roc_curve(y_test, y_prob_lr, config.BASELINE_ROC_CURVE_PATH)
        plot_precision_recall_curve(y_test, y_prob_lr, config.BASELINE_PR_CURVE_PATH)
        
        # 6. Train Random Forest Model
        logger.info("Starting Random Forest model training...")
        rf_model = train_random_forest(X_train, y_train)
        save_model(rf_model, config.RF_MODEL_PATH)
        
        # Evaluate Random Forest
        logger.info("Evaluating Random Forest model...")
        metrics_rf, y_pred_rf, y_prob_rf = evaluate(rf_model, X_test, y_test)
        save_predictions(y_test, y_pred_rf, y_prob_rf, config.RF_PREDICTIONS_PATH)
        save_metrics(metrics_rf, config.RF_METRICS_PATH)
        
        # 7. Model Comparison and Visualization
        logger.info("Running baseline vs. Random Forest model comparison...")
        df_comparison = compare_models(metrics_lr, metrics_rf, config.COMPARISON_METRICS_PATH)
        
        logger.info("Generating model comparison visualizations...")
        plot_comparison_metrics(df_comparison, config.COMPARISON_BAR_CHART_PATH)
        plot_comparison_roc_curves(
            y_test, y_prob_lr, y_prob_rf, config.COMPARISON_ROC_CURVE_PATH
        )
        
        logger.info("=========================================")
        logger.info("Pipeline Baseline Models Training & Comparison completed successfully")
        logger.info("=========================================")
        
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()