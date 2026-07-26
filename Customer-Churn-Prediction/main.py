"""
Main Entry Point for the Financial Transaction Risk & Anomaly Engine ML Pipeline.

This module orchestrates the end-to-end machine learning pipeline,
beginning with initialization of logging, loading configurations,
and running the data preprocessing pipeline.
"""

from src import config
from src.utils import setup_logger
from src.data_preprocessing import preprocess_pipeline

def main():
    """
    Execute the machine learning pipeline.
    
    This function coordinates the execution of:
    1. Initializing logging.
    2. Loading dataset configuration constants.
    3. Executing data loading, checks, and cleaning.
    """
    # 1. Initialize Logger
    logger = setup_logger("main_pipeline")
    logger.info("=========================================")
    logger.info("Starting Financial Transaction Risk Engine")
    logger.info("=========================================")
    
    try:
        # 2. Run Preprocessing Pipeline
        logger.info("Starting preprocessing step...")
        df_clean = preprocess_pipeline(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
        logger.info(f"Preprocessing completed. Cleaned dataset has {df_clean.shape[0]} rows.")
        
        logger.info("=========================================")
        logger.info("Pipeline Step 1 (Preprocessing) completed successfully")
        logger.info("=========================================")
        
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
