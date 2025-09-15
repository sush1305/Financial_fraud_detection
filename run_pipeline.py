"""
Main pipeline for the Fraud Detection System.

This script runs the complete pipeline from data loading to model training and evaluation.
"""
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from src.data.load_data import DataLoader
from src.data.process_data import DataProcessor, process_data_pipeline
from src.models.train import train_evaluate_pipeline
from src.features.engineering import FeatureEngineer
from src.utils.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fraud_detection.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration."""
    try:
        config = get_config()
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def load_and_preprocess_data(config):
    """Load and preprocess the data."""
    try:
        # Create necessary directories
        processed_dir = Path(config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Run data processing pipeline
        logger.info("Starting data processing pipeline...")
        process_data_pipeline(
            input_file=config['data']['raw_data_path'],
            output_dir=config['data']['processed_dir']
        )
        
        # Load processed data
        logger.info("Loading processed data...")
        X_train = pd.read_parquet(processed_dir / "X_train.parquet")
        X_test = pd.read_parquet(processed_dir / "X_test.parquet")
        y_train = pd.read_parquet(processed_dir / "y_train.parquet").squeeze()
        y_test = pd.read_parquet(processed_dir / "y_test.parquet").squeeze()
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in data loading/preprocessing: {e}")
        raise

def train_model(X_train, X_test, y_train, y_test, config):
    """Train and evaluate the model."""
    try:
        logger.info("Starting model training...")
        
        # Train and evaluate the model
        model, metrics = train_evaluate_pipeline(
            X_train=X_train.values,
            y_train=y_train.values,
            X_test=X_test.values,
            y_test=y_test.values,
            model_type=config['model']['type'],
            calibrate=config['model']['calibrate'],
            save_model_flag=True,
            output_dir=config['data']['model_dir']
        )
        
        logger.info(f"Model training completed. Test ROC-AUC: {metrics.get('test_roc_auc', 0):.4f}")
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def main():
    """Main function to run the pipeline."""
    try:
        logger.info("Starting Fraud Detection Pipeline...")
        
        # Load configuration
        config = load_config()
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
        
        # Train and evaluate model
        model, metrics = train_model(X_train, X_test, y_train, y_test, config)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the pipeline
    main()
