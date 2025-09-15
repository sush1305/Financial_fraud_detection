#!/usr/bin/env python
"""
Main script for the Fraud Detection System.

This script provides a command-line interface to train and evaluate fraud detection models.
"""
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Import project modules
from data.load_data import DataLoader, load_sample_data
from data.preprocess import prepare_data, FeatureEngineer
from models.train import train_evaluate_pipeline, load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fraud Detection System')
    
    # Data loading arguments
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing the data files')
    parser.add_argument('--data-file', type=str, default='creditcard.csv',
                        help='Name of the data file')
    parser.add_argument('--use-sample-data', action='store_true',
                        help='Use sample data for testing')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest', 'logistic_regression', 'isolation_forest'],
                        help='Type of model to train')
    parser.add_argument('--no-calibrate', action='store_false', dest='calibrate',
                        help='Disable probability calibration')
    
    # Training arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save the trained model')
    parser.add_argument('--no-save', action='store_false', dest='save_model',
                        help='Do not save the trained model')
    
    return parser.parse_args()

def main():
    """Main function to run the fraud detection pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(data_dir=args.data_dir)
    
    try:
        # Load data
        if args.use_sample_data:
            logger.info("Using sample data for testing")
            df = load_sample_data()
        else:
            # Load data from file
            data_path = os.path.join(args.data_dir, args.data_file)
            logger.info(f"Loading data from {data_path}")
            df = data_loader.load_data(args.data_file)
        
        # Basic data info
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Number of fraud cases: {df['Class'].sum()} out of {len(df)} ({df['Class'].mean()*100:.2f}%)")
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        feature_engineer = FeatureEngineer()
        df = feature_engineer.fit_transform(df)
        
        # Prepare data for training
        logger.info("Preparing data for training...")
        
        # Convert target to numpy array if it's a DataFrame/Series
        if hasattr(df['Class'], 'values'):
            y = df['Class'].values
        else:
            y = df['Class']
        
        # Get class distribution
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        X_train, X_test, y_train, y_test = prepare_data(
            df,
            target_col='Class',
            test_size=args.test_size,
            random_state=args.random_state,
            handle_imbalance=True,
            sampling_strategy='smote'
        )
        
        # Log class distribution after resampling
        y_train_counts = pd.Series(y_train).value_counts().to_dict()
        logger.info(f"Training set class distribution after resampling: {y_train_counts}")
        
        # Train and evaluate the model
        logger.info(f"Training {args.model_type} model...")
        
        # Store feature names for later use
        feature_names = getattr(X_train, 'columns', None)
        
        # Convert to numpy arrays if they are DataFrames
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        model, metrics = train_evaluate_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=args.model_type,
            calibrate=args.calibrate,
            save_model_flag=args.save_model,
            output_dir=args.output_dir
        )
        
        # Print evaluation metrics
        logger.info("\n=== Model Evaluation ===")
        
        # Check if metrics is a dictionary
        if not isinstance(metrics, dict):
            logger.error("Invalid metrics format received from training pipeline")
            return
        
        def format_metric(value, default='N/A'):
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
            return str(value)
        
        # Print all available metrics
        for metric_name in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc_roc', 'test_average_precision']:
            if metric_name in metrics:
                logger.info(f"{metric_name.replace('test_', '').title()}: {format_metric(metrics[metric_name])}")
        
        # Print confusion matrix if available
        if 'test_confusion_matrix' in metrics and isinstance(metrics['test_confusion_matrix'], dict):
            cm = metrics['test_confusion_matrix']
            logger.info("\n=== Confusion Matrix ===")
            logger.info(f"True Negatives: {cm.get('true_negatives', 0)}")
            logger.info(f"False Positives: {cm.get('false_positives', 0)}")
            logger.info(f"False Negatives: {cm.get('false_negatives', 0)}")
            logger.info(f"True Positives: {cm.get('true_positives', 0)}")
        
        # Log feature importances if available
        if 'feature_importances' in metrics and metrics['feature_importances']:
            logger.info("\n=== Feature Importances ===")
            # Log the first few feature importances to avoid cluttering the output
            importances = metrics['feature_importances']
            num_features = min(10, len(importances))
            logger.info(f"Top {num_features} most important features:")
            
            # Get feature names if available, otherwise use indices
            if feature_names is not None and hasattr(feature_names, 'tolist'):
                feature_names = feature_names.tolist()
            else:
                feature_names = [f'Feature {i}' for i in range(len(importances))]
            
            # Sort features by importance
            sorted_indices = np.argsort(importances)[::-1]  # Descending order
            
            for i in sorted_indices[:num_features]:
                logger.info(f"  {feature_names[i]}: {importances[i]:.4f}")
                
        logger.info("\nFraud detection pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
        
        logger.info("\nFraud detection pipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
