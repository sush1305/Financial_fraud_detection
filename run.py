#!/usr/bin/env python3
"""
Main entry point for the Fraud Detection System.

This script provides a command-line interface to run different components of the system.
"""
import argparse
import logging
import sys
from pathlib import Path
import subprocess
import webbrowser
from datetime import datetime

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

def run_data_pipeline():
    """Run the data processing pipeline."""
    try:
        logger.info("Starting data processing pipeline...")
        from src.data.process_data import process_data_pipeline
        from src.utils.config import config
        
        process_data_pipeline(
            input_file=config['data']['raw_data_path'],
            output_dir=config['data']['processed_dir']
        )
        logger.info("Data processing completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}")
        return False

def train_model():
    """Train the fraud detection model."""
    try:
        logger.info("Starting model training...")
        from src.models.train import train_evaluate_pipeline
        from src.utils.config import config
        
        # Load processed data
        import pandas as pd
        X_train = pd.read_parquet(Path(config['data']['processed_dir']) / "X_train.parquet")
        X_test = pd.read_parquet(Path(config['data']['processed_dir']) / "X_test.parquet")
        y_train = pd.read_parquet(Path(config['data']['processed_dir']) / "y_train.parquet").squeeze()
        y_test = pd.read_parquet(Path(config['data']['processed_dir']) / "y_test.parquet").squeeze()
        
        # Train and evaluate model
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
        return True
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False

def run_dashboard(host='0.0.0.0', port=8050, no_browser=False):
    """Run the dashboard application."""
    try:
        from dashboard.app import app
        
        # Open the browser if requested before starting the server
        if not no_browser:
            webbrowser.open_new(f"http://{host}:{port}")
        
        logger.info(f"Starting dashboard at http://{host}:{port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Run the app directly in the main thread
        app.run(debug=True, host=host, port=port)
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return False

def main():
    """Parse command line arguments and execute the appropriate action."""
    parser = argparse.ArgumentParser(description='Fraud Detection System')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data pipeline command
    data_parser = subparsers.add_parser('process-data', help='Run data processing pipeline')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the fraud detection model')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch the dashboard')
    dashboard_parser.add_argument('--host', default='0.0.0.0', help='Host to run the dashboard on')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    dashboard_parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all components (process data, train model, and launch dashboard)')
    all_parser.add_argument('--host', default='0.0.0.0', help='Host to run the dashboard on')
    all_parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    all_parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    if args.command == 'process-data':
        success = run_data_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.command == 'train':
        success = train_model()
        sys.exit(0 if success else 1)
    
    elif args.command == 'dashboard':
        run_dashboard(host=args.host, port=args.port, no_browser=args.no_browser)
    
    elif args.command == 'all':
        # Run data pipeline
        if not run_data_pipeline():
            logger.error("Data pipeline failed. Exiting.")
            sys.exit(1)
        
        # Train model
        if not train_model():
            logger.error("Model training failed. Exiting.")
            sys.exit(1)
        
        # Run dashboard
        run_dashboard(host=args.host, port=args.port, no_browser=args.no_browser)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
