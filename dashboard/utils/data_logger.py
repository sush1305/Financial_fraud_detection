import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Data directory set to: {DATA_DIR.absolute()}")

def log_fraudulent_transaction(transaction_data):
    """
    Log fraudulent transactions to a CSV file with timestamp
    
    Args:
        transaction_data (dict): Dictionary containing transaction details
    """
    try:
        # Add timestamp if not present
        if 'detection_time' not in transaction_data:
            transaction_data['detection_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create DataFrame from transaction data
        df = pd.DataFrame([transaction_data])
        
        # Log the transaction data for debugging
        logger.info(f"Processing transaction: {transaction_data}")
        
        # Define CSV file path with current date
        date_str = datetime.now().strftime('%Y%m%d')
        csv_filename = f'fraud_transactions_{date_str}.csv'
        csv_path = DATA_DIR / csv_filename
        logger.info(f"Attempting to write to: {csv_path.absolute()}")
        
        # Write to CSV (append if file exists, create new otherwise)
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        
        return True, str(csv_path)
    except Exception as e:
        print(f"Error logging transaction: {e}")
        return False, str(e)

def get_fraud_transactions(days_back=7):
    """
    Get all fraudulent transactions from the last N days
    
    Args:
        days_back (int): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame containing fraud transactions
    """
    try:
        all_frauds = []
        for i in range(days_back):
            date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y%m%d')
            csv_path = DATA_DIR / f'fraud_transactions_{date}.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    all_frauds.append(df)
                except pd.errors.EmptyDataError:
                    continue
        
        if all_frauds:
            return pd.concat(all_frauds, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading fraud transactions: {e}")
        return pd.DataFrame()
