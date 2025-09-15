"""
Data loading utilities for the fraud detection system.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading of financial transaction data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the DataLoader with the directory containing the data files.
        
        Args:
            data_dir: Directory containing the data files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_credit_card_data(self, filename: str = "creditcard.csv") -> pd.DataFrame:
        """Load the credit card fraud detection dataset.
        
        Args:
            filename: Name of the CSV file containing the data.
            
        Returns:
            DataFrame containing the loaded data.
        """
        file_path = self.data_dir / filename
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found at {file_path}")
            raise
    
    def load_synthetic_data(self, filename: str = "synthetic_transactions.parquet") -> pd.DataFrame:
        """Load synthetic transaction data.
        
        Args:
            filename: Name of the file containing synthetic data.
            
        Returns:
            DataFrame containing the synthetic transaction data.
        """
        file_path = self.data_dir / filename
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded synthetic data from {file_path}")
            return df
        except FileNotFoundError:
            logger.warning(f"Synthetic data file not found at {file_path}")
            return pd.DataFrame()
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            df: Input DataFrame containing features and target.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test


def load_sample_data() -> pd.DataFrame:
    """Load sample data for testing and development.
    
    Returns:
        DataFrame containing features and target ('Class' column).
    """
    from sklearn.datasets import make_classification
    
    # Generate more samples with a higher fraud rate
    n_samples = 10000
    fraud_ratio = 0.05  # 5% fraud cases
    
    # Generate features with more informative structure for fraud detection
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=15,  # More informative features
        n_redundant=5,
        n_clusters_per_class=3,  # More complex class separation
        weights=[1-fraud_ratio],
        flip_y=0.01,  # Less label noise
        class_sep=1.5,  # Better separation between classes
        random_state=42
    )
    
    # Create more realistic feature names
    feature_columns = [
        'amount', 'time_since_last_transaction', 'avg_amount_1d', 'avg_amount_7d',
        'transaction_frequency', 'merchant_risk_score', 'device_trust_score',
        'ip_risk_score', 'user_avg_amount', 'user_std_amount',
        'distance_from_home', 'distance_from_work', 'time_of_day', 'day_of_week',
        'is_weekend', 'is_night', 'is_foreign_country', 'is_high_risk_country',
        'num_transactions_1h', 'num_transactions_24h', 'avg_transaction_value',
        'card_age_days', 'account_age_days', 'num_previous_chargebacks',
        'avg_transaction_value_7d', 'transaction_amount_diff', 'velocity_1h',
        'velocity_24h', 'category_risk_score', 'user_risk_score'
    ]
    
    df = pd.DataFrame(X, columns=feature_columns)
    df['Class'] = y
    
    # Make some features more predictive of fraud
    # Higher amounts are more likely to be fraud
    df.loc[df['Class'] == 1, 'amount'] = df.loc[df['Class'] == 1, 'amount'].abs() * 2.5
    
    # Night transactions are more likely to be fraud
    df.loc[df['Class'] == 1, 'is_night'] = np.random.choice(
        [0, 1], 
        size=df['Class'].sum(), 
        p=[0.3, 0.7]  # 70% of fraud happens at night
    )
    
    # Foreign transactions are more likely to be fraud
    df.loc[df['Class'] == 1, 'is_foreign_country'] = np.random.choice(
        [0, 1], 
        size=df['Class'].sum(), 
        p=[0.4, 0.6]  # 60% of fraud is from foreign countries
    )
    
    # High-risk merchants are more likely to have fraud
    df['merchant_risk_score'] = np.random.beta(2, 5, size=len(df))  # Most merchants are low risk
    df.loc[df['Class'] == 1, 'merchant_risk_score'] = np.random.beta(5, 2, size=df['Class'].sum())  # Fraud is more likely with high-risk merchants
    
    return df
