"""
Data processing pipeline for the fraud detection system.

This module handles data cleaning, preprocessing, and preparation
for model training and evaluation.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing and preparation for fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Class'
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert Time from seconds to hours
        df['Time'] = df['Time'] / 3600
        
        # Handle missing values if any
        if df.isnull().sum().sum() > 0:
            logger.warning("Missing values found. Filling with median for numerical columns.")
            df = df.fillna(df.median())
        
        # Log class distribution
        class_dist = df[self.target_column].value_counts(normalize=True)
        logger.info(f"Class distribution: \n{class_dist}")
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split into train: {X_train.shape[0]} samples, test: {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Balanced features and target
        """
        logger.info("Handling class imbalance with SMOTE...")
        
        # Apply SMOTE only to training data
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series,
                          output_dir: str = "data/processed") -> None:
        """Save processed data to disk.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure y_train and y_test are pandas Series with proper indices
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, name='target')
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test, name='target')
            
        # Reset indices to ensure alignment
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Save features and target
        X_train.to_parquet(output_path / "X_train.parquet")
        X_test.to_parquet(output_path / "X_test.parquet")
        y_train.to_frame('target').to_parquet(output_path / "y_train.parquet")
        y_test.to_frame('target').to_parquet(output_path / "y_test.parquet")
        
        logger.info(f"Processed data saved to {output_path.absolute()}")


def process_data_pipeline(input_file: str, output_dir: str = "data/processed") -> None:
    """Complete data processing pipeline.
    
    Args:
        input_file: Path to input data file
        output_dir: Directory to save processed data
    """
    from src.data.load_data import DataLoader
    
    # Initialize components
    data_loader = DataLoader(data_dir=str(Path(input_file).parent))
    processor = DataProcessor()
    
    # Load and preprocess data
    df = data_loader.load_credit_card_data(Path(input_file).name)
    df = processor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(df)
    
    # Handle class imbalance (only on training data)
    X_train, y_train = processor.handle_imbalance(X_train, y_train)
    
    # Save processed data
    processor.save_processed_data(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    # Example usage
    process_data_pipeline("data/raw/creditcard.csv")
