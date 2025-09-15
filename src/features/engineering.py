"""
Feature engineering module for the fraud detection system.

This module provides functions for creating, transforming, and selecting features
from transaction data to improve model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    
    This class provides methods for:
    1. Creating time-based features
    2. Creating transaction amount features
    3. Creating interaction features
    4. Handling missing values
    5. Feature scaling
    6. Feature selection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary with feature engineering parameters.
        """
        self.config = config or {}
        self.scaler = None
        self.feature_selector = None
        self.numeric_features = self.config.get('numeric_features', [])
        self.categorical_features = self.config.get('categorical_features', [])
        
    def create_time_features(self, df: pd.DataFrame, time_column: str = 'Time') -> pd.DataFrame:
        """
        Create time-based features from a timestamp column.
        
        Args:
            df: Input DataFrame containing the time column.
            time_column: Name of the time column.
            
        Returns:
            DataFrame with added time-based features.
        """
        if time_column not in df.columns:
            logger.warning(f"Time column '{time_column}' not found in DataFrame")
            return df
            
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column], unit='s')
        
        # Time of day features
        df[f'{time_column}_hour'] = df[time_column].dt.hour
        df[f'{time_column}_minute'] = df[time_column].dt.minute
        df[f'{time_column}_second'] = df[time_column].dt.second
        
        # Day of week (0=Monday, 6=Sunday)
        df[f'{time_column}_dayofweek'] = df[time_column].dt.dayofweek
        
        # Weekend flag
        df[f'{time_column}_is_weekend'] = df[time_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Time since first transaction
        df[f'{time_column}_since_start'] = (df[time_column] - df[time_column].min()).dt.total_seconds()
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame, amount_column: str = 'Amount') -> pd.DataFrame:
        """
        Create features based on transaction amount.
        
        Args:
            df: Input DataFrame containing the amount column.
            amount_column: Name of the amount column.
            
        Returns:
            DataFrame with added amount-based features.
        """
        if amount_column not in df.columns:
            logger.warning(f"Amount column '{amount_column}' not found in DataFrame")
            return df
            
        df = df.copy()
        
        # Log transform of amount
        df[f'{amount_column}_log'] = np.log1p(df[amount_column])
        
        # Binned amount
        bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
        labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000+']
        df[f'{amount_column}_binned'] = pd.cut(df[amount_column], bins=bins, labels=labels)
        
        # Is high amount flag
        df[f'{amount_column}_is_high'] = (df[amount_column] > df[amount_column].quantile(0.95)).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with added interaction features.
        """
        df = df.copy()
        
        # Example interaction features
        if 'V1' in df.columns and 'V2' in df.columns:
            df['V1_V2_ratio'] = df['V1'] / (df['V2'] + 1e-6)
            
        if 'V3' in df.columns and 'V4' in df.columns:
            df['V3_V4_sum'] = df['V3'] + df['V4']
            
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with handled missing values.
        """
        df = df.copy()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
                
        return df
    
    def fit_scaler(self, X: pd.DataFrame) -> None:
        """
        Fit the feature scaler.
        
        Args:
            X: Training data to fit the scaler on.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler = StandardScaler()
        self.scaler.fit(X[numeric_cols])
    
    def transform_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Transform features using the fitted scaler.
        
        Args:
            X: Input features to transform.
            fit_scaler: Whether to fit the scaler.
            
        Returns:
            Transformed features.
        """
        X = X.copy()
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Scale numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit_scaler or self.scaler is None:
            self.fit_scaler(X)
            
        if self.scaler is not None:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X
    
    def select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                       k: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the top k most important features.
        
        Args:
            X: Input features.
            y: Target variable.
            k: Number of features to select. If None, use all features.
            
        Returns:
            Tuple of (selected features, selected feature names).
        """
        if k is None or k >= X.shape[1] or y is None:
            return X, X.columns.tolist()
            
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        logger.info(f"Selected top {k} features: {selected_features}")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Get feature importance scores.
        
        Args:
            X: Input features.
            y: Target variable.
            
        Returns:
            Series with feature importance scores.
        """
        if self.feature_selector is None:
            self.select_features(X, y, k=X.shape[1])
            
        scores = pd.Series(self.feature_selector.scores_, index=X.columns)
        return scores.sort_values(ascending=False)

# Example usage
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Sample data
    data = {
        'Time': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'Amount': np.random.lognormal(mean=4, sigma=1.5, size=100),
        'V1': np.random.normal(0, 1, 100),
        'V2': np.random.normal(0, 1, 100),
        'Class': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create features
    df = feature_engineer.create_time_features(df, 'Time')
    df = feature_engineer.create_amount_features(df, 'Amount')
    df = feature_engineer.create_interaction_features(df)
    
    # Transform features
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_transformed = feature_engineer.transform_features(X, fit_scaler=True)
    
    # Select top 5 features
    X_selected, selected_features = feature_engineer.select_features(X_transformed, y, k=5)
    
    print("Original features:", X.columns.tolist())
    print("Selected features:", selected_features)
    print("Feature importance:\n", feature_engineer.get_feature_importance(X_transformed, y))