"""
Data preprocessing utilities for the fraud detection system.
"""
from typing import Tuple, Union, Optional, Dict, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for transaction data."""
    
    def __init__(self, time_col: str = 'Time', amount_col: str = 'Amount'):
        """Initialize the feature engineering transformer.
        
        Args:
            time_col: Name of the time column.
            amount_col: Name of the amount column.
        """
        self.time_col = time_col
        self.amount_col = amount_col
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to the input data.
        
        Args:
            X: Input DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        X = X.copy()
        
        # Convert time to hours since first transaction
        if self.time_col in X.columns:
            X[f'{self.time_col}_hour'] = X[self.time_col] / 3600  # Convert to hours
            
        # Log transform amount to handle skew
        if self.amount_col in X.columns:
            X[f'log_{self.amount_col}'] = np.log1p(X[self.amount_col])
        
        # Add interaction terms (example)
        if 'V1' in X.columns and 'V2' in X.columns:
            X['V1_V2_ratio'] = X['V1'] / (X['V2'] + 1e-6)  # Add small constant to avoid division by zero
        
        return X


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str] = None,
    handle_imbalance: bool = True,
    sampling_strategy: str = 'smote',
    random_state: int = 42
) -> Pipeline:
    """Create a preprocessing pipeline for the fraud detection model.
    
    Args:
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
        handle_imbalance: Whether to handle class imbalance.
        sampling_strategy: Sampling strategy to use ('smote', 'adasyn', or 'undersample').
        random_state: Random seed for reproducibility.
        
    Returns:
        A scikit-learn Pipeline for preprocessing.
    """
    # Default to empty list if no categorical features
    if categorical_features is None:
        categorical_features = []
    
    # Numeric transformations
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # RobustScaler is less sensitive to outliers
    ])
    
    # Categorical transformations
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the pipeline
    if handle_imbalance:
        # Add resampling step for handling class imbalance
        if sampling_strategy == 'smote':
            resampler = SMOTE(random_state=random_state)
        elif sampling_strategy == 'adasyn':
            resampler = ADASYN(random_state=random_state)
        elif sampling_strategy == 'undersample':
            resampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('resampler', resampler)
        ])
    else:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    return pipeline


def detect_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to check for outliers.
        threshold: Z-score threshold for outlier detection.
        
    Returns:
        Boolean Series indicating outlier rows.
    """
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    return (z_scores > threshold).any(axis=1)


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'Class',
    test_size: float = 0.2,
    random_state: int = 42,
    handle_imbalance: bool = True,
    sampling_strategy: str = 'smote'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for model training.
    
    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        handle_imbalance: Whether to handle class imbalance.
        sampling_strategy: Sampling strategy to use ('smote', 'adasyn', or 'undersample').
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipeline without resampling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Handle class imbalance if needed
    if handle_imbalance:
        if sampling_strategy == 'smote':
            resampler = SMOTE(random_state=random_state)
        elif sampling_strategy == 'adasyn':
            resampler = ADASYN(random_state=random_state)
        elif sampling_strategy == 'undersample':
            resampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        X_train, y_train = resampler.fit_resample(X_train, y_train)
    
    logger.info(f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
