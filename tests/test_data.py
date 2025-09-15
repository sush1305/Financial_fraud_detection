"""
Tests for data loading and preprocessing.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data

def test_load_dataset():
    """Test loading the dataset."""
    # Create a sample dataset
    data = {
        'Time': [1, 2, 3, 4, 5],
        'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'V2': [1.1, 1.2, 1.3, 1.4, 1.5],
        'Amount': [100, 200, 300, 400, 500],
        'Class': [0, 0, 1, 0, 1]
    }
    test_file = Path('test_data.csv')
    pd.DataFrame(data).to_csv(test_file, index=False)
    
    # Test loading
    df = load_dataset(test_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert 'Class' in df.columns
    
    # Clean up
    test_file.unlink()

def test_preprocess_data():
    """Test data preprocessing."""
    data = {
        'Time': [1, 2, 3, 4, 5],
        'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'V2': [1.1, 1.2, 1.3, 1.4, 1.5],
        'Amount': [100, 200, 300, 400, 500],
        'Class': [0, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Test preprocessing
    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == 5
    assert 'Time' not in X.columns  # Should be dropped