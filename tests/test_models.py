"""
Tests for model training and evaluation.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.train import train_model, evaluate_model

def test_train_model():
    """Test model training."""
    # Create sample data
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    # Test training
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')

def test_evaluate_model():
    """Test model evaluation."""
    # Create sample data
    X_test = np.random.rand(50, 5)
    y_test = np.random.randint(0, 2, 50)
    
    # Train a simple model
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test evaluation
    metrics = evaluate_model(model, X_test, y_test)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics