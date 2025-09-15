"""
Utility helper functions for the fraud detection system.
"""
import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        raise

def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Use joblib or pickle based on file extension
        if filepath.endswith('.joblib'):
            import joblib
            joblib.dump(model, filepath)
        else:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")
        raise

def load_model(filepath: str) -> Any:
    """Load a trained model from disk."""
    try:
        if filepath.endswith('.joblib'):
            import joblib
            return joblib.load(filepath)
        else:
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}")
        raise

def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving metrics to {filepath}: {e}")
        raise

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise

def get_timestamp() -> str:
    """Get current timestamp in a formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")