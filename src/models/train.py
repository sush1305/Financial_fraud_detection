"""
Model training and evaluation for the fraud detection system.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import logging

# Model imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define# Model configurations
MODEL_CONFIGS = {
    'xgboost': {
        'model_class': XGBClassifier,
        'params': {
            'n_estimators': 200,  # Increased number of trees
            'max_depth': 8,  # Deeper trees for more complex patterns
            'learning_rate': 0.05,  # Lower learning rate for better generalization
            'min_child_weight': 1,  # Allow smaller leaf nodes
            'gamma': 0.1,  # Regularization
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 19,  # Approximate ratio of negative to positive classes (95/5)
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'aucpr',  # Better for imbalanced data
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0  # L2 regularization
        }
    },
    'random_forest': {
        'model_class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced_subsample',
            'max_samples': 0.8
        }
    },
    'logistic_regression': {
        'model_class': LogisticRegression,
        'params': {
            'penalty': 'l2',
            'C': 0.1,  # Stronger regularization
            'solver': 'saga',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'l1_ratio': 0.5  # For elasticnet
        }
    },
    'isolation_forest': {
        'model_class': IsolationForest,
        'params': {
            'n_estimators': 200,
            'max_samples': 256,
            'contamination': 'auto',  # Auto-detect contamination
            'max_features': 1.0,  # Use all features
            'bootstrap': False,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0
        }
    }
}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'xgboost',
    calibrate: bool = True,
    cv_folds: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    """Train a fraud detection model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: Type of model to train ('random_forest', 'xgboost', 'logistic_regression', 'isolation_forest').
        calibrate: Whether to calibrate the model's probability estimates.
        cv_folds: Number of cross-validation folds for calibration.
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    logger.info(f"Training {model_type} model...")
    
    # Initialize model
    model_config = MODEL_CONFIGS[model_type]
    model = model_config['model_class'](**model_config['params'])
    
    # Special handling for Isolation Forest (unsupervised)
    if model_type == 'isolation_forest':
        # For Isolation Forest, we don't use y_train for fitting
        model.fit(X_train)
        return model, {}
    
    # For supervised models
    if calibrate:
        # Use CalibratedClassifierCV for better probability calibration
        calibrated_model = CalibratedClassifierCV(
            model,
            method='sigmoid',
            cv=min(cv_folds, 5),  # Use min to avoid too many folds with small datasets
            n_jobs=-1
        )
        calibrated_model.fit(X_train, y_train)
        model = calibrated_model
    else:
        model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = calculate_metrics(y_train, y_pred, y_proba, 'train')
    
    # Safely format the AUC score
    auc_score = metrics.get('train_auc_roc', 'N/A')
    if isinstance(auc_score, (int, float)):
        logger.info(f"{model_type} training complete. Train AUC: {auc_score:.4f}")
    else:
        logger.info(f"{model_type} training complete. Train AUC: {auc_score}")
    
    return model, metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'xgboost',
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate a trained model on test data.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        model_type: Type of the model.
        threshold: Probability threshold for classification.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating {model_type} model...")
    logger.info(f"Input shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
    logger.info(f"Unique classes in y_test: {np.unique(y_test)}")
    
    # Special handling for Isolation Forest
    if model_type == 'isolation_forest':
        logger.info("Using Isolation Forest prediction logic")
        # For Isolation Forest, we need to invert the predictions (1 for inliers, -1 for outliers)
        y_pred = model.predict(X_test)
        logger.info(f"Raw Isolation Forest predictions: {np.unique(y_pred, return_counts=True)}")
        y_pred = (y_pred == -1).astype(int)  # Convert to 0 (inlier) and 1 (outlier)
        y_proba = None  # Isolation Forest doesn't provide probability estimates
    else:
        # For supervised models
        logger.info("Using standard prediction logic")
        try:
            y_pred = model.predict(X_test)
            logger.info(f"Prediction shape: {y_pred.shape}")
            logger.info(f"Prediction classes: {np.unique(y_pred, return_counts=True)}")
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                logger.info(f"Probability predictions shape: {y_proba.shape}")
                y_proba = y_proba[:, 1]  # Probability of positive class
            else:
                logger.info("Model does not support predict_proba")
                y_proba = None
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            # Return empty metrics if prediction fails
            return {f'test_{k}': 'N/A' for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'average_precision']}
    
    # Log class distribution in test set
    test_class_counts = pd.Series(y_test).value_counts().to_dict()
    logger.info(f"Test set class distribution: {test_class_counts}")
    
    # Log prediction distribution
    if y_pred is not None:
        unique, counts = np.unique(y_pred, return_counts=True)
        logger.info(f"Final prediction distribution: {dict(zip(unique, counts))}")
    
    # Log prediction probabilities distribution
    if y_proba is not None:
        logger.info("\n=== Prediction Probabilities ===")
        proba_df = pd.DataFrame({
            'true_class': y_test,
            'predicted_prob': y_proba
        })
        logger.info("\nProbability distribution by true class:")
        logger.info(proba_df.groupby('true_class')['predicted_prob'].describe().to_string())
        
        # Log samples where model is most uncertain (probabilities close to 0.5)
        proba_df['uncertainty'] = np.abs(proba_df['predicted_prob'] - 0.5)
        uncertain_samples = proba_df.nsmallest(5, 'uncertainty')
        logger.info("\nMost uncertain predictions (probabilities closest to 0.5):")
        logger.info(uncertain_samples.to_string())
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba, 'test', threshold)
    
    # Add model-specific metrics
    if hasattr(model, 'feature_importances_'):
        metrics['feature_importances'] = model.feature_importances_.tolist()
    
    # Log the metrics
    logger.info("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        if 'confusion_matrix' not in metric:  # Skip confusion matrix for now
            logger.info(f"{metric}: {value}")
    
    if 'test_confusion_matrix' in metrics:
        cm = metrics['test_confusion_matrix']
        logger.info("\nConfusion Matrix:")
        logger.info(f"True Negatives: {cm.get('true_negatives', 0)}")
        logger.info(f"False Positives: {cm.get('false_positives', 0)}")
        logger.info(f"False Negatives: {cm.get('false_negatives', 0)}")
        logger.info(f"True Positives: {cm.get('true_positives', 0)}")
        
        # Calculate additional metrics
        total = sum(cm.values())
        accuracy = (cm.get('true_positives', 0) + cm.get('true_negatives', 0)) / total if total > 0 else 0
        precision = cm.get('true_positives', 0) / (cm.get('true_positives', 0) + cm.get('false_positives', 1e-10))
        recall = cm.get('true_positives', 0) / (cm.get('true_positives', 0) + cm.get('false_negatives', 1e-10))
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logger.info("\nDerived Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    
    return metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    prefix: str = '',
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculate evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for binary classification).
        prefix: Prefix for metric names (e.g., 'train_', 'test_').
        threshold: Probability threshold for classification.
        
    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {}
    
    try:
        # Check if we have any predictions
        if len(y_pred) == 0:
            logger.warning("No predictions made. Check model and input data.")
            return {f'{prefix}{k}': 'N/A' for k in ['accuracy', 'precision', 'recall', 'f1']}
        
        # Check if we have any true labels
        if len(y_true) == 0:
            logger.warning("No true labels provided.")
            return {f'{prefix}{k}': 'N/A' for k in ['accuracy', 'precision', 'recall', 'f1']}
        
        # Basic metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle case where there's only one class in y_true
        if len(np.unique(y_true)) < 2:
            logger.warning(f"Only one class present in y_true: {np.unique(y_true)}. Some metrics may be undefined.")
            # Set metrics that require both classes to N/A
            for metric in ['precision', 'recall', 'f1', 'auc_roc', 'average_precision']:
                metrics[f'{prefix}{metric}'] = 'N/A'
        else:
            metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics[f'{prefix}confusion_matrix'] = {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        except ValueError as e:
            logger.warning(f"Error calculating confusion matrix: {e}")
            metrics[f'{prefix}confusion_matrix'] = {
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'true_positives': 0
            }
        
        # Probability-based metrics
        if y_proba is not None and len(np.unique(y_true)) >= 2:
            try:
                metrics[f'{prefix}auc_roc'] = roc_auc_score(y_true, y_proba)
                metrics[f'{prefix}average_precision'] = average_precision_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Error calculating probability-based metrics: {e}")
                metrics[f'{prefix}auc_roc'] = 'N/A'
                metrics[f'{prefix}average_precision'] = 'N/A'
        
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        # Return N/A for all metrics if there's an error
        metrics = {f'{prefix}{k}': 'N/A' for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'average_precision']}
        metrics[f'{prefix}confusion_matrix'] = {
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0
        }
    
    return metrics


def save_model(
    model: Any,
    metrics: Dict[str, Any],
    model_dir: str = 'models',
    model_name: str = 'fraud_detection_model',
    save_format: str = 'joblib'
) -> Dict[str, str]:
    """Save a trained model and its metrics.
    
    Args:
        model: Trained model.
        metrics: Model evaluation metrics.
        model_dir: Directory to save the model.
        model_name: Base name for the model files.
        save_format: Format to save the model ('joblib' or 'pickle').
        
    Returns:
        Dictionary with paths to saved files.
    """
    import os
    from datetime import datetime
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.{save_format}")
    
    if save_format == 'joblib':
        import joblib
        joblib.dump(model, model_path)
    elif save_format == 'pickle':
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    
    # Save metrics
    metrics_path = os.path.join(model_dir, f"{model_name}_{timestamp}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return {
        'model_path': model_path,
        'metrics_path': metrics_path
    }


def load_model(
    model_path: str,
    save_format: str = 'joblib'
) -> Any:
    """Load a saved model.
    
    Args:
        model_path: Path to the saved model file.
        save_format: Format of the saved model ('joblib' or 'pickle').
        
    Returns:
        Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    
    if save_format == 'joblib':
        import joblib
        model = joblib.load(model_path)
    elif save_format == 'pickle':
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    
    return model


def train_evaluate_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'xgboost',
    calibrate: bool = True,
    save_model_flag: bool = True,
    output_dir: str = 'models'
) -> Tuple[Any, Dict[str, Any]]:
    """Complete training and evaluation pipeline.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        model_type: Type of model to train.
        calibrate: Whether to calibrate the model.
        save_model_flag: Whether to save the trained model.
        output_dir: Directory to save the model and metrics.
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    # Train the model
    model, train_metrics = train_model(
        X_train, y_train,
        model_type=model_type,
        calibrate=calibrate
    )
    
    # Evaluate the model
    test_metrics = evaluate_model(
        model, X_test, y_test,
        model_type=model_type
    )
    
    # Combine metrics
    metrics = {**train_metrics, **test_metrics}
    
    # Save the model if requested
    if save_model_flag:
        save_model(
            model, metrics,
            model_dir=output_dir,
            model_name=f'fraud_detection_{model_type}'
        )
    
    return model, metrics
