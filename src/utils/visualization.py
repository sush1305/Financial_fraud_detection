"""
Visualization utilities for the fraud detection system.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logger = logging.getLogger(__name__)

def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution") -> None:
    """Plot the distribution of classes."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         classes: List[str] = None, 
                         title: str = "Confusion Matrix") -> None:
    """Plot a confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, 
                   roc_auc: float, title: str = "ROC Curve") -> None:
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(feature_importance: Dict[str, float], 
                          top_n: int = 20, 
                          title: str = "Feature Importance") -> None:
    """Plot feature importance."""
    # Sort features by importance
    features = sorted(feature_importance.items(), 
                     key=lambda x: x[1], 
                     reverse=True)[:top_n]
    names, values = zip(*features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(values), y=list(names))
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_anomaly_scores(scores: np.ndarray, 
                       threshold: float = None,
                       title: str = "Anomaly Scores") -> None:
    """Plot anomaly scores with an optional threshold line."""
    plt.figure(figsize=(12, 6))
    plt.plot(scores, 'o', alpha=0.5)
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='-', 
                   label=f'Threshold: {threshold:.2f}')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Anomaly Score')
    if threshold is not None:
        plt.legend()
    plt.show()