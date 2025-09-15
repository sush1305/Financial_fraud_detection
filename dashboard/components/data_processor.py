"""
Data processing utilities for the fraud detection dashboard.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and transform data for the dashboard."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.historical_data = pd.DataFrame()
        self.realtime_data = pd.DataFrame()
        self.last_processed = None
        
    def process_realtime_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single real-time transaction.
        
        Args:
            transaction: Raw transaction data
            
        Returns:
            Processed transaction data
        """
        try:
            # Ensure required fields exist
            if 'timestamp' not in transaction:
                transaction['timestamp'] = datetime.now().isoformat()
                
            if 'amount' in transaction:
                try:
                    transaction['amount'] = float(transaction['amount'])
                except (ValueError, TypeError):
                    transaction['amount'] = 0.0
            
            # Add any additional processing here
            if 'is_fraud' not in transaction:
                # Default to not fraud if not specified
                transaction['is_fraud'] = 0
                
            # Add to realtime data
            df = pd.DataFrame([transaction])
            if not self.realtime_data.empty:
                self.realtime_data = pd.concat([self.realtime_data, df], ignore_index=True)
            else:
                self.realtime_data = df
                
            return transaction
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return {}
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get combined historical and real-time data.
        
        Returns:
            Combined DataFrame with all data
        """
        if self.historical_data.empty and self.realtime_data.empty:
            return pd.DataFrame()
            
        if self.historical_data.empty:
            return self.realtime_data.copy()
            
        if self.realtime_data.empty:
            return self.historical_data.copy()
            
        # Ensure consistent columns
        common_columns = list(set(self.historical_data.columns) & set(self.realtime_data.columns))
        
        # Combine data
        combined = pd.concat([
            self.historical_data[common_columns],
            self.realtime_data[common_columns]
        ], ignore_index=True)
        
        return combined
    
    def update_historical_data(self, df: pd.DataFrame) -> None:
        """Update the historical data.
        
        Args:
            df: New historical data
        """
        if not df.empty:
            self.historical_data = df.copy()
            self.last_processed = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from the current data.
        
        Returns:
            Dictionary of metrics
        """
        df = self.get_combined_data()
        
        if df.empty:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['total_transactions'] = len(df)
        metrics['fraudulent_transactions'] = int(df.get('is_fraud', 0).sum())
        metrics['fraud_rate'] = (metrics['fraudulent_transactions'] / metrics['total_transactions'] * 100 
                               if metrics['total_transactions'] > 0 else 0)
        metrics['avg_amount'] = float(df.get('amount', 0).mean())
        
        # Time-based metrics (if timestamp is available)
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                metrics['latest_timestamp'] = df['timestamp'].max().isoformat()
                metrics['transactions_last_hour'] = len(df[df['timestamp'] > (datetime.now() - timedelta(hours=1))])
            except Exception as e:
                logger.error(f"Error processing timestamps: {e}")
        
        return metrics
    
    def get_time_series_data(self, freq: str = '1H') -> pd.DataFrame:
        """Get time series data for visualization.
        
        Args:
            freq: Resampling frequency (e.g., '1H' for hourly, '1D' for daily)
            
        Returns:
            DataFrame with time series data
        """
        df = self.get_combined_data()
        
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df = df.set_index('timestamp').sort_index()
            
            # Resample by the specified frequency
            resampled = df.resample(freq).agg({
                'amount': ['count', 'sum', 'mean'],
                'is_fraud': 'sum'
            })
            
            # Flatten column names
            resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
            
            # Reset index for plotting
            resampled = resampled.reset_index()
            
            # Calculate fraud rate
            resampled['fraud_rate'] = (resampled['is_fraud_sum'] / resampled['amount_count'] * 100)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error generating time series data: {e}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance data.
        
        Returns:
            Dictionary of feature importances
        """
        # This is a placeholder - in a real application, this would come from your model
        # For now, we'll return some mock data
        return {
            'amount': 0.35,
            'time_since_last_transaction': 0.25,
            'merchant_risk_score': 0.15,
            'transaction_frequency': 0.10,
            'location_risk': 0.08,
            'category_risk': 0.07
        }
    
    def get_model_metrics(self) -> Dict[str, float]:
        """Get model performance metrics.
        
        Returns:
            Dictionary of model metrics
        """
        # This is a placeholder - in a real application, this would come from your model
        return {
            'precision': 0.95,
            'recall': 0.88,
            'f1': 0.91,
            'auc_roc': 0.98,
            'average_precision': 0.92
        }


def preprocess_realtime_data(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a real-time transaction.
    
    Args:
        transaction: Raw transaction data
        
    Returns:
        Processed transaction data
    """
    processed = transaction.copy()
    
    # Ensure required fields
    if 'timestamp' not in processed:
        processed['timestamp'] = datetime.now().isoformat()
    
    # Convert amount to float if it's a string
    if 'amount' in processed and isinstance(processed['amount'], str):
        try:
            processed['amount'] = float(processed['amount'].replace('$', '').replace(',', ''))
        except (ValueError, TypeError):
            processed['amount'] = 0.0
    
    # Ensure is_fraud is an integer
    if 'is_fraud' in processed:
        try:
            processed['is_fraud'] = int(bool(processed['is_fraud']))
        except (ValueError, TypeError):
            processed['is_fraud'] = 0
    else:
        processed['is_fraud'] = 0
    
    return processed
