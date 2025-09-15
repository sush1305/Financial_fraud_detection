"""
WebSocket and real-time data handling utilities for the fraud detection dashboard.
"""
import os
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class RealtimeDataManager:
    """Manages real-time data updates via WebSocket."""
    
    def __init__(self, websocket_url: Optional[str] = None):
        """Initialize the real-time data manager.
        
        Args:
            websocket_url: WebSocket server URL (if None, mock data will be used)
        """
        self.websocket_url = websocket_url
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.recent_transactions: List[Dict[str, Any]] = []
        self.max_transactions = 1000
        self._stop_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the WebSocket client in a separate thread."""
        if self.websocket_url:
            self._ws_thread = threading.Thread(
                target=self._run_websocket_client,
                daemon=True
            )
            self._ws_thread.start()
        else:
            self._ws_thread = threading.Thread(
                target=self._mock_websocket_server,
                daemon=True
            )
            self._ws_thread.start()
            
        logger.info("Real-time data manager started")
    
    def stop(self) -> None:
        """Stop the WebSocket client and clean up resources."""
        self._stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        logger.info("Real-time data manager stopped")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback to be called when new data is received.
        
        Args:
            callback: Function that takes a transaction dictionary as input
        """
        if callable(callback):
            self.callbacks.append(callback)
    
    def _run_websocket_client(self) -> None:
        """Run the WebSocket client (connects to a real WebSocket server)."""
        import websocket
        
        def on_message(ws, message):
            try:
                transaction = json.loads(message)
                self._process_transaction(transaction)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode message: {message}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            logger.info(f"Connected to WebSocket server at {self.websocket_url}")
        
        ws = websocket.WebSocketApp(
            self.websocket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket client
        ws.run_forever()
    
    def _mock_websocket_server(self) -> None:
        """Run a mock WebSocket server for testing."""
        import random
        import time
        from datetime import datetime, timedelta
        
        logger.info("Starting mock WebSocket server")
        
        # Generate some initial transactions
        now = datetime.now()
        for i in range(10):
            transaction = self._generate_mock_transaction(now - timedelta(minutes=10-i))
            self._process_transaction(transaction)
        
        # Generate new transactions periodically
        while not self._stop_event.is_set():
            try:
                # Random interval between 1-5 seconds
                time.sleep(random.uniform(1, 5))
                
                # Generate a new transaction
                transaction = self._generate_mock_transaction()
                self._process_transaction(transaction)
                
            except Exception as e:
                logger.error(f"Error in mock WebSocket server: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _generate_mock_transaction(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a mock transaction for testing."""
        if timestamp is None:
            timestamp = datetime.now()
            
        merchants = ["Amazon", "Walmart", "Target", "Best Buy", "Other"]
        categories = ["Shopping", "Food", "Travel", "Entertainment", "Other"]
        
        return {
            "transaction_id": f"tx_{int(timestamp.timestamp() * 1000)}",
            "timestamp": timestamp.isoformat(),
            "amount": round(random.uniform(10, 2000), 2),
            "merchant": random.choice(merchants),
            "category": random.choice(categories),
            "is_fraud": random.choices([0, 1], weights=[0.98, 0.02])[0],
            "card_last4": f"{random.randint(1000, 9999)}",
            "location": random.choice(["US", "UK", "CA", "AU", "IN"])
        }
    
    def _process_transaction(self, transaction: Dict[str, Any]) -> None:
        """Process a new transaction and notify callbacks."""
        try:
            # Add to recent transactions
            self.recent_transactions.append(transaction)
            
            # Keep only the most recent transactions
            if len(self.recent_transactions) > self.max_transactions:
                self.recent_transactions = self.recent_transactions[-self.max_transactions:]
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(transaction)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    def get_recent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent transactions.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries, most recent first
        """
        return sorted(
            self.recent_transactions,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from the current data.
        
        Returns:
            Dictionary of metrics
        """
        if not self.recent_transactions:
            return {}
        
        df = pd.DataFrame(self.recent_transactions)
        
        metrics = {
            'total_transactions': len(df),
            'fraudulent_transactions': int(df.get('is_fraud', 0).sum()),
            'fraud_rate': 0.0,
            'avg_amount': float(df.get('amount', 0).mean()),
            'last_updated': datetime.now().isoformat()
        }
        
        if metrics['total_transactions'] > 0:
            metrics['fraud_rate'] = (
                metrics['fraudulent_transactions'] / 
                metrics['total_transactions'] * 100
            )
        
        return metrics
