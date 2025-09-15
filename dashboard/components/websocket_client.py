"""
WebSocket client for real-time data streaming in the fraud detection dashboard.
"""
import json
import threading
import logging
import websocket
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketClient:
    """WebSocket client for handling real-time transaction data."""
    
    def __init__(self, url: str):
        """Initialize the WebSocket client.
        
        Args:
            url: WebSocket server URL (e.g., 'ws://localhost:5000')
        """
        self.url = url
        self.ws = None
        self.data: List[Dict[str, Any]] = []
        self.max_data_points = 1000  # Maximum number of data points to keep in memory
        self.connected = False
        self.callbacks = []
        
    def on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            # Parse the incoming message
            transaction = json.loads(message)
            
            # Add timestamp if not present
            if 'timestamp' not in transaction:
                transaction['timestamp'] = datetime.now().isoformat()
                
            # Add to data store
            self.data.append(transaction)
            
            # Keep only the most recent data points
            if len(self.data) > self.max_data_points:
                self.data = self.data[-self.max_data_points:]
                
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(transaction)
                except Exception as e:
                    logger.error(f"Error in WebSocket callback: {e}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def on_error(self, ws, error: Exception) -> None:
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False
    
    def on_open(self, ws) -> None:
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection established")
        self.connected = True
    
    def register_callback(self, callback) -> None:
        """Register a callback function to be called on new messages.
        
        Args:
            callback: Function that takes a single argument (the message data)
        """
        if callable(callback):
            self.callbacks.append(callback)
    
    def start(self) -> None:
        """Start the WebSocket client in a separate thread."""
        try:
            websocket.enableTrace(False)  # Set to True for debugging
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.ws.on_open = self.on_open
            
            # Run WebSocket in a separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
        except Exception as e:
            logger.error(f"Error starting WebSocket client: {e}")
            self.connected = False
    
    def stop(self) -> None:
        """Stop the WebSocket client."""
        if self.ws:
            self.ws.close()
            self.connected = False
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get the current data.
        
        Returns:
            List of transaction dictionaries
        """
        return self.data.copy()
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the current data as a pandas DataFrame.
        
        Returns:
            DataFrame containing the transaction data
        """
        if not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data)


def create_mock_websocket_server(port: int = 5000) -> None:
    """Create a mock WebSocket server for testing.
    
    Args:
        port: Port to run the mock server on
    """
    from flask import Flask
    from flask_socketio import SocketIO, emit
    import random
    import time
    from threading import Thread
    
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    def generate_mock_transaction():
        """Generate a mock transaction for testing."""
        return {
            'transaction_id': f"tx_{int(datetime.now().timestamp() * 1000)}",
            'amount': round(random.uniform(1, 1000), 2),
            'timestamp': datetime.now().isoformat(),
            'merchant': random.choice(['Amazon', 'Walmart', 'Target', 'Best Buy', 'Other']),
            'category': random.choice(['Shopping', 'Food', 'Travel', 'Entertainment', 'Other']),
            'is_fraud': random.choice([0, 0, 0, 0, 1]),  # 20% chance of fraud for testing
            'card_last4': f"{random.randint(1000, 9999)}",
            'location': f"{random.choice(['US', 'UK', 'CA', 'AU', 'IN'])}"
        }
    
    def send_mock_data():
        """Send mock data at regular intervals."""
        while True:
            transaction = generate_mock_transaction()
            socketio.emit('new_transaction', transaction)
            time.sleep(2)  # Send a new transaction every 2 seconds
    
    # Start the mock data thread
    mock_thread = Thread(target=send_mock_data)
    mock_thread.daemon = True
    mock_thread.start()
    
    print(f"Starting mock WebSocket server on port {port}")
    socketio.run(app, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Start mock server in a separate thread
    import threading
    mock_thread = threading.Thread(target=create_mock_websocket_server, daemon=True)
    mock_thread.start()
    
    # Create and start WebSocket client
    client = WebSocketClient("ws://localhost:5000")
    
    def print_transaction(transaction):
        print(f"New transaction: {transaction['transaction_id']} - ${transaction['amount']:.2f}")
    
    client.register_callback(print_transaction)
    client.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        client.stop()
