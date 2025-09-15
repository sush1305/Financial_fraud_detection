"""
Kafka consumer for real-time fraud detection.
"""
import json
import logging
from kafka import KafkaConsumer
from typing import Dict, Any, Optional
from ..models.predict import predict
from ..utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionConsumer:
    """Kafka consumer for processing transaction data in real-time."""
    
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        """Initialize the Kafka consumer."""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        logger.info(f"Initialized Kafka consumer for topic: {topic}")
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction message."""
        try:
            # Make prediction
            prediction = predict(message)
            
            # Log the result
            logger.info(f"Processed transaction: {message['transaction_id']} - "
                       f"Amount: {message['amount']}, "
                       f"Prediction: {'Fraud' if prediction['is_fraud'] else 'Legitimate'}, "
                       f"Confidence: {prediction['confidence']:.4f}")
            
            return {
                'transaction_id': message['transaction_id'],
                'is_fraud': bool(prediction['is_fraud']),
                'confidence': float(prediction['confidence']),
                'timestamp': message.get('timestamp')
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
    
    def run(self) -> None:
        """Start consuming messages from Kafka."""
        logger.info("Starting fraud detection consumer...")
        try:
            for message in self.consumer:
                try:
                    result = self.process_message(message.value)
                    # Here you could send the result to another Kafka topic or store it
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.consumer.close()

def main():
    """Main function to run the consumer."""
    # Load configuration
    kafka_config = config.get('kafka', {})
    
    # Create and run consumer
    consumer = FraudDetectionConsumer(
        bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
        topic=kafka_config.get('topic', 'transactions'),
        group_id=kafka_config.get('group_id', 'fraud-detection-group')
    )
    
    consumer.run()

if __name__ == "__main__":
    main()