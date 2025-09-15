"""
Kafka producer for generating test transaction data.
"""
import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer
from typing import Dict, Any
import logging
from ..utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProducer:
    """Kafka producer for generating test transaction data."""
    
    def __init__(self, bootstrap_servers: str, topic: str):
        """Initialize the Kafka producer."""
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.topic = topic
        logger.info(f"Initialized Kafka producer for topic: {topic}")
    
    def generate_transaction(self) -> Dict[str, Any]:
        """Generate a random transaction."""
        transaction_types = ['purchase', 'withdrawal', 'transfer']
        merchants = ['amazon', 'walmart', 'target', 'netflix', 'spotify']
        
        return {
            'transaction_id': str(uuid.uuid4()),
            'user_id': f"user_{random.randint(1, 10000)}",
            'amount': round(random.uniform(1, 1000), 2),
            'merchant': random.choice(merchants),
            'transaction_type': random.choice(transaction_types),
            'location': {
                'country': 'US',
                'state': random.choice(['CA', 'NY', 'TX', 'FL', 'IL']),
                'city': 'Unknown'
            },
            'timestamp': datetime.utcnow().isoformat(),
            'device_id': f"device_{random.randint(1000, 9999)}",
            'ip_address': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        }
    
    def run(self, interval: float = 1.0) -> None:
        """Start producing test transactions."""
        logger.info(f"Starting to produce test transactions every {interval} seconds...")
        try:
            while True:
                transaction = self.generate_transaction()
                self.producer.send(self.topic, value=transaction)
                logger.info(f"Produced transaction: {transaction['transaction_id']}")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stopping producer...")
        finally:
            self.producer.close()

def main():
    """Main function to run the producer."""
    # Load configuration
    kafka_config = config.get('kafka', {})
    
    # Create and run producer
    producer = TransactionProducer(
        bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
        topic=kafka_config.get('topic', 'transactions')
    )
    
    producer.run(interval=0.5)  # Generate 2 transactions per second

if __name__ == "__main__":
    main()