"""
Configuration management for the fraud detection system.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        # Default configuration
        self._config = {
            'data': {
                'raw_dir': os.getenv('DATA_DIR', 'data/raw'),
                'processed_dir': os.getenv('PROCESSED_DIR', 'data/processed'),
                'model_dir': os.getenv('MODEL_DIR', 'data/models'),
                'raw_file': 'creditcard.csv',
                'test_size': 0.2,
                'random_state': int(os.getenv('RANDOM_STATE', 42)),
            },
            'model': {
                'type': os.getenv('MODEL_TYPE', 'xgboost'),
                'threshold': float(os.getenv('THRESHOLD', 0.5)),
                'calibrate': True,
            },
            'streaming': {
                'kafka_bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
                'kafka_topic': os.getenv('KAFKA_TOPIC', 'transactions'),
                'kafka_group_id': os.getenv('KAFKA_GROUP_ID', 'fraud-detection-group'),
                'batch_size': 1000,
                'window_size': 60,  # seconds
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            }
        }
        
        # Load YAML config if provided
        if config_path and os.path.exists(config_path):
            self._load_yaml_config(config_path)
    
    def _load_yaml_config(self, config_path: str):
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            self._merge_dicts(self._config, yaml_config)
    
    def _merge_dicts(self, base: Dict[Any, Any], update: Dict[Any, Any]) -> None:
        """Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary to update.
            update: Dictionary with updates.
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.raw_dir').
            default: Default value if key is not found.
            
        Returns:
            The configuration value or default if not found.
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary-style access."""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return self._config
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update the configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply.
        """
        self._merge_dicts(self._config, updates)

# Global configuration instance
config = Config(os.path.join('config', 'config.yaml'))

def get_config() -> Config:
    """Get the global configuration instance."""
    return config
