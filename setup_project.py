"""
Project setup script to create the required directory structure and files.
"""
import os
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    # Base directories
    base_dirs = [
        'config',
        'data/raw',
        'data/processed',
        'data/models',
        'notebooks',
        'src/features',
        'src/streaming',
        'src/utils',
        'tests',
        'dashboard/assets',
        'dashboard/components'
    ]
    
    # Create directories
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        # Add empty __init__.py for Python packages
        if dir_path.startswith('src/') and not dir_path.endswith('__pycache__'):
            with open(os.path.join(dir_path, '__init__.py'), 'w') as f:
                f.write('')
    
    # Create empty files
    empty_files = [
        'config/config.yaml',
        'notebooks/01_eda.ipynb',
        'notebooks/02_feature_engineering.ipynb',
        'notebooks/03_model_experiments.ipynb',
        'src/features/engineering.py',
        'src/streaming/consumer.py',
        'src/streaming/processor.py',
        'src/streaming/producer.py',
        'src/utils/helpers.py',
        'src/utils/visualization.py',
        'tests/__init__.py',
        'tests/test_data.py',
        'tests/test_models.py',
        'tests/test_utils.py',
        'dashboard/app.py',
        'dashboard/requirements.txt',
        '.env.example'
    ]
    
    for file_path in empty_files:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('')
    
    # Create a sample .env file
    env_content = """# Configuration for Fraud Detection System
DATA_DIR=./data/raw
PROCESSED_DIR=./data/processed
MODEL_DIR=./data/models
LOG_LEVEL=INFO
RANDOM_STATE=42

# Kafka Configuration (for real-time processing)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=transactions
KAFKA_GROUP_ID=fraud-detection-group

# Model Configuration
MODEL_TYPE=xgboost
THRESHOLD=0.5
"""
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    # Create a sample config file
    config_content = """# Model Configuration
data:
  raw_data_path: "data/raw/creditcard.csv"
  processed_path: "data/processed/processed_data.parquet"
  test_size: 0.2
  random_state: 42

model:
  name: "fraud_detection"
  type: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    scale_pos_weight: 100

preprocessing:
  numeric_features:
    - "Amount"
    - "Time"
  categorical_features: []
  handle_imbalance: true
  sampling_strategy: "smote"

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "roc_auc"
    - "average_precision"
  cv_folds: 5
"""
    with open('config/config.yaml', 'w') as f:
        f.write(config_content)

if __name__ == "__main__":
    print("Setting up project structure...")
    create_directory_structure()
    print("Project structure created successfully!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and update the values")
    print("2. Place your creditcard.csv file in data/raw/")
    print("3. Run 'python src/main.py' to start the pipeline")
