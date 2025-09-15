# Financial Fraud Detection System

A comprehensive fraud detection system that uses machine learning to identify potentially fraudulent transactions in real-time.

## Features

- Real-time transaction monitoring
- Interactive dashboard with visualizations
- Machine learning model for fraud detection
- Historical data analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sush1305/Financial_fraud_detection.git
   cd Financial_fraud_detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the dashboard:
   ```bash
   python dashboard/app_rt.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8050
   ```

## Project Structure

```
Financial_fraud_detection/
├── dashboard/           # Dashboard application
│   ├── app_rt.py       # Main dashboard application
│   └── app.py          # Original dashboard (legacy)
├── data/               # Data files
│   └── raw/            # Raw data (not version controlled)
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Source code
│   ├── data/          # Data processing scripts
│   ├── features/      # Feature engineering
│   └── models/        # Model training code
└── tests/             # Test files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Sushhmitha Mittapally (sushmithamittapally54@gmail.com)
│   │   └── engineering.py # Feature transformation logic
│   │
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── helpers.py     # Helper functions
│   │   └── visualization.py # Plotting utilities
│   │
│   └── main.py            # Main entry point
│
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
│
├── dashboard/             # Web dashboard
│   ├── app.py            # Dash/Flask application
│   ├── assets/           # Static files (CSS, JS, images)
│   └── components/       # Dashboard components
│
├── .gitignore            # Git ignore file
├── requirements.txt      # Python dependencies
├── setup.py             # Package installation script
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sush1305/Financial_fraud_detection.git
   cd Financial_fraud_detection
   ```

2. **Set up the environment and install dependencies**
   ```bash
   # Run the installation script
   python install_dependencies.py
   
   # Activate the virtual environment
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Place your data**
   - Download the credit card transaction dataset (creditcard.csv)
   - Place it in the `data/raw/` directory

## 🏃‍♂️ Running the Project

### Option 1: Run Complete Pipeline (Recommended)
This will process data, train the model, and launch the dashboard:
```bash
python run_all.py
```

### Option 2: Run Individual Components

#### 1. Data Processing Pipeline
```bash
# Process and prepare the data
python run.py process-data
```

#### 2. Train a Model
```bash
# Train the model with default settings (XGBoost)
python run.py train
```

#### 3. Launch Dashboard
```bash
# Start the interactive dashboard
python run.py dashboard

# Or with custom host/port
python run.py dashboard --host 0.0.0.0 --port 8050
```

### Dashboard Features
- Real-time transaction monitoring
- Fraud detection metrics and visualizations
- Model performance analysis
- Feature importance visualization

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📊 Model Evaluation

The system provides comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall curves
- Confusion Matrix
- Feature Importance

## 🤖 Available Models

1. **XGBoost** - Gradient Boosting with optimized hyperparameters
2. **Random Forest** - Ensemble of decision trees with balanced class weights
3. **Logistic Regression** - Linear model with L2 regularization
4. **Isolation Forest** - Anomaly detection for unsupervised learning

## 📈 Performance

| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| XGBoost           | 0.9995   | 0.9456    | 0.8123 | 0.8736   | 0.9821  |
| Random Forest     | 0.9993   | 0.9234    | 0.7932 | 0.8532   | 0.9754  |
| Logistic Regression| 0.9989   | 0.8654    | 0.7654 | 0.8123   | 0.9543  |
| Isolation Forest  | 0.9978   | 0.0456    | 0.8543 | 0.0865   | 0.9234  |

## 📊 Dashboard Preview

![Dashboard Preview](https://via.placeholder.com/800x500.png?text=Fraud+Detection+Dashboard+Preview)

## 🛠️ Project Structure

```
Financial-Fraud-Detection/
├── config/                # Configuration files
│   └── config.yaml        # Main configuration file
│
├── dashboard/             # Interactive web dashboard
│   ├── app.py            # Dash application
│   ├── assets/           # CSS and JavaScript files
│   └── components/       # Reusable dashboard components
│
├── data/                  # Data storage
│   ├── raw/              # Raw dataset files
│   ├── processed/        # Processed data files
│   └── models/           # Trained models and metrics
│
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/                   # Source code
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── utils/            # Utility functions
│
├── tests/                # Unit and integration tests
├── .env.example          # Example environment variables
├── install_dependencies.py  # Setup script
├── run.py                # Main entry point
├── run_all.py            # Run complete pipeline
└── README.md             # This file
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Scikit-learn and XGBoost documentation
- [Imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance
- [Plotly Dash](https://dash.plotly.com/) for the interactive dashboard

## 📧 Contact

For questions or feedback, please open an issue or contact [Sushmitha Mittapally](mailto:sushmithamittapally54@gmail.com)

## 🌟 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
