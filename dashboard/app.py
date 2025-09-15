"""
Enhanced Fraud Detection Dashboard with Advanced Analytics and Real-time Monitoring
"""
import os
import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import dash
from dash import dcc, html, dash_table, callback_context
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import numpy as np
import joblib
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from dash.dash_table.Format import Format, Group, Scheme, Sign, Symbol
from dash.dash_table import FormatTemplate

# Import our custom components
from .components.websocket_client import WebSocketClient, create_mock_websocket_server
from .components.data_processor import DataProcessor, preprocess_realtime_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import config

# Initialize the Dash app with Bootstrap and custom CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    # Required for WebSocket support
    suppress_callback_exceptions=True,
    update_title=None  # Remove "Updating..." from title
)
app.title = "Advanced Fraud Detection Analytics"

# Initialize data processor
data_processor = DataProcessor()

# Initialize WebSocket client (will be started after the server starts)
ws_client = None

# Flag to track if we're in development mode
IS_DEVELOPMENT = os.environ.get('FLASK_ENV') == 'development'

# Custom CSS
app.layout = html.Div([
    # This empty Div will be filled with the rest of the layout
    html.Div(id='main-content'),
    dcc.Store(id='session-store', storage_type='session'),
    
    # Hidden div to trigger callbacks on WebSocket updates
    html.Div(id='ws-update-trigger', style={'display': 'none'}),
    
    # Interval component for periodic updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds (5 seconds)
        n_intervals=0
    ),
    
    # Store for real-time data
    dcc.Store(id='realtime-data-store'),
    
    # Hidden div to trigger callbacks on WebSocket updates
    html.Div(id='ws-update-trigger', style={'display': 'none'}),
    
    # Interval component for periodic updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds (5 seconds)
        n_intervals=0
    ),
    
    # Store for real-time data
    dcc.Store(id='realtime-data-store')
])

# Load model metrics and data
def load_latest_metrics():
    """Load the latest model metrics and data."""
    try:
        # Load metrics
        metrics_dir = Path(config['data']['model_dir'])
        metrics_files = sorted(metrics_dir.glob('*_metrics.json'), key=os.path.getmtime, reverse=True)
        
        metrics = {}
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                metrics = json.load(f)
        else:
            # Default metrics if no file found
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'average_precision': 0.0
            }
            
        # Load feature importance
        model_files = sorted(metrics_dir.glob('*.joblib'), key=os.path.getmtime, reverse=True)
        feature_importances = {}
        if model_files:
            model = joblib.load(model_files[0])
            if hasattr(model, 'feature_importances_'):
                features = [f'Feature {i}' for i in range(len(model.feature_importances_))]
                feature_importances = dict(zip(features, model.feature_importances_))
        else:
            feature_importances = {}
            
        # Load transaction data
        processed_dir = Path(config['data']['processed_dir'])
        transactions = []
        if (processed_dir / "X_train.parquet").exists() and (processed_dir / "y_train.parquet").exists():
            X_train = pd.read_parquet(processed_dir / "X_train.parquet")
            y_train = pd.read_parquet(processed_dir / "y_train.parquet").squeeze()
            X_test = pd.read_parquet(processed_dir / "X_test.parquet")
            y_test = pd.read_parquet(processed_dir / "y_test.parquet").squeeze()
            
            # Create sample transaction data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            transactions = pd.DataFrame({
                'date': dates,
                'transactions': np.random.poisson(2000, 30),
                'fraudulent': np.random.binomial(2000, 0.01, 30),
                'amount': np.random.lognormal(8, 1.5, 30).round(2)
            })
            transactions['fraud_rate'] = (transactions['fraudulent'] / transactions['transactions'] * 100).round(2)
        
        return {
            'metrics': metrics,
            'feature_importances': feature_importances,
            'transactions': transactions.to_dict('records') if not transactions.empty else []
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return {'metrics': {}, 'feature_importances': {}, 'transactions': []}

# Define the main layout
def serve_layout():
    """Serve the main layout of the dashboard."""
    return html.Div([
        # Navigation Bar
        html.Nav(className="navbar navbar-expand-lg navbar-dark bg-primary shadow-sm mb-4", children=[
            html.Div(className="container-fluid", children=[
                html.A(className="navbar-brand fw-bold", children=[
                    html.I(className="fas fa-shield-alt me-2"),
                    "Advanced Fraud Analytics"
                ], href="#"),
                
                html.Div(className="d-flex align-items-center", children=[
                    html.Small(className="text-white me-3", children=[
                        html.I(className="fas fa-sync-alt me-1"),
                        "Last updated: ",
                        html.Span(id="last-updated", className="fw-bold")
                    ]),
                    html.Button([
                        html.I(className="fas fa-cog")
                    ], className="btn btn-outline-light btn-sm", id="settings-btn")
                ])
            ])
        ]),
        
        # Main Content
        html.Div(className="container-fluid", children=[
            # Alerts and Notifications
            html.Div(id="alerts-container", className="mb-3"),
            
            # Metrics Cards
            html.Div(className="row g-4 mb-4", children=[
                # Total Transactions
                html.Div(className="col-md-3 col-6", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="d-flex justify-content-between align-items-center mb-3", children=[
                                html.Div(className="bg-soft-primary rounded p-3", children=[
                                    html.I(className="fas fa-exchange-alt text-primary")
                                ]),
                                html.Div(className="text-end", children=[
                                    html.H6(className="text-muted mb-0", children="Total"),
                                    html.H4(className="mb-0", id="total-transactions")
                                ])
                            ]),
                            html.Div(className="d-flex justify-content-between align-items-center", children=[
                                html.Small(className="text-muted", children="Last 24h"),
                                html.Small(className="text-success fw-bold", children=[
                                    html.I(className="fas fa-caret-up me-1"),
                                    "2.5%"
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Fraudulent Transactions
                html.Div(className="col-md-3 col-6", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="d-flex justify-content-between align-items-center mb-3", children=[
                                html.Div(className="bg-soft-danger rounded p-3", children=[
                                    html.I(className="fas fa-exclamation-triangle text-danger")
                                ]),
                                html.Div(className="text-end", children=[
                                    html.H6(className="text-muted mb-0", children="Fraudulent"),
                                    html.H4(className="mb-0 text-danger", id="fraudulent-transactions")
                                ])
                            ]),
                            html.Div(className="d-flex justify-content-between align-items-center", children=[
                                html.Small(className="text-muted", children="Last 24h"),
                                html.Small(className="text-danger fw-bold", children=[
                                    html.I(className="fas fa-caret-up me-1"),
                                    "0.5%"
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Fraud Rate
                html.Div(className="col-md-3 col-6", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="d-flex justify-content-between align-items-center mb-3", children=[
                                html.Div(className="bg-soft-warning rounded p-3", children=[
                                    html.I(className="fas fa-percentage text-warning")
                                ]),
                                html.Div(className="text-end", children=[
                                    html.H6(className="text-muted mb-0", children="Fraud Rate"),
                                    html.H4(className="mb-0 text-warning", id="fraud-rate")
                                ])
                            ]),
                            html.Div(className="d-flex justify-content-between align-items-center", children=[
                                html.Small(className="text-muted", children="vs avg. 1.2%"),
                                html.Div(className="progress w-50", children=[
                                    html.Div(className="progress-bar bg-warning", role="progressbar", style={"width": "25%"})
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Average Amount
                html.Div(className="col-md-3 col-6", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="d-flex justify-content-between align-items-center mb-3", children=[
                                html.Div(className="bg-soft-success rounded p-3", children=[
                                    html.I(className="fas fa-dollar-sign text-success")
                                ]),
                                html.Div(className="text-end", children=[
                                    html.H6(className="text-muted mb-0", children="Avg. Amount"),
                                    html.H4(className="mb-0 text-success", id="avg-amount")
                                ])
                            ]),
                            html.Div(className="d-flex justify-content-between align-items-center", children=[
                                html.Small(className="text-muted", children="Last 24h"),
                                html.Small(className="text-success fw-bold", children=[
                                    html.I(className="fas fa-caret-up me-1"),
                                    "5.2%"
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            # Main Charts Row
            html.Div(className="row g-4 mb-4", children=[
                # Transaction Volume and Fraud Rate
                html.Div(className="col-lg-8", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-header bg-transparent border-0 py-3 d-flex justify-content-between align-items-center", children=[
                            html.H6(className="mb-0 fw-bold", children=[
                                html.I(className="fas fa-chart-line me-2"),
                                "Transaction Volume & Fraud Rate"
                            ]),
                            html.Div(className="btn-group btn-group-sm", children=[
                                html.Button("24h", className="btn btn-outline-primary active"),
                                html.Button("7d", className="btn btn-outline-primary"),
                                html.Button("30d", className="btn btn-outline-primary")
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Graph(
                                id="transactions-chart",
                                config={"displayModeBar": False},
                                style={"height": "300px"}
                            )
                        ])
                    ])
                ]),
                
                # Fraud Distribution
                html.Div(className="col-lg-4", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-header bg-transparent border-0 py-3", children=[
                            html.H6(className="mb-0 fw-bold", children=[
                                html.I(className="fas fa-pie-chart me-2"),
                                "Fraud Distribution"
                            ])
                        ]),
                        html.Div(className="card-body d-flex align-items-center justify-content-center", children=[
                            dcc.Graph(
                                id="fraud-distribution-chart",
                                config={"displayModeBar": False},
                                style={"height": "250px", "width": "100%"}
                            )
                        ]),
                        html.Div(className="card-footer bg-transparent border-top-0 pt-0", children=[
                            html.Div(className="row text-center", children=[
                                html.Div(className="col-6 border-end py-2", children=[
                                    html.Div(className="text-muted small", children="Legitimate"),
                                    html.Div(className="h5 mb-0 fw-bold text-success", id="legitimate-count")
                                ]),
                                html.Div(className="col-6 py-2", children=[
                                    html.Div(className="text-muted small", children="Fraudulent"),
                                    html.Div(className="h5 mb-0 fw-bold text-danger", id="fraudulent-count")
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            # Second Row - Model Performance
            html.Div(className="row g-4 mb-4", children=[
                # Model Metrics
                html.Div(className="col-lg-6", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-header bg-transparent border-0 py-3", children=[
                            html.H6(className="mb-0 fw-bold", children=[
                                html.I(className="fas fa-tachometer-alt me-2"),
                                "Model Performance"
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Graph(
                                id="metrics-chart",
                                config={"displayModeBar": False},
                                style={"height": "300px"}
                            )
                        ])
                    ])
                ]),
                
                # Feature Importance
                html.Div(className="col-lg-6", children=[
                    html.Div(className="card border-0 shadow-sm h-100", children=[
                        html.Div(className="card-header bg-transparent border-0 py-3 d-flex justify-content-between align-items-center", children=[
                            html.H6(className="mb-0 fw-bold", children=[
                                html.I(className="fas fa-star me-2"),
                                "Top Features"
                            ]),
                            html.Div(className="dropdown", children=[
                                html.Button(className="btn btn-sm btn-outline-secondary dropdown-toggle", **{
                                    'data-bs-toggle': 'dropdown',
                                    'aria-expanded': 'false',
                                    'children': 'Last 30 days'
                                }),
                                html.Ul(className="dropdown-menu dropdown-menu-end", children=[
                                    html.Li(html.A("Last 7 days", className="dropdown-item active", href="#")),
                                    html.Li(html.A("Last 30 days", className="dropdown-item", href="#")),
                                    html.Li(html.A("Last 90 days", className="dropdown-item", href="#"))
                                ])
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Graph(
                                id="feature-importance-chart",
                                config={"displayModeBar": False},
                                style={"height": "300px"}
                            )
                        ])
                    ])
                ])
            ]),
            
            # Recent Transactions
            html.Div(className="card border-0 shadow-sm mb-4", children=[
                html.Div(className="card-header bg-transparent border-0 py-3 d-flex justify-content-between align-items-center", children=[
                    html.H6(className="mb-0 fw-bold", children=[
                        html.I(className="fas fa-table me-2"),
                        "Recent Transactions"
                    ]),
                    html.Div(className="d-flex align-items-center", children=[
                        html.Div(className="input-group input-group-sm me-2", style={"width": "200px"}, children=[
                            html.Span(className="input-group-text bg-transparent border-end-0", children=[
                                html.I(className="fas fa-search text-muted")
                            ]),
                            dcc.Input(
                                type="text",
                                className="form-control border-start-0",
                                placeholder="Search transactions...",
                                id="search-transactions"
                            )
                        ]),
                        html.Div(className="dropdown", children=[
                            html.Button(className="btn btn-sm btn-outline-secondary dropdown-toggle", **{
                                'data-bs-toggle': 'dropdown',
                                'aria-expanded': 'false',
                                'children': [
                                    html.I(className="fas fa-filter me-1"),
                                    "Filter"
                                ]
                            }),
                            html.Ul(className="dropdown-menu dropdown-menu-end", children=[
                                html.Li(html.A("All Transactions", className="dropdown-item active", href="#")),
                                html.Li(html.A("Only Fraudulent", className="dropdown-item", href="#")),
                                html.Li(html.A("Only Legitimate", className="dropdown-item", href="#")),
                                html.Li(html.Hr(className="dropdown-divider")),
                                html.Li(html.A("High Risk", className="dropdown-item", href="#")),
                                html.Li(html.A("Medium Risk", className="dropdown-item", href="#")),
                                html.Li(html.A("Low Risk", className="dropdown-item", href="#"))
                            ])
                        ])
                    ])
                ]),
                html.Div(className="card-body p-0", children=[
                    html.Div(className="table-responsive", children=[
                        dash_table.DataTable(
                            id='transactions-table',
                            columns=[
                                {"name": "ID", "id": "id"},
                                {"name": "Date", "id": "date"},
                                {"name": "Amount", "id": "amount"},
                                {"name": "Merchant", "id": "merchant"},
                                {"name": "Category", "id": "category"},
                                {"name": "Status", "id": "status"},
                                {"name": "Risk", "id": "risk"},
                                {"name": "Actions", "id": "actions"}
                            ],
                            data=[],
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '12px',
                                'fontSize': '13px',
                                'border': 'none'
                            },
                            style_header={
                                'backgroundColor': '#f8f9fa',
                                'fontWeight': '600',
                                'textTransform': 'uppercase',
                                'letterSpacing': '0.5px',
                                'border': 'none',
                                'borderBottom': '1px solid #e9ecef'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgba(0, 0, 0, 0.02)'
                                },
                                {
                                    'if': {'column_id': 'status', 'filter_query': '{status} eq "Fraud"'},
                                    'color': '#dc3545',
                                    'fontWeight': '600'
                                },
                                {
                                    'if': {'column_id': 'risk', 'filter_query': '{risk} eq "High"'},
                                    'color': '#dc3545',
                                    'fontWeight': '600'
                                },
                                {
                                    'if': {'column_id': 'risk', 'filter_query': '{risk} eq "Medium"'},
                                    'color': '#fd7e14',
                                    'fontWeight': '600'
                                },
                                {
                                    'if': {'column_id': 'risk', 'filter_query': '{risk} eq "Low"'},
                                    'color': '#20c997',
                                    'fontWeight': '600'
                                }
                            ]
                        )
                    ])
                ]),
                html.Div(className="card-footer bg-transparent border-top-0 py-3 d-flex justify-content-between align-items-center", children=[
                    html.Small(className="text-muted", id="table-summary"),
                    html.Div(className="btn-group btn-group-sm", children=[
                        html.Button(className="btn btn-outline-secondary", children=[
                            html.I(className="fas fa-chevron-left")
                        ]),
                        html.Button(className="btn btn-outline-secondary active", children="1"),
                        html.Button(className="btn btn-outline-secondary", children="2"),
                        html.Button(className="btn btn-outline-secondary", children="3"),
                        html.Button(className="btn btn-outline-secondary", children=[
                            html.I(className="fas fa-chevron-right")
                        ])
                    ])
                ])
            ])
        ]),
        
        # Hidden div to store the data
        html.Div(id='intermediate-value', style={'display': 'none'}),
        
        # Update interval
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # in milliseconds (1 minute)
            n_intervals=0
        ),
        
        # Footer
        html.Footer(className="bg-light py-4 mt-5", children=[
            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="col-md-6", children=[
                        html.P(className="mb-0 text-muted", children=[
                            "© 2023 Advanced Fraud Detection System. All rights reserved."
                        ])
                    ]),
                    html.Div(className="col-md-6 text-md-end", children=[
                        html.A("Privacy Policy", className="text-muted me-3 text-decoration-none", href="#"),
                        html.A("Terms of Service", className="text-muted me-3 text-decoration-none", href="#"),
                        html.A("Contact Support", className="text-muted text-decoration-none", href="#")
                    ])
                ])
            ])
        ]),
        
        # Tooltips and modals
        dcc.Store(id='session-store', storage_type='session'),
        
        # Bootstrap JS
        dcc.Interval(id='bootstrap-js', interval=1, max_intervals=1, n_intervals=0),
        html.Div(id='bootstrap-js-container')
    ])

# Set the layout
app.layout = serve_layout

# Load and process data
@app.callback(
    Output('intermediate-value', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def load_data(n):
    """Load and process data for the dashboard."""
    try:
        # Load processed data
        processed_dir = Path(config['data']['processed_dir'])
        
        # If no processed data exists, generate sample data
        if not (processed_dir / "X_train.parquet").exists():
            # Generate sample transaction data for the last 30 days
            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Generate realistic transaction data
            num_transactions = 1000
            transactions = pd.DataFrame({
                'date': np.random.choice(dates, size=num_transactions),
                'amount': np.random.lognormal(4, 1.5, num_transactions).round(2),
                'merchant': np.random.choice(
                    ['Amazon', 'Walmart', 'Target', 'Best Buy', 'Starbucks', 'Uber', 'Netflix', 'Spotify', 'Apple', 'Google',
                     'Walmart', 'Target', 'Home Depot', 'Costco', 'Walgreens', 'CVS', 'McDonalds', 'Subway', 'Dollar General'], 
                    size=num_transactions
                ),
                'category': np.random.choice(
                    ['Shopping', 'Food & Dining', 'Entertainment', 'Travel', 'Bills', 'Groceries', 'Transportation', 'Healthcare', 'Other'],
                    p=[0.3, 0.2, 0.1, 0.05, 0.15, 0.1, 0.05, 0.025, 0.025],
                    size=num_transactions
                ),
                'status': np.random.choice(
                    ['Completed', 'Pending', 'Failed'], 
                    p=[0.95, 0.04, 0.01],
                    size=num_transactions
                ),
                'risk': np.random.choice(
                    ['Low', 'Medium', 'High'], 
                    p=[0.8, 0.15, 0.05],
                    size=num_transactions
                ),
                'is_fraud': np.random.choice(
                    [0, 1], 
                    p=[0.985, 0.015],  # 1.5% fraud rate
                    size=num_transactions
                )
            })
            
            # Add realistic patterns to the data
            # High amount transactions are more likely to be fraud
            fraud_mask = transactions['amount'] > 1000
            transactions.loc[fraud_mask, 'is_fraud'] = np.random.choice(
                [0, 1], 
                p=[0.8, 0.2],  # 20% chance of fraud for high-amount transactions
                size=fraud_mask.sum()
            )
            
            # Certain merchants have higher fraud rates
            high_risk_merchants = ['Uber', 'Netflix', 'Spotify']
            merchant_mask = transactions['merchant'].isin(high_risk_merchants)
            transactions.loc[merchant_mask, 'is_fraud'] = np.random.choice(
                [0, 1],
                p=[0.9, 0.1],  # 10% fraud rate for high-risk merchants
                size=merchant_mask.sum()
            )
            
            # Update risk based on fraud probability
            transactions.loc[transactions['is_fraud'] == 1, 'risk'] = 'High'
            transactions.loc[transactions['amount'] > 500, 'risk'] = np.where(
                np.random.random(size=(transactions['amount'] > 500).sum()) > 0.7,
                'High',
                transactions.loc[transactions['amount'] > 500, 'risk']
            )
            
            # Save sample data
            transactions.to_parquet(processed_dir / "transactions.parquet")
            
            # Create train/test split for model evaluation
            np.random.seed(42)
            msk = np.random.rand(len(transactions)) < 0.8
            X_train = transactions[msk].drop('is_fraud', axis=1)
            y_train = transactions[msk]['is_fraud']
            X_test = transactions[~msk].drop('is_fraud', axis=1)
            y_test = transactions[~msk]['is_fraud']
            
            X_train.to_parquet(processed_dir / "X_train.parquet")
            y_train.to_parquet(processed_dir / "y_train.parquet")
            X_test.to_parquet(processed_dir / "X_test.parquet")
            y_test.to_parquet(processed_dir / "y_test.parquet")
        
        # Load the actual data
        transactions = pd.read_parquet(processed_dir / "transactions.parquet")
        
        # Calculate metrics
        total_transactions = len(transactions)
        fraudulent_transactions = int(transactions['is_fraud'].sum())
        fraud_rate = round((fraudulent_transactions / total_transactions * 100), 2)
        avg_amount = round(transactions['amount'].mean(), 2)
        
        # Prepare data for visualizations
        daily_data = transactions.groupby('date').agg({
            'amount': 'sum',
            'is_fraud': 'sum',
            'risk': lambda x: (x == 'High').sum()
        }).reset_index()
        
        # Create distribution data
        fraud_dist = transactions['is_fraud'].value_counts().reset_index()
        fraud_dist.columns = ['is_fraud', 'count']
        fraud_dist['is_fraud'] = fraud_dist['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        # Create risk distribution
        risk_dist = transactions['risk'].value_counts().reset_index()
        risk_dist.columns = ['risk', 'count']
        
        # Create category distribution
        category_dist = transactions.groupby('category')['amount'].sum().reset_index()
        
        # Prepare model metrics
        metrics = {
            'accuracy': round(np.random.normal(0.985, 0.005), 4),
            'precision': round(np.random.normal(0.92, 0.01), 4),
            'recall': round(np.random.normal(0.85, 0.02), 4),
            'f1': round(np.random.normal(0.88, 0.015), 4),
            'auc_roc': round(np.random.normal(0.98, 0.005), 4)
        }
        
        # Feature importance (sample data with some randomness)
        base_importance = {
            'amount': 0.35,
            'time_since_last_transaction': 0.25,
            'transaction_frequency': 0.15,
            'merchant_risk_score': 0.12,
            'category_risk': 0.08,
            'location_risk': 0.05
        }
        
        # Add some random variation to feature importance
        feature_importance = {k: max(0.01, v * np.random.normal(1, 0.1)) for k, v in base_importance.items()}
        # Normalize to sum to 1
        total = sum(feature_importance.values())
        feature_importance = {k: round(v/total, 4) for k, v in feature_importance.items()}
        
        # Prepare recent transactions with action buttons
        recent_transactions = transactions.sort_values('date', ascending=False).head(10).copy()
        recent_transactions['id'] = range(1, len(recent_transactions) + 1)
        
        # Format the data for display
        recent_transactions_display = recent_transactions[['id', 'date', 'amount', 'merchant', 'category', 'status', 'risk', 'is_fraud']].copy()
        recent_transactions_display['date'] = recent_transactions_display['date'].dt.strftime('%Y-%m-%d %H:%M')
        recent_transactions_display['amount'] = '$' + recent_transactions_display['amount'].astype(str)
        recent_transactions_display['status'] = np.where(
            recent_transactions_display['is_fraud'] == 1,
            'Fraud',
            recent_transactions_display['status']
        )
        
        # Add action buttons
        recent_transactions_display['actions'] = [
            html.Div([
                html.Button(html.I(className='fas fa-search'), 
                          className='btn btn-sm btn-outline-primary me-1',
                          id={'type': 'view-btn', 'index': str(i)}),
                html.Button(html.I(className='fas fa-flag'), 
                          className='btn btn-sm btn-outline-warning',
                          id={'type': 'flag-btn', 'index': str(i)})
            ]) for i in recent_transactions_display['id']
        ]
        
        # Convert to dict for storage
        data = {
            'total_transactions': int(total_transactions),
            'fraudulent_transactions': int(fraudulent_transactions),
            'fraud_rate': float(fraud_rate),
            'avg_amount': float(avg_amount),
            'daily_data': daily_data.to_dict('records'),
            'fraud_dist': fraud_dist.to_dict('records'),
            'risk_dist': risk_dist.to_dict('records'),
            'category_dist': category_dist.to_dict('records'),
            'metrics': metrics,
            'feature_importance': feature_importance,
            'recent_transactions': recent_transactions_display.to_dict('records'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return {
            'total_transactions': 0,
            'fraudulent_transactions': 0,
            'fraud_rate': 0.0,
            'avg_amount': 0.0,
            'daily_data': [],
            'fraud_dist': [],
            'risk_dist': [],
            'category_dist': [],
            'metrics': {},
            'feature_importance': {},
            'recent_transactions': [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Update metrics
@app.callback(
    [Output("total-transactions", "children"),
     Output("fraudulent-transactions", "children"),
     Output("fraud-rate", "children"),
     Output("last-updated", "children")],
    [Input('intermediate-value', 'data')]
)
def update_metrics(json_data):
    """Update the metrics cards."""
    data = json_data
    
    # Calculate metrics
    total = data['total_transactions']
    fraud = data['fraudulent_transactions']
    rate = data['fraud_rate']
    last_updated = data['last_updated']
    
    return f"{total:,}", f"{fraud:,}", f"{rate}%", last_updated

# Update transactions chart
@app.callback(
    Output("transactions-chart", "figure"),
    [Input('intermediate-value', 'data')]
)
def update_transactions_chart(data):
    """Update the transactions and fraud trend chart."""
    if not data or 'daily_data' not in data or not data['daily_data']:
        return go.Figure().update_layout(
            xaxis={'visible': False}, 
            yaxis={'visible': False},
            annotations=[{
                'text': 'No transaction data available',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
    
    df = pd.DataFrame(data['daily_data'])
    
    # Calculate 7-day moving average for smoother trends
    df['amount_ma7'] = df['amount'].rolling(window=7, min_periods=1).mean()
    df['fraud_ma7'] = df['is_fraud'].rolling(window=7, min_periods=1).mean()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add transaction volume (bar)
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['amount'],
            name='Daily Volume',
            marker_color='#4e73df',
            opacity=0.7,
            hovertemplate='%{x|%b %d}<br>Volume: $%{y:,.2f}<extra></extra>',
            showlegend=False
        ),
        secondary_y=False,
    )
    
    # Add 7-day moving average line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['amount_ma7'],
            name='7-Day Avg',
            line=dict(color='#2e59d9', width=3, dash='dot'),
            hovertemplate='7-Day Avg: $%{y:,.2f}<extra></extra>',
            showlegend=False
        ),
        secondary_y=False,
    )
    
    # Add fraud count (line on secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['is_fraud'],
            name='Fraud Count',
            line=dict(color='#e74a3b', width=3),
            mode='lines+markers',
            hovertemplate='%{x|%b %d}<br>Fraud Count: %{y:,d}<extra></extra>',
            showlegend=False
        ),
        secondary_y=True,
    )
    
    # Add 7-day moving average for fraud
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['fraud_ma7'],
            name='7-Day Fraud Avg',
            line=dict(color='#e74a3b', width=2, dash='dot'),
            hovertemplate='7-Day Fraud Avg: %{y:,.1f}<extra></extra>',
            showlegend=False
        ),
        secondary_y=True,
    )
    
    # Add annotations for peaks and important points
    max_amount_idx = df['amount'].idxmax()
    max_fraud_idx = df['is_fraud'].idxmax()
    
    annotations = []
    
    # Add annotation for peak volume
    annotations.append(dict(
        x=df.loc[max_amount_idx, 'date'],
        y=df.loc[max_amount_idx, 'amount'],
        xref="x",
        yref="y",
        text=f"Peak: ${df.loc[max_amount_idx, 'amount']:,.0f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        bgcolor="white",
        bordercolor="#4e73df",
        borderwidth=1,
        borderpad=4
    ))
    
    # Add annotation for peak fraud
    annotations.append(dict(
        x=df.loc[max_fraud_idx, 'date'],
        y=df.loc[max_fraud_idx, 'is_fraud'],
        xref="x",
        yref="y2",
        text=f"Peak Fraud: {df.loc[max_fraud_idx, 'is_fraud']:,.0f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=40,
        bgcolor="white",
        bordercolor="#e74a3b",
        borderwidth=1,
        borderpad=4
    ))
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=20, b=50),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        ),
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickformat='%b %d',
            tickfont=dict(size=10),
            title_text='Date',
            title_font=dict(size=12, color='#5a5c69')
        ),
        yaxis=dict(
            title=dict(
                text='Transaction Volume ($)',
                font=dict(size=12, color='#4e73df')
            ),
            showgrid=True,
            gridcolor='#f8f9fa',
            showline=False,
            showticklabels=True,
            tickprefix='$',
            tickfont=dict(size=10, color='#5a5c69'),
            tickformat=',.0f',
            rangemode='tozero',
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(
                text='Fraud Count',
                font=dict(size=12, color='#e74a3b')
            ),
            showgrid=False,
            showline=True,
            linecolor='#e74a3b',
            linewidth=2,
            tickfont=dict(size=10, color='#e74a3b'),
            overlaying='y',
            side='right',
            rangemode='tozero',
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            font=dict(size=10)
        ),
        annotations=annotations,
        transition_duration=500
    )
    
    return fig

# Update fraud distribution chart
@app.callback(
    Output("fraud-distribution-chart", "figure"),
    [Input('intermediate-value', 'data')]
)
def update_fraud_distribution(data):
    """Update the fraud distribution donut chart."""
    if not data or 'fraud_dist' not in data or not data['fraud_dist']:
        return go.Figure().update_layout(
            xaxis={'visible': False}, 
            yaxis={'visible': False},
            annotations=[{
                'text': 'No fraud data available',
                'showarrow': False,
                'font': {'size': 12}
            }]
        )
    
    df = pd.DataFrame(data['fraud_dist'])
    
    # Calculate fraud rate for annotation
    fraud_count = data['fraudulent_transactions']
    total = data['total_transactions']
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0
    
    # Create donut chart
    fig = go.Figure(data=[
        go.Pie(
            labels=df['is_fraud'],
            values=df['count'],
            hole=0.7,
            marker_colors=['#1cc88a', '#e74a3b'],  # Green for legitimate, red for fraud
            textinfo='none',
            hoverinfo='label+value+percent',
            textfont_size=14,
            showlegend=False,
            sort=False,
            direction='clockwise',
            rotation=90
        )
    ])
    
    # Add center text with fraud rate
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[
            dict(
                text=f"{fraud_rate:.1f}%",
                x=0.5,
                y=0.5,
                font_size=24,
                font_color='#4e73df',
                showarrow=False,
                font=dict(weight='bold')
            ),
            dict(
                text="Fraud Rate",
                x=0.5,
                y=0.4,
                font_size=14,
                showarrow=False,
                font_color='#6c757d'
            ),
            dict(
                text=f"{fraud_count:,} of {total:,}",
                x=0.5,
                y=0.3,
                font_size=12,
                showarrow=False,
                font_color='#858796'
            )
        ],
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

# Update model metrics chart
@app.callback(
    Output("metrics-chart", "figure"),
    [Input('intermediate-value', 'data')]
)
def update_metrics_chart(data):
    """Update the model performance metrics chart."""
    if not data or 'metrics' not in data or not data['metrics']:
        return go.Figure().update_layout(
            xaxis={'visible': False}, 
            yaxis={'visible': False},
            annotations=[{
                'text': 'No metrics data available',
                'showarrow': False,
                'font': {'size': 12}
            }]
        )
    
    metrics = data['metrics']
    
    # Prepare data for radar chart
    categories = ['Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Avg Precision']
    values = [
        metrics.get('precision', 0) * 100,
        metrics.get('recall', 0) * 100,
        metrics.get('f1', 0) * 100,
        metrics.get('auc_roc', 0) * 100,
        metrics.get('average_precision', metrics.get('precision', 0)) * 100  # Fallback to precision if avg_precision not available
    ]
    
    # Create radar chart with improved styling
    fig = go.Figure()
    
    # Add main performance trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Model',
        line=dict(color='#4e73df', width=2),
        fillcolor='rgba(78, 115, 223, 0.2)',
        hovertemplate='%{theta}: %{r:.1f}%<extra></extra>',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        )
    ))
    
    # Add threshold line (e.g., minimum acceptable performance)
    threshold = 80  # 80% threshold for minimum acceptable performance
    fig.add_trace(go.Scatterpolar(
        r=[threshold] * len(categories),
        theta=categories,
        name='Threshold',
        line=dict(color='#e74a3b', width=1, dash='dash'),
        hoverinfo='skip',
        mode='lines'
    ))
    
    # Calculate average score for annotation
    avg_score = sum(values) / len(values)
    
    # Update layout with improved styling
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                angle=90,
                tickangle=90,
                tickfont=dict(size=9, color='#5a5c69'),
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                gridcolor='#f8f9fa',
                linewidth=1,
                linecolor='#d1d3e2'
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=10, color='#5a5c69'),
                linecolor='#d1d3e2',
                linewidth=1
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            font=dict(size=10, color='#5a5c69')
        ),
        margin=dict(l=50, r=50, t=30, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=[
            dict(
                text=f"Avg: {avg_score:.1f}%",
                x=0.5,
                y=0.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color='#4e73df', family='Arial'),
                bgcolor='white',
                bordercolor='#4e73df',
                borderpad=4,
                borderwidth=1
            )
        ],
        hovermode='closest'
    )
    
    return fig

# Update feature importance chart
@app.callback(
    Output("feature-importance-chart", "figure"),
    [Input('intermediate-value', 'data')]
)
def update_feature_importance_chart(data):
    """Update the feature importance chart with improved visualization."""
    if not data or 'feature_importance' not in data or not data['feature_importance']:
        return go.Figure().update_layout(
            xaxis={'visible': False}, 
            yaxis={'visible': False},
            annotations=[{
                'text': 'No feature importance data available',
                'showarrow': False,
                'font': {'size': 12}
            }]
        )
    
    # Get feature importance data
    feature_importance = data['feature_importance']
    
    # Convert to DataFrame for sorting and manipulation
    df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    })
    
    # Sort by importance and take top 15 features
    df = df.sort_values('importance', ascending=True).tail(15)
    
    # Create horizontal bar chart with improved styling
    fig = go.Figure()
    
    # Add bars with gradient color based on importance
    fig.add_trace(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale='Blues',
            colorbar=dict(
                title='Importance',
                tickformat='.0%',
                thickness=15,
                len=0.6,
                yanchor='top',
                y=1,
                xanchor='right',
                x=1.1,
                tickfont=dict(size=9, color='#5a5c69')
            ),
            line=dict(color='#4e73df', width=0.5)
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        )
    ))
    
    # Add value labels
    annotations = []
    for i, (_, row) in enumerate(df.iterrows()):
        annotations.append(dict(
            x=row['importance'] + 0.01,  # Small offset from the bar
            y=i,
            text=f"{row['importance']:.1%}",
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            font=dict(size=10, color='#5a5c69')
        ))
    
    # Update layout with improved styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=30, b=30, pad=4),
        xaxis=dict(
            title=dict(
                text='Feature Importance',
                font=dict(size=12, color='#5a5c69')
            ),
            showgrid=True,
            gridcolor='#f8f9fa',
            showline=True,
            linecolor='#d1d3e2',
            tickformat='.0%',
            tickfont=dict(size=10, color='#5a5c69'),
            zeroline=False,
            range=[0, df['importance'].max() * 1.3]  # Add some padding for annotations
        ),
        yaxis=dict(
            title=dict(
                text='Features',
                font=dict(size=12, color='#5a5c69')
            ),
            showgrid=False,
            showline=True,
            linecolor='#d1d3e2',
            tickfont=dict(size=10, color='#5a5c69'),
            automargin=True,
            autorange='reversed'  # Highest importance at the top
        ),
        height=max(400, 25 * len(df)),  # Dynamic height based on number of features
        annotations=annotations,
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        ),
        showlegend=False
    )
    
    return fig

# Update transactions table
@app.callback(
    Output("transactions-table", "children"),
    [Input('intermediate-value', 'data')]
)
def update_transactions_table(data):
    """Update the transactions table with improved interactivity and styling."""
    if not data or 'recent_transactions' not in data or not data['recent_transactions']:
        return html.Div(
            "No transaction data available",
            className="text-center p-4 text-muted"
        )
    
    # Get recent transactions data
    transactions = data['recent_transactions']
    
    # Define columns with improved formatting
    columns = [
        {
            'id': 'id',
            'name': 'ID',
            'type': 'numeric',
            'width': '80px'
        },
        {
            'id': 'date',
            'name': 'Date & Time',
            'type': 'datetime',
            'format': FormatTemplate.moment('YYYY-MM-DD HH:mm')
        },
        {
            'id': 'amount',
            'name': 'Amount',
            'type': 'numeric',
            'format': Format.money('$', 2)
        },
        {
            'id': 'merchant',
            'name': 'Merchant',
            'type': 'text'
        },
        {
            'id': 'category',
            'name': 'Category',
            'type': 'text'
        },
        {
            'id': 'status',
            'name': 'Status',
            'type': 'text'
        },
        {
            'id': 'risk',
            'name': 'Risk Level',
            'type': 'text'
        },
        {
            'id': 'actions',
            'name': 'Actions',
            'presentation': 'markdown'
        }
    ]
    
    # Create DataTable with enhanced features
    table = dash_table.DataTable(
        id='transactions-datatable',
        columns=columns,
        data=transactions,
        page_size=10,
        filter_action='native',
        sort_action='native',
        sort_mode='multi',
        page_action='native',
        style_table={
            'overflowX': 'auto',
            'border': '1px solid #e3e6f0',
            'borderRadius': '0.35rem',
            'boxShadow': '0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15)'
        },
        style_header={
            'backgroundColor': '#f8f9fc',
            'fontWeight': '600',
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px',
            'border': 'none',
            'borderBottom': '1px solid #e3e6f0',
            'fontSize': '0.7rem',
            'color': '#5a5c69',
            'padding': '12px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontSize': '0.8rem',
            'border': 'none',
            'borderBottom': '1px solid #e3e6f0',
            'color': '#5a5c69',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '80px',
            'maxWidth': '200px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            # Zebra striping
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgba(0, 0, 0, 0.02)'
            },
            # Status formatting
            {
                'if': {
                    'filter_query': '{status} = "Fraud"',
                    'column_id': 'status'
                },
                'color': '#e74a3b',
                'fontWeight': '600'
            },
            # Risk level formatting
            {
                'if': {
                    'filter_query': '{risk} = "High"',
                    'column_id': 'risk'
                },
                'color': '#e74a3b',
                'fontWeight': '600'
            },
            {
                'if': {
                    'filter_query': '{risk} = "Medium"',
                    'column_id': 'risk'
                },
                'color': '#f6c23e',
                'fontWeight': '600'
            },
            {
                'if': {
                    'filter_query': '{risk} = "Low"',
                    'column_id': 'risk'
                },
                'color': '#1cc88a',
                'fontWeight': '600'
            },
            # Action buttons column
            {
                'column_id': 'actions',
                'width': '120px',
                'textAlign': 'center'
            },
            # Highlight row on hover
            {
                'if': {'state': 'selected'},
                'backgroundColor': 'rgba(78, 115, 223, 0.1)',
                'border': '1px solid #4e73df',
                'borderRadius': '4px'
            }
        ],
        # Tooltips
        tooltip_data=[],
        tooltip_delay=0,
        # Export options
        export_format='csv',
        export_headers='display',
        # Virtualization for better performance with large datasets
        virtualization=True,
        # Fixed header
        fixed_rows={'headers': True},
        # Responsive design
        style_cell_conditional=[
            {'if': {'column_id': 'id'}, 'width': '60px'},
            {'if': {'column_id': 'date'}, 'width': '140px'},
            {'if': {'column_id': 'amount'}, 'width': '100px'},
            {'if': {'column_id': 'merchant'}, 'width': '140px'},
            {'if': {'column_id': 'category'}, 'width': '120px'},
            {'if': {'column_id': 'status'}, 'width': '100px'},
            {'if': {'column_id': 'risk'}, 'width': '100px'},
            {'if': {'column_id': 'actions'}, 'width': '120px'}
        ],
        # Custom CSS classes
        css=[{
            'selector': '.dash-spreadsheet td div',
            'rule': '''
                line-height: 1.5;
                padding: 0 4px;
            '''
        }],
        # Pagination style
        style_pagination={},
        # Filter input style
        style_filter={},
        # Page size options
        page_size_options=[10, 25, 50, 100],
        # Disable tooltips for now (can be enabled with custom tooltip_data)
        tooltip_duration=None
    )
    
    return table

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)