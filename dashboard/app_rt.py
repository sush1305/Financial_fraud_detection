"""
Real-time Fraud Detection Dashboard
"""
import os
import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import hashlib
from scipy import stats
import re

# Use relative import for the data_logger module
from dashboard.utils.data_logger import log_fraudulent_transaction, get_fraud_transactions

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

# Advanced fraud detection patterns
class FraudDetector:
    def __init__(self):
        self.suspicious_merchants = ['Unknown', 'International', 'Overseas']
        self.high_risk_categories = ['Gambling', 'Cryptocurrency', 'High-End Retail']
        self.known_fraud_patterns = {
            'amount': {
                'micro_transaction': (0.01, 1.00),  # Common in card testing
                'just_below_threshold': (950, 1000)  # Just below reporting threshold
            },
            'time_window': {
                'rapid_transactions': (5, 60)  # Multiple transactions in X seconds
            }
        }
        
    def check_amount_patterns(self, amount):
        for pattern, (min_val, max_val) in self.known_fraud_patterns['amount'].items():
            if min_val <= amount <= max_val:
                return pattern
        return None
    
    def check_merchant_risk(self, merchant):
        if merchant in self.suspicious_merchants:
            return 'high_risk_merchant'
        return None
    
    def check_category_risk(self, category):
        if category in self.high_risk_categories:
            return 'high_risk_category'
        return None
    
    def check_time_based_anomaly(self, transaction_time, last_transaction_time):
        if last_transaction_time:
            time_diff = (transaction_time - last_transaction_time).total_seconds()
            if time_diff < self.known_fraud_patterns['time_window']['rapid_transactions'][1]:
                return 'rapid_transaction'
        return None

# Initialize fraud detector
detector = FraudDetector()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    update_title=None
)
app.title = "Real-time Fraud Detection Dashboard"

# Sample data for testing
def generate_sample_data():
    """Generate sample transaction data for testing."""
    np.random.seed(42)
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(100)]
    
    data = {
        'timestamp': timestamps,
        'amount': np.random.lognormal(mean=4, sigma=1, size=100).round(2),
        'is_fraud': np.random.choice([0, 1], size=100, p=[0.98, 0.02]),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Best Buy', 'Other'], size=100),
        'category': np.random.choice(['Shopping', 'Food', 'Travel', 'Entertainment', 'Other'], size=100)
    }
    return pd.DataFrame(data)

# Initialize sample data
df = generate_sample_data()

# Create directory for data storage if it doesn't exist
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# Function to save transactions to CSV
def save_transactions(transactions):
    """Save transactions to a CSV file."""
    try:
        df = pd.DataFrame(transactions)
        # Add current timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = DATA_DIR / f'transactions_{timestamp}.csv'
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} transactions to {filename}")
        return str(filename)
    except Exception as e:
        logger.error(f"Error saving transactions: {e}")
        return None

# Layout
app.layout = html.Div([
    # Navigation
    html.Nav([
        html.Div([
            html.A("Fraud Detection Dashboard", className="navbar-brand"),
            html.Div([
                html.Span("Real-time ", className="me-2"),
                html.Span(className="realtime-dot")
            ], className="realtime-indicator")
        ], className="container")
    ], className="navbar navbar-expand-lg navbar-dark bg-primary"),
    
    # Main content container
    html.Div([
        html.Div(className="row g-0", children=[
            # Left sidebar
            html.Div(className="col-md-3 bg-light", children=[
                html.Div(className="p-3", children=[
                    html.H4("Filters", className="sidebar-heading"),
                    html.Hr(),
                    
                    # Time Range Filter
                    html.Label("Time Range"),
                    dcc.Dropdown(
                        id='time-range',
                        options=[
                            {'label': 'Last 24 hours', 'value': '24h'},
                            {'label': 'Last 7 days', 'value': '7d'},
                            {'label': 'All time', 'value': 'all'}
                        ],
                        value='24h',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    # Transaction Type Filter
                    html.Label("Transaction Type"),
                    dcc.Checklist(
                        id='transaction-type',
                        options=[
                            {'label': ' Show only suspicious', 'value': 'suspicious'},
                            {'label': ' Show only high risk', 'value': 'high_risk'}
                        ],
                        value=[],
                        className="mb-3"
                    ),
                    
                    html.Hr(),
                    
                    # Last Updated
                    html.Div([
                        html.Small("Last updated: "),
                        html.Span(id='last-updated', className="text-muted")
                    ], className="text-center")
                ])
            ]),  # End sidebar
            
            # Main content area
            html.Div(className="col-md-9 p-4", children=[
                # Metrics cards will go here
                html.Div(id='metrics-container'),
                
                # Charts will go here
                html.Div(id='charts-container'),
                
                # Data table will go here
                html.Div(id='data-table-container')
            ])  # End main content
        ]),  # End row
        
        # Interval component for real-time updates
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # 5 seconds
            n_intervals=0
        ),
        
        # Sidebar
        html.Div(className="col-md-3 bg-light p-3", children=[
            # Last Updated
            html.Div([
                html.Small("Last updated: "),
                html.Span(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    id='last-updated',
                    className="text-muted"
                )
            ], className="text-center mb-3"),
                
                # Transaction Verification
                html.Div([
                    html.H6("Verify Transaction", className="sidebar-heading"),
                    # Verification Form
                    html.Div([
                        html.Label("Amount ($)", className="form-label"),
                        dcc.Input(
                            id='verify-amount',
                            type='number',
                            step='0.01',
                            className="form-control mb-2"
                        ),
                        
                        html.Label("Merchant", className="form-label"),
                        dcc.Dropdown(
                            id='verify-merchant',
                            options=[
                                {'label': 'Amazon', 'value': 'Amazon'},
                                {'label': 'Walmart', 'value': 'Walmart'},
                                {'label': 'Target', 'value': 'Target'},
                                {'label': 'Best Buy', 'value': 'Best Buy'},
                                {'label': 'Other', 'value': 'Other'}
                            ],
                            className="mb-3"
                        ),
                        
                        html.Label("Category", className="form-label"),
                        dcc.Dropdown(
                            id='verify-category',
                            options=[
                                {'label': 'Shopping', 'value': 'Shopping'},
                                {'label': 'Food', 'value': 'Food'},
                                {'label': 'Travel', 'value': 'Travel'},
                                {'label': 'Entertainment', 'value': 'Entertainment'},
                                {'label': 'Other', 'value': 'Other'}
                            ],
                            className="mb-3"
                        ),
                        
                        html.Button(
                            "Verify Transaction",
                            id='verify-button',
                            className="btn btn-primary w-100"
                        ),
                        
                        html.Div(id='verification-result', className="mt-3")
                    ])
                ])
            ]),  # End sidebar
            # Metrics Cards Row
            html.Div(className="row mb-4", children=[
                # Total Transactions Card
                html.Div(className="col-xl-3 col-md-6 mb-4", children=[
                    html.Div(className="card border-left-primary shadow h-100 py-2", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="row no-gutters align-items-center", children=[
                                html.Div(className="col mr-2", children=[
                                    html.Div(
                                        "Total Transactions",
                                        className="text-xs font-weight-bold text-primary text-uppercase mb-1"
                                    ),
                                    html.Div(
                                        id='total-transactions',
                                        className="h5 mb-0 font-weight-bold text-gray-800"
                                    )
                                ]),
                                html.Div(className="col-auto", children=[
                                    html.I(className="fas fa-exchange-alt fa-2x text-gray-300")
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Fraudulent Transactions Card
                html.Div(className="col-xl-3 col-md-6 mb-4", children=[
                    html.Div(className="card border-left-danger shadow h-100 py-2", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="row no-gutters align-items-center", children=[
                                html.Div(className="col mr-2", children=[
                                    html.Div(
                                        "Fraudulent Transactions",
                                        className="text-xs font-weight-bold text-danger text-uppercase mb-1"
                                    ),
                                    html.Div(
                                        id='fraud-count',
                                        className="h5 mb-0 font-weight-bold text-gray-800"
                                    )
                                ]),
                                html.Div(className="col-auto", children=[
                                    html.I(className="fas fa-exclamation-triangle fa-2x text-gray-300")
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Average Transaction Card
                html.Div(className="col-xl-3 col-md-6 mb-4", children=[
                    html.Div(className="card border-left-success shadow h-100 py-2", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="row no-gutters align-items-center", children=[
                                html.Div(className="col mr-2", children=[
                                    html.Div(
                                        "Avg. Transaction",
                                        className="text-xs font-weight-bold text-success text-uppercase mb-1"
                                    ),
                                    html.Div(
                                        id='avg-amount',
                                        className="h5 mb-0 font-weight-bold text-gray-800"
                                    )
                                ]),
                                html.Div(className="col-auto", children=[
                                    html.I(className="fas fa-dollar-sign fa-2x text-gray-300")
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Fraud Rate Card
                html.Div(className="col-xl-3 col-md-6 mb-4", children=[
                    html.Div(className="card border-left-warning shadow h-100 py-2", children=[
                        html.Div(className="card-body", children=[
                            html.Div(className="row no-gutters align-items-center", children=[
                                html.Div(className="col mr-2", children=[
                                    html.Div(
                                        "Fraud Rate",
                                        className="text-xs font-weight-bold text-warning text-uppercase mb-1"
                                    ),
                                    html.Div(
                                        id='fraud-rate',
                                        className="h5 mb-0 font-weight-bold text-gray-800"
                                    )
                                ]),
                                html.Div(className="col-auto", children=[
                                    html.I(className="fas fa-percentage fa-2x text-gray-300")
                                ])
                            ])
                        ])
                    ])
                ])
            ]),  # End Metrics Cards Row
            
            # Charts Row
            html.Div(className="row mb-4", children=[
                # Transactions Over Time Chart
                html.Div(className="col-lg-8 mb-4", children=[
                    html.Div(className="card shadow h-100", children=[
                        html.Div(className="card-header py-3 d-flex flex-row align-items-center justify-content-between", children=[
                            html.H6("Transactions Over Time", className="m-0 font-weight-bold text-primary")
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Graph(
                                id='transactions-chart',
                                className="chart-area",
                                config={'displayModeBar': False}
                            )
                        ])
                    ])
                ]),
                
                # Risk Distribution Chart
                html.Div(className="col-lg-4 mb-4", children=[
                    html.Div(className="card shadow h-100", children=[
                        html.Div(className="card-header py-3 d-flex flex-row align-items-center justify-content-between", children=[
                            html.H6("Risk Distribution", className="m-0 font-weight-bold text-primary")
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Graph(
                                id='risk-chart',
                                className="chart-pie pt-4 pb-2"
                            )
                        ])
                    ])
                ])
            ]),  # End Charts Row
            
            # Transactions Table
            html.Div(className="card shadow mb-4", children=[
                html.Div(className="card-header py-3 d-flex flex-row align-items-center justify-content-between", children=[
                    html.H6("Suspicious Transactions", className="m-0 font-weight-bold text-primary"),
                    html.Div(children=[
                        dbc.Button("Verify Selected", id="verify-btn", color="primary", size="sm", className="me-2"),
                        dbc.Button("Export to CSV", id="export-btn", color="secondary", size="sm")
                    ])
                ]),
                html.Div(className="card-body", children=[
                    dash_table.DataTable(
                        id='fraud-table',
                        columns=[
                            {"name": "ID", "id": "id"},
                            {"name": "Amount", "id": "amount", "type": "numeric", "format": {"specifier": "$"}},
                            {"name": "Merchant", "id": "merchant"},
                            {"name": "Category", "id": "category"},
                            {"name": "Risk Score", "id": "risk_score", "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Status", "id": "status"},
                            {"name": "Date", "id": "date"}
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': 'rgb(248, 249, 250)',
                            'fontWeight': 'bold',
                            'border': '1px solid #e3e6f0'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            },
                            {
                                'if': {
                                    'filter_query': "{status} = 'High Risk'",
                                    'column_id': 'status'
                                },
                                'color': 'white',
                                'backgroundColor': '#e74a3b',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': "{status} = 'Suspicious'",
                                    'column_id': 'status'
                                },
                                'color': 'white',
                                'backgroundColor': '#f6c23e',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': "{status} = 'Verified'",
                                    'column_id': 'status'
                                },
                                'color': 'white',
                                'backgroundColor': '#1cc88a',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': "{risk_score} > 0.7",
                                    'column_id': 'risk_score'
                                },
                                'backgroundColor': 'rgba(231, 74, 59, 0.1)',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': "{risk_score} > 0.3 && {risk_score} <= 0.7",
                                    'column_id': 'risk_score'
                                },
                                'backgroundColor': 'rgba(246, 194, 62, 0.1)'
                            }
                        ],
                        page_size=10,
                        sort_action='native',
                        filter_action='native',
                        row_selectable='single',
                        selected_rows=[],
                        style_cell_conditional=[
                            {'if': {'column_id': 'id'}, 'width': '10%'},
                            {'if': {'column_id': 'amount'}, 'width': '15%'},
                            {'if': {'column_id': 'merchant'}, 'width': '20%'},
                            {'if': {'column_id': 'category'}, 'width': '15%'},
                            {'if': {'column_id': 'risk_score'}, 'width': '15%'},
                            {'if': {'column_id': 'status'}, 'width': '15%'},
                            {'if': {'column_id': 'date'}, 'width': '10%'}
                        ],
                        style_data={
                            'border': '1px solid #e3e6f0',
                            'fontSize': '14px'
                        },
                        style_header_conditional=[{
                            'fontFamily': 'Arial, sans-serif',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        }]
                    )
                ])
            ]),  # End Transactions Table
        ]),  # End main content area
        
        # Interval component for real-time updates
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=0
        )
    ]),  # End row
    
    # Verification Modal
    dbc.Modal(
        [
            dbc.ModalHeader("Verify Transaction"),
            dbc.ModalBody(
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Transaction ID"),
                            dbc.Input(
                                id="verify-tx-id",
                                type="text",
                                disabled=True,
                                className="mb-3"
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Amount"),
                            dbc.Input(
                                id="verify-amount",
                                type="number",
                                step="0.01",
                                className="mb-3"
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Merchant"),
                            dcc.Dropdown(
                                id='verify-merchant',
                                options=[
                                    {'label': 'Amazon', 'value': 'Amazon'},
                                    {'label': 'eBay', 'value': 'eBay'},
                                    {'label': 'Walmart', 'value': 'Walmart'},
                                    {'label': 'Best Buy', 'value': 'Best Buy'},
                                    {'label': 'Target', 'value': 'Target'}
                                ],
                                className="mb-3"
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Category"),
                            dcc.Dropdown(
                                id='verify-category',
                                options=[
                                    {'label': 'Retail', 'value': 'Retail'},
                                    {'label': 'Electronics', 'value': 'Electronics'},
                                    {'label': 'Grocery', 'value': 'Grocery'},
                                    {'label': 'Entertainment', 'value': 'Entertainment'},
                                    {'label': 'Other', 'value': 'Other'}
                                ],
                                className="mb-3"
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Notes"),
                            dbc.Textarea(
                                id="verify-notes",
                                className="mb-3"
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id="verify-is-fraud",
                                options=[{"label": "Mark as Fraudulent", "value": True}],
                                value=[],
                                switch=True,
                                className="mb-3"
                            )
                        ])
                    ])
                ])
            ),
            dbc.ModalFooter([
                dbc.Button(
                    "Close",
                    id="verify-close",
                    className="ms-auto",
                    color="secondary",
                    style={"margin-right": "10px"}
                ),
                dbc.Button(
                    "Submit",
                    id="verify-submit",
                    color="primary"
                )
            ])
        ],
        id="verification-modal",
        size="lg",
        is_open=False
    ),
    
    # Hidden stores

    
    dcc.Store(id='transactions-store'),
    dcc.Store(id='verification-store')
])  # End container div

# Callback to update the transactions chart
@app.callback(
    Output('transactions-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('transaction-type', 'value')]
)
def update_transactions_chart(n, transaction_types):
    """Update the transactions chart with sample data."""
    global df
    filtered_df = df.copy()
    
    # Filter data based on transaction type
    if transaction_types:
        if 'suspicious' in transaction_types:
            filtered_df = filtered_df[filtered_df['amount'] > 1000]
        if 'high_risk' in transaction_types:
            filtered_df = filtered_df[filtered_df['amount'] > 5000]
    
    # Add a new random transaction
    is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])
    new_row = {
        'transaction_id': f'TX{int(datetime.now().timestamp())}',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'amount': np.random.lognormal(mean=4, sigma=1) * 10,
        'is_fraud': is_fraud,
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Best Buy', 'Other']),
        'category': np.random.choice(['Shopping', 'Food', 'Travel', 'Entertainment', 'Other']),
        'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'IN', 'JP', 'DE', 'FR', 'Other']),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer']),
        'risk_score': np.random.randint(0, 100) if is_fraud else np.random.randint(0, 30)
    }
    
    # Log fraudulent transactions
    if is_fraud:
        log_fraudulent_transaction(new_row)
    
    # Append the new row to the dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Keep only the last 1000 transactions to prevent memory issues
    if len(df) > 1000:
        df = df.tail(1000)
    
    # Create figure
    fig = go.Figure()
    
    # Add transactions
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['amount'],
        mode='markers',
        name='Transactions',
        marker=dict(
            size=8,
            color=df['is_fraud'].map({0: '#4e73df', 1: '#e74a3b'}),
            opacity=0.7
        ),
        hoverinfo='text',
        hovertext=df.apply(
            lambda x: f"Amount: ${x['amount']:.2f}<br>" +
                     f"Merchant: {x['merchant']}<br>" +
                     f"Category: {x['category']}<br>" +
                     f"Fraud: {'Yes' if x['is_fraud'] else 'No'}",
            axis=1
        )
    ))
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=30, b=50),
        hovermode='closest',
        xaxis=dict(
            title='Time',
            showgrid=False,
            showline=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2
        ),
        yaxis=dict(
            title='Amount ($)',
            showgrid=True,
            gridcolor='rgb(230, 230, 230)',
            showline=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2
        ),
        showlegend=False
    )
    
    return fig

# Callback to update the fraud distribution chart
@app.callback(
    Output('fraud-distribution-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_fraud_distribution(n):
    """Update the fraud distribution chart."""
    global df
    
    # Calculate fraud distribution
    fraud_count = df['is_fraud'].sum()
    legit_count = len(df) - fraud_count
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Legitimate', 'Fraudulent'],
        values=[legit_count, fraud_count],
        hole=0.6,
        marker_colors=['#1cc88a', '#e74a3b'],
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont_size=14
    )])
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

# Callback to update metrics
# Transaction verification callback
@app.callback(
    Output('verification-result', 'children'),
    [Input('verify-button', 'n_clicks')],
    [State('verify-amount', 'value'),
     State('verify-merchant', 'value'),
     State('verify-category', 'value')]
)
def verify_transaction(n_clicks, amount, merchant, category):
    if n_clicks is None or amount is None or not merchant or not category:
        return ""
    
    risk_factors = []
    risk_score = 0
    
    # Check amount patterns
    amount_pattern = detector.check_amount_patterns(amount)
    if amount_pattern:
        risk_factors.append(f"Suspicious amount pattern: {amount_pattern}")
        risk_score += 30
    
    # Check merchant risk
    merchant_risk = detector.check_merchant_risk(merchant)
    if merchant_risk:
        risk_factors.append(f"High-risk merchant: {merchant}")
        risk_score += 25
    
    # Check category risk
    category_risk = detector.check_category_risk(category)
    if category_risk:
        risk_factors.append(f"High-risk category: {category}")
        risk_score += 25
    
    # Amount-based risk
    if amount > 5000:
        risk_factors.append("High transaction amount")
        risk_score += 20
    elif amount > 1000:
        risk_factors.append("Moderate transaction amount")
        risk_score += 10
    
    # Determine risk level
    if risk_score > 50:
        risk_level = "High Risk"
        alert_class = "alert-danger"
    elif risk_score > 25:
        risk_level = "Medium Risk"
        alert_class = "alert-warning"
    else:
        risk_level = "Low Risk"
        alert_class = "alert-success"
    
    # Create result message
    result = [
        html.H5(f"Risk Assessment: {risk_level}", className=f"alert-heading {alert_class}"),
        html.P(f"Risk Score: {risk_score}/100")
    ]
    
    if risk_factors:
        result.append(html.P("Risk Factors:"))
        result.extend([html.Li(factor) for factor in risk_factors])
    
    # Add recommendation
    if risk_score > 50:
        result.append(html.P("Recommendation: Review manually and consider blocking", 
                           className="font-weight-bold"))
        
        # Log high-risk transactions
        transaction_data = {
            'transaction_id': f'VERIFY-{int(datetime.now().timestamp())}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'amount': amount,
            'is_fraud': 1,  # Mark as potential fraud
            'merchant': merchant,
            'category': category,
            'risk_score': risk_score,
            'verification_result': 'High Risk',
            'risk_factors': ', '.join(risk_factors) if risk_factors else 'None'
        }
        log_fraudulent_transaction(transaction_data)
    
    return html.Div(result, className=f"alert {alert_class}")

@app.callback(
    [Output('total-transactions', 'children'),
     Output('fraud-count', 'children'),
     Output('avg-amount', 'children'),
     Output('fraud-rate', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('transaction-type', 'value')]
)
def update_metrics(n, transaction_types):
    """Update the metrics cards with the latest data."""
    global df
    
    # Filter data based on transaction type
    filtered_df = df.copy()
    if transaction_types:
        if 'suspicious' in transaction_types:
            # Filter for potentially suspicious transactions
            filtered_df = filtered_df[
                (filtered_df['amount'] > 1000) |
                (filtered_df['merchant'].isin(detector.suspicious_merchants)) |
                (filtered_df['category'].isin(detector.high_risk_categories))
            ]
        if 'high_risk' in transaction_types:
            # Filter for high-risk transactions
            filtered_df = filtered_df[
                (filtered_df['amount'] > 5000) |
                (filtered_df['is_fraud'] == 1) |
                (filtered_df['merchant'].isin(detector.suspicious_merchants))
            ]
    
    # Calculate metrics on filtered data
    total = len(filtered_df)
    fraud_count = filtered_df['is_fraud'].sum()
    avg_amount = filtered_df['amount'].mean()
    fraud_rate = (fraud_count / total) * 100 if total > 0 else 0
    
    return (
        f"{total:,}",
        f"{fraud_count:,}",
        f"${avg_amount:,.2f}",
        f"{fraud_rate:.2f}%"
    )

# Callback to update the merchant distribution chart
@app.callback(
    Output('merchant-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('transaction-type', 'value')]
)
def update_merchant_chart(n, transaction_types):
    """Update the merchant distribution chart."""
    global df
    
    # Filter data based on transaction type
    filtered_df = df.copy()
    if transaction_types:
        if 'suspicious' in transaction_types:
            filtered_df = filtered_df[filtered_df['amount'] > 1000]
        if 'high_risk' in transaction_types:
            filtered_df = filtered_df[filtered_df['amount'] > 5000]
    
    # Group by merchant and count transactions
    merchant_counts = filtered_df['merchant'].value_counts().reset_index()
    merchant_counts.columns = ['merchant', 'count']
    
    # Create bar chart
    fig = px.bar(
        merchant_counts, 
        x='merchant', 
        y='count',
        title='Transactions by Merchant',
        labels={'merchant': 'Merchant', 'count': 'Number of Transactions'},
        color='merchant'
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None
    )
    
    return fig

# Callback to update the category distribution chart
@app.callback(
    Output('category-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('transaction-type', 'value')]
)
def update_category_chart(n, transaction_types):
    """Update the category distribution chart with enhanced filtering and error handling."""
    try:
        # Create a copy of the dataframe to avoid modifying the original
        filtered_df = df.copy()
        
        # Apply transaction type filters
        if transaction_types:
            if 'suspicious' in transaction_types:
                filtered_df = filtered_df[filtered_df['amount'] > 1000]
            if 'high_risk' in transaction_types:
                filtered_df = filtered_df[filtered_df['amount'] > 5000]
        
        # Ensure we have data to plot
        if filtered_df.empty:
            return {
                'data': [],
                'layout': {
                    'title': 'No data available for the selected filters',
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': 'No data available',
                        'showarrow': False
                    }]
                }
            }
        
        # Group by category and count transactions
        category_counts = filtered_df['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        
        # Create pie chart with better styling
        fig = px.pie(
            category_counts,
            names='category',
            values='count',
            title='Transactions by Category',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu,
            labels={'count': 'Number of Transactions'},
            hover_data={'count': ':.0f'}
        )
        
        # Update layout for better readability
        fig.update_layout(
            margin=dict(t=50, b=0, l=0, r=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update traces for better hover info
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Count: %{value}<br>" +
                "Percentage: %{percent}<br>" +
                "<extra></extra>"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in update_category_chart: {str(e)}")
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }

# Callback to update the fraud details table
@app.callback(
    Output('fraud-details-table', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_fraud_details(n):
    """Update the fraudulent transactions table with error handling and data validation."""
    global df
    
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure 'is_fraud' column exists and is numeric
        if 'is_fraud' not in df_copy.columns:
            return html.Div("Error: 'is_fraud' column not found in data", className="text-danger")
            
        # Convert 'is_fraud' to numeric if it's not already
        df_copy['is_fraud'] = pd.to_numeric(df_copy['is_fraud'], errors='coerce')
        
        # Filter fraudulent transactions
        fraud_df = df_copy[df_copy['is_fraud'] == 1].copy()
        
        if fraud_df.empty:
            return html.Div("No fraudulent transactions found.", className="text-muted")
        
        # Ensure required columns exist
        if 'risk_score' not in fraud_df.columns:
            fraud_df['risk_score'] = np.random.uniform(0.1, 0.95, size=len(fraud_df))
            
        # Function to format risk score and determine risk level
        def format_risk_score(score):
            try:
                score = float(score)
                if score >= 0.8:
                    return {'value': f"{score:.2f} ⚠️", 'level': 'Critical', 'color': '#dc3545'}
                elif score >= 0.6:
                    return {'value': f"{score:.2f} ⚠️", 'level': 'High', 'color': '#fd7e14'}
                elif score >= 0.4:
                    return {'value': f"{score:.2f} ⚠️", 'level': 'Medium', 'color': '#ffc107'}
                else:
                    return {'value': f"{score:.2f}", 'level': 'Low', 'color': '#28a745'}
            except (ValueError, TypeError):
                return {'value': 'N/A', 'level': 'Unknown', 'color': '#6c757d'}
                
        # Apply risk score formatting and create display columns
        risk_data = fraud_df['risk_score'].apply(format_risk_score)
        fraud_df['risk_value'] = risk_data.apply(lambda x: x['value'])
        fraud_df['risk_level'] = risk_data.apply(lambda x: x['level'])
        fraud_df['risk_color'] = risk_data.apply(lambda x: x['color'])
        
        # Format amount as currency
        fraud_df['amount'] = fraud_df['amount'].apply(lambda x: f'${float(x):,.2f}')
        
        # Create the data table with enhanced styling and verification button
        return html.Div([
            # Verification status message
            html.Div(id="verification-status"),
            
            # Main content div
            html.Div([
                html.Div([
                    html.H5("🚨 Suspicious Transactions", className="mb-0"),
                    html.Button("Verify Selected Transaction", 
                              id="verify-btn", 
                              className="btn btn-outline-danger btn-sm ms-2",
                              n_clicks=0,
                              disabled=True,
                              style={"display": "none"})  # Hidden by default, shown by callback
                ], className="d-flex align-items-center mb-3"),
                
                # Risk level legend
                html.Div([
                    html.Small("Risk Levels: ", className="me-2"),
                    html.Span("Critical (>0.8)", className="badge bg-danger me-2"),
                    html.Span("High (0.6-0.8)", className="badge bg-warning me-2 text-dark"),
                    html.Span("Medium (0.4-0.6)", className="badge bg-info me-2"),
                    html.Span("Low (<0.4)", className="badge bg-success me-2")
                ], className="mb-3"),
                
                # Main data table
                dash_table.DataTable(
                    id='fraud-table',
                    columns=[
                        {"name": "Time", "id": "timestamp"},
                        {"name": "Amount", "id": "amount"},
                        {"name": "Merchant", "id": "merchant"},
                        {"name": "Category", "id": "category"},
                        {"name": "Risk Score", "id": "risk_value"},
                        {"name": "Risk Level", "id": "risk_level"}
                    ],
                    data=fraud_df.to_dict('records'),
                    style_table={
                        'overflowX': 'auto',
                        'maxHeight': '500px',
                        'overflowY': 'auto',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': '1px solid #e9ecef',
                        'backgroundColor': '#fff',
                        'fontFamily': 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px 12px',
                        'fontFamily': 'inherit',
                        'border': 'none',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'minWidth': '100px',
                        'maxWidth': '200px',
                        'borderBottom': '1px solid #e9ecef',
                        'fontSize': '14px',
                        'color': '#212529',
                        'verticalAlign': 'middle'
                    },
                    style_data_conditional=[
                        # Row styling
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgba(0, 0, 0, 0.02)'
                        },
                        {
                            'if': {'state': 'selected'},
                            'backgroundColor': 'rgba(13, 110, 253, 0.1)',
                            'border': 'none',
                            'borderLeft': '3px solid #0d6efd'
                        },
                        # Risk-based row styling
                        {
                            'if': {'filter_query': '{risk_score} >= 0.8'},
                            'borderLeft': '3px solid #dc3545',
                            'backgroundColor': 'rgba(220, 53, 69, 0.05)'
                        },
                        {
                            'if': {'filter_query': '{risk_score} >= 0.6 && {risk_score} < 0.8'},
                            'borderLeft': '3px solid #fd7e14',
                            'backgroundColor': 'rgba(253, 126, 20, 0.05)'
                        },
                        {
                            'if': {'filter_query': '{risk_score} >= 0.4 && {risk_score} < 0.6'},
                            'borderLeft': '3px solid #ffc107',
                            'backgroundColor': 'rgba(255, 193, 7, 0.05)'
                        },
                        {
                            'if': {'filter_query': '{risk_score} < 0.4'},
                            'borderLeft': '3px solid #28a745',
                            'backgroundColor': 'rgba(40, 167, 69, 0.05)'
                        },
                        # Column alignments
                        {
                            'if': {'column_id': 'amount'},
                            'textAlign': 'right',
                            'fontWeight': '600',
                            'color': '#dc3545',
                            'fontFamily': 'monospace',
                            'fontSize': '14px'
                        },
                        {
                            'if': {'column_id': 'risk_score'},
                            'textAlign': 'center',
                            'fontWeight': '600',
                            'fontSize': '14px'
                        },
                        {
                            'if': {'column_id': 'risk_level'},
                            'textAlign': 'center',
                            'fontWeight': '500',
                            'fontSize': '13px'
                        },
                        {
                            'if': {'column_id': 'timestamp'},
                            'minWidth': '140px',
                            'color': '#6c757d',
                            'fontSize': '13px'
                        },
                        # Dynamic risk level colors
                        {
                            'if': {
                                'filter_query': "{risk_level} = 'Critical'",
                                'column_id': 'risk_level'
                            },
                            'color': '#dc3545',
                            'fontWeight': '600',
                            'backgroundColor': 'rgba(220, 53, 69, 0.1)'
                        },
                        {
                            'if': {
                                'filter_query': "{risk_level} = 'High'",
                                'column_id': 'risk_level'
                            },
                            'color': '#fd7e14',
                            'fontWeight': '500',
                            'backgroundColor': 'rgba(253, 126, 20, 0.1)'
                        },
                        {
                            'if': {
                                'filter_query': "{risk_level} = 'Medium'",
                                'column_id': 'risk_level'
                            },
                            'color': '#ffc107',
                            'fontWeight': '500',
                            'backgroundColor': 'rgba(255, 193, 7, 0.1)'
                        },
                        {
                            'if': {
                                'filter_query': "{risk_level} = 'Low'",
                                'column_id': 'risk_level'
                            },
                            'color': '#28a745',
                            'fontWeight': '500',
                            'backgroundColor': 'rgba(40, 167, 69, 0.1)'
                        },
                        # Hover effects
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': 'rgba(13, 110, 253, 0.05)',
                            'border': 'none',
                            'borderLeft': '3px solid #0d6efd'
                        },
                        {
                            'if': {'state': 'active', 'column_id': 'risk_level'},
                            'backgroundColor': 'rgba(13, 110, 253, 0.05)'
                        }
                    ],
                    page_size=10,
                    page_action='native',
                    filter_action='native',
                    sort_action='native',
                    sort_mode='multi',
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'timestamp'}, 'minWidth': '180px'},
                        {'if': {'column_id': 'amount'}, 'textAlign': 'right'},
                    ],
                    tooltip_data=[
                        {
                            column: {'value': str(value), 'type': 'markdown'}
                            for column, value in row.items()
                        } for row in fraud_df.to_dict('records')
                    ],
                    tooltip_duration=None
                )
            ]),
            
            # Verification Modal

            
            dbc.Modal([
                dbc.ModalHeader("Verify Transaction"),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Transaction ID", width=4, className="fw-bold"),
                            dbc.Col(dbc.Input(id="verify-tx-id", type="text", disabled=True))
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Amount", width=4, className="fw-bold"),
                            dbc.Col(dbc.Input(id="verify-amount", type="text", disabled=True))
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Merchant", width=4, className="fw-bold"),
                            dbc.Col(dbc.Input(id="verify-merchant", type="text"))
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Category", width=4, className="fw-bold"),
                            dbc.Col(dbc.Select(id="verify-category", options=[
                                {"label": "Shopping", "value": "shopping"},
                                {"label": "Grocery", "value": "grocery"},
                                {"label": "Bills", "value": "bills"},
                                {"label": "Entertainment", "value": "entertainment"},
                                {"label": "Other", "value": "other"}
                            ]))
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Notes", width=4, className="fw-bold"),
                            dbc.Col(dbc.Textarea(id="verify-notes", rows=3))
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Is this fraud?", width=4, className="fw-bold"),
                            dbc.Col(dbc.RadioItems(
                                id="verify-fraud",
                                options=[
                                    {"label": "Confirm Fraud", "value": "fraud"},
                                    {"label": "False Positive", "value": "legitimate"}
                                ],
                                inline=True
                            ))
                        ])
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Cancel", id="verify-cancel", className="ms-auto"),
                    dbc.Button("Submit", id="verify-submit", color="primary")
                ])
            ], id="verify-modal", is_open=False)
        ])  # Close the main html.Div
        
    except Exception as e:
        logger.error(f"Error in update_fraud_details: {str(e)}", exc_info=True)
        return html.Div(f"An error occurred: {str(e)}", className="text-danger")

# Callback to handle CSV export
@app.callback(
    Output("download-fraud-csv", "data"),
    [Input("export-fraud-btn", "n_clicks")],
    prevent_initial_call=True,
)
def export_fraud_data(n_clicks):
    """Export fraudulent transactions to CSV."""
    global df
    
    if n_clicks is None:
        return dash.no_update
    
    # Filter fraudulent transactions
    fraud_df = df[df['is_fraud'] == 1].copy()
    
    if fraud_df.empty:
        return dash.no_update
    
    # Save to CSV and return the file
    filename = save_transactions(fraud_df)
    if filename:
        return dcc.send_file(filename)
    return dash.no_update

# Callback to update the transactions table
@app.callback(
    [Output('transactions-table', 'data'),
     Output('last-updated-transactions', 'children')],
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_transactions_table(n):
    """Update the transactions table with the latest data."""
    global df
    
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure timestamp is in datetime format
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Format the data for the table
        table_data = df_copy.sort_values('timestamp', ascending=False).head(10).copy()
        
        # Format timestamp for display
        table_data['timestamp'] = table_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format amount as currency and map fraud status
        table_data['amount'] = table_data['amount'].apply(lambda x: f'${float(x):,.2f}')
        table_data['is_fraud'] = table_data['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
        
        # Ensure risk_score is included if it exists
        if 'risk_score' not in table_data.columns:
            table_data['risk_score'] = 'N/A'
        
        # Get the last updated time
        last_updated = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return table_data.to_dict('records'), last_updated
        
    except Exception as e:
        logging.error(f"Error in update_transactions_table: {str(e)}")
        return [], f"Error: {str(e)}"

# Callback to show/hide verify button based on row selection
@app.callback(
    Output("verify-btn", "style"),
    [Input("fraud-table", "active_cell")]
)
def toggle_verify_button(active_cell):
    if active_cell is not None:
        return {"display": "inline-block"}
    return {"display": "none"}

# Callback for verification modal
@app.callback(
    [Output("verify-modal", "is_open"),
     Output("verify-tx-id", "value"),
     Output("verify-amount", "value"),
     Output("verify-merchant", "value"),
     Output("verify-category", "value"),
     Output("verify-notes", "value"),
     Output("verify-fraud", "value")],
    [Input("verify-btn", "n_clicks"),
     Input("verify-cancel", "n_clicks"),
     Input("verify-submit", "n_clicks")],
    [State("verify-modal", "is_open"),
     State("fraud-table", "active_cell"),
     State("fraud-table", "data")],
    prevent_initial_call=True
)
def toggle_modal(open_btn, close_btn, submit_btn, is_open, active_cell, data):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return [dash.no_update] * 7
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle modal close or submit
    if button_id in ["verify-cancel", "verify-submit"] or not any([open_btn, close_btn, submit_btn]):
        return not is_open, "", "", "", "", "", ""
    
    # Get selected row data when opening modal
    if active_cell and data and button_id == "verify-btn":
        row = data[active_cell['row']]
        return (
            not is_open,  # Toggle modal open
            str(row.get('id', 'N/A')),  # Ensure ID is string
            str(row.get('amount', 'N/A')),
            str(row.get('merchant', '')),
            str(row.get('category', 'other')).lower(),
            "",  # Empty notes by default
            ""   # Reset radio button
        )
    
    return [dash.no_update] * 7

# Callback to handle verification submission
@app.callback(
    Output("verification-status", "children"),
    [Input("verify-submit", "n_clicks")],
    [State("verify-tx-id", "value"),
     State("verify-merchant", "value"),
     State("verify-category", "value"),
     State("verify-notes", "value"),
     State("verify-fraud", "value")],
    prevent_initial_call=True
)
def handle_verification(n_clicks, tx_id, merchant, category, notes, is_fraud):
    if not n_clicks:
        return ""
    
    # Here you would typically save this to a database
    verification_data = {
        'transaction_id': tx_id,
        'timestamp': datetime.now().isoformat(),
        'merchant': merchant,
        'category': category,
        'notes': notes,
        'verified_as': is_fraud,
        'verified_by': 'user'  # In a real app, this would be the logged-in user
    }
    
    # Log the verification
    logging.info(f"Transaction verified: {verification_data}")
    
    # Return success message
    return dbc.Alert(
        f"Transaction {tx_id} marked as {'FRAUD' if is_fraud == 'fraud' else 'LEGITIMATE'}",
        color="success" if is_fraud == 'legitimate' else "danger",
        dismissable=True,
        className="mt-2"
    )

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
