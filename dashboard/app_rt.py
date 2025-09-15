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
from dash import dcc, html, dash_table
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

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
    
    # Main content
    html.Div([
        # Left sidebar
        html.Div([
            html.Div([
                html.H4("Filters", className="sidebar-heading"),
                html.Hr(),
                html.Label("Time Range"),
                dcc.Dropdown(
                    id='time-range',
                    options=[
                        {'label': 'Last 24 hours', 'value': '24h'},
                        {'label': 'Last 7 days', 'value': '7d'},
                        {'label': 'All time', 'value': 'all'}
                    ],
                    value='24h',
                    clearable=False
                ),
                html.Br(),
                html.Label("Transaction Type"),
                dcc.Checklist(
                    id='transaction-type',
                    options=[
                        {'label': ' Show only suspicious', 'value': 'suspicious'},
                        {'label': ' Show only high risk', 'value': 'high_risk'}
                    ],
                    value=[]
                ),
                html.Hr(),
                html.Div([
                    html.Small("Last updated: "),
                    html.Span(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                             id='last-updated', 
                             className="text-muted")
                ], className="text-center")
            ], className="sidebar-sticky")
        ], className="sidebar"),
        
        # Main content area
        html.Div([
            # Metrics cards
            html.Div([
                # Total Transactions Card
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-exchange-alt fa-2x text-gray-300"),
                                html.Div([
                                    html.Div("Total Transactions", 
                                            className="text-xs font-weight-bold text-primary text-uppercase mb-1"),
                                    html.Div(f"{len(df):,}", 
                                            className="h5 mb-0 font-weight-bold text-gray-800")
                                ], className="col mr-2")
                            ], className="row no-gutters align-items-center"),
                        ], className="card-body")
                    ], className="card border-left-primary shadow h-100 py-2"),
                ], className="col-xl-3 col-md-6 mb-4"),
                
                # Fraudulent Transactions Card
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle fa-2x text-gray-300"),
                                html.Div([
                                    html.Div("Fraudulent Transactions", 
                                            className="text-xs font-weight-bold text-danger text-uppercase mb-1"),
                                    html.Div(f"{df['is_fraud'].sum():,}", 
                                            className="h5 mb-0 font-weight-bold text-gray-800")
                                ], className="col mr-2")
                            ], className="row no-gutters align-items-center"),
                        ], className="card-body")
                    ], className="card border-left-danger shadow h-100 py-2"),
                ], className="col-xl-3 col-md-6 mb-4"),
                
                # Add more metric cards here...
                
            ], className="row"),
            
            # Charts
            html.Div([
                # Transactions over time
                html.Div([
                    html.Div([
                        html.Div("Transactions Over Time", 
                                className="card-header py-3 d-flex flex-row align-items-center justify-content-between"),
                        html.Div([
                            dcc.Graph(id='transactions-chart')
                        ], className="card-body")
                    ], className="card shadow mb-4")
                ], className="col-lg-8"),
                
                # Fraud distribution
                html.Div([
                    html.Div([
                        html.Div("Fraud Distribution", 
                                className="card-header py-3"),
                        html.Div([
                            dcc.Graph(id='fraud-distribution-chart')
                        ], className="card-body")
                    ], className="card shadow mb-4")
                ], className="col-lg-4"),
                
                # Transactions table
                html.Div([
                    html.Div([
                        html.Div("Recent Transactions", 
                                className="card-header py-3"),
                        html.Div([
                            dash_table.DataTable(
                                id='transactions-table',
                                columns=[
                                    {"name": "Time", "id": "timestamp"},
                                    {"name": "Amount", "id": "amount"},
                                    {"name": "Merchant", "id": "merchant"},
                                    {"name": "Category", "id": "category"},
                                    {"name": "Status", "id": "is_fraud"}
                                ],
                                data=df.sort_values('timestamp', ascending=False).head(10).to_dict('records'),
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'fontFamily': 'Arial, sans-serif',
                                    'border': '1px solid #e3e6f0'
                                },
                                style_header={
                                    'backgroundColor': '#f8f9fc',
                                    'fontWeight': 'bold',
                                    'borderBottom': '1px solid #e3e6f0',
                                    'textAlign': 'center'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {
                                            'filter_query': '{is_fraud} = 1',
                                            'column_id': 'is_fraud'
                                        },
                                        'backgroundColor': '#f8d7da',
                                        'color': '#721c24',
                                        'fontWeight': 'bold'
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{is_fraud} = 0',
                                            'column_id': 'is_fraud'
                                        },
                                        'backgroundColor': '#d1e7dd',
                                        'color': '#0f5132'
                                    }
                                ]
                            )
                        ], className="card-body")
                    ], className="card shadow mb-4")
                ], className="col-12")
                
            ], className="row")
            
        ], className="main-content")
        
    ], className="wrapper"),
    
    # Hidden div to trigger callbacks
    html.Div(id='hidden-div', style={'display': 'none'}),
    
    # Interval component for periodic updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds (5 seconds)
        n_intervals=0
    )
])

# Callback to update the transactions chart
@app.callback(
    Output('transactions-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_transactions_chart(n):
    """Update the transactions chart with sample data."""
    global df
    
    # Add a new random transaction
    new_row = {
        'timestamp': datetime.now(),
        'amount': np.random.lognormal(mean=4, sigma=1) * 10,
        'is_fraud': np.random.choice([0, 1], p=[0.98, 0.02]),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Best Buy', 'Other']),
        'category': np.random.choice(['Shopping', 'Food', 'Travel', 'Entertainment', 'Other'])
    }
    
    # Add to dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Keep only the most recent 100 transactions
    if len(df) > 100:
        df = df.tail(100)
    
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

# Callback to update the transactions table
@app.callback(
    [Output('transactions-table', 'data'),
     Output('last-updated', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_transactions_table(n):
    """Update the transactions table with the latest data."""
    global df
    
    # Format the data for the table
    table_data = df.sort_values('timestamp', ascending=False).head(10).copy()
    table_data['timestamp'] = table_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    table_data['amount'] = table_data['amount'].apply(lambda x: f'${x:,.2f}')
    table_data['is_fraud'] = table_data['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    
    # Get the current time for the last updated timestamp
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return table_data.to_dict('records'), last_updated

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
