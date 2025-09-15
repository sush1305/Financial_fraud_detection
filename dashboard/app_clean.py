"""
Real-time Fraud Detection Dashboard - Clean Version
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fraud_detection.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True  # Add this to suppress callback exceptions
)
app.title = "Real-time Fraud Detection Dashboard"

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class FraudDetector:
    def __init__(self):
        self.suspicious_merchants = ['Unknown', 'International', 'Overseas', 'CryptoExchange', 'Offshore']
        self.high_risk_categories = ['Gambling', 'Cryptocurrency', 'High-End Retail', 'Digital Currency', 'Forex']
        self.merchants = ['Amazon', 'Walmart', 'Target', 'eBay', 'Best Buy', 'Apple', 'Google', 'Microsoft', 'Netflix', 'Uber', 'Lyft']
        self.categories = ['Electronics', 'Clothing', 'Food', 'Entertainment', 'Travel', 'Groceries', 'Utilities']
        self.last_timestamp = datetime.now()
        self.transaction_counter = 0
        
    def generate_sample_data(self, n=1):
        """Generate sample transaction data for testing."""
        np.random.seed(None)  # Remove fixed seed for randomness
        
        # Generate timestamps with increasing time
        self.last_timestamp += timedelta(minutes=5)
        timestamps = [self.last_timestamp - timedelta(minutes=i*5) for i in range(n)]
        
        # Generate random data with more variation
        amounts = np.random.lognormal(4, 1.5, n).round(2)
        
        # More realistic merchant and category distribution
        merchants = np.random.choice(
            self.merchants + self.suspicious_merchants,
            n,
            p=[0.85/len(self.merchants)]*len(self.merchants) + [0.15/len(self.suspicious_merchants)]*len(self.suspicious_merchants)
        )
        
        categories = []
        is_fraud = []
        
        for merchant in merchants:
            if merchant in self.suspicious_merchants:
                # Higher chance of fraud for suspicious merchants
                cat = np.random.choice(self.high_risk_categories)
                fraud = np.random.binomial(1, 0.4)  # 40% chance of fraud
            else:
                cat = np.random.choice(self.categories)
                fraud = np.random.binomial(1, 0.02)  # 2% chance of fraud for normal merchants
            categories.append(cat)
            is_fraud.append(fraud)
        
        # Create DataFrame
        df = pd.DataFrame({
            'transaction_id': [f"TX{self.transaction_counter + i:06d}" for i in range(n)],
            'timestamp': timestamps,
            'amount': amounts,
            'merchant': merchants,
            'category': categories,
            'is_fraud': is_fraud,
            'verified': [False] * n,
            'verification_notes': [''] * n,
            'verification_status': ['pending'] * n
        })
        
        self.transaction_counter += n
        
        # Add risk scores
        df['risk_score'] = df.apply(self._calculate_risk_score, axis=1)
        return df
    
    def _calculate_risk_score(self, row):
        """Calculate risk score based on transaction features."""
        score = 0.0
        
        # Base score based on amount (0-0.4)
        amount_factor = min(1.0, row['amount'] / 10000)  # Cap at 10,000 for scoring
        score += 0.4 * amount_factor
        
        # Merchant-based risk (0-0.3)
        if row['merchant'] in self.suspicious_merchants:
            score += 0.3
            
        # Category-based risk (0-0.2)
        if row['category'] in self.high_risk_categories:
            score += 0.2
            
        # Fraud indicator (0.5 if fraud, 0 otherwise)
        if row.get('is_fraud', 0) == 1:
            score = max(score, 0.7)  # Ensure fraud transactions have high risk
        
        # Add some randomness (0-0.1)
        score += np.random.uniform(0, 0.1)
        
        # Apply sigmoid function to create more natural distribution
        # This will create an S-curve that's flatter at the extremes
        def sigmoid(x):
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
            
        # Scale the score to be between 0.05 and 0.95
        score = 0.05 + 0.9 * sigmoid(score - 0.5)
        
        # Add some noise to prevent too many identical scores
        score = min(0.99, max(0.01, score + np.random.normal(0, 0.02)))
        
        return float(score)

# Initialize components
detector = FraudDetector()
df = detector.generate_sample_data(1000)

# Layout

app.layout = dbc.Container(fluid=True, children=[
    # Navigation
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
            dbc.NavItem(dbc.NavLink("Alerts", href="#")),
            dbc.NavItem(dbc.NavLink("Reports", href="#")),
        ],
        brand="Fraud Detection Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Metrics Row
    dbc.Row([
        # Total Transactions
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6("Total Transactions", className="card-subtitle mb-2 text-muted"),
                    html.H2(id="total-transactions", className="card-title"),
                    html.P("All transactions processed", className="card-text")
                ])
            ),
            md=3
        ),
        
        # Fraudulent Transactions
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6("Fraudulent Transactions", className="card-subtitle mb-2 text-muted"),
                    html.H2(id="fraud-count", className="card-title text-danger"),
                    html.P("Suspicious activities detected", className="card-text")
                ])
            ),
            md=3
        ),
        
        # High Risk Transactions
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6("High Risk", className="card-subtitle mb-2 text-muted"),
                    html.H2(id="high-risk-count", className="card-title text-warning"),
                    html.P("Transactions requiring review", className="card-text")
                ])
            ),
            md=3
        ),
        
        # Total Amount at Risk
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6("Amount at Risk", className="card-subtitle mb-2 text-muted"),
                    html.H2(id="amount-at-risk", className="card-title text-success"),
                    html.P("Potential financial exposure", className="card-text")
                ])
            ),
            md=3
        )
    ], className="mb-4"),
    
    # Main Content
    dbc.Row([
        # Left Column - Charts
        dbc.Col(md=8, children=[
            # Transactions Over Time
            dbc.Card(className="mb-4", children=[
                dbc.CardHeader("Transaction Activity Over Time"),
                dbc.CardBody([
                    dcc.Graph(id="transactions-chart")
                ])
            ]),
            
            # Recent Transactions
            dbc.Card(children=[
                dbc.CardHeader([
                    html.Div([
                        html.Span("Recent Transactions", className="h6 mb-0"),
                        dbc.Button("Export", id="export-btn", color="primary", size="sm", className="ms-2")
                    ], className="d-flex justify-content-between align-items-center")
                ]),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='recent-transactions',
                        columns=[
                            {"name": "Time", "id": "timestamp"},
                            {"name": "Amount", "id": "amount"},
                            {"name": "Merchant", "id": "merchant"},
                            {"name": "Category", "id": "category"},
                            {"name": "Risk", "id": "risk_score", "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Status", "id": "status"}
                        ],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontSize': '12px',
                            'font-family': 'sans-serif',
                            'maxWidth': '150px',
                            'textOverflow': 'ellipsis',
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
                                    'filter_query': '{risk_score} > 0.7',
                                    'column_id': 'risk_score'
                                },
                                'backgroundColor': 'rgba(231, 74, 59, 0.1)',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': '{status} = "High Risk"',
                                    'column_id': 'status'
                                },
                                'color': 'white',
                                'backgroundColor': '#e74a3b',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': '{status} = "Suspicious"',
                                    'column_id': 'status'
                                },
                                'color': 'white',
                                'backgroundColor': '#f6c23e',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {
                                    'filter_query': '{status} = "Verified"',
                                    'column_id': 'status'
                                },
                                'color': 'white',
                                'backgroundColor': '#1cc88a',
                                'fontWeight': 'bold'
                            }
                        ]
                    )
                ])
            ])
        ]),
        
        # Right Column - Fraud Alerts
        dbc.Col(md=4, children=[
            dbc.Card(className="mb-4", children=[
                dbc.CardHeader("Fraud Alerts"),
                dbc.CardBody([
                    html.Div(id="fraud-alerts", className="list-group")
                ])
            ]),
            
            # Risk Distribution and Analysis
            dbc.Card(children=[
                dbc.CardHeader([
                    dbc.Tabs([
                        dbc.Tab(label="Risk Distribution", tab_id="tab-risk"),
                        dbc.Tab(label="By Category", tab_id="tab-category"),
                        dbc.Tab(label="By Merchant", tab_id="tab-merchant"),
                    ], id="analysis-tabs", active_tab="tab-risk")
                ]),
                dbc.CardBody([
                    html.Div(id="analysis-content")
                ])
            ]),
        ])
    ]),
    
    # Hidden components
    dcc.Store(id='transactions-store'),
    dcc.Store(id='selected-transaction', data=None),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
    
    # Verification Modal for transaction verification
    dbc.Modal([
        dbc.ModalHeader("Verify Transaction"),
        dbc.ModalBody([
            dbc.Form([
                dbc.Row([
                    dbc.Label("Transaction ID", width=4),
                    dbc.Col(dbc.Input(id='verify-tx-id', type='text', readonly=True))
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Amount", width=4),
                    dbc.Col(dbc.Input(id='verify-amount', type='number', readonly=True))
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Merchant", width=4),
                    dbc.Col(dbc.Input(id='verify-merchant', type='text', readonly=True))
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Category", width=4),
                    dbc.Col(dbc.Input(id='verify-category', type='text', readonly=True))
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Status", width=4),
                    dbc.Col(dcc.Dropdown(
                        id='verify-status',
                        options=[
                            {"label": "Verified Legitimate", "value": "legitimate"},
                            {"label": "Confirmed Fraud", "value": "fraud"},
                            {"label": "Needs Review", "value": "review"}
                        ],
                        value=""
                    ))
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("Notes", width=4),
                    dbc.Col(dbc.Textarea(id='verify-notes', placeholder="Add verification notes...", style={'height': '100px'}))
                ], className="mb-3")
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="verify-cancel", className="me-2", color="secondary"),
            dbc.Button("Submit", id="verify-submit", color="primary")
        ])
    ], id="verification-modal", is_open=False, size="lg"),
    
    # Status message container
    html.Div(id="verification-status", className="mt-3")
])

# Initialize a global counter for transaction IDs
global transaction_id_counter
transaction_id_counter = 0

# Callback to update metrics and recent transactions
@app.callback(
    [
        Output('total-transactions', 'children'),
        Output('fraud-count', 'children'),
        Output('high-risk-count', 'children'),
        Output('amount-at-risk', 'children'),
        Output('recent-transactions', 'data'),
        Output('transactions-store', 'data')
    ],
    [Input('interval-component', 'n_intervals')],
    [State('transactions-store', 'data')]
)
def update_metrics(n, current_data):
    """Update the dashboard metrics."""
    global df, transaction_id_counter
    
    try:
        # Initialize empty data if none exists or invalid format
        if not current_data or not isinstance(current_data, list):
            current_data = []
            print("Initialized empty transaction data list")
        
        # Generate new transaction with sequential ID
        try:
            # Generate 1-3 new transactions at a time
            num_new = np.random.randint(1, 4)
            new_data = detector.generate_sample_data(num_new).to_dict('records')
            
            # Add transaction IDs and timestamps
            for i, tx in enumerate(new_data):
                transaction_id_counter += 1
                tx['transaction_id'] = f"TX{transaction_id_counter:06d}"
                tx['timestamp'] = pd.Timestamp.now().isoformat()
                
                # Ensure all required fields are present
                required_fields = ['amount', 'merchant', 'category', 'risk_score', 'is_fraud']
                for field in required_fields:
                    if field not in tx:
                        if field == 'amount':
                            tx[field] = round(np.random.uniform(10, 1000), 2)
                        elif field == 'risk_score':
                            tx[field] = min(0.99, max(0.01, np.random.beta(2, 5)))  # Skewed towards lower risk
                        elif field == 'is_fraud':
                            tx[field] = 1 if tx.get('risk_score', 0) > 0.8 and np.random.random() < 0.3 else 0
                        else:
                            tx[field] = 'Unknown'
            
            current_data.extend(new_data)
            print(f"Added {len(new_data)} new transactions")
            
        except Exception as e:
            print(f"Error generating sample data: {str(e)}")
            if not current_data:  # If no data exists, create a default entry
                current_data = [{
                    'transaction_id': f"TX{transaction_id_counter:06d}",
                    'amount': round(np.random.uniform(10, 1000), 2),
                    'merchant': 'Default Merchant',
                    'category': 'Other',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'risk_score': round(np.random.uniform(0.1, 0.3), 2),
                    'is_fraud': 0
                }]
        
        # Keep only the last 100 transactions for performance
        current_data = current_data[-100:]
        print(f"Current data length: {len(current_data)}")
        
        # Convert to DataFrame for calculations
        df = pd.DataFrame(current_data)
        print("DataFrame columns:", df.columns.tolist())
        
        # Ensure required columns exist with proper types
        required_columns = {
            'risk_score': ('risk_score', float, lambda: np.random.random()),
            'is_fraud': ('is_fraud', int, lambda: 0),
            'amount': ('amount', float, lambda: round(np.random.uniform(10, 1000), 2)),
            'timestamp': ('timestamp', 'datetime64[ns]', pd.Timestamp.now),
            'merchant': ('merchant', str, lambda: 'Unknown'),
            'category': ('category', str, lambda: 'Other'),
            'transaction_id': ('transaction_id', str, lambda: f'TX{transaction_id_counter:06d}')
        }
        
        # Ensure all required columns exist and have correct types
        for col, (col_name, dtype, default_func) in required_columns.items():
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                df[col] = [default_func() for _ in range(len(df))]
            
            try:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Error converting {col} to {dtype}: {str(e)}")
                df[col] = [default_func() for _ in range(len(df))]
        
        # Add risk level for visualization
        try:
            df['risk_level'] = pd.cut(
                df['risk_score'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        except Exception as e:
            print(f"Error creating risk levels: {str(e)}")
            df['risk_level'] = 'Medium'
        
        # Add risk level for visualization
        try:
            df['risk_level'] = pd.cut(
                df['risk_score'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        except Exception as e:
            print(f"Error creating risk levels: {str(e)}")
            df['risk_level'] = 'Medium'
        
        # Calculate metrics with error handling
        try:
            total_transactions = len(df)
            fraud_count = int(df['is_fraud'].sum())
            high_risk_count = int((df['risk_score'] > 0.7).sum())
            amount_at_risk = float(df[df['risk_score'] > 0.7]['amount'].sum())
            
            # Debug output
            print("\n=== Metrics Debug ===")
            print(f"Total transactions: {total_transactions}")
            print(f"Fraud count: {fraud_count}")
            print(f"High risk count: {high_risk_count}")
            print(f"Amount at risk: ${amount_at_risk:,.2f}")
            print("Sample data:", df[['transaction_id', 'amount', 'merchant', 'risk_score', 'is_fraud']].head(2).to_dict('records'))
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            total_transactions = 0
            fraud_count = 0
            high_risk_count = 0
            amount_at_risk = 0.0
        
        # Prepare recent transactions data
        try:
            if 'timestamp' in df.columns:
                recent_transactions = df.sort_values('timestamp', ascending=False).head(10).to_dict('records')
            else:
                recent_transactions = df.head(10).to_dict('records')
        except Exception as e:
            print(f"Error preparing recent transactions: {str(e)}")
            recent_transactions = []
        
        # Ensure all required fields are in each transaction
        for tx in recent_transactions:
            for field in ['transaction_id', 'amount', 'merchant', 'category', 'risk_score', 'is_fraud']:
                if field not in tx:
                    if field == 'amount':
                        tx[field] = 0.0
                    elif field == 'risk_score':
                        tx[field] = 0.0
                    elif field == 'is_fraud':
                        tx[field] = 0
                    else:
                        tx[field] = ''
        
        # Ensure timestamps are strings for JSON serialization
        for tx in current_data:
            if 'timestamp' in tx and isinstance(tx['timestamp'], (pd.Timestamp, datetime)):
                tx['timestamp'] = tx['timestamp'].isoformat()
        
        # Debug: Print the first transaction to verify format
        if current_data:
            print("First transaction in store:", {k: v for k, v in current_data[0].items() if k != 'timestamp'})
        
        # Return all outputs in the correct order
        return (
            str(total_transactions),
            str(fraud_count),
            str(high_risk_count),
            f"${amount_at_risk:,.2f}",
            recent_transactions,
            current_data  # This is the transactions-store data
        )
        
    except Exception as e:
        print(f"Critical error in update_metrics: {str(e)}")
        # Return safe default values
        return ["0", "0", "0", "$0.00", [], []]

# Callback to handle transaction selection for verification
@app.callback(
    [
        Output('verification-modal', 'is_open'),
        Output('verify-tx-id', 'value'),
        Output('verify-amount', 'value'),
        Output('verify-merchant', 'value'),
        Output('verify-category', 'value'),
        Output('selected-transaction', 'data')
    ],
    [
        Input('recent-transactions', 'active_cell'),
        Input('verify-cancel', 'n_clicks'),
        Input('verify-submit', 'n_clicks')
    ],
    [
        State('verification-modal', 'is_open'),
        State('recent-transactions', 'data'),
        State('selected-transaction', 'data')
    ]
)
def handle_verification(active_cell, cancel_clicks, submit_clicks, is_open, rows, selected_tx):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return [dash.no_update] * 6
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # When a transaction is clicked
    if button_id == 'recent-transactions' and active_cell and rows:
        row = rows[active_cell['row']]
        selected_tx = {
            'index': active_cell['row'],
            'id': row.get('transaction_id', str(active_cell['row'])),
            'amount': row.get('amount', ''),
            'merchant': row.get('merchant', ''),
            'category': row.get('category', '')
        }
        return [True, selected_tx['id'], str(row['amount']), row['merchant'], 
                row['category'], selected_tx]
    
    # When cancel or submit is clicked
    elif button_id in ['verify-cancel', 'verify-submit']:
        return [False, None, None, None, None, None]
    
    return [is_open, dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update, dash.no_update]

# Callback to handle verification submission and update status
@app.callback(
    Output('verification-status', 'children'),
    [Input('verify-submit', 'n_clicks')],
    [
        State('verify-tx-id', 'value'),
        State('verify-status', 'value'),
        State('verify-notes', 'value')
    ]
)
def update_verification_status(submit_clicks, tx_id, status, notes):
    if not submit_clicks or not tx_id:
        return ""
    
    status_msg = f"Transaction {tx_id} "
    if status == 'legitimate':
        status_msg += "marked as legitimate"
    elif status == 'fraud':
        status_msg += "reported as fraud"
    else:
        status_msg += "marked for review"
        
    if notes:
        status_msg += f" with notes: {notes}"
    
    return dbc.Alert(status_msg, color="success", dismissable=True, className="mt-3")

# Callback to update fraud alerts
@app.callback(
    Output('fraud-alerts', 'children'),
    [Input('transactions-store', 'data')]
)
def update_fraud_alerts(transactions):
    """Update the fraud alerts list."""
    if not transactions or not isinstance(transactions, list) or len(transactions) == 0:
        return ["No recent alerts"]
    
    try:
        df = pd.DataFrame(transactions)
        
        # Get high risk or fraud transactions
        alerts = df[(df['risk_score'] > 0.7) | (df['is_fraud'] == 1)].sort_values('timestamp', ascending=False).head(5)
        
        if len(alerts) == 0:
            return ["No recent alerts"]
            
        alert_items = []
        for _, row in alerts.iterrows():
            alert_type = 'danger' if row['is_fraud'] else 'warning'
            alert_text = f"${row['amount']:.2f} at {row['merchant']}"
            alert_time = pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')
            alert_items.append(
                dbc.Alert(
                    [
                        html.Div([
                            html.Strong(f"{alert_time} - "),
                            html.Span(alert_text)
                        ]),
                        html.Small(f"Risk: {row['risk_score']:.2f}", className="text-muted d-block mt-1")
                    ],
                    color=alert_type,
                    className="mb-2"
                )
            )
            
        return alert_items
        
    except Exception as e:
        print(f"Error updating fraud alerts: {str(e)}")
        return ["Error loading alerts"]

# Callback to update the analysis content based on selected tab
@app.callback(
    Output('analysis-content', 'children'),
    [Input('analysis-tabs', 'active_tab'),
     Input('transactions-store', 'data')]
)
def update_analysis_content(active_tab, data):
    if not data or not isinstance(data, list) or len(data) == 0:
        return dbc.Alert("No transaction data available yet. Please wait for transactions to be generated.", 
                        color="info", className="mt-3")
    
    try:
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        if 'risk_level' not in df.columns:
            df['risk_level'] = pd.cut(df['risk_score'], 
                                    bins=[0, 0.3, 0.7, 1.0], 
                                    labels=['Low', 'Medium', 'High'],
                                    include_lowest=True)
        
        if active_tab == 'tab-risk':
            # Risk Distribution Pie Chart
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            fig = px.pie(risk_counts, values='Count', names='Risk Level', 
                         title='Transaction Risk Distribution',
                         color_discrete_map={
                             'Low': '#2ecc71',
                             'Medium': '#f39c12',
                             'High': '#e74c3c'
                         })
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
            )
            
            return dcc.Graph(figure=fig, className="mt-3")
        
        elif active_tab == 'tab-category':
            # Category Analysis Bar Chart
            if 'category' not in df.columns:
                return dbc.Alert("Category data not available.", color="warning")
                
            category_risk = df.groupby('category')['risk_score'].mean().sort_values(ascending=False).reset_index()
            
            fig = px.bar(category_risk, 
                        x='category', 
                        y='risk_score',
                        title='Average Risk Score by Category',
                        labels={'risk_score': 'Average Risk Score', 'category': 'Category'})
            
            fig.update_layout(
                xaxis_tickangle=-45,
                coloraxis_showscale=False
            )
            
            return dcc.Graph(figure=fig, className="mt-3")
        
        elif active_tab == 'tab-merchant':
            # Merchant Analysis Scatter Plot
            if 'merchant' not in df.columns or 'amount' not in df.columns:
                return dbc.Alert("Merchant or amount data not available.", color="warning")
                
            merchant_risk = df.groupby('merchant').agg({
                'risk_score': 'mean',
                'amount': 'mean',
                'transaction_id': 'count',
                'is_fraud': 'sum'
            }).reset_index()
            
            if len(merchant_risk) == 0:
                return dbc.Alert("No merchant data available.", color="warning")
                
            merchant_risk = merchant_risk[merchant_risk['transaction_id'] > 5]  # Only show merchants with >5 transactions
                
            fig = px.scatter(merchant_risk, 
                            x='risk_score', 
                            y='amount',
                            size='transaction_id', 
                            color='is_fraud',
                            title='Merchant Risk Analysis',
                            hover_data=['merchant', 'transaction_id'],
                            labels={
                                'risk_score': 'Average Risk Score',
                                'amount': 'Average Amount',
                                'is_fraud': 'Fraud Count',
                                'transaction_id': 'Transaction Count'
                            })
            
            return dcc.Graph(figure=fig, className="mt-3")
        
        return dbc.Alert("Select an analysis tab", color="info")
        
    except Exception as e:
        return dbc.Alert(f"Error generating visualization: {str(e)}", color="danger")

# Callback for transactions chart
@app.callback(
    Output('transactions-chart', 'figure'),
    [Input('transactions-store', 'data')]
)
def update_transactions_chart(data):
    """Update the transactions over time chart."""
    print("\n=== update_transactions_chart called ===")
    
    # Check if we have data
    if not data or not isinstance(data, list) or len(data) == 0:
        print("No data available for chart")
        return {
            'data': [],
            'layout': {
                'title': 'No transaction data available',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'Waiting for transaction data...',
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            }
        }
    
    # Create empty figure with error message
    def create_error_figure(message):
        print(f"Creating error figure: {message}")
        return {
            'data': [],
            'layout': {
                'title': 'Transaction Data Issue',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': message,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14}
                }],
                'height': 400,
                'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40}
            }
        }
    
    # Check input data
    if not data or not isinstance(data, list):
        return create_error_figure("No transaction data received")
    
    if len(data) == 0:
        return create_error_figure("Transaction data is empty")
    
    try:
        print(f"Processing {len(data)} transactions...")
        
        # Create DataFrame and ensure proper data types
        df = pd.DataFrame(data)
        print("Original columns:", df.columns.tolist())
        
        # Ensure required columns with proper types
        required_columns = {
            'timestamp': 'datetime64[ns]',
            'amount': float,
            'is_fraud': int,
            'risk_score': float,
            'transaction_id': str,
            'merchant': str,
            'category': str
        }
        
        # Add missing columns with defaults
        for col, col_type in required_columns.items():
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                elif col == 'is_fraud':
                    df[col] = 0
                elif col == 'risk_score':
                    df[col] = np.random.random()
                elif col == 'amount':
                    df[col] = np.random.uniform(10, 1000).round(2)
                else:
                    df[col] = ''
            
            # Convert to correct type
            try:
                if col_type == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(col_type)
            except Exception as e:
                print(f"Error converting {col} to {col_type}: {str(e)}")
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                else:
                    df[col] = 0 if col_type in [int, float] else ''
        
        print("Processed columns:", df.columns.tolist())
        
        # Ensure we have at least one data point
        if len(df) == 0:
            return create_error_figure("No transaction data available")
        
        # Create time-based grouping
        df = df.sort_values('timestamp')
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Create transaction type column
        df['transaction_type'] = df['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
        
        # Ensure we have both transaction types
        if 'transaction_type' not in df.columns:
            df['transaction_type'] = 'Legitimate'
        
        print(f"Transaction types: {df['transaction_type'].unique().tolist()}")
        
        # Create hourly aggregates
        try:
            hourly = df.groupby(['hour', 'transaction_type']).size().unstack(fill_value=0)
            
            # Ensure both transaction types exist
            for t in ['Legitimate', 'Fraud']:
                if t not in hourly.columns:
                    print(f"Adding missing transaction type: {t}")
                    hourly[t] = 0
            
            # Reset index for plotting
            plot_data = hourly.reset_index()
            
            print(f"Plot data shape: {plot_data.shape}")
            print(f"Columns: {plot_data.columns.tolist()}")
            
            # Create the figure using Plotly Express
            import plotly.express as px
            
            # Convert to long format for Plotly Express
            plot_data_long = pd.melt(plot_data, 
                                  id_vars=['hour'], 
                                  value_vars=['Legitimate', 'Fraud'],
                                  var_name='Transaction Type',
                                  value_name='Count')
            
            # Create the line chart
            fig = px.line(plot_data_long, 
                         x='hour', 
                         y='Count',
                         color='Transaction Type',
                         title='Transaction Volume Over Time',
                         labels={'hour': 'Time', 'Count': 'Number of Transactions'},
                         color_discrete_map={
                             'Legitimate': '#2ecc71',
                             'Fraud': '#e74c3c'
                         })
            
            # Update layout
            fig.update_layout(
                hovermode='x unified',
                height=400,
                margin={'l': 50, 'r': 30, 't': 60, 'b': 50},
                legend_title_text='Transaction Type'
            )
            
            print("Chart generated successfully")
            return fig
            
        except Exception as e:
            error_msg = f"Error creating chart: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return create_error_figure(error_msg)
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return create_error_figure(error_msg)

# Prediction form
app.layout.children.extend([
    html.Div([
        html.H4("Fraud Prediction", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Amount ($)", html_for="predict-amount"),
                dbc.Input(id="predict-amount", type="number", placeholder="Enter amount", className="mb-3")
            ], md=4),
            dbc.Col([
                dbc.Label("Category", html_for="predict-category"),
                dcc.Dropdown(
                    id="predict-category",
                    options=[
                        {"label": category, "value": category} for category in [
                            "Shopping", "Food & Dining", "Bills & Utilities", "Travel",
                            "Entertainment", "Health", "Other"
                        ]
                    ],
                    placeholder="Select category",
                    className="mb-3"
                )
            ], md=4),
            dbc.Col([
                dbc.Label("Merchant", html_for="predict-merchant"),
                dbc.Input(id="predict-merchant", type="text", placeholder="Enter merchant name", className="mb-3")
            ], md=4)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Button("Check Fraud Risk", id="predict-button", color="primary", className="w-100")
            ])
        ]),
        html.Div(id="prediction-result", className="mt-3")
    ], className="mt-5 p-4 border rounded")
])

# Callback for fraud prediction
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('predict-amount', 'value'),
        State('predict-category', 'value'),
        State('predict-merchant', 'value')
    ]
)
def predict_fraud(n_clicks, amount, category, merchant):
    if n_clicks is None or amount is None or not category or not merchant:
        return ""
    
    # Create a transaction row for prediction
    transaction = {
        'amount': float(amount),
        'category': category,
        'merchant': merchant,
        'is_fraud': 0  # Default, will be predicted
    }
    
    # Calculate risk score
    risk_score = detector._calculate_risk_score(transaction)
    
    # Determine if likely fraud
    is_high_risk = risk_score > 0.7
    is_suspicious = risk_score > 0.4
    
    # Create result message
    if is_high_risk or merchant in detector.suspicious_merchants or category in detector.high_risk_categories:
        color = "danger"
        message = f"⚠️ High Fraud Risk ({risk_score*100:.1f}%)"
        details = [
            html.P(f"• Merchant '{merchant}' is marked as suspicious" if merchant in detector.suspicious_merchants else 
                   f"• Category '{category}' is considered high risk" if category in detector.high_risk_categories else
                   f"• High risk score indicates potential fraud"),
            html.P(f"• Amount: ${amount:,.2f} is above average" if amount > 1000 else "")
        ]
    elif is_suspicious:
        color = "warning"
        message = f"⚠️ Suspicious Transaction ({risk_score*100:.1f}% risk)"
        details = [
            html.P("• This transaction shows some suspicious patterns"),
            html.P(f"• Consider verifying this transaction")
        ]
    else:
        color = "success"
        message = f"✅ Low Risk ({risk_score*100:.1f}%)"
        details = [
            html.P("• This transaction appears to be legitimate"),
            html.P(f"• Normal spending pattern detected")
        ]
    
    return dbc.Card([
        dbc.CardHeader(html.Strong("Fraud Prediction Result")),
        dbc.CardBody([
            html.H4(message, className=f"text-{color}", style={"margin-bottom": "1rem"}),
            html.Div(details, className="text-muted")
        ])
    ])

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
