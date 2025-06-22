import yfinance as yf
import pandas as pd
import numpy as np

# This script reuses the core logic from signal_generator.py to produce
# table snapshots for the final research report.

# --- 1. Configuration (Mirrors the main script) ---
CONFIG = {
    "ticker": "EURUSD=X",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "future_period": 5,
    "price_change_threshold": 0.01
}

def fetch_data(config):
    """Fetches historical data from Yahoo Finance."""
    print("Step 1: Fetching data...")
    data = yf.download(config['ticker'], start=config['start_date'], end=config['end_date'])
    if data.empty:
        raise ValueError("No data fetched.")
    print("Data fetched successfully.\n")
    return data

def engineer_features(df):
    """Creates technical indicator and date-based features."""
    print("Step 2: Engineering features...")
    # --- Technical Indicators ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

    # --- Date-Based Features ---
    df['day_of_week'] = df.index.dayofweek
    df['month_of_year'] = df.index.month
    df['day_of_month'] = df.index.day

    df_featured = df.dropna()
    print("Features engineered successfully.\n")
    return df_featured

def define_target(df, config):
    """Creates the target variable (Signal)."""
    print("Step 3: Defining target variable...")
    df_copy = df.copy()
    future_close_prices = df_copy['Close'].shift(-config['future_period'])
    price_change = (future_close_prices - df_copy['Close']) / df_copy['Close']

    conditions = [
        (price_change > config['price_change_threshold']),
        (price_change < -config['price_change_threshold'])
    ]
    choices = [1, -1]
    df_copy['Signal'] = np.select(conditions, choices, default=0)

    df_labeled = df_copy.dropna()
    df_labeled['Signal'] = df_labeled['Signal'].astype(int)
    print("Target variable defined successfully.\n")
    return df_labeled

def generate_snapshots():
    """Main function to generate and print the table snapshots."""
    # Execute the data processing pipeline
    raw_data = fetch_data(CONFIG)
    featured_data = engineer_features(raw_data)
    final_labeled_data = define_target(featured_data, CONFIG)

    # --- Generate Snapshot 1: Data with Engineered Features ---
    print("="*80)
    print(">>> TABLE 1: Snapshot of Data with Engineered Features <<<")
    print("="*80)
    # Select a readable subset of columns for the report
    cols_snapshot1 = [
        'Open', 'High', 'Low', 'Close', 'SMA_20', 'RSI', 'MACD', 'Volatility',
        'day_of_week', 'month_of_year'
    ]
    snapshot1 = featured_data[cols_snapshot1].head()
    # Format numbers for better readability in the report
    print(snapshot1.to_markdown(floatfmt=".4f"))
    print("\n\n")


    # --- Generate Snapshot 2: Final Labeled Dataset for Training ---
    print("="*80)
    print(">>> TABLE 2: Sample of the Final Labeled Dataset for Model Training <<<")
    print("="*80)
    # Define the final features list, mirroring the main script
    final_features = [
        'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Volatility',
        'day_of_week', 'month_of_year', 'day_of_month'
    ]
    cols_snapshot2 = final_features + ['Signal']
    snapshot2 = final_labeled_data[cols_snapshot2].head()
    print(snapshot2.to_markdown(floatfmt=".4f"))

if __name__ == "__main__":
    # Set pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    generate_snapshots()