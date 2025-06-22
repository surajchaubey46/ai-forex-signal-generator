import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- 1. Configuration ---
# This is the central control panel for the project, allowing for easy experimentation.
CONFIG = {
    "ticker": "EURUSD=X",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "features": [
        'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Volatility', # Technical indicators
        'day_of_week', 'month_of_year', 'day_of_month'                  # Time-based features
    ],
    "future_period": 5,          # Corresponds to a one-week trading horizon
    "price_change_threshold": 0.01, # Defines a significant market move
    "test_size": 0.2,            # 80% for training, 20% for out-of-sample testing
    "model_params": {
        "n_estimators": 100,     # Number of trees in the forest
        "random_state": 42,      # Ensures reproducibility
        "class_weight": "balanced" # Critical for handling imbalanced financial data
    },
    "output_dir": "output"
}

class ForexSignalGenerator:
    """
    Encapsulates the entire machine learning pipeline for generating and evaluating
    Forex trading signals, addressing key scientific and technical considerations.
    """
    def __init__(self, config):
        self.config = config
        self.data = None
        self.model = None
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])

    def fetch_data(self):
        """Fetches and prepares the raw OHLC data from the Yahoo Finance API."""
        print(f"Fetching data for {self.config['ticker']}...")
        self.data = yf.download(self.config['ticker'], start=self.config['start_date'], end=self.config['end_date'])
        if self.data.empty:
            raise ValueError("No data fetched. Check ticker or date range.")
        print("Data fetched successfully.")

    def engineer_features(self):
        """Constructs a feature set from raw data for model training."""
        print("Engineering features...")
        df = self.data
        # Technical Indicators
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
        # Time-Based Features
        df['day_of_week'] = df.index.dayofweek
        df['month_of_year'] = df.index.month
        df['day_of_month'] = df.index.day
        self.data = df.dropna()
        print("Features engineered.")

    def define_target(self):
        """Creates the target variable ('Signal') based on future price movements."""
        print("Defining target variable...")
        df = self.data.copy()
        future_close_prices = df['Close'].shift(-self.config['future_period'])
        price_change = (future_close_prices - df['Close']) / df['Close']
        conditions = [
            (price_change > self.config['price_change_threshold']),
            (price_change < -self.config['price_change_threshold'])
        ]
        choices = [1, -1] # 1 for BUY, -1 for SELL
        df['Signal'] = np.select(conditions, choices, default=0) # 0 for HOLD
        self.data = df.dropna()
        self.data['Signal'] = self.data['Signal'].astype(int)
        print("Target variable defined.")

    def train_model(self):
        """
        Trains the Random Forest model using a chronological split to prevent data leakage.
        """
        print("\n--- Model Training & Evaluation ---")
        X = self.data[self.config['features']]
        y = self.data['Signal']
        
        # Chronological split is crucial for time-series data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], shuffle=False
        )
        
        print(f"Training model on {len(X_train)} samples...")
        self.model = RandomForestClassifier(**self.config['model_params'])
        self.model.fit(X_train, y_train)
        print("Model training complete.")
        
        model_path = os.path.join(self.config['output_dir'], 'forex_model.joblib')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        y_pred = self.model.predict(X_test)
        self.evaluate(y_test, y_pred)
        
        self.test_data = self.data.loc[X_test.index].copy()
        self.test_data['Predicted_Signal'] = y_pred

    def evaluate(self, y_true, y_pred):
        """Calculates and presents model performance metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy on Test Set: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0))

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['SELL', 'HOLD', 'BUY'], yticklabels=['SELL', 'HOLD', 'BUY'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Signal')
        plt.xlabel('Predicted Signal')
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix.png'))
        plt.close()

        # Feature Importance
        importances = pd.Series(self.model.feature_importances_, index=self.config['features'])
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.sort_values().plot(kind='barh', ax=ax)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'feature_importance.png'))
        plt.close()

    def backtest(self):
        """
        Performs a simple vectorized backtest to evaluate the practical
        profitability of the model's signals.
        """
        print("\n--- Backtesting Performance ---")
        df_test = self.test_data
        df_test['Market_Return'] = df_test['Close'].pct_change()
        df_test['Strategy_Return'] = df_test['Market_Return'] * df_test['Predicted_Signal'].shift(1)
        df_test['Cumulative_Market_Return'] = (1 + df_test['Market_Return']).cumprod()
        df_test['Cumulative_Strategy_Return'] = (1 + df_test['Strategy_Return']).cumprod()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        df_test['Cumulative_Market_Return'].plot(ax=ax, label='Market (Buy and Hold)')
        df_test['Cumulative_Strategy_Return'].plot(ax=ax, label='AI Strategy')
        ax.set_title('Backtest: AI Strategy vs. Market Performance')
        ax.set_ylabel('Cumulative Returns')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(self.config['output_dir'], 'backtest_performance.png'))
        plt.close()
        
        final_market_return = df_test['Cumulative_Market_Return'].iloc[-1]
        final_strategy_return = df_test['Cumulative_Strategy_Return'].iloc[-1]
        print(f"\nFinal Cumulative Market Return: {final_market_return:.2f}")
        print(f"Final Cumulative Strategy Return: {final_strategy_return:.2f}")

    def run(self):
        """Executes the full pipeline from data fetching to backtesting."""
        self.fetch_data()
        self.engineer_features()
        self.define_target()
        self.train_model()
        self.backtest()
        print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    generator = ForexSignalGenerator(CONFIG)
    generator.run()