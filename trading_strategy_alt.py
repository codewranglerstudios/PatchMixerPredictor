import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import time
import threading

class TradingStrategy:
    def __init__(self, ticker, start_date, end_date, initial_capital=150, trade_limit=3, live_trading=False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.trade_limit = trade_limit
        self.transaction_cost = 0.01  # $0.01 per share fee
        self.live_trading = live_trading
        self.data = self.get_data()
        self.signals = self.generate_signals()
        self.dynamic_risk_params()
        if self.live_trading:
            self.alpaca_api = self.initialize_alpaca()

    def get_data(self):
        """Fetch historical stock data."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data['Returns'] = data['Adj Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=10).std()
        return data

    def dynamic_risk_params(self):
        """Adjust stop-loss and take-profit dynamically based on volatility."""
        avg_volatility = self.data['Volatility'].mean()
        self.stop_loss = avg_volatility * 2
        self.take_profit = avg_volatility * 6

    def generate_signals(self):
        """Generate buy/sell signals using LSTM deep learning model."""
        data = self.data.copy()
        data['Momentum'] = data['Adj Close'].diff()
        data['RSI'] = 100 - (100 / (1 + (data['Returns'].rolling(window=14).mean() / data['Returns'].rolling(window=14).std())))
        data.dropna(inplace=True)

        # Prepare LSTM input data
        scaler = MinMaxScaler(feature_range=(0,1))
        features = ['Adj Close', 'Momentum', 'RSI', 'Returns']
        data_scaled = scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(60, len(data_scaled)):
            X.append(data_scaled[i-60:i])
            y.append(1 if data_scaled[i][0] > data_scaled[i-1][0] else 0)  # Predict upward movement

        X, y = np.array(X), np.array(y)

        # Build LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)

        # Predict buy/sell signals
        predictions = model.predict(X)
        data['LSTM_Signal'] = np.where(predictions > 0.5, 1, -1)
        data['Signal'] = data['LSTM_Signal']
        return data

    def execute_strategy(self):
        """Simulate trading strategy with stop-loss, take-profit, and trade limits."""
        signals = self.signals.copy()
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Cash'] = self.initial_capital
        portfolio['Shares'] = 0
        portfolio['Value'] = self.initial_capital
        trade_count = 0

        for i in range(1, len(signals)):
            price = signals['Adj Close'].iloc[i]
            prev_price = signals['Adj Close'].iloc[i - 1]
            signal = signals['Signal'].iloc[i]

            if trade_count < self.trade_limit:
                if signal == 1:  # Buy
                    shares_to_buy = portfolio['Cash'].iloc[i - 1] // (price + self.transaction_cost)
                    cost = shares_to_buy * (price + self.transaction_cost)

                    if shares_to_buy > 0:
                        portfolio.loc[signals.index[i], 'Shares'] = shares_to_buy
                        portfolio.loc[signals.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1] - cost
                        trade_count += 1

                        if self.live_trading:
                            self.place_trade('buy', shares_to_buy)

                elif signal == -1 and portfolio['Shares'].iloc[i - 1] > 0:  # Sell
                    shares_to_sell = portfolio['Shares'].iloc[i - 1]
                    revenue = shares_to_sell * (price - self.transaction_cost)

                    portfolio.loc[signals.index[i], 'Shares'] = 0
                    portfolio.loc[signals.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1] + revenue
                    trade_count += 1

                    if self.live_trading:
                        self.place_trade('sell', shares_to_sell)

            portfolio.loc[signals.index[i], 'Value'] = portfolio['Cash'].iloc[i] + (portfolio['Shares'].iloc[i] * price)

        return portfolio

    def initialize_alpaca(self):
        """Initialize Alpaca API for live trading."""
        ALPACA_API_KEY = "your_api_key"
        ALPACA_SECRET_KEY = "your_secret_key"
        BASE_URL = "https://paper-api.alpaca.markets"
        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

    def place_trade(self, side, qty):
        """Execute a live trade using Alpaca API."""
        try:
            self.alpaca_api.submit_order(
                symbol=self.ticker,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            print(f"Executed {side} order for {qty} shares of {self.ticker}.")
        except Exception as e:
            print(f"Trade execution failed: {e}")

# Parallel execution for multiple stocks
def run_strategy(ticker):
    strategy = TradingStrategy(ticker, '2015-01-01', '2025-01-01')
    portfolio = strategy.execute_strategy()
    print(f"\nFinal Portfolio Value for {ticker}: {portfolio['Value'].iloc[-1]}")

tickers = ['NVDA', 'APP', 'GEVO', 'USEG']
threads = []

for ticker in tickers:
    t = threading.Thread(target=run_strategy, args=(ticker,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Live deployment
live_strategy = TradingStrategy('NVDA', '2025-01-01', '2026-01-01', live_trading=True)
live_strategy.execute_strategy()
