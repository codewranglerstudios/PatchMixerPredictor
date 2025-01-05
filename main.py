import asyncio
from datetime import datetime, timedelta

import pandas as pd
import requests
import talib
from bs4 import BeautifulSoup

from data.scraper_reddit import RedditScraper
from data.stock_data_fetcher import StockDataFetcher
from data.trading_patterns import TradingPatterns
from data.scraper_google import GoogleScraper
import os
import pandas as pd
import yfinance as yf
from trading_env import DQNAgent
from trading_env import TradingEnvironment
from trading_strategy import TradingStrategy
from robinhood_trading import RobinhoodTrading
from informer import Informer  # Assuming this is a custom Informer implementation.

def string_to_float(s: str) -> float:
    # Define a mapping for each character to a unique integer
    CHAR_MAP = {chr(i): i - 32 for i in range(32, 127)}
    # Reverse mapping to decode integer back to string
    REVERSE_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}
    # Maximum length of string we want to encode
    MAX_LENGTH = 6
    # Ensure the string is at most MAX_LENGTH characters long
    if len(s) > MAX_LENGTH:
        raise ValueError(f"String length must be at most {MAX_LENGTH} characters")

    # Encode the string into a unique integer
    encoded_val = 0
    for char in s:
        encoded_val = encoded_val * len(CHAR_MAP) + CHAR_MAP[char]

    # Normalize the integer to a float between 0 and 1 with up to 6 decimal places
    max_val = (len(CHAR_MAP) ** MAX_LENGTH) - 1
    normalized_float = round(encoded_val / max_val, 6)

    return normalized_float


# returns all tickers on the nyse (a-z)Z
def get_ticker_symbols():
    base_url = "https://www.eoddata.com/stocklist/NYSE/"
    tickers = []
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        url = f"{base_url}{letter}.htm"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'quotes'})
        if table:
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if cols:
                    ticker = cols[0].text.strip()
                    tickers.append(ticker)
    return tickers


async def get_data(symbol, days=2000):
    # Fetch stock data from historical CSV
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    fetcher = StockDataFetcher(symbol)
    stock_data = fetcher.fetch_historical_stock_data(start_date, end_date)
    analyzer = SentimentAnalyzer(symbol)
    stock_data = analyzer.add_sentiment(stock_data)
    # Convert 'Datetime' to datetime object and sort as index
    if 'Datetime' in stock_data.index.names:
        stock_data.index.name = 'date'
        stock_data.reset_index(inplace=True)
    if 'Datetime' in stock_data.columns:
        stock_data['date'] = stock_data['Datetime']
        stock_data.drop('Datetime', axis=1, inplace=True)
    stock_data['Close'] = stock_data['Adj Close'].where(stock_data['Adj Close'] != 0, stock_data['Close'])
    stock_data.sort_index(inplace=True)
    if 'date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)
    stock_data['id'] = string_to_float(symbol)
    stock_data['RSI'] = 0
    stock_data['EMA'] = 0
    # Reverse the DataFrame
    reversed_stock_data = stock_data.iloc[::-1]
    # Forward fill 'RSI' 'EMA' column
    stock_data['RSI'] = stock_data['RSI'].ffill()
    stock_data['EMA'] = stock_data['EMA'].ffill()
    stock_data = stock_data.ffill()

    # Replace any remaining NaN values in 'RSI' column with 0
    stock_data['RSI'].fillna(0, inplace=True)
    stock_data['EMA'].fillna(0, inplace=True)

    google_scraper = GoogleScraper()
    articles_google = await google_scraper.fetch_google_articles(stock_symbol=symbol, num_articles=200000)
    articles_google = pd.DataFrame(articles_google)
    stock_data = analyzer.add_sentiment(stock_data=stock_data, news_articles=articles_google)

    # get reddit sentiment
    scraper = RedditScraper(symbol)
    subreddits = await scraper.get_reddit_sentiment_by_days(days=days, verify_content=True)
    stock_data = analyzer.add_sentiment(stock_data=stock_data, news_articles=subreddits)

    data = stock_data
    print(f"data: {data}")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)
    root_path = f'./data/stock_data/'
    data_path = f'{symbol}_stock_data.csv'
    data.to_csv(f"{root_path}{data_path}", index=True, index_label='date')
    return data


def fix_indicators(stock_data, symbol):
    print(f"getting indicators for {stock_data}")
    print(f"getting indicators for {stock_data.columns}")
    stock_data['RSI'] = 0
    stock_data['EMA'] = 0
    # Calculate RSI
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=28)

    # Calculate EMA
    stock_data['EMA'] = talib.EMA(stock_data['Close'], timeperiod=41)
    stock_data['SMA'] = talib.SMA(stock_data['Close'], timeperiod=128)

    # Calculate Bollinger Bands
    upper, middle, lower = talib.BBANDS(stock_data['Close'], timeperiod=48)
    stock_data['BB_Upper'] = upper
    stock_data['BB_Middle'] = middle
    stock_data['BB_Lower'] = lower
    stock_data['RSI'] = stock_data['RSI'].ffill()
    stock_data['EMA'] = stock_data['EMA'].ffill()

    patterns = TradingPatterns(stock_data)
    stock_data = patterns.find_all_patterns()
    stock_data = stock_data.ffill()
    stock_data = stock_data.fillna(0)

    # Replace any remaining NaN values in 'RSI' column with 0
    stock_data['RSI'] = stock_data['RSI'].fillna(0)
    stock_data['EMA'] = stock_data['EMA'].fillna(0)
    stock_data['BB_Upper'] = stock_data['BB_Upper'].fillna(0)
    stock_data['BB_Lower'] = stock_data['BB_Lower'].fillna(0)
    stock_data['BB_Middle'] = stock_data['BB_Middle'].fillna(0)
    stock_data['support'] = stock_data['support'].fillna(0)
    stock_data['resistance'] = stock_data['resistance'].fillna(0)

    print(f"fixed indicators for {stock_data}")
    print(f"columns: {stock_data.columns}")
    return stock_data.select_dtypes(include=['number'])


# Ensure necessary directories exist
os.makedirs('./data/stock_data', exist_ok=True)

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date='2010-01-01', end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    print(f"Fetching data for {ticker} from Yahoo Finance...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}.")
        data.reset_index(inplace=True)
        file_path = f'./data/stock_data/{ticker}_stock_data.csv'
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Main execution function
def run():
    i = 0
    ticker = 'AAPL'  # Example ticker
    symbol = ticker.upper()

    print(f"Starting data training on {symbol}...")
    load_saved = False if i == 0 else True

    # Fetch stock data
    data_path = get_stock_data(symbol)
    if not data_path:
        print(f"Unable to fetch data for {symbol}. Exiting.")
        return

    # Initialize and configure the Informer model
    informer = Informer(
        symbol=symbol,
        batch_size=256,
        seq_len=512,
        label_len=0,
        pred_len=128,
        train_epochs=50,
        d_model=256,
        affin=0,
        patients=50,
        output_size=1,
        target='Close',
        root_path='./data/stock_data/',
        data_path=f'{symbol}_stock_data.csv',
        use_saved_data=True,
        load_saved_model=load_saved,
        add_unique_model_name=False,
        train_now=False  # Set to True to train the Informer model
    )

    # Initialize the RL Environment
    env = TradingEnvironment(data_path=data_path, initial_balance=10000)

    # Initialize the RL Agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Train the RL Agent
    print("Training the RL agent...")
    agent.train(env, episodes=100)

    # Simulate and test the RL strategy
    print("Simulating trading strategy...")
    strategy = TradingStrategy(env, agent)
    strategy.execute_strategy(episodes=10)

    # Optional: Integrate Robinhood for live trading
    robinhood_username = os.getenv("ROBINHOOD_USERNAME")
    robinhood_password = os.getenv("ROBINHOOD_PASSWORD")
    if robinhood_username and robinhood_password:
        trader = RobinhoodTrading(robinhood_username, robinhood_password)
        print("Robinhood trading integration is ready.")
    else:
        print("Robinhood credentials not set. Skipping live trading integration.")

if __name__ == '__main__':
    run()

