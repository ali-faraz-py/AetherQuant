import yfinance as yf
import numpy as np
import pandas as pd

def add_indicators(df):
    if df.empty:
        return df

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['EMA_12'] - df['EMA_26']

    df['BBM'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BBU'] = df['BBM'] + (2 * std)
    df['BBL'] = df['BBM'] - (2 * std)
    df['BB_width'] = df['BBU'] - df['BBL']

    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(10).std()
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Change'] = df['Close'].pct_change()
    df['EMA_diff'] = df['EMA_12'] - df['EMA_26']
    df['Close_to_SMA'] = df['Close'] / df['SMA_20'] - 1
    df['SMA_diff'] = df['Close'] - df['SMA_20']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

def get_live_data(ticker):
    df = yf.download(ticker, period='60d', interval='1h')
    df.columns = df.columns.get_level_values(0)
    df = add_indicators(df)
    return df