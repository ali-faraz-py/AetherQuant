import pandas_ta as ta
import yfinance as yf
import numpy as np

def add_indicators(df):
    if df.empty:
        return df

    df["SMA_20"] = ta.sma(df['Close'], length=20)
    df["RSI_14"] = ta.rsi(df['Close'], length=14)
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_26'] = ta.ema(df['Close'], length=26)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(10).std()
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Change'] = df['Close'].pct_change()
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['VWAP'] = ta.vwap(df['High'], df['Low'],
                          df['Close'], df['Volume'])

    bbands = ta.bbands(df['Close'], length=20, std=2)
    bbl_col = [c for c in bbands.columns if 'BBL' in c][0]
    bbm_col = [c for c in bbands.columns if 'BBM' in c][0]
    bbu_col = [c for c in bbands.columns if 'BBU' in c][0]
    df['BBL'] = bbands[bbl_col]
    df['BBM'] = bbands[bbm_col]
    df['BBU'] = bbands[bbu_col]
    df['BB_width'] = df['BBU'] - df['BBL']
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
