import pandas_ta as ta
import yfinance as yf


def add_indicators(df):
    if df.empty:
        return df

    df["SMA_20"] = df.ta.sma(length=20)

    df["RSI_14"] = df.ta.rsi(length=14)

    df = df.dropna()

    return df


def get_live_data(ticker):
    df = yf.download(ticker, period='60d', interval='1h')
    df.columns = df.columns.get_level_values(0)
    df = add_indicators(df)
    return df

