import pandas_ta as ta
import yfinance as yf


def add_indicators(df):
    if df.empty:
        return df

    df["SMA_20"] = df.ta.sma(length=20)

    df["RSI_14"] = df.ta.rsi(length=14)

    df = df.dropna()

    bbands = ta.bbands(df['Close'], length=20, std=2)

    bbl_col = [c for c in bbands.columns if 'BBL' in c][0]
    bbm_col = [c for c in bbands.columns if 'BBM' in c][0]
    bbu_col = [c for c in bbands.columns if 'BBU' in c][0]

    df['BBL'] = bbands[bbl_col]
    df['BBM'] = bbands[bbm_col]
    df['BBU'] = bbands[bbu_col]

    return df


def get_live_data(ticker):
    df = yf.download(ticker, period='60d', interval='1h')
    df.columns = df.columns.get_level_values(0)
    df = add_indicators(df)
    return df


