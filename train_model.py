import yfinance as yf
import pandas_ta as ta
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

def train_and_save():
    df = yf.download("BTC-USD", period="2y", interval="1h")
    df.columns = df.columns.get_level_values(0)

    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
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
    df['BBL'] = bbands.iloc[:, 0]
    df['BBM'] = bbands.iloc[:, 1]
    df['BBU'] = bbands.iloc[:, 2]
    df['BB_width'] = df['BBU'] - df['BBL']
    df['EMA_diff'] = df['EMA_12'] - df['EMA_26']
    df['Close_to_SMA'] = df['Close'] / df['SMA_20'] - 1
    df['SMA_diff'] = df['Close'] - df['SMA_20']

    df['Target'] = (
        df['Close'].shift(-1) > df['Close'] * 1.001
    ).astype(int)
    df.dropna(inplace=True)

    FEATURES = [
        'Close', 'SMA_20', 'RSI_14',
        'BBL', 'BBM', 'BBU', 'BB_width',
        'High_Low_Range', 'Volume_Change',
        'EMA_12', 'EMA_26', 'EMA_diff',
        'Momentum', 'Volatility', 'SMA_diff',
        'Close_to_SMA', 'Price_Change',
        'EMA_50', 'MACD', 'ATR', 'OBV', 'VWAP'
    ]

    X = df[FEATURES]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    fill_values = X_train.median()
    X_train = X_train.fillna(fill_values)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        random_state=42,
        eval_metric='logloss'
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', model)
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'aether_model.pkl')
    print("Model trained and saved!")
    return pipeline

if __name__ == "__main__":
    train_and_save()