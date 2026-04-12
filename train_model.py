from sklearn.metrics import accuracy_score
import yfinance as yf
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

def train_and_save():
    df = yf.download("BTC-USD", period="730d", interval="1h")
    df.columns = df.columns.get_level_values(0)

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain/loss))

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

    df['Target'] = (
        df['Close'].shift(-1) > df['Close'] * 1.002
    ).astype(int)
    df.dropna(inplace=True)

    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    FEATURES = [
        'Close', 'SMA_20', 'RSI_14',
        'BBL', 'BBM', 'BBU', 'BB_width',
        'High_Low_Range', 'Volume_Change',
        'EMA_12', 'EMA_26', 'EMA_diff',
        'Momentum', 'Volatility', 'SMA_diff',
        'Close_to_SMA', 'Price_Change',
        'EMA_50', 'MACD', 'ATR', 'OBV', 'VWAP',
        'Hour', 'DayOfWeek'
    ]

    X = df[FEATURES]
    y = df['Target']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=1,
        random_state=42,
        eval_metric='logloss'
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', model)
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'aether_model.pkl')
    print("Model saved!")
    return pipeline

if __name__ == "__main__":
    train_and_save()