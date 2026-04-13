from sklearn.metrics import accuracy_score
import yfinance as yf
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from engine import add_indicators

def train_and_save():
    df = yf.download("BTC-USD", period="730d", interval="1h")
    df.columns = df.columns.get_level_values(0)

    df = add_indicators(df)

    df['Target'] = (df['Close'].shift(-1) > df['Close'] * 1.002).astype(int)
    df.dropna(inplace=True)

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