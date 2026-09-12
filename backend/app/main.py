from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
from .engine import get_live_data

model = joblib.load("app/aether_model.pkl")

app = FastAPI(title="AetherQuant Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://aether-quant.vercel.app"],
    allow_origin_regex=r"https://aether-quant-.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "Close", "SMA_20", "RSI_14", "BBL", "BBM", "BBU", "BB_width",
    "High_Low_Range", "Volume_Change", "EMA_12", "EMA_26", "EMA_diff",
    "Momentum", "Volatility", "SMA_diff", "Close_to_SMA", "Price_Change",
    "EMA_50", "MACD", "ATR", "OBV", "VWAP", "Hour", "DayOfWeek",
]

ALLOWED_TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"]


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/predict")
def predict(ticker: str = Query(default="BTC-USD")):
    if ticker not in ALLOWED_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker: {ticker}")

    try:
        df = get_live_data(ticker)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not fetch market data: {e}")

    if df.empty or len(df) < 2:
        raise HTTPException(status_code=503, detail="Not enough market data returned")

    latest = df.tail(1)
    features = latest[FEATURES]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    buy_confidence = float(probability[1])

    history = df.tail(60)[["Close", "SMA_20", "BBU", "BBL", "RSI_14"]].copy()
    history.index = history.index.astype(str)
    history_records = history.reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records")

    return {
        "ticker": ticker,
        "reliable": ticker == "BTC-USD",
        "as_of": str(latest.index[0]),
        "signal": "BUY" if prediction == 1 else "SELL",
        "confidence": round(buy_confidence if prediction == 1 else 1 - buy_confidence, 4),
        "price": float(latest["Close"].values[0]),
        "sma_20": float(latest["SMA_20"].values[0]),
        "rsi_14": float(latest["RSI_14"].values[0]),
        "history": history_records,
    }