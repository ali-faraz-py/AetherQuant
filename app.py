import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from engine import get_live_data

st.set_page_config(
    page_title="AetherQuant AI",
    page_icon="🏹",
    layout="wide"
)
st.title("🏹 AetherQuant: AI Trading Dashboard")

pipeline = joblib.load("aether_model.pkl")

st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.selectbox(
    "Select Asset",
    ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"]
)
run_btn = st.sidebar.button("🚀 Generate Signal")

FEATURES = [
    'Close', 'SMA_20', 'RSI_14',
    'BBL', 'BBM', 'BBU', 'BB_width',
    'High_Low_Range', 'Volume_Change',
    'EMA_12', 'EMA_26', 'EMA_diff',
    'Momentum', 'Volatility', 'SMA_diff',
    'Close_to_SMA', 'Price_Change',
    'EMA_50', 'MACD', 'ATR', 'OBV', 'VWAP'
]

if run_btn:
    with st.spinner("Fetching live data..."):
        df = get_live_data(ticker)

    latest = df.tail(1)
    features = latest[FEATURES]

    prediction = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0]
    buy_prob = probability[1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${latest['Close'].values[0]:,.2f}")
    col2.metric("SMA 20", f"${latest['SMA_20'].values[0]:,.2f}")
    col3.metric("RSI 14", f"{latest['RSI_14'].values[0]:.2f}")
    col4.metric("Buy Confidence", f"{buy_prob:.2%}")

    st.progress(float(buy_prob))

    if prediction == 1:
        st.success(f"🎯 BUY SIGNAL ({buy_prob:.2%} confidence)")
    else:
        st.error(f"⚠️ SELL/AVOID ({1-buy_prob:.2%} confidence)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='Price'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'], name='SMA 20'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BBU'],
        name='Upper Band',
        line=dict(dash='dash', color='red')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BBL'],
        name='Lower Band',
        line=dict(dash='dash', color='green')
    ))
    fig.update_layout(title=f"{ticker} Price Chart")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.tail(10))

st.caption("⚠️ Educational purposes only. Not financial advice!")
