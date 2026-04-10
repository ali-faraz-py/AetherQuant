import streamlit as st
import joblib
import pandas as pd
from engine import get_live_data

st.set_page_config(page_title="AetherQuant AI", layout="wide")
st.title("🏹 AetherQuant: AI Trading Dashboard")


model = joblib.load("aether_model.pkl")

ticker = st.sidebar.text_input("Enter Ticker (e.g., BTC-USD)", value="BTC-USD")
run_btn = st.sidebar.button("Generate Signal")

if run_btn:
    df = get_live_data(ticker)
    latest_data = df.tail(1)

    features = latest_data[["Close", "SMA_20", "RSI_14"]]
    prediction = model.predict(features)[0]

    st.subheader(f"Analysis for {ticker}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${latest_data['Close'].values[0]:,.2f}")
    col2.metric("SMA 20", f"${latest_data['SMA_20'].values[0]:,.2f}")
    col3.metric("RSI 14", f"{latest_data['RSI_14'].values[0]:.2f}")

    if prediction == 1:
        st.success("🎯 AI SIGNAL: BUY")
    else:
        st.error("⚠️ AI SIGNAL: SELL / AVOID")
            
        st.dataframe(df.tail(10))