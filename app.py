import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from engine import get_live_data
import os


st.set_page_config(
    page_title="AetherQuant AI",
    page_icon="🏹",
    layout="wide"
)
st.title("🏹 AetherQuant: AI Trading Dashboard")

@st.cache_resource
def get_model():
    if not os.path.exists('aether_model.pkl'):
        from train_model import train_and_save
        pipeline = train_and_save()
    else:
        pipeline = joblib.load('aether_model.pkl')
    return pipeline

pipeline = get_model()

with st.sidebar:
    st.selectbox(
        "Select Asset",
        ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"],
        key="ticker"
    )
    run_btn = st.button("🚀 Generate Signal",
                         use_container_width=True)
    st.divider()
    st.markdown("#### 📊 Model Info")

    info = {
        "🤖 Model": "XGBoost",
        "🎯 Accuracy": "66.81%",
        "📐 Features": "24",
        "📅 Training": "2 Years"
    }

    for key, value in info.items():
        col1, col2 = st.columns([1.5, 1])
        col1.markdown(f"**{key}**")
        col2.markdown(value)


ticker = st.session_state.ticker

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

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df.index, y=df['RSI_14'],
        name='RSI', line=dict(color='purple')
    ))
    fig2.add_hline(y=70, line_dash="dash",
                   line_color="red",
                   annotation_text="Overbought")
    fig2.add_hline(y=30, line_dash="dash",
                   line_color="green",
                   annotation_text="Oversold")
    fig2.update_layout(title="RSI Indicator")
    st.plotly_chart(fig2, use_container_width=True)

    df['Signal'] = pipeline.predict(df[FEATURES])
    df['Signal_Label'] = df['Signal'].map(
        {1: '🟢 BUY', 0: '🔴 SELL'}
    )


    if '-USD' in ticker or '-' in ticker:
        display_cols = ['Close', 'RSI_14', 
                        'SMA_20', 'Signal_Label']
    else:
        display_cols = ['Close', 'RSI_14', 
                        'SMA_20', 'Volume', 
                        'Signal_Label']

    st.subheader("📋 Recent Signals")
    st.dataframe(
        df[display_cols].tail(10),
        use_container_width=True
    )


st.caption("⚠️ Educational purposes only. Not financial advice!")
