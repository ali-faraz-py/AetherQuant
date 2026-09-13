# 🏹 AetherQuant AI: Advanced Crypto Predictive Engine

A machine learning-powered market signal dashboard that classifies short-term price direction using live technical indicators. Built with a **FastAPI** backend and a **Next.js** frontend. Originally a Streamlit app, rebuilt into a full separate backend/frontend architecture.

---

## 🚀 Live Demo
**[Try the live app](https://aether-quant-blush.vercel.app/)**

---

## ✨ Features
* **AI-powered signals** — an **XGBoost Classifier** trained on 2 years of hourly BTC-USD data classifies market trend as BUY or SELL.
* **Live market data** — every request fetches fresh data directly from Yahoo Finance (`yfinance`) and computes all 24 indicators in real time.
* **Real price & RSI charts** — interactive line charts (price with SMA/Bollinger Bands, and RSI with overbought/oversold reference lines) built from actual live indicator history.
* **Multi-asset support** — BTC-USD, ETH-USD, AAPL, GOOGL, TSLA. Since the model is trained only on BTC-USD, predictions for other assets are clearly flagged as less reliable.
* **Dark mode** — toggle between a soft light "market brief" theme and a dark variant, persisted across visits.

## 🛠️ Tech Stack
* **Backend:** FastAPI, XGBoost, scikit-learn, pandas, yfinance, deployed on **Render**
* **Frontend:** Next.js (App Router, JavaScript, Tailwind CSS), Recharts, deployed on **Vercel**
* **Model:** XGBoost Classifier, ~66.81% accuracy, trained on 730 days of hourly BTC-USD data, 24 engineered features

## 🚀 Installation & Local Setup

### Backend

    cd backend
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Runs at `http://127.0.0.1:8000`.

### Frontend

    cd frontend
    npm install
    npm run dev

Runs at `http://localhost:3000`. Requires a `.env.local` file containing:

    NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

## 📂 Project Structure

    AetherQuant/
    ├── backend/
    │   ├── app/
    │   │   ├── main.py          # FastAPI app: loads model, fetches live data, exposes /predict
    │   │   ├── engine.py        # Feature engineering: computes all 24 indicators
    │   │   └── aether_model.pkl # Pre-trained XGBoost pipeline
    │   └── requirements.txt
    ├── frontend/
    │   ├── app/
    │   │   ├── layout.js        # Fonts, metadata
    │   │   ├── page.js          # Main UI: ticker selector, signal card, price/RSI charts
    │   │   └── globals.css      # Design tokens (light + dark theme variables)
    │   └── package.json
    ├── NoteBook/
    │   └── explore.ipynb        # Data exploration
    ├── assets/
    │   └── AetherQuant.gif
    ├── engine.py                # Root copy, used by train_model.py for reproducibility
    └── train_model.py           # Model training script

## 🧠 Model Insights
The model is trained on 730 days of hourly BTC-USD data and achieves **~66.81% accuracy** on unseen test data.

It analyzes 24 features spanning **trend, volatility, momentum, and volume-weighted indicators** (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, plus time-of-day/day-of-week) to classify whether price is likely to rise over the next hour.

**Important caveat:** the model is trained exclusively on BTC-USD. Predictions for other assets (ETH-USD, AAPL, GOOGL, TSLA) use the same model but on data it was never trained on — the app flags these as less reliable rather than presenting them with equal confidence.

---

### 👤 Author
**Syed Ali Faraz** — [GitHub Profile](https://github.com/ali-faraz-py)

*If you found this tool insightful, please give the repository a ⭐!*