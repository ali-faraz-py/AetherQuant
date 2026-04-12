# 🏹 AetherQuant AI: Advanced Crypto Predictive Engine

A professional, machine learning-powered predictive dashboard built with **Python** and **Streamlit**. This tool utilizes an **XGBoost Classifier** and **24 technical indicators** to predict Bitcoin (BTC-USD) price movements with high-precision temporal awareness.

---

## 🚀 Live Demo
**[Click here to try the Live App](https://aether-quant.streamlit.app/)**

---

## 📺 Demo Preview
**[Diabetes Detector Demo](assets/AetherQuant.gif)**

---

## ✨ Features
* **AI-Powered Signals:** Uses an XGBoost pipeline to classify market trends into "BUY" or "SELL" signals.
* **Feature Engineering:** Real-time calculation of 24 indicators including RSI, MACD, Bollinger Bands, and VWAP.
* **Temporal Awareness:** Incorporates time-series features (Hour, Day of Week) to capture cyclical market patterns.
* **Interactive Visualizations:** High-fidelity price charts and indicator overlays powered by `Plotly`.
* **Live Market Data:** Direct integration with the Yahoo Finance API (`yfinance`) for up-to-the-minute accuracy.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Framework:** Streamlit (Web UI)
* **Machine Learning:** Scikit-learn & XGBoost
* **Data Handling:** Pandas & NumPy
* **Visualization:** Plotly Graph Objects
* **Deployment:** Streamlit Community Cloud

## 🚀 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ali-faraz-py/AetherQuant](https://github.com/ali-faraz-py/AetherQuant)
   cd AetherQuant

2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt

3. **Run the application:**
   ```bash
    streamlit run app.py

## 📂 Project Structure

```text
AetherQuant/
├── app.py              # Streamlit Web Application and UI logic
├── engine.py           # Technical indicator and data processing engine
├── train_model.py      # Model training, feature engineering, and validation
├── aether_model.pkl    # Pre-trained XGBoost Pipeline (24 features)
├── requirements.txt    # Project dependencies
├── .gitignore          # Prevents tracking of temporary files
└── .gitattributes      # LFS tracking for the model file
```

## 🧠 Model Insights
The model is trained on 730 days of hourly data and currently achieves a **66.81% accuracy rate** on unseen test sets.

The engine analyzes 24 unique dimensions including **Trend, Volatility, Momentum, and Volume-Weighted indicators** to minimize false signals in volatile crypto markets.

---

### 👤 Author
**Syed Ali Faraz** - [GitHub Profile](https://github.com/ali-faraz-py)

*If you found this tool insightful, please give the repository a ⭐!*