import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Price Prediction Dashboard")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Settings")
stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL")

# ---------- LOADING ANIMATION ----------
loading_container = st.empty()

with loading_container.container():

    st.subheader("📡 Loading Market Data...")
    progress = st.progress(0)
    chart_placeholder = st.empty()

    dummy_data = [100]

    for i in range(50):
        # smooth small fluctuation (-1 to +1)
        change = np.random.uniform(-1, 1)
        new_value = dummy_data[-1] + change
        dummy_data.append(new_value)

        fig, ax = plt.subplots()

        # dark theme
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')

        # clean bold zig-zag (not too thick)
        for j in range(1, len(dummy_data)):
            color = "green" if dummy_data[j] >= dummy_data[j-1] else "red"
            ax.plot(
                [j-1, j],
                [dummy_data[j-1], dummy_data[j]],
                color=color,
                linewidth=1.8
            )

        # clean UI
        ax.set_title("📈 Live Market Fluctuations...", color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        chart_placeholder.pyplot(fig)
        progress.progress((i + 1) * 2)

        time.sleep(0.03)

    st.success("✅ Data Loaded Successfully!")

# ---------- REMOVE LOADING ----------
time.sleep(0.5)
loading_container.empty()

# ---------- DOWNLOAD REAL DATA ----------
data = yf.download(stock, start="2015-01-01", end="2024-01-01")

if data.empty:
    st.error("❌ Invalid stock symbol or no data found!")
    st.stop()

data.columns = data.columns.get_level_values(0)

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)

col1.metric("📊 Latest Price", f"${data['Close'].iloc[-1]:.2f}")
col2.metric("📈 Highest", f"${data['High'].max():.2f}")
col3.metric("📉 Lowest", f"${data['Low'].min():.2f}")

# ---------- RAW DATA ----------
with st.expander("📄 Show Raw Data"):
    st.write(data.tail())

# ---------- TECHNICAL INDICATORS ----------
data['MA20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
data['MACD'] = ta.trend.MACD(close=data['Close']).macd()

data.dropna(inplace=True)

features = data[['Close','Volume','MA20','RSI','MACD']]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# ---------- CREATE SEQUENCES ----------
sequence_length = 60
X = []

for i in range(sequence_length, len(scaled)):
    X.append(scaled[i-sequence_length:i])

X = np.array(X)

# ---------- LOAD MODEL ----------
try:
    model = load_model("models/lstm_model.h5")

    predictions = model.predict(X)

    dummy = np.zeros((len(predictions),5))
    dummy[:,0] = predictions[:,0]

    predicted_prices = scaler.inverse_transform(dummy)[:,0]

except:
    st.warning("⚠️ Model not found. Using dummy prediction.")
    predicted_prices = data['Close'][60:].values * 1.01

actual_prices = data['Close'][60:].values

# ---------- PLOT ----------
st.subheader("📊 Stock Prediction Graph")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(actual_prices, label="Actual Price", color="blue")
ax.plot(predicted_prices, label="Predicted Price", color="orange")

ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)