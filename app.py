import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta

st.title("📈 AI Stock Price Prediction Dashboard")

# Stock selection
stock = st.text_input("Enter Stock Symbol", "AAPL")

# Download data
data = yf.download(stock, start="2015-01-01", end="2024-01-01")

data.columns = data.columns.get_level_values(0)

st.subheader("Raw Stock Data")
st.write(data.tail())

# Add technical indicators
data['MA20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
data['MACD'] = ta.trend.MACD(close=data['Close']).macd()

data.dropna(inplace=True)

features = data[['Close','Volume','MA20','RSI','MACD']]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# Create sequences
sequence_length = 60
X = []

for i in range(sequence_length, len(scaled)):
    X.append(scaled[i-sequence_length:i])

X = np.array(X)

# Load trained model
model = load_model("models/lstm_model.h5")

# Predict
predictions = model.predict(X)

dummy = np.zeros((len(predictions),5))
dummy[:,0] = predictions[:,0]

predicted_prices = scaler.inverse_transform(dummy)[:,0]

actual_prices = data['Close'][60:].values

# Plot
st.subheader("Stock Prediction Graph")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(actual_prices, label="Actual Price")
ax.plot(predicted_prices, label="Predicted Price")

ax.set_xlabel("Time")
ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)