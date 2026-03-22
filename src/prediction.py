import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load processed dataset
data = pd.read_csv("data/processed/processed_data.csv")

# Select features
features = data[['Close','Volume','MA20','RSI','MACD']]

# Normalize data (same scaler used earlier)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# Load LSTM input sequences
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

# Load trained model
model = load_model("models/lstm_model.h5")

# Make predictions
predictions = model.predict(X)

# Convert predictions back to real price
dummy = np.zeros((len(predictions), 5))
dummy[:,0] = predictions[:,0]

predicted_prices = scaler.inverse_transform(dummy)[:,0]

# Actual stock prices
actual_prices = data['Close'][60:].values

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print("RMSE:", rmse)

# Plot results
plt.figure(figsize=(12,6))

plt.plot(actual_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")

plt.title("Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.show()