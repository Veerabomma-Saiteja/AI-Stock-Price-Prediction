import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load processed data
file_path = "data/processed/processed_data.csv"
data = pd.read_csv(file_path)

# Select features
features = data[['Close','Volume','MA20','RSI','MACD']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(features)

# Sequence length
sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i,0])  # Predict Close price

X = np.array(X)
y = np.array(y)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Save arrays
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("Preprocessed data saved successfully!")