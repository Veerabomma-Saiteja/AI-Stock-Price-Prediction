import pandas as pd
import ta
import os

# Load raw data
file_path = "data/raw/stock_data.csv"

data = pd.read_csv(file_path)

# Convert numeric columns properly
numeric_cols = ['Open','High','Low','Close','Volume']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Moving Average
data['MA20'] = data['Close'].rolling(window=20).mean()

# RSI
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()

# MACD
macd = ta.trend.MACD(close=data['Close'])
data['MACD'] = macd.macd()

# Remove missing rows
data.dropna(inplace=True)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/processed_data.csv"

data.to_csv(output_path, index=False)

print("Technical indicators added successfully!")
print(data.head())