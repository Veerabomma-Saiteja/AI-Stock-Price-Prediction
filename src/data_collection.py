import yfinance as yf
import os

stock_symbol = "AAPL"

data = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")

data.reset_index(inplace=True)

os.makedirs("data/raw", exist_ok=True)

data.to_csv("data/raw/stock_data.csv", index=False)

print("Stock data downloaded successfully!")
print(data.head())