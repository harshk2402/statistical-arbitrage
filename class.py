import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

ticker = yf.Ticker("VIX")

price_history = ticker.history(start="2020-01-01", interval="1d")

log_returns = [
    pd.Series(np.log(price_history["Close"])).diff(periods=i) for i in range(1, 10)
]

plt.figure(figsize=(12, 6))
plt.plot(price_history.index, log_returns, label="Log Returns", color="blue")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.title("Log Returns of VIX Over Time")
plt.grid(True)
plt.legend()
plt.show()

mu, sigma = norm.fit(log_returns[0].dropna())

plt.hist(log_returns, bins=50, density=True, alpha=0.6, color="steelblue")

x = np.linspace(log_returns[0].min(), log_returns[0].max(), 200)
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf, "r", linewidth=2)

plt.title(f"Log Returns vs Fitted Normal\nμ={mu:.4f}, σ={sigma:.4f}")
plt.xlabel("Log return")
plt.ylabel("Density")

s = skew(log_returns, nan_policy="omit")
k = kurtosis(log_returns, nan_policy="omit")
print(f"Skewness: {s:.4f}, Kurtosis: {k:.4f}")

plt.show()

q = [1, 0.9, 0.5, 1.5, 2]
