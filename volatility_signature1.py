import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ticker = yf.Ticker("^SPX")


def get_volatility_signature(ticker: yf.Ticker, start_date="", end_date="") -> list:
    price_history: pd.DataFrame = ticker.history(start=start_date, end=end_date)

    # Calculate log returns for n = 1 to 30 days diff
    log_returns = [
        pd.Series(np.log(price_history["Close"])).diff(periods=i) for i in range(1, 31)
    ]

    # Std of log returns
    sigma = [np.nanstd(lr, ddof=1) for lr in log_returns]

    # Annulized volatility for n = 1 to 30 days diff
    volatility_signature = [s * np.sqrt(252 / k) for k, s in enumerate(sigma, start=1)]
    return volatility_signature


def plot_volatility_signature(volatility_signature, start_date=None, end_date=None):
    plt.plot(range(1, 31), volatility_signature)
    plt.xlabel("Days Diff")
    plt.ylabel("Annualized Volatility")
    plt.title(
        f"Volatility Signature Plot for S&P 500 from {start_date} to {end_date if end_date else 'Present'}"
    )
    plt.grid(True)
    plt.show()


def plot_rolling_volatility_statistics(
    mean_vs_k, median_vs_k, quantile_25_vs_k, quantile_75_vs_k
):
    k_values = range(1, 31)
    plt.plot(k_values, mean_vs_k, label="Mean", color="blue")
    plt.plot(k_values, median_vs_k, label="Median", color="orange")
    plt.fill_between(
        k_values,
        quantile_25_vs_k,
        quantile_75_vs_k,
        color="gray",
        alpha=0.5,
        label="25th-75th Percentile",
    )
    plt.xlabel("Days Diff (k)")
    plt.ylabel("Annualized Volatility")
    plt.title("Rolling 1-Year Volatility Signature Statistics for S&P 500")
    plt.legend()
    plt.grid(True)
    plt.show()


# Full volatility signature from 2000-01-01 to present
volatility_signature_full = get_volatility_signature(ticker, "2000-01-01")
plot_volatility_signature(volatility_signature_full, "2000-01-01")

# Volatility signature from 2000-01-01 to 2007-12-31
volatility_signature_07 = get_volatility_signature(ticker, "2000-01-01", "2007-12-31")
plot_volatility_signature(volatility_signature_07, "2000-01-01", "2007-12-31")

# Volatility signature from 2007-01-01 to 2014-12-31
volatility_signature_14 = get_volatility_signature(ticker, "2007-01-01", "2014-12-31")
plot_volatility_signature(volatility_signature_14, "2007-01-01", "2014-12-31")

# Rolling 1-year volatility signature from 2000-01-01 to present, calculated monthly
rolling_volatility_signatures = [
    (
        start,
        get_volatility_signature(
            ticker,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=(start + pd.DateOffset(years=1)).strftime("%Y-%m-%d"),
        ),
    )
    for start in pd.date_range(
        start="2000-01-01", end=pd.Timestamp.today() - pd.DateOffset(years=1), freq="MS"
    )
]  # Contains tuples of (start_date, volatility_signature)

mean_vs_k = []
median_vs_k = []
quantile_25_vs_k = []
quantile_75_vs_k = []

# Aggregating statistics for each k from 1 to 30
for k in range(1, 31):
    k_volatilities = [vs[k - 1] for _, vs in rolling_volatility_signatures]
    mean_vs_k.append(np.mean(k_volatilities))
    median_vs_k.append(np.median(k_volatilities))
    quantile_25_vs_k.append(np.quantile(k_volatilities, 0.25))
    quantile_75_vs_k.append(np.quantile(k_volatilities, 0.75))

print(len(mean_vs_k), len(median_vs_k), len(quantile_25_vs_k), len(quantile_75_vs_k))

plot_rolling_volatility_statistics(
    mean_vs_k, median_vs_k, quantile_25_vs_k, quantile_75_vs_k
)
