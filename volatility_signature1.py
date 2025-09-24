# Retrieve all the data using Bloomberg, Quandl, Yahoo Finance, Google Finance and WRDS as discussed in class
# Volatility signature:
# using the standard approach (squared root of variance of log returns adjusted for the sampling frequency: we assume 252 trading days per year) to compute volatility

#      1-1) Download intra-day and daily data for foreign currencies, S&P, DJA futures or ETFs  data using  either R, Python, C++, Java etc..
#      1-2) Compute the annualized volatility graph  with a sampling frequency from 1 day returns to 30 days returns
#      1-2-a) Produce the graph at different time periods i.e. 2000 to 2007, and 2007 to 2014

#      1-2-b) compute the mean, the median, the 25% and 75% quantile for the rolling volatility signature (i.e 6months or 1 year window) for each frequency and plot the corresponding graphs
#       1-3) compute the volatility signature using intraday with annualized returns. the sampling frequency could range from seconds or minutes to few hours.
#       1-3-a) compute the mean, the median,  the 25% and 75% quantile for the rolling volatility signature (i.e 6months or 1 year window) for each frequency and plot the corresponding graphs

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

ticker = yf.Ticker("SPY")


def get_volatility_signature(
    is_intraday=False,
    data=pd.DataFrame([]),
) -> list:

    price_history: pd.DataFrame = data if not data.empty else pd.DataFrame()

    if price_history.empty:
        raise ValueError("No data fetched for the given date range and interval.")

    # Calculate log returns for n = 1 to 30 days/mins diff
    log_returns = [
        pd.Series(np.log(price_history["Close"])).diff(periods=i) for i in range(1, 31)
    ]

    # Std of log returns
    sigma = [np.nanstd(lr, ddof=1) for lr in log_returns]

    # Annulized volatility for n = 1 to 30 min diff, we assume 252*3900 = 98000 minutes in a year
    if is_intraday:
        volatility_signature = [
            s * np.sqrt(98280 / k) for k, s in enumerate(sigma, start=1)
        ]
    else:
        # Annulized volatility for n = 1 to 30 days diff, we assume 252 trading days in a year
        volatility_signature = [
            s * np.sqrt(252 / k) for k, s in enumerate(sigma, start=1)
        ]

    return volatility_signature


def plot_volatility_signature(
    volatility_signature, start_date=None, end_date=None, is_intraday=False
):
    plt.plot(range(1, 31), volatility_signature)
    plt.xlabel(f"{'Mins' if is_intraday else 'Days'} Diff (k)")
    plt.ylabel("Annualized Volatility")
    plt.title(
        f"Volatility Signature Plot for S&P 500 from {start_date} to {end_date if end_date else 'Present'}"
    )
    plt.grid(True)
    plt.show()


def plot_rolling_volatility_statistics(
    mean_vs_k, median_vs_k, quantile_25_vs_k, quantile_75_vs_k, is_intraday=False
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
    plt.xlabel(f"{'Mins' if is_intraday else 'Days'} Diff")
    plt.ylabel("Annualized Volatility")
    plt.title(
        f"Rolling {'20 day' if is_intraday else '1-Year'} Volatility Signature Statistics for S&P 500"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def get_intraday_data(intraday_ticker, start_date, end_date, interval="1m"):
    # Split the date range into chunks of 7 days
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_ranges = [
        (start + pd.DateOffset(days=i), min(start + pd.DateOffset(days=i + 7), end))
        for i in range(0, (end - start).days, 7)
    ]
    intraday_data = pd.DataFrame()
    for start_date, end_date in date_ranges:
        temp_data = intraday_ticker.history(
            start=start_date, end=end_date, interval=interval
        )
        intraday_data = pd.concat(
            [intraday_data, temp_data],
            ignore_index=False,
        )
        time.sleep(0.5)

    return intraday_data


price_history = ticker.history(start="2000-01-01", interval="1d")

# Remove timezone info for consistency
if price_history.index.tz is not None:  # type: ignore
    price_history.index = price_history.index.tz_localize(None)  # type: ignore
price_history = price_history.sort_index()

# Intraday data for the last month (yfinance provides ~1 month of intraday data)
intraday_data: pd.DataFrame = get_intraday_data(ticker, "2025-08-26", "2025-09-25")

# Remove timezone info for consistency
if intraday_data.index.tz is not None:  # type: ignore
    intraday_data.index = intraday_data.index.tz_localize(None)  # type: ignore
intraday_data = intraday_data.sort_index()

# Full volatility signature from 2000-01-01 to present
volatility_signature_full = get_volatility_signature(data=price_history)
plot_volatility_signature(volatility_signature_full, "2000-01-01")

# Volatility signature from 2000-01-01 to 2007-12-31
volatility_signature_07 = get_volatility_signature(
    data=price_history.loc["2000-01-01":"2007-12-31"]
)
plot_volatility_signature(volatility_signature_07, "2000-01-01", "2007-12-31")

# Volatility signature from 2007-01-01 to 2014-12-31
volatility_signature_14 = get_volatility_signature(
    data=price_history.loc["2007-01-01":"2014-12-31"]
)
plot_volatility_signature(volatility_signature_14, "2007-01-01", "2014-12-31")

# Rolling 1-year volatility signature from 2000-01-01 to present, calculated monthly
rolling_volatility_signatures = [
    (
        start,
        get_volatility_signature(
            data=price_history.loc[start : (start + pd.DateOffset(years=1))],
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

plot_rolling_volatility_statistics(
    mean_vs_k, median_vs_k, quantile_25_vs_k, quantile_75_vs_k
)

volatility_signature_intraday = get_volatility_signature(
    is_intraday=True, data=intraday_data
)
# The intraday volatility signature plot from 2024-08-26 to 2024-09-24, as we only have ~1 month of data through yfinance
# (Data from before ~30 days ago is not available for intraday data in yfinance)

plot_volatility_signature(
    volatility_signature_intraday, "2025-08-26", "2025-09-24", True
)


# Rolling 20 day volatility signature from 2025-08-26 to 2025-09-24, calculated daily
rolling_intraday_volatility_signatures = [
    (
        start,
        get_volatility_signature(
            is_intraday=True,
            data=intraday_data.loc[start : (start + pd.DateOffset(days=20))],
        ),
    )
    for start in pd.date_range(
        start=pd.Timestamp("2025-08-26"),
        end=pd.Timestamp("2025-09-24") - pd.DateOffset(days=20),
        freq="D",
    )
]  # Contains tuples of (start_date, volatility_signature)

mean_vs_k = []
median_vs_k = []
quantile_25_vs_k = []
quantile_75_vs_k = []

# Aggregating statistics for each k from 1 to 30
for k in range(1, 31):
    k_volatilities = [vs[k - 1] for _, vs in rolling_intraday_volatility_signatures]
    mean_vs_k.append(np.mean(k_volatilities))
    median_vs_k.append(np.median(k_volatilities))
    quantile_25_vs_k.append(np.quantile(k_volatilities, 0.25))
    quantile_75_vs_k.append(np.quantile(k_volatilities, 0.75))

plot_rolling_volatility_statistics(
    mean_vs_k, median_vs_k, quantile_25_vs_k, quantile_75_vs_k, True
)
