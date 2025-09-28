import pandas as pd
import yfinance as yf
import time
import numpy as np


# Get 5 min intraday data in chunks to avoid yfinance limitations
def get_intraday_data(
    intraday_ticker, start: pd.Timestamp, end: pd.Timestamp, interval="1m"
):
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


ticker = yf.Ticker("SPY")
N = 1  # Notional

# To comply with yfinance limitations, we fetch 5 min data in chunks of 7 days up to a max of 60 days
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(days=60)

intraday_data_5min = get_intraday_data(ticker, start_date, end_date, interval="5m")

daily_data = ticker.history(start=start_date, end=end_date, interval="1d")

if intraday_data_5min.index.tz is not None:  # type: ignore
    intraday_data_5min.index = intraday_data_5min.index.tz_localize(None)  # type: ignore
if daily_data.index.tz is not None:  # type: ignore
    daily_data.index = daily_data.index.tz_localize(None)  # type: ignore

intraday_data_5min["log_ret_5m"] = pd.Series(np.log(intraday_data_5min["Close"])).diff()
daily_data["log_ret_1d"] = pd.Series(np.log(daily_data["Close"])).diff()

intraday_data_5min["date"] = pd.to_datetime(intraday_data_5min.index.date)  # type: ignore
daily_data["date"] = pd.to_datetime(daily_data.index.date)  # type: ignore

# Realized variance from 5min data
RV_1d = intraday_data_5min.groupby("date")["log_ret_5m"].apply(lambda x: np.sum(x**2))
RV_1d.index = pd.to_datetime(RV_1d.index)

# Weekly realized variance from daily data
RV_1w = RV_1d.rolling(window=5).mean()

panel = daily_data[["Close", "log_ret_1d", "date"]].copy()
panel = panel.merge(RV_1d.rename("RV_1d"), left_on="date", right_index=True, how="left")
panel["RV_1w"] = panel["RV_1d"].rolling(window=5, min_periods=3).mean()

panel["vol_signal_raw"] = pd.Series(np.sign(panel["RV_1d"] - panel["RV_1w"])).fillna(
    0.0
)
# Shifting vol_signal to avoid lookahead bias
panel["vol_signal"] = panel["vol_signal_raw"].shift(1).fillna(0.0)

panel = panel.sort_index()

# Making daily and weekly anchors
panel["S_D_prev"] = panel["Close"].shift(1)

panel["week"] = panel.index.to_period("W-FRI")  # type: ignore
weekly_close = panel.groupby("week")["Close"].last()
panel["S_W_prev"] = panel["week"].shift(1).map(weekly_close).ffill()

panel["position"] = panel["vol_signal"] * (
    N / panel["S_D_prev"] - N / panel["S_W_prev"]
)

panel["dS"] = panel["Close"] - panel["Close"].shift(1)
panel["pnl_gross"] = panel["position"] * panel["dS"]
panel["ret"] = panel["pnl_gross"] / N

daily_ret = panel["ret"].dropna()
ann_factor = 252

hit_rate = (daily_ret > 0).mean()

# Expected gain/loss expressed as dollars per single dollar invested
expected_gain_per_hit = daily_ret[daily_ret > 0].mean()  # per $1 invested
expected_loss_per_loss = daily_ret[daily_ret < 0].mean()  # per $1 invested

print(f"Hit rate: {hit_rate:.2%}")
print(f"Expected gain per hit (per $1 invested): ${expected_gain_per_hit:.6f}")
print(f"Expected loss per loss (per $1 invested): ${expected_loss_per_loss:.6f}")

# It's important here to note that one of the reasons the hit rate and expected gain/loss per hit is low is due to the fact that we only have access to
# about 2 months of 5-minute intraday data. With a longer history, we would be able to better estimate the weekly realized variance and thus improve the strategy's performance.
