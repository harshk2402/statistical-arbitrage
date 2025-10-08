"""Data preparation and initial cointegration diagnostics for the pairs trading assignment."""

from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller

TICKERS = ["AAPL", "GOOG", "IBM", "SPY", "DIA"]
YEARS = 10
INSAMPLE_RATIO = 0.8
PRIMARY, SECONDARY = "IBM", "SPY"
ROLLING_WINDOW = 60  # ~3 months of trading days
ENTRY_Z = 2.0
EXIT_Z = 0.5

end_date = date.today()
start_date = end_date - timedelta(days=365 * YEARS)

market_data: Dict[str, pd.DataFrame] = {}

for ticker in TICKERS:
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=False,
        progress=False,
        actions=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker} between {start_date} and {end_date}")

    df.index.name = "Date"
    null_rows = df.isna().any(axis=1).sum()
    if null_rows:
        print(f"Warning: {ticker} has {null_rows} rows with missing values")

    market_data[ticker] = df
    print(
        f"Loaded {ticker}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}"
    )

adj_close = pd.concat(
    [market_data[t]["Adj Close"] for t in TICKERS],
    axis=1,
    keys=TICKERS,
).dropna()
# Flatten MultiIndex columns created by concat(keys=...) so each column is just the ticker
if isinstance(adj_close.columns, pd.MultiIndex):
    adj_close.columns = adj_close.columns.get_level_values(0)

print(
    f"Aligned adjusted closes: {adj_close.shape[0]} shared dates "
    f"from {adj_close.index[0].date()} to {adj_close.index[-1].date()}"
)

split_idx = int(len(adj_close) * INSAMPLE_RATIO)
insample = adj_close.iloc[:split_idx]
outsample = adj_close.iloc[split_idx:]

print(
    f"In-sample (80%): {insample.shape[0]} rows, {insample.index[0].date()} -> {insample.index[-1].date()}"
)
print(
    f"Out-of-sample (20%): {outsample.shape[0]} rows, {outsample.index[0].date()} -> {outsample.index[-1].date()}"
)

# Engle–Granger step for the primary pair (AAPL vs GOOG)
if PRIMARY not in insample.columns or SECONDARY not in insample.columns:
    raise KeyError("Primary/secondary tickers not present after alignment")

y = insample[PRIMARY]
x = add_constant(insample[SECONDARY])
model = OLS(y, x).fit()
hedge_ratio = model.params[SECONDARY]
intercept = model.params["const"]

spread_in = y - (hedge_ratio * insample[SECONDARY] + intercept)
adf_stat, adf_pval, *_ = adfuller(spread_in)

print(
    f"OLS hedge ratio {PRIMARY}/{SECONDARY}: beta={hedge_ratio:.4f}, intercept={intercept:.4f}"
)
print(f"ADF test on in-sample spread: statistic={adf_stat:.3f}, p-value={adf_pval:.4f}")

# Construct spread and rolling z-scores for both in- and out-of-sample periods
spread_out = outsample[PRIMARY] - (hedge_ratio * outsample[SECONDARY] + intercept)
spread = pd.concat([spread_in, spread_out])
rolling_mean = spread.rolling(ROLLING_WINDOW).mean()
rolling_std = spread.rolling(ROLLING_WINDOW).std()
zscore = (spread - rolling_mean) / rolling_std

print(
    f"Rolling z-score (window={ROLLING_WINDOW}) available from {zscore.dropna().index[0].date()} onward"
)

# Half-life estimation using the in-sample spread
spread_lag = spread_in.shift(1).dropna()
spread_lag.name = "lag"
spread_ret_in = (spread_in - spread_in.shift(1)).dropna()
reg = OLS(spread_ret_in, add_constant(spread_lag)).fit()
phi = reg.params["lag"]
if phi >= 0:
    print(
        "Warning: estimated mean-reversion speed is non-negative; half-life undefined"
    )
else:
    half_life = -np.log(2) / phi
    print(f"Estimated half-life: {half_life:.2f} days")

# Generate trading signals based on z-score thresholds
signal = pd.Series(0, index=zscore.index, dtype="int8")
position = 0

for idx, z_val in zip(zscore.index, zscore.to_numpy()):
    z_val = float(z_val) if not np.isnan(z_val) else np.nan
    if np.isnan(z_val):
        signal.at[idx] = position
        continue

    if position == 0:
        if z_val > ENTRY_Z:
            position = -1
        elif z_val < -ENTRY_Z:
            position = 1
    elif abs(z_val) < EXIT_Z:
        position = 0

    signal.at[idx] = position

print(
    "Signal counts (full sample): "
    + ", ".join(
        f"{state}:{count}"
        for state, count in signal.value_counts().sort_index().items()
    )
)

# Strategy returns in spread units (no costs)
spread_ret = spread.diff()
strategy_ret = (signal.shift(1) * spread_ret).dropna()

cum_pnl = strategy_ret.cumsum()
print(f"Full-sample cumulative spread PnL: {cum_pnl.iloc[-1]:.2f} (spread units)")

# Out-of-sample slice diagnostics
split_date = outsample.index[0]
oos_signal = signal.loc[split_date:]
oos_ret = strategy_ret.loc[split_date:]

if not oos_ret.empty:
    oos_cum = oos_ret.cumsum()
    print(f"Out-of-sample cumulative spread PnL: {oos_cum.iloc[-1]:.2f} (spread units)")
    print(f"Out-of-sample trading days active: {int((oos_signal != 0).sum())}")
else:
    print("No out-of-sample returns available; check signal alignment.")

# Trade count summary
trade_count = int((signal.diff().abs() == 1).sum())
print(f"Trades opened (full sample): {trade_count}")
