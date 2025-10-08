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
PAIRS = [("IBM", "SPY"), ("DIA", "SPY"), ("AAPL", "GOOG")]
ROLLING_WINDOW = 60
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


def run_pair(primary: str, secondary: str) -> dict:
    if primary not in insample.columns or secondary not in insample.columns:
        raise KeyError("Primary/secondary tickers not present after alignment")

    y = insample[primary]
    x = add_constant(insample[secondary])
    model = OLS(y, x).fit()
    beta = model.params[secondary]
    intercept = model.params["const"]

    spread_in = y - (beta * insample[secondary] + intercept)
    adf_stat, adf_pval, *_ = adfuller(spread_in)

    spread_out = outsample[primary] - (beta * outsample[secondary] + intercept)
    spread = pd.concat([spread_in, spread_out])
    rolling_mean = spread.rolling(ROLLING_WINDOW).mean()
    rolling_std = spread.rolling(ROLLING_WINDOW).std()
    zscore = (spread - rolling_mean) / rolling_std

    spread_lag = spread_in.shift(1).dropna()
    spread_lag.name = "lag"
    spread_ret_in = (spread_in - spread_in.shift(1)).dropna()
    reg = OLS(spread_ret_in, add_constant(spread_lag)).fit()
    phi = reg.params["lag"]
    hl = None
    if phi < 0:
        hl = -np.log(2) / phi

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

    trades_opened = int((signal.diff().abs() == 1).sum())

    spread_ret = spread.diff()
    strategy_ret = (signal.shift(1) * spread_ret).dropna()

    cum_pnl_full = strategy_ret.cumsum().iloc[-1]

    split_date = outsample.index[0]
    oos_ret = strategy_ret.loc[split_date:]

    cum_pnl_oos = oos_ret.cumsum().iloc[-1] if not oos_ret.empty else None

    return {
        "pair": (primary, secondary),
        "beta": beta,
        "intercept": intercept,
        "adf_stat": float(adf_stat),
        "adf_pval": float(adf_pval),
        "half_life": float(hl) if hl is not None else None,
        "trades_opened": trades_opened,
        "cum_pnl_full": cum_pnl_full,
        "cum_pnl_oos": cum_pnl_oos,
    }


print("\nRunning baseline diagnostics and backtests for selected pairs:")
results = []
for primary, secondary in PAIRS:
    try:
        res = run_pair(primary, secondary)
        results.append(res)
        print(
            f"Pair {primary}/{secondary}: beta={res['beta']:.4f}, ADF stat={res['adf_stat']:.3f}, "
            f"p-val={res['adf_pval']:.4f}, half-life={res['half_life']}, trades={res['trades_opened']}, "
            f"cum pnl full={res['cum_pnl_full']:.2f}, cum pnl oos={res['cum_pnl_oos']}"
        )
    except Exception as e:
        print(f"Failed for pair {primary}/{secondary}: {e}")

print("\nSummary table:")
print(
    pd.DataFrame(results)[
        [
            "pair",
            "beta",
            "intercept",
            "adf_stat",
            "adf_pval",
            "half_life",
            "trades_opened",
            "cum_pnl_full",
            "cum_pnl_oos",
        ]
    ]
)
