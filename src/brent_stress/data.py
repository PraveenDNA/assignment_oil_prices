"""
Data fetching and preprocessing for Brent crude log-returns.

Ticker: BZ=F on Yahoo Finance, via yfinance.

Log-returns (rather than simple returns) are used throughout because:
  - They are additive over time: r_{0→T} = sum r_t
  - They are better approximated by symmetric distributions on short horizons
  - GARCH models are conventionally estimated on log-returns scaled by 100
    (i.e. percentage log-returns), which keeps parameter magnitudes tractable
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_brent(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Fetch Brent crude (BZ=F) daily close prices from Yahoo Finance.

    Returns a DataFrame with columns:
        Close      : unadjusted daily close price (USD/bbl)
        log_return : 100 * log(Close_t / Close_{t-1})  [percentage log-return]

    Parameters
    ----------
    start : str
        Start date in ISO format.  Must give at least 10 years of history.
    end : str or None
        End date.  Defaults to today.
    """
    raw = yf.download("BZ=F", start=start, end=end, auto_adjust=False, progress=False)

    # yfinance ≥ 0.2.x may return multi-level columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close_col = [c for c in raw.columns if c[0] == "Close"]
        prices = raw[close_col[0]].squeeze()
    else:
        prices = raw["Close"].squeeze()

    prices = prices.dropna()
    prices.name = "Close"
    prices.index = pd.to_datetime(prices.index)

    log_ret = 100.0 * np.log(prices / prices.shift(1))
    log_ret.name = "log_return"

    df = pd.concat([prices, log_ret], axis=1).dropna()
    return df


def load_returns(start: str = "2015-01-01") -> pd.Series:
    """
    Convenience wrapper: return the percentage log-return Series only.

    This is the series fed into GARCH estimation and all diagnostic functions.
    """
    return fetch_brent(start=start)["log_return"]
