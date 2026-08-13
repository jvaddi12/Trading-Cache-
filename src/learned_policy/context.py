"""
context.py
----------
Fast context-feature helpers for the learned cache.

The model uses rolling volatility and hour-of-day. Rolling volatility is still
computed once with pandas, but online lookup is array-based and returns a fixed
feature vector without building dictionaries or DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _timestamp_array(values) -> np.ndarray:
    ts = pd.to_datetime(values, utc=True)
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    return ts.tz_convert(None).to_numpy(dtype="datetime64[ns]")


def _timestamp_scalar(value) -> np.datetime64:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return np.datetime64(ts.to_datetime64(), "ns")


@dataclass(slots=True)
class ContextData:
    timestamps: np.ndarray
    roll_vol: np.ndarray
    hour_of_day: np.ndarray


def compute_rolling_vol_column(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Adds a 'roll_vol' column: rolling realized volatility (std of log returns) per symbol."""
    df = df.sort_values(["symbol", "timestamp"], kind="mergesort").copy()
    log_ret = np.log(df["close"] / df.groupby("symbol")["close"].shift(1))
    df["log_ret"] = log_ret
    df["roll_vol"] = df.groupby("symbol")["log_ret"].transform(
        lambda s: s.rolling(window, min_periods=2).std()
    )
    df["roll_vol"] = df["roll_vol"].fillna(0.0)
    return df.drop(columns=["log_ret"])


def build_context_index(df_with_vol: pd.DataFrame) -> Dict[str, ContextData]:
    """Builds a per-symbol array lookup for fast binary-search access."""
    index: Dict[str, ContextData] = {}
    for symbol, sub in df_with_vol.groupby("symbol", sort=False):
        sub = sub.sort_values("timestamp", kind="mergesort")
        timestamps = _timestamp_array(sub["timestamp"])
        # Precompute hour once instead of constructing pd.Timestamp in every cache access.
        hours = pd.to_datetime(sub["timestamp"], utc=True).dt.hour.to_numpy(dtype=np.int16)
        index[str(symbol)] = ContextData(
            timestamps=timestamps,
            roll_vol=sub["roll_vol"].to_numpy(dtype=np.float64, copy=True),
            hour_of_day=hours,
        )
    return index


def get_context(context_index: Dict[str, ContextData], symbol: str, timestamp) -> Tuple[float, int]:
    """Returns (roll_vol, hour_of_day) for a (symbol, timestamp) pair via binary search."""
    entry = context_index.get(symbol)
    fallback_hour = pd.Timestamp(timestamp).hour
    if entry is None:
        return 0.0, int(fallback_hour)
    idx = int(np.searchsorted(entry.timestamps, _timestamp_scalar(timestamp), side="right")) - 1
    if idx < 0:
        return 0.0, int(fallback_hour)
    return float(entry.roll_vol[idx]), int(entry.hour_of_day[idx])
