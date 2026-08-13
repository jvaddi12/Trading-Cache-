"""
features.py
-----------
Fast trading-signal computation for cache simulations.

The original implementation already avoided full-dataframe scans by grouping by
symbol, but each request still sliced pandas DataFrames and built pandas Series.
That overhead dominates at simulation scale. This version stores per-symbol OHLCV
columns as contiguous numpy arrays and computes each feature directly on array
slices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
class _SymbolData:
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class SymbolIndex:
    """
    Pre-groups the raw OHLCV frame by symbol and keeps each symbol's rows sorted
    by timestamp as numpy arrays.

    Hot-path lookup is now:
      1. one binary search into the symbol's timestamp array
      2. simple numpy slices over OHLCV arrays

    That removes pandas object creation from per-request feature computation.
    """

    __slots__ = ("_by_symbol",)

    def __init__(self, df: pd.DataFrame):
        self._by_symbol: Dict[str, _SymbolData] = {}
        needed = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        missing = needed.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        for symbol, sub in df.groupby("symbol", sort=False):
            sub = sub.sort_values("timestamp", kind="mergesort")
            self._by_symbol[str(symbol)] = _SymbolData(
                timestamps=_timestamp_array(sub["timestamp"]),
                open=sub["open"].to_numpy(dtype=np.float64, copy=True),
                high=sub["high"].to_numpy(dtype=np.float64, copy=True),
                low=sub["low"].to_numpy(dtype=np.float64, copy=True),
                close=sub["close"].to_numpy(dtype=np.float64, copy=True),
                volume=sub["volume"].to_numpy(dtype=np.float64, copy=True),
            )

    def _window_arrays(self, symbol: str, timestamp, window: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        entry = self._by_symbol.get(symbol)
        if entry is None:
            return None
        idx = int(np.searchsorted(entry.timestamps, _timestamp_scalar(timestamp), side="right"))
        if idx <= 0:
            return None
        start = max(0, idx - window)
        return (
            entry.high[start:idx],
            entry.low[start:idx],
            entry.close[start:idx],
            entry.volume[start:idx],
        )

    def lookback_window(self, symbol: str, timestamp, window: int) -> pd.DataFrame:
        """
        Compatibility helper for notebooks/debugging. The optimized production
        path calls compute_features(), which avoids constructing this DataFrame.
        """
        entry = self._by_symbol.get(symbol)
        if entry is None:
            return pd.DataFrame()
        idx = int(np.searchsorted(entry.timestamps, _timestamp_scalar(timestamp), side="right"))
        if idx <= 0:
            return pd.DataFrame()
        start = max(0, idx - window)
        return pd.DataFrame(
            {
                "timestamp": entry.timestamps[start:idx],
                "symbol": symbol,
                "open": entry.open[start:idx],
                "high": entry.high[start:idx],
                "low": entry.low[start:idx],
                "close": entry.close[start:idx],
                "volume": entry.volume[start:idx],
            }
        )


def _order_book_imbalance_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> float:
    if close.size == 0:
        return 0.0
    price_range = high - low
    position = np.zeros_like(close, dtype=np.float64)
    valid = price_range != 0.0
    position[valid] = ((close[valid] - low[valid]) / price_range[valid]) * 2.0 - 1.0
    vol_sum = float(volume.sum())
    if vol_sum > 0.0:
        return float(np.dot(position, volume) / vol_sum)
    return float(position.mean())


def _realized_volatility_np(close: np.ndarray) -> float:
    if close.size < 3:
        return 0.0
    log_ret = np.diff(np.log(close))
    if log_ret.size < 2:
        return 0.0
    return float(log_ret.std(ddof=1))


def _vwap_deviation_np(close: np.ndarray, volume: np.ndarray) -> float:
    if close.size == 0:
        return 0.0
    vol_sum = float(volume.sum())
    if vol_sum == 0.0:
        return 0.0
    vwap = float(np.dot(close, volume) / vol_sum)
    if vwap == 0.0:
        return 0.0
    return float((close[-1] - vwap) / vwap)


def _momentum_np(close: np.ndarray, short_window: int = 5) -> float:
    if close.size < short_window + 1:
        return 0.0
    start_price = float(close[-(short_window + 1)])
    if start_price == 0.0:
        return 0.0
    return float((close[-1] - start_price) / start_price)


# Public pandas-based helpers kept for readability/tests/backward compatibility.
def compute_order_book_imbalance(window_df: pd.DataFrame) -> float:
    if window_df.empty:
        return 0.0
    return _order_book_imbalance_np(
        window_df["high"].to_numpy(dtype=np.float64),
        window_df["low"].to_numpy(dtype=np.float64),
        window_df["close"].to_numpy(dtype=np.float64),
        window_df["volume"].to_numpy(dtype=np.float64),
    )


def compute_realized_volatility(window_df: pd.DataFrame) -> float:
    if len(window_df) < 3:
        return 0.0
    return _realized_volatility_np(window_df["close"].to_numpy(dtype=np.float64))


def compute_vwap_deviation(window_df: pd.DataFrame) -> float:
    if window_df.empty:
        return 0.0
    return _vwap_deviation_np(
        window_df["close"].to_numpy(dtype=np.float64),
        window_df["volume"].to_numpy(dtype=np.float64),
    )


def compute_momentum(window_df: pd.DataFrame, short_window: int = 5) -> float:
    if len(window_df) < short_window + 1:
        return 0.0
    return _momentum_np(window_df["close"].to_numpy(dtype=np.float64), short_window=short_window)


def compute_features(index: SymbolIndex, symbol: str, timestamp, window: int = 60) -> dict:
    """
    Returns features for one (symbol, timestamp) request using only numpy arrays
    in the hot path.
    """
    arrays = index._window_arrays(symbol, timestamp, window)
    if arrays is None:
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "order_book_imbalance": 0.0,
            "realized_volatility": 0.0,
            "vwap_deviation": 0.0,
            "momentum": 0.0,
        }

    high, low, close, volume = arrays
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "order_book_imbalance": _order_book_imbalance_np(high, low, close, volume),
        "realized_volatility": _realized_volatility_np(close),
        "vwap_deviation": _vwap_deviation_np(close, volume),
        "momentum": _momentum_np(close),
    }


def benchmark_feature_cost(index: SymbolIndex, symbol: str, all_timestamps, n_calls: int = 200, window: int = 60):
    """Times one compute_features call for README/results writeups."""
    rng = np.random.default_rng(1)
    replace = len(all_timestamps) < n_calls
    sample_ts = rng.choice(all_timestamps, size=n_calls, replace=replace)
    start = time.perf_counter()
    for ts in sample_ts:
        compute_features(index, symbol, ts, window=window)
    elapsed = time.perf_counter() - start
    per_call_ms = (elapsed / n_calls) * 1000
    print(f"Avg compute_features cost over {n_calls} calls: {per_call_ms:.4f} ms/call")
    return per_call_ms


if __name__ == "__main__":
    from pathlib import Path
    from io_utils import read_ohlcv

    processed = Path(__file__).resolve().parent.parent / "data" / "processed" / "ohlcv_synthetic"
    df = read_ohlcv(processed)
    index = SymbolIndex(df)

    sample_symbol = df["symbol"].iloc[0]
    sym_timestamps = df[df["symbol"] == sample_symbol]["timestamp"].values
    sample_ts = sym_timestamps[1000]

    feats = compute_features(index, sample_symbol, sample_ts)
    print("Sample features:", feats)

    benchmark_feature_cost(index, sample_symbol, sym_timestamps, n_calls=500)
