"""
simulate.py
-----------
Generates realistic feature-request streams and runs cache variants through the
same stream, logging hit rate and latency stats for apples-to-apples comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from features import SymbolIndex, compute_features
from io_utils import read_ohlcv
from cache.no_cache import NoCache
from cache.lru_cache import LRUCache
from cache.learned_cache import LearnedCache

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _timestamp_array(values) -> np.ndarray:
    ts = pd.to_datetime(values, utc=True)
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    return ts.tz_convert(None).to_numpy(dtype="datetime64[ns]")


def _symbol_timestamp_arrays(df: pd.DataFrame) -> Tuple[List[str], Dict[str, np.ndarray]]:
    symbols = sorted(str(s) for s in df["symbol"].unique())
    symbol_timestamps = {
        s: _timestamp_array(df.loc[df["symbol"].astype(str) == s, "timestamp"])
        for s in symbols
    }
    return symbols, symbol_timestamps


def generate_request_stream(df: pd.DataFrame, n_requests: int, window: int = 60, seed: int = 7):
    """
    Zipfian + bursty workload: a few symbols dominate and requests cluster around
    nearby timestamps. This favors caching via temporal locality.
    """
    rng = np.random.default_rng(seed)
    symbols, symbol_timestamps = _symbol_timestamp_arrays(df)
    n_symbols = len(symbols)

    ranks = np.arange(1, n_symbols + 1)
    weights = 1.0 / ranks
    weights /= weights.sum()

    requests = []
    i = 0
    while i < n_requests:
        symbol = str(rng.choice(symbols, p=weights))
        ts_pool = symbol_timestamps[symbol]
        burst_len = int(rng.integers(1, 8))
        anchor_idx = int(rng.integers(window, len(ts_pool)))
        for _ in range(burst_len):
            if i >= n_requests:
                break
            offset = int(rng.integers(-3, 4))
            idx = int(np.clip(anchor_idx + offset, window, len(ts_pool) - 1))
            requests.append((symbol, ts_pool[idx]))
            i += 1
    return requests


def generate_cyclical_request_stream(
    df: pd.DataFrame,
    n_requests: int,
    window: int = 60,
    p_hot: float = 0.7,
    seed: int = 7,
):
    """
    Cyclical workload: each hour of day has a designated hot symbol, so
    hour_of_day is a real signal for the learned policy.
    """
    rng = np.random.default_rng(seed)
    symbols, symbol_timestamps = _symbol_timestamp_arrays(df)
    n_symbols = len(symbols)

    ranks = np.arange(1, n_symbols + 1)
    zipf_weights = 1.0 / ranks
    zipf_weights /= zipf_weights.sum()
    all_timestamps = np.unique(_timestamp_array(df["timestamp"]))

    requests = []
    i = 0
    while i < n_requests:
        clock_ts = all_timestamps[int(rng.integers(window, len(all_timestamps)))]
        hour = pd.Timestamp(clock_ts).hour
        hot_symbol = symbols[hour % n_symbols]

        symbol = hot_symbol if rng.random() < p_hot else str(rng.choice(symbols, p=zipf_weights))
        ts_pool = symbol_timestamps[symbol]
        anchor_idx = int(np.searchsorted(ts_pool, clock_ts))
        anchor_idx = int(np.clip(anchor_idx, window, len(ts_pool) - 1))

        burst_len = int(rng.integers(1, 8))
        for _ in range(burst_len):
            if i >= n_requests:
                break
            offset = int(rng.integers(-3, 4))
            idx = int(np.clip(anchor_idx + offset, window, len(ts_pool) - 1))
            requests.append((symbol, ts_pool[idx]))
            i += 1
    return requests


def run_simulation(df: pd.DataFrame, requests: Sequence[Tuple[str, object]], cache_variants: dict, window: int = 60):
    """Runs identical requests through each cache variant."""
    index = SymbolIndex(df)
    results = {}

    for name, cache in cache_variants.items():
        cache.reset_stats()
        for symbol, timestamp in requests:
            key = (symbol, timestamp)
            cache.access(key, lambda s=symbol, t=timestamp: compute_features(index, s, t, window=window))
        results[name] = cache.stats()

    return results


def default_variants(df: pd.DataFrame, capacity: int, model_path: Path) -> dict:
    variants = {
        "no_cache": NoCache(capacity=capacity),
        "lru": LRUCache(capacity=capacity),
    }
    if model_path.exists():
        variants["learned_cache"] = LearnedCache(capacity=capacity, model_path=model_path, raw_df=df)
    else:
        print(f"(No trained model found at {model_path} -- skipping learned_cache. Run train.py first.)")
    return variants


def results_to_frame(results: dict) -> pd.DataFrame:
    rows = []
    for name, stats in results.items():
        rows.append({"variant": name, **stats})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = read_ohlcv(PROCESSED_DIR / "ohlcv_synthetic")

    N_REQUESTS = 20_000
    WINDOW = 60
    CACHE_CAPACITY = 50
    MODEL_PATH = Path(__file__).resolve().parent / "learned_policy" / "model.pkl"

    print(f"Generating {N_REQUESTS:,} requests across {df['symbol'].nunique()} symbols (cyclical workload)...")
    requests = generate_cyclical_request_stream(df, n_requests=N_REQUESTS, window=WINDOW, p_hot=0.7, seed=99)

    print("Running simulation...")
    results = run_simulation(df, requests, default_variants(df, CACHE_CAPACITY, MODEL_PATH), window=WINDOW)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = results_to_frame(results)
    for name, stats in results.items():
        print(f"\n{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    out = RESULTS_DIR / "baseline_metrics.csv"
    frame.to_csv(out, index=False)
    print(f"\nSaved metrics -> {out}")
