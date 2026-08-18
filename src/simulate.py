"""
simulate.py
-----------
Generates realistic feature-request streams and runs cache variants through the
same stream, logging hit rate and latency stats for apples-to-apples comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from features import SymbolIndex, compute_features
from io_utils import read_ohlcv
from cache.no_cache import NoCache
from cache.lru_cache import LRUCache
from cache.learned_cache import LearnedAdmissionLRUCache, LearnedCache


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _timestamp_array(values) -> np.ndarray:
    ts = pd.to_datetime(values, utc=True)

    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    return ts.tz_convert(None).to_numpy(dtype="datetime64[ns]")


def _symbol_timestamp_arrays(
    df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    symbols = sorted(str(symbol) for symbol in df["symbol"].unique())

    symbol_timestamps = {
        symbol: _timestamp_array(
            df.loc[df["symbol"].astype(str) == symbol, "timestamp"]
        )
        for symbol in symbols
    }

    return symbols, symbol_timestamps


def generate_request_stream(
    df: pd.DataFrame,
    n_requests: int,
    window: int = 60,
    seed: int = 7,
):
    """
    Zipfian + bursty workload.

    A few symbols dominate the workload and requests cluster around nearby
    timestamps. This creates temporal locality and gives caching an opportunity
    to improve request latency.
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
            idx = int(
                np.clip(
                    anchor_idx + offset,
                    window,
                    len(ts_pool) - 1,
                )
            )

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
    Cyclical workload.

    Each hour of the day has a designated hot symbol. This makes hour_of_day
    useful information for the learned cache policy.
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
        clock_ts = all_timestamps[
            int(rng.integers(window, len(all_timestamps)))
        ]

        hour = pd.Timestamp(clock_ts).hour
        hot_symbol = symbols[hour % n_symbols]

        if rng.random() < p_hot:
            symbol = hot_symbol
        else:
            symbol = str(
                rng.choice(
                    symbols,
                    p=zipf_weights,
                )
            )

        ts_pool = symbol_timestamps[symbol]

        anchor_idx = int(np.searchsorted(ts_pool, clock_ts))
        anchor_idx = int(
            np.clip(
                anchor_idx,
                window,
                len(ts_pool) - 1,
            )
        )

        burst_len = int(rng.integers(1, 8))

        for _ in range(burst_len):
            if i >= n_requests:
                break

            offset = int(rng.integers(-3, 4))
            idx = int(
                np.clip(
                    anchor_idx + offset,
                    window,
                    len(ts_pool) - 1,
                )
            )

            requests.append((symbol, ts_pool[idx]))
            i += 1

    return requests


def run_simulation(
    df: pd.DataFrame,
    requests: Sequence[Tuple[str, object]],
    cache_variants: dict,
    window: int = 60,
):
    """
    Runs the exact same request stream through every cache variant.

    Using the same requests for every policy makes the resulting hit rate,
    latency, and throughput metrics directly comparable.
    """
    index = SymbolIndex(df)
    results = {}

    for name, cache in cache_variants.items():
        cache.reset_stats()

        for symbol, timestamp in requests:
            key = (symbol, timestamp)

            cache.access(
                key,
                lambda s=symbol, t=timestamp: compute_features(
                    index,
                    s,
                    t,
                    window=window,
                ),
            )

        results[name] = cache.stats()

    return results


def default_variants(
    df: pd.DataFrame,
    capacity: int,
    model_path: Path,
    admission_threshold: float = 0.5,
) -> dict:
    """
    Creates the cache policies used in the benchmark.

    The learned variants are included only when a trained model is available.
    """
    variants = {
        "no_cache": NoCache(capacity=capacity),
        "lru": LRUCache(capacity=capacity),
    }

    if model_path.exists():
        variants["learned_cache"] = LearnedCache(
            capacity=capacity,
            model_path=model_path,
            raw_df=df,
        )

        variants["learned_admission_lru"] = LearnedAdmissionLRUCache(
            capacity=capacity,
            model_path=model_path,
            raw_df=df,
            score_threshold=admission_threshold,
        )

    else:
        print(
            f"(No trained model found at {model_path} -- "
            "skipping learned cache variants. Run train.py first.)"
        )

    return variants


def results_to_frame(results: dict) -> pd.DataFrame:
    """Converts cache statistics into a DataFrame for CSV output."""
    rows = []

    for name, stats in results.items():
        rows.append(
            {
                "variant": name,
                **stats,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = read_ohlcv(
        PROCESSED_DIR / "ohlcv_synthetic"
    )

    N_REQUESTS = 20_000
    WINDOW = 60
    CACHE_CAPACITY = 50
    ADMISSION_THRESHOLD = 0.5

    MODEL_PATH = (
        Path(__file__).resolve().parent
        / "learned_policy"
        / "model.pkl"
    )

    print(
        f"Generating {N_REQUESTS:,} requests across "
        f"{df['symbol'].nunique()} symbols "
        "(cyclical workload)..."
    )

    requests = generate_cyclical_request_stream(
        df,
        n_requests=N_REQUESTS,
        window=WINDOW,
        p_hot=0.7,
        seed=99,
    )

    cache_variants = default_variants(
        df=df,
        capacity=CACHE_CAPACITY,
        model_path=MODEL_PATH,
        admission_threshold=ADMISSION_THRESHOLD,
    )

    print("Running simulation...")

    results = run_simulation(
        df=df,
        requests=requests,
        cache_variants=cache_variants,
        window=WINDOW,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = results_to_frame(results)

    for name, stats in results.items():
        print(f"\n{name}:")

        for key, value in stats.items():
            print(f"  {key}: {value}")

    out = RESULTS_DIR / "baseline_metrics.csv"

    frame.to_csv(
        out,
        index=False,
    )

    print(f"\nSaved metrics -> {out}")