"""
benchmark.py
------------
Repeatable benchmark harness for the optimized trading feature cache.

It measures:
  - raw compute_features() cost
  - no-cache / LRU / learned-cache hit rate
  - avg, p50, p95, p99 latency
  - throughput and learned-model scoring counters
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from features import SymbolIndex, benchmark_feature_cost
from ingest import generate_synthetic_ohlcv, save_processed
from io_utils import read_ohlcv
from simulate import (
    PROCESSED_DIR,
    RESULTS_DIR,
    default_variants,
    generate_cyclical_request_stream,
    generate_request_stream,
    results_to_frame,
    run_simulation,
)


def ensure_data() -> pd.DataFrame:
    try:
        return read_ohlcv(PROCESSED_DIR / "ohlcv_synthetic")
    except FileNotFoundError:
        symbols = [f"SIM{i}/USDT" for i in range(15)]
        df = generate_synthetic_ohlcv(symbols, n_periods=60 * 24 * 90)
        save_processed(df, name="ohlcv_synthetic")
        return df


def main():
    df = ensure_data()
    model_path = Path(__file__).resolve().parent / "learned_policy" / "model.pkl"
    window = 60
    capacity = 50
    n_requests = 20_000

    index = SymbolIndex(df)
    sample_symbol = str(df["symbol"].iloc[0])
    sym_timestamps = df.loc[df["symbol"].astype(str) == sample_symbol, "timestamp"].to_numpy()

    feature_ms = benchmark_feature_cost(index, sample_symbol, sym_timestamps, n_calls=2_000, window=window)

    all_rows = []
    for workload_name, request_fn in [
        ("zipf_bursty", lambda: generate_request_stream(df, n_requests=n_requests, window=window, seed=99)),
        ("cyclical", lambda: generate_cyclical_request_stream(df, n_requests=n_requests, window=window, p_hot=0.7, seed=99)),
    ]:
        requests = request_fn()
        start = time.perf_counter()
        results = run_simulation(df, requests, default_variants(df, capacity, model_path), window=window)
        wall_s = time.perf_counter() - start
        frame = results_to_frame(results)
        frame.insert(0, "workload", workload_name)
        frame.insert(1, "feature_compute_ms", feature_ms)
        frame.insert(2, "benchmark_wall_s", wall_s)
        all_rows.append(frame)
        print(f"\n=== {workload_name} ({n_requests:,} requests, capacity={capacity}) ===")
        print(frame.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "optimized_benchmark_metrics.csv"
    pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
    print(f"\nSaved benchmark metrics -> {out}")


if __name__ == "__main__":
    main()
