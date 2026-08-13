# ML-Augmented Cache for a Low-Latency Trading Feature Store

A feature-serving pipeline over tick-level market data with interchangeable
cache policies: no-cache, hand-built O(1) LRU, and a LightGBM-backed learned
admission/eviction policy. The project tests whether a learned reuse-prediction
model can beat standard LRU when serving expensive trading features.

## Current status

The system runs end-to-end and now has a much faster online path. The main
optimization result is that feature computation moved from pandas-heavy slices
to numpy array slices, and learned-cache inference moved from pandas DataFrames
plus default LightGBM threading to dense numpy matrices plus fixed-thread Booster
prediction.

The learned cache is still an honest negative-result baseline: after optimizing
it, it is much faster than before, but LRU remains better on both hit rate and
latency for the synthetic workloads tested. That is a useful systems finding,
not a bug: the model's current features do not add enough signal beyond recency
to justify scoring overhead.

## Architecture

```text
data ingestion (ingest.py)
        |
feature computation (features.py) -- order-book imbalance proxy,
        |                             realized vol, VWAP deviation, momentum
        v
cache layer:
  - no_cache.py       control group, always recomputes
  - lru_cache.py      hand-built O(1) LRU, doubly linked list + hashmap
  - learned_cache.py  LightGBM-scored learned eviction/admission
        |
simulate.py / benchmark.py -- generate request streams, run cache comparisons,
                             write benchmark metrics
```

## What was optimized

### 1. Feature computation hot path

Old implementation:
- binary searched into each symbol's timestamp array
- sliced a pandas DataFrame
- built pandas Series for every feature calculation

New implementation:
- stores each symbol's `open/high/low/close/volume/timestamp` columns as
  contiguous numpy arrays
- computes every feature directly on numpy slices
- keeps `lookback_window()` only as a compatibility/debug helper

Result on the synthetic 15-symbol / 90-day dataset:

| Metric | Before | After |
|---|---:|---:|
| `compute_features()` average cost | 0.5766 ms/call | 0.0256 ms/call |
| Speedup | — | **22.5x** |

Correctness check: optimized features were compared against the original
pandas implementation on 1,000 random requests. Maximum absolute differences
were floating-point noise only: `order_book_imbalance <= 1.7e-16`,
`realized_volatility <= 4.6e-17`, `vwap_deviation <= 6.3e-16`, `momentum = 0`.

### 2. Learned-cache scoring path

Old implementation:
- built Python dictionaries for every key
- converted feature dictionaries into pandas DataFrames
- called `model.predict_proba()` through the sklearn wrapper with default
  LightGBM thread behavior on thousands of tiny batches

New implementation:
- uses slotted metadata objects instead of nested dictionaries
- builds a dense `numpy.ndarray` in `FEATURE_COLUMNS` order
- scores all residents plus candidate in one batch
- calls `model.booster_.predict(..., num_threads=4)`, which avoids huge
  auto-threading overhead on small batches
- records `model_calls`, `scored_keys`, `evictions`, and
  `admission_rejections` in benchmark output

### 3. Benchmark instrumentation

`BaseCache.stats()` now reports:
- hit rate
- average latency
- p50 / p95 / p99 latency
- throughput in requests/sec

`src/benchmark.py` runs both the Zipfian/bursty and cyclical workloads and
writes `results/optimized_benchmark_metrics.csv`.

### 4. Storage fallback

The repo now includes `src/io_utils.py`, so it still runs in environments where
`pyarrow` or `fastparquet` are unavailable. It tries parquet first and falls
back to pickle for local synthetic data.

## Optimized benchmark results

Test setup: synthetic data, 15 symbols, 90 days of 1-minute bars, 20,000
requests, cache capacity 50, 60-bar lookback window.

### Workload 1: Zipfian + bursty

| Variant | Hit Rate | Avg Latency | p95 Latency | Throughput |
|---|---:|---:|---:|---:|
| No Cache | 0.00% | 0.0245 ms | 0.0259 ms | 40,867 req/s |
| LRU | 24.10% | 0.0202 ms | 0.0271 ms | 49,397 req/s |
| Learned Cache | 17.30% | 0.3816 ms | 0.5181 ms | 2,620 req/s |

### Workload 2: Cyclical hour-hot-symbol workload

| Variant | Hit Rate | Avg Latency | p95 Latency | Throughput |
|---|---:|---:|---:|---:|
| No Cache | 0.00% | 0.0283 ms | 0.0467 ms | 35,291 req/s |
| LRU | 23.82% | 0.0236 ms | 0.0500 ms | 42,317 req/s |
| Learned Cache | 17.54% | 0.4046 ms | 0.6449 ms | 2,471 req/s |

## Interpretation

- The production-fast path is currently **LRU + optimized numpy feature
  computation**.
- The learned policy is now much more efficient, but it still loses because it
  scores many keys and its current feature set is not predictive enough.
- This is a strong project story if presented honestly: you built the learned
  system, optimized it, benchmarked it, and showed empirically that a simpler
  O(1) policy wins under these workloads.

## Reproducing

```bash
pip install -r requirements.txt

# 1. Generate synthetic OHLCV data
python src/ingest.py

# 2. Sanity-check feature computation cost
python src/features.py

# 3. Optional: retrain the learned model
python src/learned_policy/train.py

# 4. Run one cache comparison
python src/simulate.py

# 5. Run the full benchmark suite
python src/benchmark.py
```

## Repo structure

```text
trading-feature-cache/
├── requirements.txt
├── data/processed/                 # generated locally; excluded from this zip
├── results/
│   ├── baseline_metrics.csv
│   └── optimized_benchmark_metrics.csv
├── src/
│   ├── benchmark.py                # repeatable benchmark harness
│   ├── ingest.py                   # real/synthetic OHLCV ingestion
│   ├── io_utils.py                 # parquet/pickle read-write fallback
│   ├── features.py                 # optimized numpy feature path
│   ├── simulate.py                 # request streams + cache comparison
│   ├── cache/
│   │   ├── base_cache.py
│   │   ├── no_cache.py
│   │   ├── lru_cache.py
│   │   └── learned_cache.py
│   └── learned_policy/
│       ├── context.py
│       ├── label_generation.py
│       ├── train.py
│       └── model.pkl
```

## Next high-leverage work

1. Add richer model features: recent symbol burst size, reuse distance buckets,
   per-symbol hit-rate trend, volatility shock flags, cross-asset sector/group
   features, and rolling request entropy.
2. Add a true backtest consumer so cache quality can be tied to downstream
   strategy latency / P&L / Sharpe.
3. Try a cheap learned-admission + LRU-eviction policy only if the model becomes
   meaningfully predictive; otherwise the exact learned eviction baseline is not
   worth the scoring overhead.
