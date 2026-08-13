"""
ingest.py
---------
Two data paths:

1. `fetch_real_ohlcv(...)`  — pulls real 1-minute OHLCV bars from Binance via ccxt.
   Run this on your own machine (needs live internet access to Binance).

2. `generate_synthetic_ohlcv(...)` — produces realistic-looking synthetic tick data
   (geometric Brownian motion + volatility clustering + occasional volume bursts)
   so the rest of the pipeline can be built, tested, and demoed without live network
   access. Swap this out for `fetch_real_ohlcv` once you're running locally.

Both paths return the same schema so nothing downstream needs to change:
    columns: [timestamp, symbol, open, high, low, close, volume]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from io_utils import write_ohlcv

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def fetch_real_ohlcv(symbols, timeframe="1m", since_days=90, limit_per_call=1000):
    """
    Pulls real OHLCV data from Binance using ccxt. Requires internet access —
    run this on your own machine, not in a network-restricted sandbox.

    symbols: list like ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    """
    import ccxt  # imported lazily so the synthetic path doesn't require it installed correctly

    exchange = ccxt.binance()
    since = exchange.parse8601(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=since_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    all_rows = []
    for symbol in symbols:
        cursor = since
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
            if not batch:
                break
            for row in batch:
                all_rows.append(
                    {
                        "timestamp": pd.to_datetime(row[0], unit="ms", utc=True),
                        "symbol": symbol,
                        "open": row[1],
                        "high": row[2],
                        "low": row[3],
                        "close": row[4],
                        "volume": row[5],
                    }
                )
            cursor = batch[-1][0] + 1
            if len(batch) < limit_per_call:
                break

    df = pd.DataFrame(all_rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def generate_synthetic_ohlcv(symbols, n_periods=60 * 24 * 90, freq="1min", seed=42):
    """
    Generates realistic synthetic 1-minute OHLCV bars for each symbol using:
      - geometric Brownian motion for the price path
      - a GARCH-like volatility regime that clusters (so volatility isn't iid noise)
      - Poisson-distributed volume with occasional bursts (simulates news/volume spikes)

    This is NOT real market data — it exists purely so the rest of the pipeline
    (features, caching, learned admission, backtest) can be built and verified
    end-to-end before you plug in real Binance/LOBSTER data.

    n_periods=60*24*90 -> 90 days of 1-minute bars.
    """
    rng = np.random.default_rng(seed)
    all_dfs = []

    start_ts = pd.Timestamp("2026-01-01", tz="UTC")
    timestamps = pd.date_range(start_ts, periods=n_periods, freq=freq)

    for i, symbol in enumerate(symbols):
        # each symbol gets a different vol regime + drift so they aren't identical
        base_vol = 0.0005 + 0.0003 * i
        drift = rng.normal(0, 0.00002)

        # volatility clustering via a simple mean-reverting vol process
        vol = np.zeros(n_periods)
        vol[0] = base_vol
        for t in range(1, n_periods):
            vol[t] = max(
                1e-6,
                vol[t - 1] + 0.05 * (base_vol - vol[t - 1]) + rng.normal(0, base_vol * 0.1),
            )

        returns = rng.normal(drift, vol)
        log_price = np.cumsum(returns) + np.log(100 + i * 50)  # different starting price per symbol
        close = np.exp(log_price)

        # derive open/high/low from close with small intrabar noise
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, base_vol * 0.5, n_periods)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, base_vol * 0.5, n_periods)))

        # volume: baseline Poisson + occasional bursts correlated with |returns|
        base_volume = rng.poisson(lam=50, size=n_periods).astype(float)
        burst_mask = np.abs(returns) > (2.5 * vol)
        base_volume[burst_mask] *= rng.uniform(3, 8, size=burst_mask.sum())

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": base_volume,
            }
        )
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return full


def save_processed(df: pd.DataFrame, name: str = "ohlcv"):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = write_ohlcv(df, PROCESSED_DIR / name)
    print(f"Saved {len(df):,} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    # Default: generate synthetic data for 15 symbols, 90 days of 1-minute bars.
    # Swap this block for fetch_real_ohlcv(...) once running with live internet access.
    symbols = [f"SIM{i}/USDT" for i in range(15)]
    df = generate_synthetic_ohlcv(symbols, n_periods=60 * 24 * 90)
    save_processed(df, name="ohlcv_synthetic")
    print(df.head())
    print(df.groupby("symbol").size())
