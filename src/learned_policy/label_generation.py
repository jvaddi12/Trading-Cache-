"""
label_generation.py
--------------------
Turns a historical request stream into a supervised learning problem:

  Given a request for key=(symbol, timestamp) at stream position `pos`, will
  this *exact* key be requested again within the next `lookahead` requests?

Features are built only from information available *at* `pos` (no lookahead
leakage into the features themselves -- only the label looks forward):
  - key_repeat_count_so_far : how many times this exact key has been seen before
  - symbol_count_so_far     : how many times this symbol has been seen before
  - pos_since_last_symbol   : positions since this symbol was last requested
  - pos_since_last_key      : positions since this exact key was last requested
  - roll_vol                : rolling realized volatility of the symbol at this timestamp
  - hour_of_day             : time-of-day bucket
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from learned_policy.context import compute_rolling_vol_column, build_context_index, get_context
from io_utils import read_ohlcv

BIG = 10 ** 9  # sentinel for "never seen before"


def build_training_frame(requests: list, df: pd.DataFrame, lookahead: int = 20) -> pd.DataFrame:
    """
    requests: list of (symbol, timestamp) tuples in stream order (as produced by
              simulate.generate_request_stream).
    df: the raw OHLCV frame (used to compute volatility context).
    lookahead: how many future requests count as "reused soon" for the label.
    """
    stream = pd.DataFrame(requests, columns=["symbol", "timestamp"])
    stream["pos"] = np.arange(len(stream))
    stream["key"] = list(zip(stream["symbol"], stream["timestamp"]))

    # --- recency/frequency features, computed causally (no future leakage) ---
    stream["key_repeat_count_so_far"] = stream.groupby("key").cumcount()
    stream["symbol_count_so_far"] = stream.groupby("symbol").cumcount()

    stream["pos_since_last_symbol"] = (
        stream["pos"] - stream.groupby("symbol")["pos"].shift(1)
    ).fillna(BIG)
    stream["pos_since_last_key"] = (
        stream["pos"] - stream.groupby("key")["pos"].shift(1)
    ).fillna(BIG)

    # --- label: does this exact key reappear within the next `lookahead` requests? ---
    next_pos = stream.groupby("key")["pos"].shift(-1)
    gap = next_pos - stream["pos"]
    stream["label"] = ((gap > 0) & (gap <= lookahead)).astype(int)

    # --- context features: volatility regime + time of day ---
    df_vol = compute_rolling_vol_column(df, window=60)
    ctx_index = build_context_index(df_vol)
    roll_vols, hours = [], []
    for symbol, ts in zip(stream["symbol"], stream["timestamp"]):
        rv, hr = get_context(ctx_index, symbol, ts)
        roll_vols.append(rv)
        hours.append(hr)
    stream["roll_vol"] = roll_vols
    stream["hour_of_day"] = hours

    return stream.drop(columns=["key"])


FEATURE_COLUMNS = [
    "key_repeat_count_so_far",
    "symbol_count_so_far",
    "pos_since_last_symbol",
    "pos_since_last_key",
    "roll_vol",
    "hour_of_day",
]


if __name__ == "__main__":
    from simulate import generate_request_stream

    processed = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "ohlcv_synthetic"
    df = read_ohlcv(processed)

    requests = generate_request_stream(df, n_requests=20_000, window=60)
    training_frame = build_training_frame(requests, df, lookahead=20)

    print(training_frame[["symbol", "pos"] + FEATURE_COLUMNS + ["label"]].head(10))
    print("\nLabel balance:")
    print(training_frame["label"].value_counts(normalize=True))
