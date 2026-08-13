"""
io_utils.py
-----------
Small storage helpers so the project runs even when optional parquet engines
(pyarrow/fastparquet) are unavailable in a restricted environment.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_ohlcv(df: pd.DataFrame, path_without_suffix: Path) -> Path:
    """Prefer parquet; fall back to pickle if parquet support is unavailable."""
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        pkl_path = path_without_suffix.with_suffix(".pkl")
        df.to_pickle(pkl_path)
        print(f"Parquet unavailable ({exc.__class__.__name__}); saved pickle fallback instead.")
        return pkl_path


def read_ohlcv(path_without_suffix: Path) -> pd.DataFrame:
    """Read parquet if present and supported; otherwise read the pickle fallback."""
    parquet_path = path_without_suffix.with_suffix(".parquet")
    pkl_path = path_without_suffix.with_suffix(".pkl")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            if not pkl_path.exists():
                raise RuntimeError(
                    f"Found {parquet_path}, but pandas could not read it and no pickle fallback exists. "
                    "Install pyarrow/fastparquet or regenerate data."
                ) from exc
            print(f"Parquet read unavailable ({exc.__class__.__name__}); using pickle fallback.")
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    raise FileNotFoundError(f"Could not find {parquet_path} or {pkl_path}")
