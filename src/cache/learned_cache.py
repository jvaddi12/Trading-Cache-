"""
learned_cache.py
----------------
Fast learned cache admission/eviction policy.

Optimization pass:
  - removes pandas DataFrame creation from online model inference
  - builds dense numpy feature matrices in FEATURE_COLUMNS order
  - uses slotted metadata objects instead of nested dictionaries
  - batches candidate + resident scoring into one LightGBM call per eviction
  - records model-call / eviction / rejection counts for benchmark analysis
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from cache.base_cache import BaseCache
from learned_policy.context import build_context_index, get_context, compute_rolling_vol_column
from learned_policy.label_generation import FEATURE_COLUMNS

BIG = 10 ** 9
N_FEATURES = len(FEATURE_COLUMNS)


@dataclass(slots=True)
class _AccessMeta:
    last_pos: int = -1
    count: int = 0


class LearnedCache(BaseCache):
    def __init__(self, capacity: int, model_path: Path, raw_df: pd.DataFrame, admission_margin: float = 0.0, predict_threads: int = 4):
        super().__init__(capacity)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        self.model = joblib.load(model_path)
        df_vol = compute_rolling_vol_column(raw_df, window=60)
        self.ctx_index = build_context_index(df_vol)

        self._store: Dict[Tuple[str, object], object] = {}
        self._key_meta: Dict[Tuple[str, object], _AccessMeta] = {}
        self._symbol_meta: Dict[str, _AccessMeta] = {}
        self._pos = 0
        self.admission_margin = float(admission_margin)
        self.predict_threads = int(predict_threads)

        # Extra instrumentation beyond BaseCache stats.
        self.model_calls = 0
        self.scored_keys = 0
        self.evictions = 0
        self.admission_rejections = 0

    def reset_stats(self):
        super().reset_stats()
        self.model_calls = 0
        self.scored_keys = 0
        self.evictions = 0
        self.admission_rejections = 0

    def _feature_vector_for(self, key, out: np.ndarray | None = None) -> np.ndarray:
        """Build one feature vector in FEATURE_COLUMNS order without dictionaries/pandas."""
        symbol, timestamp = key
        key_meta = self._key_meta.get(key)
        sym_meta = self._symbol_meta.get(symbol)

        key_count = key_meta.count if key_meta is not None else 0
        sym_count = sym_meta.count if sym_meta is not None else 0
        pos_since_last_key = (self._pos - key_meta.last_pos) if key_meta is not None and key_meta.last_pos >= 0 else BIG
        pos_since_last_symbol = (self._pos - sym_meta.last_pos) if sym_meta is not None and sym_meta.last_pos >= 0 else BIG
        roll_vol, hour_of_day = get_context(self.ctx_index, symbol, timestamp)

        if out is None:
            out = np.empty(N_FEATURES, dtype=np.float64)
        out[0] = key_count
        out[1] = sym_count
        out[2] = pos_since_last_symbol
        out[3] = pos_since_last_key
        out[4] = roll_vol
        out[5] = hour_of_day
        return out

    def _score_batch(self, keys: List[Tuple[str, object]]) -> np.ndarray:
        """Score keys using one dense numpy matrix and one model call."""
        X = np.empty((len(keys), N_FEATURES), dtype=np.float64)
        for i, key in enumerate(keys):
            self._feature_vector_for(key, X[i])
        self.model_calls += 1
        self.scored_keys += len(keys)
        # The sklearn wrapper defaults to OpenMP auto-threading, which is
        # surprisingly slow for thousands of tiny batches. The underlying
        # LightGBM Booster with a small fixed thread count is much faster here.
        booster = getattr(self.model, "booster_", None)
        if booster is not None:
            return booster.predict(X, num_threads=self.predict_threads)
        return self.model.predict_proba(X)[:, 1]

    def _record_access(self, key):
        symbol, _ = key
        key_meta = self._key_meta.get(key)
        if key_meta is None:
            key_meta = _AccessMeta()
            self._key_meta[key] = key_meta

        sym_meta = self._symbol_meta.get(symbol)
        if sym_meta is None:
            sym_meta = _AccessMeta()
            self._symbol_meta[symbol] = sym_meta

        key_meta.last_pos = self._pos
        key_meta.count += 1
        sym_meta.last_pos = self._pos
        sym_meta.count += 1
        self._pos += 1

    def get(self, key):
        value = self._store.get(key)
        self._record_access(key)
        return value

    def put(self, key, value):
        if key in self._store:
            self._store[key] = value
            return

        if len(self._store) < self.capacity:
            self._store[key] = value
            return

        # Exact learned eviction: one batched call for all residents + candidate.
        resident_keys = list(self._store.keys())
        all_keys = resident_keys + [key]
        scores = self._score_batch(all_keys)
        resident_scores = scores[:-1]
        candidate_score = float(scores[-1])

        weakest_idx = int(np.argmin(resident_scores))
        weakest_key = resident_keys[weakest_idx]
        weakest_score = float(resident_scores[weakest_idx])

        # Optional margin prevents churn when the model only has a tiny preference.
        if candidate_score > weakest_score + self.admission_margin:
            del self._store[weakest_key]
            self._store[key] = value
            self.evictions += 1
        else:
            self.admission_rejections += 1

    def stats(self):
        stats = super().stats()
        stats.update(
            {
                "model_calls": self.model_calls,
                "scored_keys": self.scored_keys,
                "evictions": self.evictions,
                "admission_rejections": self.admission_rejections,
            }
        )
        return stats


class LearnedAdmissionLRUCache(LearnedCache):
    """
    Low-latency hybrid: keep LRU eviction's O(1) hot path, but use the model as
    an admission filter on full-cache misses.

    Exact LearnedCache is useful as a research baseline but scores every resident
    on every eviction. This variant scores only the miss candidate, so it is much
    closer to production latency while still allowing learned admission control.
    """

    def __init__(
        self,
        capacity: int,
        model_path: Path,
        raw_df: pd.DataFrame,
        score_threshold: float = 0.0,
        predict_threads: int = 4,
    ):
        super().__init__(capacity, model_path, raw_df, admission_margin=0.0, predict_threads=predict_threads)
        from cache.lru_cache import LRUCache

        self._lru = LRUCache(capacity)
        self.score_threshold = float(score_threshold)

    def get(self, key):
        value = self._lru.get(key)
        self._record_access(key)
        return value

    def put(self, key, value):
        if key in self._lru._map:
            self._lru.put(key, value)
            return

        if len(self._lru._map) < self.capacity:
            self._lru.put(key, value)
            return

        # One model score instead of capacity+1 scores. A threshold of 0.0 makes
        # this equivalent to plain LRU admission, which is useful for ablations.
        score = float(self._score_batch([key])[0])
        if score >= self.score_threshold:
            self._lru.put(key, value)
            self.evictions += 1
        else:
            self.admission_rejections += 1
