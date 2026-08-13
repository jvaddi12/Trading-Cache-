"""
base_cache.py
-------------
Shared cache interface plus timing/quality instrumentation for benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Callable, Any

import numpy as np


class BaseCache(ABC):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.total_latency_s = 0.0
        self._latency_samples_ms = []

    @abstractmethod
    def get(self, key):
        """Return cached value or None if not present (a miss)."""
        raise NotImplementedError

    @abstractmethod
    def put(self, key, value):
        """Insert/update a value, evicting per this cache's policy if at capacity."""
        raise NotImplementedError

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.total_latency_s = 0.0
        self._latency_samples_ms.clear()

    def access(self, key, compute_fn: Callable[[], Any]):
        """
        Unified access path used by the simulator: try cache first, on miss call
        compute_fn() and populate the cache. Tracks hit/miss counts plus p50/p95/p99 latency.
        """
        start = time.perf_counter()
        cached = self.get(key)
        if cached is not None:
            self.hits += 1
            elapsed = time.perf_counter() - start
            self.total_latency_s += elapsed
            self._latency_samples_ms.append(elapsed * 1000.0)
            return cached, True

        self.misses += 1
        value = compute_fn()
        self.put(key, value)
        elapsed = time.perf_counter() - start
        self.total_latency_s += elapsed
        self._latency_samples_ms.append(elapsed * 1000.0)
        return value, False

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        avg_latency_ms = (self.total_latency_s / total * 1000) if total > 0 else 0.0
        if self._latency_samples_ms:
            lat = np.asarray(self._latency_samples_ms, dtype=np.float64)
            p50_latency_ms = float(np.percentile(lat, 50))
            p95_latency_ms = float(np.percentile(lat, 95))
            p99_latency_ms = float(np.percentile(lat, 99))
            throughput_req_s = float(total / self.total_latency_s) if self.total_latency_s > 0 else 0.0
        else:
            p50_latency_ms = p95_latency_ms = p99_latency_ms = throughput_req_s = 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "avg_latency_ms": avg_latency_ms,
            "p50_latency_ms": p50_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "throughput_req_s": throughput_req_s,
        }
