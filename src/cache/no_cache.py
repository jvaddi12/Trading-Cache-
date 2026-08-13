"""
no_cache.py
-----------
Control-group baseline: always recomputes, never caches. Every access() call is
a guaranteed miss. This gives us the "no caching at all" latency floor to
compare LRU and the learned cache against.
"""

from .base_cache import BaseCache


class NoCache(BaseCache):
    def get(self, key):
        return None  # always a miss by design

    def put(self, key, value):
        pass  # never stores anything
