"""
lru_cache.py
------------
Hand-built O(1) LRU cache using a doubly linked list + hashmap (not
collections.OrderedDict) so both get() and put() are true O(1) regardless of
cache size, and so the internals are fully explainable in an interview.

This supersedes the earlier array-shifting LRU stack (O(n) per update) from
the cache-lab-style implementation -- that's the concrete "O(n) -> O(1)"
optimization worth mentioning in a writeup/resume bullet.
"""

from .base_cache import BaseCache


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache(BaseCache):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._map = {}  # key -> _Node
        # sentinel head/tail nodes to avoid null-checks at the boundaries
        self.head = _Node()  # head.next = most recently used
        self.tail = _Node()  # tail.prev = least recently used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_at_front(self, node: _Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self._map.get(key)
        if node is None:
            return None
        # promote to most-recently-used: O(1) unlink + O(1) reinsert
        self._remove(node)
        self._insert_at_front(node)
        return node.value

    def put(self, key, value):
        if key in self._map:
            node = self._map[key]
            node.value = value
            self._remove(node)
            self._insert_at_front(node)
            return

        if len(self._map) >= self.capacity:
            lru_node = self.tail.prev
            self._remove(lru_node)
            del self._map[lru_node.key]

        node = _Node(key, value)
        self._map[key] = node
        self._insert_at_front(node)
