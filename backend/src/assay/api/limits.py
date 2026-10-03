"""A small in-process sliding-window limiter (PRD 16, API security: rate limiting).

State is per process. With several API workers each enforces its own window, so the effective limit
is the per-worker limit times the worker count; a shared limiter (for example at the gateway) is the
production answer. This one keeps a runaway client or a guessing attacker from hurting the service.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit, self.window, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit. None means allowed; otherwise the seconds until a slot frees up (the hit
        is not recorded, so a client that keeps retrying does not extend its own lockout)."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return self.window - (now - q[0])
            q.append(now)
            return None
