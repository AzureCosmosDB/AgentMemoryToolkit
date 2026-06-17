"""Concurrent read/write probe for shared-memory consistency measurement.

Drives N agent clients against a shared store, recording every operation into a
:class:`~benchmarks.consistency.trace.TraceLog` for offline analysis. Two store
adapters are provided:

* :class:`InMemoryStoreAdapter` — dependency-free, with an optional visibility
  delay to emulate eventual consistency (useful for tests and demos).
* :class:`CosmosStoreAdapter` — backs onto a ``CosmosMemoryClient`` and a shared
  ``(user_id, thread_id)``, exercising the real shared-memory path. Vary the
  Cosmos *consistency level* on the client to compare staleness across levels.

A logical *key* is a register slot; each write appends a new version and a read
returns the highest version currently visible. Measuring how far behind that
visible version lags the latest durable write is exactly the staleness the
analyzer quantifies.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Optional, Protocol, Sequence

from .trace import TraceLog


class StoreAdapter(Protocol):
    """Minimal register interface the probe drives."""

    def write(self, key: str, version: int, agent_id: str) -> None: ...

    def read_latest(self, key: str, agent_id: str) -> Optional[int]: ...


class InMemoryStoreAdapter:
    """Thread-safe in-memory register store with optional visibility delay.

    A non-zero ``visibility_delay`` emulates eventual consistency: a write only
    becomes observable ``visibility_delay`` time units after it is issued, so
    reads in that window observe an older version (or none).
    """

    def __init__(self, visibility_delay: float = 0.0, clock=time.perf_counter) -> None:
        self._writes: dict[str, list[tuple[int, float]]] = {}
        self._delay = visibility_delay
        self._clock = clock
        self._lock = threading.Lock()

    def write(self, key: str, version: int, agent_id: str) -> None:
        visible_at = self._clock() + self._delay
        with self._lock:
            self._writes.setdefault(key, []).append((version, visible_at))

    def read_latest(self, key: str, agent_id: str) -> Optional[int]:
        now = self._clock()
        with self._lock:
            visible = [v for (v, t) in self._writes.get(key, []) if t <= now]
        return max(visible) if visible else None


class CosmosStoreAdapter:
    """Register store backed by a ``CosmosMemoryClient`` on a shared thread.

    Each logical key maps to metadata ``key``/``v`` (and, when supported, a
    ``key:<key>`` tag). ``read_latest`` returns the highest version a query
    currently surfaces — the quantity whose lag the analyzer scores. Set the
    Cosmos consistency level on the injected client to benchmark different
    levels.
    """

    def __init__(self, client, user_id: str, thread_id: str) -> None:
        self._c = client
        self._u = user_id
        self._t = thread_id
        import inspect

        try:
            params = inspect.signature(client.add_cosmos).parameters
            self._tags_ok = "tags" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            self._tags_ok = False

    def write(self, key: str, version: int, agent_id: str) -> None:
        kwargs = dict(
            user_id=self._u,
            role="agent",
            content=f"{key}={version}",
            memory_type="episodic",
            thread_id=self._t,
            metadata={"key": key, "v": version, "agent_id": agent_id},
        )
        if self._tags_ok:
            kwargs["tags"] = [f"key:{key}"]
        self._c.add_cosmos(**kwargs)

    def read_latest(self, key: str, agent_id: str) -> Optional[int]:
        recs = self._c.get_memories(
            user_id=self._u, thread_id=self._t, memory_types=["episodic"]
        )
        versions = [
            (r.get("metadata") or {}).get("v")
            for r in recs
            if (r.get("metadata") or {}).get("key") == key
        ]
        versions = [v for v in versions if isinstance(v, int)]
        return max(versions) if versions else None


def run_probe(
    store: StoreAdapter,
    *,
    agents: Sequence[str],
    keys: Sequence[str],
    ops_per_agent: int,
    read_ratio: float = 0.5,
    seed: int = 0,
    clock=time.perf_counter,
) -> TraceLog:
    """Drive concurrent agents against ``store`` and return the operation trace.

    Each agent runs in its own thread, performing ``ops_per_agent`` operations,
    each a read (probability ``read_ratio``) or a write of a fresh version to a
    randomly chosen key. All agents share ``store``, so the resulting trace
    captures cross-agent visibility — the essence of shared memory.
    """
    trace = TraceLog()
    keys = list(keys)
    counters = {k: 0 for k in keys}
    counter_lock = threading.Lock()

    def next_version(key: str) -> int:
        with counter_lock:
            counters[key] += 1
            return counters[key]

    def worker(agent_id: str, wseed: int) -> None:
        rng = random.Random(wseed)
        for _ in range(ops_per_agent):
            key = rng.choice(keys)
            if rng.random() < read_ratio:
                t0 = clock()
                observed = store.read_latest(key, agent_id)
                t1 = clock()
                trace.read(key, agent_id, observed, t0, t1)
            else:
                version = next_version(key)
                t0 = clock()
                store.write(key, version, agent_id)
                t1 = clock()
                trace.write(key, agent_id, version, t0, t1)

    threads = [
        threading.Thread(target=worker, args=(agent, seed + i + 1))
        for i, agent in enumerate(agents)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return trace
