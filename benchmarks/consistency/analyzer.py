"""Offline client-centric consistency analysis over an operation trace.

Implements observable staleness and session-guarantee metrics inspired by
Golab et al., "Client-Centric Benchmarking of Eventual Consistency for Cloud
Storage Systems" (ICDCS 2014) and the Delta/Gamma consistency line of work
(Golab, Li, Shah, PODC 2011):

* **Delta** (time staleness) — for a stale read, how long a newer,
  already-durable version had been available when the read began. The trace's
  Delta score is the worst case; the full distribution is also reported.
* **k** (version staleness) — how many newer durable versions a read missed.
* **Read-your-writes** / **monotonic-reads** anomaly counts (per agent), the
  session guarantees most relevant to multi-agent shared memory.

This computes the operational *t-visibility* staleness that Delta-atomicity
bounds. Exact minimal Delta-atomicity (the graph/LP construction in the source
papers) is intentionally out of scope; the goal here is a cheap, comparable
score across consistency levels and concurrency settings.

The model assumes a register-per-key abstraction with monotonically increasing
versions, so a read can be matched unambiguously to the version it observed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Union

from .trace import Operation, OpType, TraceLog


def _percentile(values_sorted: list[float], q: float) -> float:
    if not values_sorted:
        return 0.0
    if len(values_sorted) == 1:
        return float(values_sorted[0])
    pos = q * (len(values_sorted) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values_sorted[int(pos)])
    frac = pos - lo
    return float(values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac)


@dataclass
class ConsistencyReport:
    """Aggregate consistency metrics derived from a trace."""

    total_ops: int = 0
    reads: int = 0
    writes: int = 0
    reads_scored: int = 0
    stale_reads: int = 0
    read_misses: int = 0
    stale_read_rate: float = 0.0
    delta_max: float = 0.0
    delta_mean: float = 0.0
    delta_p50: float = 0.0
    delta_p95: float = 0.0
    k_max: int = 0
    k_mean: float = 0.0
    read_your_writes_violations: int = 0
    monotonic_reads_violations: int = 0
    per_key: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"ops={self.total_ops} reads={self.reads} writes={self.writes} "
            f"stale={self.stale_reads}/{self.reads_scored} "
            f"({self.stale_read_rate:.1%}) "
            f"delta_max={self.delta_max:.4g} delta_p95={self.delta_p95:.4g} "
            f"k_max={self.k_max} RYW_viol={self.read_your_writes_violations} "
            f"MR_viol={self.monotonic_reads_violations}"
        )


def analyze(trace: Union[TraceLog, Iterable[Operation]]) -> ConsistencyReport:
    """Compute a :class:`ConsistencyReport` from a trace or operation iterable."""
    ops = trace.operations if isinstance(trace, TraceLog) else list(trace)
    writes = [o for o in ops if o.op is OpType.write]
    reads = [o for o in ops if o.op is OpType.read]

    writes_by_key: dict[str, list[Operation]] = defaultdict(list)
    for w in writes:
        writes_by_key[w.key].append(w)

    deltas: list[float] = []
    ks: list[int] = []
    stale = 0
    misses = 0
    per_key_stale: dict[str, int] = defaultdict(int)
    per_key_delta_max: dict[str, float] = defaultdict(float)

    for r in reads:
        # Writes guaranteed durable before this read began.
        durable = [w for w in writes_by_key.get(r.key, []) if w.t_response <= r.t_invoke]
        if not durable:
            # Nothing was guaranteed visible yet; staleness is undefined.
            continue
        newest = max(w.version for w in durable)
        observed = r.version if r.version is not None else 0
        k = newest - observed
        if k <= 0:
            # Read saw a version at least as fresh as the newest durable one.
            deltas.append(0.0)
            ks.append(0)
            continue

        # Stale read: it missed at least one already-durable newer version.
        stale += 1
        if r.version is None:
            misses += 1
        ks.append(k)
        newer_durable = [w for w in durable if w.version > observed]
        oldest_missed = min(newer_durable, key=lambda w: w.version)
        d = max(0.0, r.t_invoke - oldest_missed.t_response)
        deltas.append(d)
        per_key_stale[r.key] += 1
        per_key_delta_max[r.key] = max(per_key_delta_max[r.key], d)

    # Read-your-writes: an agent should observe its own latest durable write.
    ryw = 0
    for r in reads:
        mine = [
            w
            for w in writes_by_key.get(r.key, [])
            if w.agent_id == r.agent_id and w.t_response <= r.t_invoke
        ]
        if mine:
            my_latest = max(w.version for w in mine)
            observed = r.version if r.version is not None else 0
            if observed < my_latest:
                ryw += 1

    # Monotonic reads: an agent's successive reads of a key must not go back.
    mr = 0
    reads_by_ak: dict[tuple[str, str], list[Operation]] = defaultdict(list)
    for r in reads:
        reads_by_ak[(r.agent_id, r.key)].append(r)
    for seq in reads_by_ak.values():
        seq.sort(key=lambda r: r.t_invoke)
        high = 0
        for r in seq:
            observed = r.version if r.version is not None else 0
            if observed < high:
                mr += 1
            else:
                high = observed

    deltas_sorted = sorted(deltas)
    return ConsistencyReport(
        total_ops=len(ops),
        reads=len(reads),
        writes=len(writes),
        reads_scored=len(deltas),
        stale_reads=stale,
        read_misses=misses,
        stale_read_rate=(stale / len(deltas)) if deltas else 0.0,
        delta_max=max(deltas) if deltas else 0.0,
        delta_mean=(sum(deltas) / len(deltas)) if deltas else 0.0,
        delta_p50=_percentile(deltas_sorted, 0.50),
        delta_p95=_percentile(deltas_sorted, 0.95),
        k_max=max(ks) if ks else 0,
        k_mean=(sum(ks) / len(ks)) if ks else 0.0,
        read_your_writes_violations=ryw,
        monotonic_reads_violations=mr,
        per_key={
            key: {
                "stale_reads": per_key_stale.get(key, 0),
                "delta_max": per_key_delta_max.get(key, 0.0),
            }
            for key in writes_by_key
        },
    )
