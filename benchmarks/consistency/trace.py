"""Operation-trace model for client-centric consistency measurement.

Records a flat, append-only log of read/write operations as observed by the
clients (agents) sharing a memory store. The log is the sole input to the
offline analyzer in :mod:`benchmarks.consistency.analyzer`, mirroring the
client-centric methodology of Golab et al., "Client-Centric Benchmarking of
Eventual Consistency for Cloud Storage Systems" (ICDCS 2014): consistency is
derived purely from externally observable operation timings and values, with
no access to server internals.

Each operation targets a logical *key* (a register slot) and carries a
*version*: for writes it is the version written; for reads it is the version
observed (``None`` denotes a read that returned nothing).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OpType(str, Enum):
    """The two operation kinds the analyzer understands."""

    write = "write"
    read = "read"


@dataclass(frozen=True)
class Operation:
    """A single observed operation with invocation/response timestamps."""

    op: OpType
    key: str
    agent_id: str
    t_invoke: float
    t_response: float
    version: Optional[int] = None
    value: Optional[str] = None

    def __post_init__(self) -> None:
        if self.t_response < self.t_invoke:
            raise ValueError("t_response must be >= t_invoke")
        if self.op is OpType.write and self.version is None:
            raise ValueError("write operations require a version")


class TraceLog:
    """Thread-safe, append-only collection of :class:`Operation` records."""

    def __init__(self) -> None:
        self._ops: list[Operation] = []
        self._lock = threading.Lock()

    def record(self, op: Operation) -> None:
        with self._lock:
            self._ops.append(op)

    def write(
        self,
        key: str,
        agent_id: str,
        version: int,
        t_invoke: float,
        t_response: float,
        value: Optional[str] = None,
    ) -> None:
        self.record(
            Operation(OpType.write, key, agent_id, t_invoke, t_response, version, value)
        )

    def read(
        self,
        key: str,
        agent_id: str,
        version: Optional[int],
        t_invoke: float,
        t_response: float,
        value: Optional[str] = None,
    ) -> None:
        self.record(
            Operation(OpType.read, key, agent_id, t_invoke, t_response, version, value)
        )

    @property
    def operations(self) -> list[Operation]:
        with self._lock:
            return list(self._ops)

    def __len__(self) -> int:
        with self._lock:
            return len(self._ops)
