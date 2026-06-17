"""Client-centric consistency benchmarking for shared agent memory.

Trace logging, an offline staleness/anomaly analyzer, and a concurrent probe
for measuring eventual-consistency behavior of a shared memory store, inspired
by Golab et al., "Client-Centric Benchmarking of Eventual Consistency for Cloud
Storage Systems" (ICDCS 2014).
"""

from .analyzer import ConsistencyReport, analyze
from .probe import (
    CosmosStoreAdapter,
    InMemoryStoreAdapter,
    StoreAdapter,
    run_probe,
)
from .sweep import (
    SIMULATED_DELAYS,
    SweepRow,
    cosmos_store_factory,
    run_sweep,
    simulated_store_factory,
    write_leaderboard_csv,
)
from .trace import Operation, OpType, TraceLog

__all__ = [
    "Operation",
    "OpType",
    "TraceLog",
    "ConsistencyReport",
    "analyze",
    "CosmosStoreAdapter",
    "InMemoryStoreAdapter",
    "StoreAdapter",
    "run_probe",
    "SIMULATED_DELAYS",
    "SweepRow",
    "cosmos_store_factory",
    "run_sweep",
    "simulated_store_factory",
    "write_leaderboard_csv",
]
