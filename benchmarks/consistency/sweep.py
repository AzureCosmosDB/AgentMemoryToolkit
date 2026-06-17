"""Sweep consistency levels x concurrency and emit a leaderboard CSV.

Runs the consistency :func:`~benchmarks.consistency.probe.run_probe` workload
for every ``(level, concurrency)`` cell, scores each trace with
:func:`~benchmarks.consistency.analyzer.analyze`, and writes one CSV row per
cell with the Delta/k staleness and session-anomaly metrics.

Two execution modes:

* ``simulated`` (default, no Azure needed) — maps each Cosmos consistency level
  label to a representative visibility delay via :data:`SIMULATED_DELAYS` and
  runs the in-memory store. Useful for demonstrating the methodology and for
  CI.
* ``cosmos`` — runs against a real ``CosmosMemoryClient`` on a shared thread per
  cell. Cosmos consistency is configured at the client/account level (the
  package client does not expose a per-call override), so the ``level`` column
  is a label for the account's configured level unless you supply a custom
  store factory that varies it.

CLI::

    python -m benchmarks.consistency.sweep --mode simulated --out leaderboard.csv
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Callable, Optional, Sequence

from .analyzer import analyze
from .probe import CosmosStoreAdapter, InMemoryStoreAdapter, StoreAdapter, run_probe

#: Representative per-level visibility delays (seconds) for ``simulated`` mode,
#: ordered strong -> eventual. These are illustrative, not measured constants.
SIMULATED_DELAYS: dict[str, float] = {
    "strong": 0.0,
    "bounded_staleness": 0.0005,
    "session": 0.001,
    "consistent_prefix": 0.002,
    "eventual": 0.005,
}

StoreFactory = Callable[[str, int], StoreAdapter]


@dataclass
class SweepRow:
    """One leaderboard row: metrics for a single (level, concurrency) cell."""

    level: str
    concurrency: int
    reads: int
    writes: int
    stale_reads: int
    stale_read_rate: float
    delta_max: float
    delta_mean: float
    delta_p95: float
    k_max: int
    k_mean: float
    read_your_writes_violations: int
    monotonic_reads_violations: int


def _row_from_report(level: str, concurrency: int, rep) -> SweepRow:
    return SweepRow(
        level=level,
        concurrency=concurrency,
        reads=rep.reads,
        writes=rep.writes,
        stale_reads=rep.stale_reads,
        stale_read_rate=round(rep.stale_read_rate, 6),
        delta_max=round(rep.delta_max, 9),
        delta_mean=round(rep.delta_mean, 9),
        delta_p95=round(rep.delta_p95, 9),
        k_max=rep.k_max,
        k_mean=round(rep.k_mean, 6),
        read_your_writes_violations=rep.read_your_writes_violations,
        monotonic_reads_violations=rep.monotonic_reads_violations,
    )


def run_sweep(
    store_factory: StoreFactory,
    *,
    levels: Sequence[str],
    concurrency_levels: Sequence[int],
    keys: int = 4,
    ops_per_agent: int = 50,
    read_ratio: float = 0.5,
    seed: int = 0,
    clock=time.perf_counter,
) -> list[SweepRow]:
    """Run the probe for every (level, concurrency) cell and return rows.

    The same ``seed`` is reused across levels so a given concurrency uses the
    identical operation schedule, making cross-level comparisons apples-to-apples.
    """
    keys_list = [f"k{j}" for j in range(keys)]
    rows: list[SweepRow] = []
    for level in levels:
        for concurrency in concurrency_levels:
            store = store_factory(level, concurrency)
            agents = [f"agent{i}" for i in range(concurrency)]
            trace = run_probe(
                store,
                agents=agents,
                keys=keys_list,
                ops_per_agent=ops_per_agent,
                read_ratio=read_ratio,
                seed=seed,
                clock=clock,
            )
            rows.append(_row_from_report(level, concurrency, analyze(trace)))
    return rows


def write_leaderboard_csv(rows: Sequence[SweepRow], path: str) -> str:
    """Write ``rows`` to ``path`` as CSV and return the path."""
    field_names = [f.name for f in fields(SweepRow)]
    with open(str(path), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return str(path)


def simulated_store_factory(
    delays: Optional[dict[str, float]] = None, clock=time.perf_counter
) -> StoreFactory:
    """Factory producing in-memory stores whose delay maps to the level label."""
    table = dict(SIMULATED_DELAYS)
    if delays:
        table.update(delays)

    def factory(level: str, concurrency: int) -> StoreAdapter:
        return InMemoryStoreAdapter(visibility_delay=table.get(level, 0.0), clock=clock)

    return factory


def cosmos_store_factory(
    client, *, user_id: str, run_id: str = "sweep"
) -> StoreFactory:
    """Factory producing Cosmos-backed stores on a unique thread per cell."""

    def factory(level: str, concurrency: int) -> StoreAdapter:
        thread_id = f"consistency::{level}::c{concurrency}::{run_id}"
        return CosmosStoreAdapter(client, user_id=user_id, thread_id=thread_id)

    return factory


def _build_cosmos_client_from_env():
    try:
        from azure.cosmos.agent_memory import CosmosMemoryClient
    except ImportError:  # pragma: no cover
        from agent_memory_toolkit import CosmosMemoryClient

    return CosmosMemoryClient(
        cosmos_endpoint=os.environ.get("COSMOS_DB_ENDPOINT"),
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER"),
        cosmos_counter_container=os.environ.get("COSMOS_DB_COUNTERS_CONTAINER"),
        cosmos_lease_container=os.environ.get("COSMOS_DB_LEASE_CONTAINER"),
        cosmos_throughput_mode=os.environ.get("COSMOS_DB_THROUGHPUT_MODE"),
        ai_foundry_endpoint=os.environ.get("AI_FOUNDRY_ENDPOINT"),
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY") or None,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
        adf_endpoint=os.environ.get("ADF_ENDPOINT"),
        adf_key=os.environ.get("ADF_KEY"),
    )


def _print_table(rows: Sequence[SweepRow]) -> None:
    header = (
        f"{'level':<20}{'conc':>5}{'stale%':>9}{'d_p95':>11}"
        f"{'k_max':>7}{'RYW':>6}{'MR':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.level:<20}{r.concurrency:>5}{r.stale_read_rate * 100:>8.1f}%"
            f"{r.delta_p95:>11.4g}{r.k_max:>7}"
            f"{r.read_your_writes_violations:>6}{r.monotonic_reads_violations:>5}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep consistency levels x concurrency and emit a leaderboard.csv"
    )
    parser.add_argument("--mode", choices=["simulated", "cosmos"], default="simulated")
    parser.add_argument(
        "--levels",
        default="strong,bounded_staleness,session,consistent_prefix,eventual",
        help="Comma-separated consistency level labels.",
    )
    parser.add_argument(
        "--concurrency", default="1,2,4", help="Comma-separated agent counts."
    )
    parser.add_argument("--keys", type=int, default=4)
    parser.add_argument("--ops", type=int, default=50, help="Operations per agent.")
    parser.add_argument("--read-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="leaderboard.csv")
    parser.add_argument("--run-id", default="sweep")
    parser.add_argument("--user-id", default="consistency-probe")
    args = parser.parse_args(argv)

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]

    client = None
    if args.mode == "cosmos":
        client = _build_cosmos_client_from_env()
        factory = cosmos_store_factory(client, user_id=args.user_id, run_id=args.run_id)
    else:
        factory = simulated_store_factory()

    try:
        rows = run_sweep(
            factory,
            levels=levels,
            concurrency_levels=concurrency,
            keys=args.keys,
            ops_per_agent=args.ops,
            read_ratio=args.read_ratio,
            seed=args.seed,
        )
        out = write_leaderboard_csv(rows, args.out)
        _print_table(rows)
        print(f"\nWrote {len(rows)} rows to {out}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
