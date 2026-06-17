"""Real Cosmos DB consistency-level sweep on a single account.

Cosmos keeps four replicas per partition even in a single region. Under
**Eventual** consistency a read is routed to any replica and may observe a
secondary that has not yet received the latest write, producing genuine,
measurable staleness. Under **Session** the SDK tracks a session token across
operations on the shared client, restoring read-your-writes. Sweeping the
levels the account permits therefore surfaces a real staleness gradient — no
multi-region account required.

An account can only serve a consistency level **at or weaker than** its default.
``agentmemorytest`` defaults to Session, so the achievable levels are
``Session``, ``ConsistentPrefix`` and ``Eventual`` (Strong / BoundedStaleness
would be rejected).

Run::

    $env:COSMOS_ENDPOINT=...; $env:COSMOS_MASTER_KEY=...
    python -m benchmarks.cosmos_levels --agents 6 --ops 50
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from typing import Optional

from azure.cosmos import CosmosClient, PartitionKey

from .consistency.analyzer import analyze
from .consistency.probe import run_probe
from .consistency.sweep import SweepRow, write_leaderboard_csv

#: Levels an account can serve are those at or weaker than its default. Ordered
#: strongest -> weakest among the Session-or-weaker tier.
ACHIEVABLE_LEVELS = ("Session", "ConsistentPrefix", "Eventual")


class DirectCosmosRegisterStore:
    """Register store over raw Cosmos with a fixed client consistency level.

    Writes upsert ``{id, pk=<ns:key>, key, version, agent_id}``; reads aggregate
    the max version for a key. All keys are namespaced per run so repeated
    sweeps never interfere and no cleanup is required between levels.
    """

    def __init__(
        self,
        endpoint: str,
        key: str,
        consistency_level: str,
        database: str = "consistency_bench",
        container: str = "registers",
        namespace: str = "",
    ) -> None:
        self.framework_name = "Cosmos"
        self.backend_type = f"cosmos-{consistency_level.lower()}"
        self._ns = namespace or uuid.uuid4().hex[:8]
        # One shared client at the requested level — its session token (Session)
        # or lack thereof (Eventual) is exactly what we are benchmarking.
        self._client = CosmosClient(
            url=endpoint, credential=key, consistency_level=consistency_level
        )
        db = self._client.create_database_if_not_exists(id=database)
        self._container = db.create_container_if_not_exists(
            id=container, partition_key=PartitionKey(path="/pk")
        )

    def _pk(self, key: str) -> str:
        return f"{self._ns}:{key}"

    def write(self, key: str, version: int, agent_id: str) -> None:
        pk = self._pk(key)
        self._container.upsert_item(
            {
                "id": f"{pk}:{uuid.uuid4().hex}",
                "pk": pk,
                "key": key,
                "version": version,
                "agent_id": agent_id,
            }
        )

    def read_latest(self, key: str, agent_id: str) -> Optional[int]:
        pk = self._pk(key)
        items = self._container.query_items(
            query="SELECT VALUE c.version FROM c WHERE c.pk=@pk",
            parameters=[{"name": "@pk", "value": pk}],
            partition_key=pk,
        )
        versions = [v for v in items if isinstance(v, int)]
        return max(versions) if versions else None


def _row(level: str, concurrency: int, rep) -> SweepRow:
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Real Cosmos consistency-level sweep.")
    parser.add_argument("--agents", type=int, default=6, help="Concurrent agents.")
    parser.add_argument("--ops", type=int, default=50, help="Operations per agent.")
    parser.add_argument("--keys", type=int, default=3, help="Distinct register keys.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(ACHIEVABLE_LEVELS),
        help=f"Consistency levels to sweep (achievable: {ACHIEVABLE_LEVELS}).",
    )
    parser.add_argument(
        "--read-ratio", type=float, default=0.5, help="Fraction of ops that are reads."
    )
    parser.add_argument("--out", default="cosmos_levels.csv", help="CSV output path.")
    args = parser.parse_args(argv)

    endpoint = os.getenv("COSMOS_ENDPOINT")
    key = os.getenv("COSMOS_MASTER_KEY")
    if not endpoint or not key:
        print("COSMOS_ENDPOINT and COSMOS_MASTER_KEY must be set.")
        return 1

    keys = [f"key{i}" for i in range(args.keys)]
    agent_ids = [f"agent_{i}" for i in range(args.agents)]
    rows: list[SweepRow] = []

    for level in args.levels:
        print(f"\n=== Cosmos level: {level} ===")
        store = DirectCosmosRegisterStore(
            endpoint, key, consistency_level=level, namespace=uuid.uuid4().hex[:8]
        )
        t0 = time.perf_counter()
        trace = run_probe(
            store=store,
            agents=agent_ids,
            keys=keys,
            ops_per_agent=args.ops,
            read_ratio=args.read_ratio,
        )
        wall = time.perf_counter() - t0
        rep = analyze(trace)
        print(f"  {rep.summary()}")
        print(f"  wall={wall:.2f}s")
        rows.append(_row(level, args.agents, rep))

    write_leaderboard_csv(rows, args.out)
    print(f"\nWrote {args.out}")

    # Console summary table.
    print("\n" + "=" * 96)
    print("REAL COSMOS DB CONSISTENCY-LEVEL SWEEP (single account, West US 3)")
    print("=" * 96)
    print(
        f"{'Level':<18}{'Stale%':>9}{'d_max(ms)':>12}{'d_p95(ms)':>12}"
        f"{'k_max':>8}{'RYW':>7}{'MR':>6}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r.level:<18}{r.stale_read_rate * 100:>9.1f}{r.delta_max * 1000:>12.3f}"
            f"{r.delta_p95 * 1000:>12.3f}{r.k_max:>8}{r.read_your_writes_violations:>7}"
            f"{r.monotonic_reads_violations:>6}"
        )
    print("=" * 96)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
