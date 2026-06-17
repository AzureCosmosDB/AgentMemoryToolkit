"""Behavioral degradation of shared agent memory under concurrency.

The consistency probe (``consistency/``) measures whether an individual *read*
observes the latest durable *write* (Delta / k staleness). But agents rarely
just read — they **read-modify-write** shared memory: read the current state,
reason over it, then write an updated state back. That pattern loses updates
under concurrency *even when every individual read is perfectly consistent*,
because two agents can read the same value, both reason, and both write back —
silently dropping one agent's contribution.

Lost updates are the **behavioral** failure that matters: an agent's work
vanishes from shared memory, so downstream agents reason over an incorrect
state. Crucially this is an *application-level* race, not a storage-replication
issue, so it appears even under Strong consistency in a single region.

This module quantifies that degradation as a function of agent concurrency for
four shared-memory patterns:

* ``mutable``   — last-writer-wins cell (mutate-in-place). Loses updates.
* ``locked``    — same cell guarded by a global lock. Correct, but serialized.
* ``cas``       — compare-and-swap with retry (optimistic concurrency). Correct.
* ``append``    — append-only log (the AgentMemoryToolkit ``add_cosmos`` model).
                  Inherently correct for "latest value" / aggregation reads.

A ``--cosmos`` mode reproduces the same result on a live Cosmos DB item:
a naive read-modify-write replace loses updates, while an ETag-guarded replace
(optimistic concurrency with retry) does not.

Run::

    python -m benchmarks.behavioral --increments 50
    $env:COSMOS_ENDPOINT=...; $env:COSMOS_MASTER_KEY=...
    python -m benchmarks.behavioral --increments 50 --cosmos
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Shared-memory cell patterns
# ---------------------------------------------------------------------------
class MutableCell:
    """Last-writer-wins cell modelling mutate-in-place shared memory."""

    def __init__(self) -> None:
        self._v = 0

    def read(self) -> int:
        return self._v

    def write(self, value: int) -> None:
        self._v = value

    def value(self) -> int:
        return self._v


class LockedCell(MutableCell):
    """Mutable cell whose read-modify-write is serialized by a global lock."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()


class CASCell:
    """Compare-and-swap cell modelling optimistic concurrency (Cosmos ETags).

    ``compare_and_set`` succeeds only if the stored version still matches the
    one the caller read; otherwise the caller must re-read and retry. This is
    the in-process analogue of an ETag / ``If-Match`` guarded write.
    """

    def __init__(self) -> None:
        self._v = 0
        self._ver = 0
        self._lock = threading.Lock()

    def read(self) -> tuple[int, int]:
        with self._lock:
            return self._v, self._ver

    def compare_and_set(self, expected_ver: int, value: int) -> bool:
        with self._lock:
            if self._ver != expected_ver:
                return False
            self._v = value
            self._ver += 1
            return True

    def value(self) -> int:
        return self._v


class AppendLog:
    """Append-only log modelling the toolkit's ``add_cosmos`` pattern.

    Each contribution is a distinct entry; the aggregate ("how many?") is
    derived by counting. There is no read-modify-write, so nothing is lost.
    """

    def __init__(self) -> None:
        self._entries: list = []
        self._lock = threading.Lock()

    def append(self, item) -> None:
        with self._lock:
            self._entries.append(item)

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Concurrent task: each agent records `increments` contributions to shared memory
# ---------------------------------------------------------------------------
@dataclass
class BehavioralResult:
    pattern: str
    concurrency: int
    expected: int
    observed: int
    lost_updates: int
    lost_update_rate: float
    retries: int
    wall_seconds: float


def _run_workers(n_agents: int, work: Callable[[int], None]) -> float:
    barrier = threading.Barrier(n_agents)
    threads = []

    def runner(idx: int) -> None:
        barrier.wait()  # release all agents together to maximize contention
        work(idx)

    t0 = time.perf_counter()
    for i in range(n_agents):
        t = threading.Thread(target=runner, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def run_pattern(
    pattern: str,
    *,
    concurrency: int,
    increments_per_agent: int,
    think_delay: float,
) -> BehavioralResult:
    """Drive `concurrency` agents, each contributing `increments_per_agent`."""
    expected = concurrency * increments_per_agent
    retries = 0
    retries_lock = threading.Lock()

    if pattern == "mutable":
        cell = MutableCell()

        def work(_idx: int) -> None:
            for _ in range(increments_per_agent):
                cur = cell.read()
                time.sleep(think_delay)  # models agent reasoning between R and W
                cell.write(cur + 1)

        wall = _run_workers(concurrency, work)
        observed = cell.value()

    elif pattern == "locked":
        cell = LockedCell()

        def work(_idx: int) -> None:
            for _ in range(increments_per_agent):
                with cell.lock:
                    cur = cell.read()
                    time.sleep(think_delay)
                    cell.write(cur + 1)

        wall = _run_workers(concurrency, work)
        observed = cell.value()

    elif pattern == "cas":
        cas = CASCell()

        def work(_idx: int) -> None:
            nonlocal retries
            for _ in range(increments_per_agent):
                while True:
                    cur, ver = cas.read()
                    time.sleep(think_delay)
                    if cas.compare_and_set(ver, cur + 1):
                        break
                    with retries_lock:
                        retries += 1

        wall = _run_workers(concurrency, work)
        observed = cas.value()

    elif pattern == "append":
        log = AppendLog()

        def work(idx: int) -> None:
            for _ in range(increments_per_agent):
                time.sleep(think_delay)
                log.append((idx, uuid.uuid4().hex))

        wall = _run_workers(concurrency, work)
        observed = log.count()

    else:
        raise ValueError(f"unknown pattern: {pattern}")

    lost = expected - observed
    return BehavioralResult(
        pattern=pattern,
        concurrency=concurrency,
        expected=expected,
        observed=observed,
        lost_updates=lost,
        lost_update_rate=round(lost / expected, 4) if expected else 0.0,
        retries=retries,
        wall_seconds=round(wall, 3),
    )


# ---------------------------------------------------------------------------
# Real Cosmos read-modify-write: naive replace vs. ETag optimistic concurrency
# ---------------------------------------------------------------------------
def run_cosmos_rmw(
    *, concurrency: int, increments_per_agent: int, guarded: bool, think_delay: float
) -> BehavioralResult:
    """Concurrent read-modify-write of a single Cosmos item.

    ``guarded=False`` replaces the item with no concurrency check (lost
    updates). ``guarded=True`` uses ETag / ``If-Match`` optimistic concurrency
    and retries on the 412 precondition failure (no lost updates).
    """
    import os

    from azure.cosmos import CosmosClient, PartitionKey, exceptions
    from azure.core import MatchConditions

    endpoint = os.environ["COSMOS_ENDPOINT"]
    key = os.environ["COSMOS_MASTER_KEY"]
    client = CosmosClient(url=endpoint, credential=key, consistency_level="Session")
    db = client.create_database_if_not_exists(id="consistency_bench")
    container = db.create_container_if_not_exists(
        id="rmw_counter", partition_key=PartitionKey(path="/pk")
    )

    item_id = uuid.uuid4().hex[:8]
    container.upsert_item({"id": item_id, "pk": item_id, "value": 0})

    retries = 0
    retries_lock = threading.Lock()

    def work(_idx: int) -> None:
        nonlocal retries
        for _ in range(increments_per_agent):
            while True:
                item = container.read_item(item=item_id, partition_key=item_id)
                item["value"] = item["value"] + 1
                time.sleep(think_delay)
                try:
                    if guarded:
                        container.replace_item(
                            item=item_id,
                            body=item,
                            etag=item["_etag"],
                            match_condition=MatchConditions.IfNotModified,
                        )
                    else:
                        container.replace_item(item=item_id, body=item)
                    break
                except exceptions.CosmosAccessConditionFailedError:
                    with retries_lock:
                        retries += 1
                    continue  # ETag mismatch: re-read and retry

    wall = _run_workers(concurrency, work)
    final = container.read_item(item=item_id, partition_key=item_id)
    observed = final["value"]
    expected = concurrency * increments_per_agent
    lost = expected - observed
    return BehavioralResult(
        pattern="cosmos-etag" if guarded else "cosmos-naive",
        concurrency=concurrency,
        expected=expected,
        observed=observed,
        lost_updates=lost,
        lost_update_rate=round(lost / expected, 4) if expected else 0.0,
        retries=retries,
        wall_seconds=round(wall, 3),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_table(rows: list[BehavioralResult], title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(
        f"{'Pattern':<14}{'Agents':>7}{'Expected':>10}{'Observed':>10}"
        f"{'Lost':>7}{'Lost%':>8}{'Retries':>9}{'wall(s)':>9}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r.pattern:<14}{r.concurrency:>7}{r.expected:>10}{r.observed:>10}"
            f"{r.lost_updates:>7}{r.lost_update_rate * 100:>7.1f}%{r.retries:>9}"
            f"{r.wall_seconds:>9.2f}"
        )
    print("=" * 92)


def _write_csv(rows: list[BehavioralResult], path: str) -> None:
    import csv
    from dataclasses import asdict

    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\nWrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure behavioral degradation (lost updates) of shared memory under concurrency."
    )
    parser.add_argument("--increments", type=int, default=50, help="Increments per agent.")
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32",
        help="Comma-separated agent counts to sweep.",
    )
    parser.add_argument(
        "--think-delay",
        type=float,
        default=0.0002,
        help="Seconds an agent 'reasons' between read and write (widens the race window).",
    )
    parser.add_argument("--cosmos", action="store_true", help="Also run the live Cosmos RMW demo.")
    parser.add_argument("--out", default="behavioral.csv", help="CSV output path.")
    args = parser.parse_args(argv)

    concurrency_levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    patterns = ["mutable", "locked", "cas", "append"]

    rows: list[BehavioralResult] = []
    for pattern in patterns:
        for c in concurrency_levels:
            rows.append(
                run_pattern(
                    pattern,
                    concurrency=c,
                    increments_per_agent=args.increments,
                    think_delay=args.think_delay,
                )
            )

    _print_table(
        rows,
        "IN-PROCESS: SHARED-MEMORY CORRECTNESS vs CONCURRENCY (lost updates = dropped agent contributions)",
    )

    if args.cosmos:
        import os

        if not os.getenv("COSMOS_ENDPOINT") or not os.getenv("COSMOS_MASTER_KEY"):
            print("\n[skip] --cosmos set but COSMOS_ENDPOINT/COSMOS_MASTER_KEY missing.")
        else:
            cosmos_rows: list[BehavioralResult] = []
            for guarded in (False, True):
                for c in [c for c in concurrency_levels if c <= 8]:
                    cosmos_rows.append(
                        run_cosmos_rmw(
                            concurrency=c,
                            increments_per_agent=min(args.increments, 20),
                            guarded=guarded,
                            think_delay=0.0,
                        )
                    )
            _print_table(
                cosmos_rows,
                "LIVE COSMOS DB: read-modify-write of one item (naive replace vs ETag optimistic concurrency)",
            )
            rows.extend(cosmos_rows)

    _write_csv(rows, args.out)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
