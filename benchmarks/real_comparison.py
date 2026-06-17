"""Real cross-framework shared-memory consistency benchmark.

Unlike ``comparison_harness.py`` (which sketches stub adapters), this module
drives the *actual* storage engines each framework uses, so the staleness it
measures is real:

* **LangChain**  -> ``SQLChatMessageHistory`` on SQLite (ACID / strong).
* **ChromaDB**   -> local persistent vector store (HNSW indexing -> real lag).
* **AgentMemoryToolkit** -> live Cosmos DB via ``CosmosStoreAdapter`` (session).

Each adapter implements the probe's register interface
(``write(key, version, agent_id)`` / ``read_latest(key, agent_id)``) so the same
Golab et al. analyzer scores every framework identically.

Run::

    python -m benchmarks.real_comparison --agents 4 --ops 60
    # include live Cosmos (needs COSMOS_ENDPOINT + COSMOS_MASTER_KEY):
    python -m benchmarks.real_comparison --agents 4 --ops 60 --cosmos
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import threading
import time
from typing import Optional

from .consistency.analyzer import analyze
from .consistency.probe import run_probe


# ---------------------------------------------------------------------------
# LangChain — SQLite-backed chat history (real ACID persistence)
# ---------------------------------------------------------------------------
class LangChainSQLiteAdapter:
    """Register store over LangChain ``SQLChatMessageHistory`` on SQLite.

    Each write appends a ``key=version`` message to a shared session; a read
    parses the message history and returns the highest version seen for a key.
    SQLite is transactional, so this represents a strong-consistency baseline
    backed by real disk I/O (not an in-memory dict).
    """

    framework_name = "LangChain"
    backend_type = "sqlite"

    def __init__(self, db_path: str, session_id: str = "shared_thread") -> None:
        from langchain_community.chat_message_histories import SQLChatMessageHistory

        # busy timeout lets concurrent writers wait for SQLite's single-writer
        # lock instead of erroring out.
        self._db_url = f"sqlite:///{db_path}?timeout=30"
        self._session_id = session_id
        self._History = SQLChatMessageHistory
        # One history handle per thread (SQLite connections aren't shareable).
        self._local = threading.local()
        # SQLite serializes writes; an explicit lock avoids table-creation races
        # and "database is locked" errors under concurrency.
        self._write_lock = threading.Lock()
        # Bootstrap the schema once up front so worker threads never race on
        # CREATE TABLE.
        self._History(session_id=self._session_id, connection=self._db_url)

    def _history(self):
        h = getattr(self._local, "h", None)
        if h is None:
            h = self._History(session_id=self._session_id, connection=self._db_url)
            self._local.h = h
        return h

    def write(self, key: str, version: int, agent_id: str) -> None:
        with self._write_lock:
            self._history().add_user_message(f"{key}={version}|{agent_id}")

    def read_latest(self, key: str, agent_id: str) -> Optional[int]:
        best: Optional[int] = None
        for msg in self._history().messages:
            content = getattr(msg, "content", "")
            if content.startswith(f"{key}="):
                try:
                    v = int(content.split("=", 1)[1].split("|", 1)[0])
                except (ValueError, IndexError):
                    continue
                if best is None or v > best:
                    best = v
        return best


# ---------------------------------------------------------------------------
# ChromaDB — local persistent vector store (real HNSW indexing -> staleness)
# ---------------------------------------------------------------------------
class _HashEmbedding:
    """Deterministic 16-dim embedding so Chroma never downloads an ONNX model.

    Embeds on the *key prefix* (text before ``=``) so every version of a key
    maps to the same vector. That lets the vector-read path retrieve all
    versions of a key by similarity, while still avoiding the heavyweight ONNX
    model. Metadata reads do not depend on the embedding at all.
    """

    _DIM = 16

    def name(self) -> str:  # Chroma >=0.5 requires a stable name.
        return "hash16"

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        key = text.split("=", 1)[0]
        h = hashlib.sha256(key.encode("utf-8")).digest()
        return [h[i] / 255.0 for i in range(self._DIM)]

    def __call__(self, input):
        return [self._embed_one(t) for t in input]

    # Newer Chroma splits document vs. query embedding entry points.
    def embed_documents(self, input):
        return [self._embed_one(t) for t in input]

    def embed_query(self, input):
        return [self._embed_one(t) for t in input]


class ChromaDBAdapter:
    """Register store over a real ChromaDB collection.

    Writes upsert a document carrying ``{key, version, agent_id}`` metadata.
    Two read paths are exposed:

    * ``read_mode="metadata"`` — a synchronous ``get(where=...)`` metadata
      filter that never touches the vector index, so it is strongly consistent.
    * ``read_mode="vector"`` — a similarity ``query(...)`` that goes through the
      HNSW index. Chroma indexes asynchronously (controlled by
      ``hnsw:sync_threshold``), so freshly added vectors can briefly be absent
      from query results — surfacing genuine async-index staleness.
    """

    framework_name = "ChromaDB"

    def __init__(
        self,
        path: str,
        collection: str = "shared_memory",
        read_mode: str = "metadata",
        sync_threshold: int = 1000,
    ) -> None:
        import chromadb

        self._read_mode = read_mode
        self.backend_type = f"chroma-{read_mode}"
        self._client = chromadb.PersistentClient(path=path)
        self._embed = _HashEmbedding()
        # A high sync threshold keeps writes in the in-memory brute-force buffer
        # longer before they are flushed into the HNSW graph, widening the
        # window in which vector queries can miss recent writes.
        self._col = self._client.get_or_create_collection(
            name=collection,
            embedding_function=self._embed,
            metadata={"hnsw:sync_threshold": sync_threshold},
        )
        self._seq = 0
        self._seq_lock = threading.Lock()

    def _next_id(self) -> str:
        with self._seq_lock:
            self._seq += 1
            return f"doc_{self._seq}"

    def write(self, key: str, version: int, agent_id: str) -> None:
        self._col.add(
            ids=[self._next_id()],
            documents=[f"{key}={version}"],
            metadatas=[{"key": key, "version": version, "agent_id": agent_id}],
        )

    def read_latest(self, key: str, agent_id: str) -> Optional[int]:
        if self._read_mode == "vector":
            return self._read_vector(key)
        return self._read_metadata(key)

    def _read_metadata(self, key: str) -> Optional[int]:
        res = self._col.get(where={"key": key})
        metas = res.get("metadatas") or []
        versions = [m.get("version") for m in metas if isinstance(m.get("version"), int)]
        return max(versions) if versions else None

    def _read_vector(self, key: str) -> Optional[int]:
        # Similarity query routed through the HNSW index. n_results is generous
        # so all of a key's versions can be returned once indexed.
        res = self._col.query(
            query_texts=[key],
            n_results=256,
            where={"key": key},
        )
        metas_batches = res.get("metadatas") or [[]]
        metas = metas_batches[0] if metas_batches else []
        versions = [m.get("version") for m in metas if isinstance(m.get("version"), int)]
        return max(versions) if versions else None


# ---------------------------------------------------------------------------
# AgentMemoryToolkit — live Cosmos DB
# ---------------------------------------------------------------------------
def build_cosmos_adapter():
    """Create a real ``CosmosStoreAdapter`` from environment credentials."""
    from .consistency.probe import CosmosStoreAdapter

    try:
        from azure.cosmos.agent_memory import CosmosMemoryClient
    except ImportError:
        from agent_memory_toolkit import CosmosMemoryClient

    endpoint = os.environ["COSMOS_ENDPOINT"]
    master_key = os.environ["COSMOS_MASTER_KEY"]
    client = CosmosMemoryClient(
        cosmos_endpoint=endpoint,
        cosmos_key=master_key,
        cosmos_database="agent_memory_db",
        cosmos_container="memory",
        use_default_credential=False,
        ai_foundry_endpoint=None,
    )
    adapter = CosmosStoreAdapter(client, user_id="bench_user", thread_id="bench_thread")
    adapter.framework_name = "AgentMemoryToolkit"
    adapter.backend_type = "cosmos-session"
    return adapter


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def benchmark(adapter, *, agents: int, ops_per_agent: int, keys: list[str]) -> dict:
    name = getattr(adapter, "framework_name", adapter.__class__.__name__)
    backend = getattr(adapter, "backend_type", "?")
    print(f"\nBenchmarking {name} ({backend})...")

    agent_ids = [f"agent_{i}" for i in range(agents)]
    t0 = time.perf_counter()
    trace = run_probe(
        store=adapter,
        agents=agent_ids,
        keys=keys,
        ops_per_agent=ops_per_agent,
    )
    wall = time.perf_counter() - t0
    rep = analyze(trace)
    print(f"  {rep.summary()}")
    print(f"  wall={wall:.2f}s")

    return {
        "framework": name,
        "backend": backend,
        "reads": rep.reads,
        "writes": rep.writes,
        "stale_reads": rep.stale_reads,
        "stale_read_pct": round(rep.stale_read_rate * 100, 2),
        "delta_max_ms": round(rep.delta_max * 1000, 3),
        "delta_mean_ms": round(rep.delta_mean * 1000, 3),
        "delta_p95_ms": round(rep.delta_p95 * 1000, 3),
        "k_max": rep.k_max,
        "k_mean": round(rep.k_mean, 3),
        "ryw_violations": rep.read_your_writes_violations,
        "monotonic_reads_violations": rep.monotonic_reads_violations,
        "wall_seconds": round(wall, 3),
    }


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("REAL MULTI-AGENT SHARED MEMORY CONSISTENCY (Golab et al. ICDCS 2014)")
    print("=" * 110)
    header = (
        f"{'Framework':<20}{'Backend':<16}{'Stale%':>8}{'d_max(ms)':>11}"
        f"{'d_p95(ms)':>11}{'k_max':>7}{'RYW':>6}{'MR':>5}{'wall(s)':>9}"
    )
    print(header)
    print("-" * 110)
    for r in rows:
        print(
            f"{r['framework']:<20}{r['backend']:<16}{r['stale_read_pct']:>8.1f}"
            f"{r['delta_max_ms']:>11.3f}{r['delta_p95_ms']:>11.3f}{r['k_max']:>7}"
            f"{r['ryw_violations']:>6}{r['monotonic_reads_violations']:>5}"
            f"{r['wall_seconds']:>9.2f}"
        )
    print("=" * 110)


def write_csv(rows: list[dict], path: str) -> None:
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Real cross-framework consistency benchmark.")
    parser.add_argument("--agents", type=int, default=4, help="Concurrent agents.")
    parser.add_argument("--ops", type=int, default=60, help="Operations per agent.")
    parser.add_argument("--keys", type=int, default=3, help="Distinct register keys.")
    parser.add_argument("--cosmos", action="store_true", help="Include live Cosmos DB.")
    parser.add_argument(
        "--control-delay",
        type=float,
        default=0.002,
        help="Visibility delay (s) for the in-memory positive control.",
    )
    parser.add_argument("--out", default="real_comparison.csv", help="CSV output path.")
    args = parser.parse_args(argv)

    keys = [f"key{i}" for i in range(args.keys)]
    workdir = tempfile.mkdtemp(prefix="amt_bench_")
    rows: list[dict] = []

    try:
        # Positive control: in-memory register with an injected visibility delay.
        # Proves the harness actually detects staleness when it exists, so a 0%
        # result from a real backend means "consistent", not "blind".
        from .consistency.probe import InMemoryStoreAdapter

        control = InMemoryStoreAdapter(visibility_delay=args.control_delay)
        control.framework_name = "Control(eventual)"
        control.backend_type = f"in-mem+{int(args.control_delay * 1000)}ms"
        rows.append(benchmark(control, agents=args.agents, ops_per_agent=args.ops, keys=keys))

        # LangChain / SQLite
        lc = LangChainSQLiteAdapter(db_path=os.path.join(workdir, "langchain.db"))
        rows.append(benchmark(lc, agents=args.agents, ops_per_agent=args.ops, keys=keys))

        # ChromaDB — synchronous metadata read (strongly consistent path)
        chroma_meta = ChromaDBAdapter(
            path=os.path.join(workdir, "chroma_meta"), read_mode="metadata"
        )
        rows.append(
            benchmark(chroma_meta, agents=args.agents, ops_per_agent=args.ops, keys=keys)
        )

        # ChromaDB — vector similarity read (async HNSW index -> real staleness)
        chroma_vec = ChromaDBAdapter(
            path=os.path.join(workdir, "chroma_vec"), read_mode="vector"
        )
        rows.append(
            benchmark(chroma_vec, agents=args.agents, ops_per_agent=args.ops, keys=keys)
        )

        # AgentMemoryToolkit / Cosmos (optional)
        if args.cosmos:
            if not os.getenv("COSMOS_ENDPOINT") or not os.getenv("COSMOS_MASTER_KEY"):
                print("\n[skip] Cosmos requested but COSMOS_ENDPOINT/COSMOS_MASTER_KEY not set.")
            else:
                cosmos = build_cosmos_adapter()
                rows.append(
                    benchmark(cosmos, agents=args.agents, ops_per_agent=args.ops, keys=keys)
                )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print_table(rows)
    write_csv(rows, args.out)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
