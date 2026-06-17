# Multi-Agent Shared Memory Evaluation

This folder contains a benchmarking framework for evaluating shared memory performance in multi-agent systems, integrated with [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench).

## Overview

**Three core pieces:**

1. **Adapter** (`memoryagentbench/adapter.py`) — Extensions to the MAB agent to support multi-agent shared-memory modes:
   - `memory_toolkit_writer_agents` — multiple agents write to the same shared `(user_id, thread_id)`, round-robin attributed.
   - `memory_toolkit_query_agents` — optionally restrict retrieval to a subset (per-agent-filtered) or empty (shared).

2. **Consistency Harness** (`consistency/`) — Client-centric staleness and anomaly metrics inspired by Golab et al. (ICDCS 2014):
   - **Δ-atomicity** (time staleness) — how long a read lags behind the newest durable version.
   - **k-atomicity** (version staleness) — how many newer versions a read missed.
   - **Session guarantees** — read-your-writes and monotonic-reads anomaly counts.

3. **Orchestrator** (`orchestrate.py`) — Single command that runs the full pipeline: MAB accuracy → consistency sweep → combined report.

## Quick Start

### Single-Command Pipeline

```bash
python -m benchmarks.orchestrate \
    --mab-agent-config path/to/agent_config.yaml \
    --mab-dataset-config path/to/dataset_config.yaml \
    --out-dir ./eval_run
```

Produces:
- `eval_run/leaderboard.csv` — consistency levels × concurrency staleness/anomalies
- `eval_run/combined_report.json` — per-run accuracy + consistency alignment
- `eval_run/combined.csv` — flat per-run summary

### Individual Steps

**MemoryAgentBench accuracy run:**
```bash
cd /path/to/MemoryAgentBench
python -m benchmarks.memoryagentbench.run_benchmark \
    --memoryagentbench . \
    --agent-config path/to/agent_config.yaml \
    --dataset-config path/to/dataset_config.yaml
```

**Consistency sweep:**
```bash
python -m benchmarks.consistency.sweep \
    --mode simulated \
    --levels strong,session,eventual \
    --concurrency 1,2,4 \
    --ops 200 \
    --out leaderboard.csv
```

**Combine results:**
```bash
python -m benchmarks.combine_report \
    --mab outputs/agent_memory_toolkit \
    --leaderboard leaderboard.csv \
    --level eventual \
    --out combined_report.json --csv combined.csv
```

## Multi-Agent Shared Memory Mode

In MemoryAgentBench agent config, enable multi-agent shared memory:

```yaml
# Standard MAB config...
agent_name: agent_memory_toolkit
model: gpt-4o-mini

# Multi-agent knobs (optional)
memory_toolkit_writer_agents: "researcher,planner"  # Round-robin writes on shared thread
memory_toolkit_query_agents: ""                      # Empty = shared read; or "planner" for filtered
memory_toolkit_store_mode: facts_only               # Or turns_only, summary_plus_facts, user_summary
memory_toolkit_search_mode: hybrid                  # Or vector
```

When `writer_agents` has >1 agent, each turn written to MAB is attributed to one of those agents (round-robin) on the **same** `(user_id, thread_id)`. This exercises the toolkit's actual shared-memory model (hierarchical Cosmos partition).

## Consistency Metrics

Each consistency row reports, for one `(level, concurrency)` cell:

| Metric | Meaning |
|--------|---------|
| `stale_read_rate` | Fraction of reads that returned a version older than the newest durable one |
| `delta_max` | Longest time a stale read lagged (seconds) |
| `delta_p95` | 95th percentile lag (seconds) |
| `k_max` | Most versions a single read missed |
| `k_mean` | Average versions missed per stale read |
| `read_your_writes_violations` | Agent failed to see its own prior writes |
| `monotonic_reads_violations` | Agent's reads went backwards in version order |

## Combined Report Schema

**JSON** (`combined_report.json`):
```json
{
  "generated_at": "2026-06-16T...",
  "run_id_filter": null,
  "consistency_level": "eventual",
  "runs": [
    {
      "run_id": "smoke",
      "agent_name": "agent_memory_toolkit",
      "dataset": "Ruler",
      "concurrency": 2,
      "accuracy": { "accuracy": 80.5, "sub_em": 75.2 },
      "consistency_summary": {
        "level": "eventual",
        "concurrency": 2,
        "stale_read_rate": 0.95,
        "delta_p95": 0.00087,
        "k_max": 33,
        ...
      },
      ...
    }
  ],
  "consistency": [
    { "level": "strong", "concurrency": 1, ... },
    { "level": "eventual", "concurrency": 4, ... },
    ...
  ]
}
```

**CSV** (`combined.csv`):
Flat per-run rows with columns: `run_id`, `agent_name`, `dataset`, `concurrency`, `primary_accuracy_metric`, `primary_accuracy_value`, `cons_level`, `cons_stale_read_rate`, `cons_delta_p95`, etc.

## Cross-Framework Comparison (Real Backends)

`real_comparison.py` drives the *actual* storage engines other frameworks ship
with, scoring each through the same Golab et al. analyzer:

| Framework | Backend exercised |
|-----------|-------------------|
| LangChain | `SQLChatMessageHistory` on SQLite (ACID) |
| ChromaDB | local persistent vector store (metadata read) |
| AgentMemoryToolkit | live Cosmos DB (session) via `--cosmos` |
| Control (eventual) | in-memory register + injected visibility delay |

```bash
# Local backends only:
python -m benchmarks.real_comparison --agents 4 --ops 40 --keys 3

# Add live Cosmos DB (needs COSMOS_ENDPOINT + COSMOS_MASTER_KEY):
python -m benchmarks.real_comparison --agents 4 --ops 40 --keys 3 --cosmos
```

The **control** is a positive control: an in-memory register with a small
artificial visibility delay. It reports ~100% staleness, validating that the
harness detects staleness when it exists — so a 0% result from a real backend
means *consistent*, not *unmeasured*. A measured sample is checked in at
[results/real_comparison_sample.csv](results/real_comparison_sample.csv); see
[../Docs/shared_memory_consistency_research.md](../Docs/shared_memory_consistency_research.md)
for the full analysis.

## Behavioral Degradation (Lost Updates)

Read-consistency metrics (Δ, k) ask "does this read see the latest write?" But
agents **read-modify-write** shared memory, and that pattern loses contributions
under concurrency *even when every read is consistent*. `behavioral.py` measures
this directly — the application-level outcome that actually degrades agents.

```bash
# In-process sweep across mutate-in-place vs. mitigations:
python -m benchmarks.behavioral --increments 50 --concurrency 1,2,4,8,16,32

# Add the live Cosmos read-modify-write demo (naive vs ETag):
python -m benchmarks.behavioral --increments 15 --concurrency 1,4,8 --cosmos
```

| Pattern | Model | Lost updates under concurrency |
|---------|-------|-------------------------------|
| `mutable` | mutate-in-place (last-writer-wins) | **up to ~97%** |
| `locked` | serialized read-modify-write | 0% (throughput-limited) |
| `cas` | optimistic concurrency / ETag retry | 0% (retry cost) |
| `append` | append-only (toolkit `add_cosmos`) | 0% (fast) |
| `cosmos-naive` | live Cosmos replace, no check | **50–70%** |
| `cosmos-etag` | live Cosmos `If-Match` retry | 0% |

**Takeaway:** prefer append-only shared memory; if mutating in place, use
optimistic concurrency. Read-consistency alone does not prevent lost agent work.
Sample: [results/behavioral_sample.csv](results/behavioral_sample.csv).

## Testing

```bash
python -m pytest benchmarks/tests -q
```

Currently 32 tests covering adapter multi-agent modes, consistency analysis, probe, sweep, report combination, and behavioral lost-update patterns.

## Implementation Notes

- **Clock precision**: The probe uses `time.perf_counter` (sub-microsecond resolution) to measure operation order. `time.monotonic` (~15ms on Windows) produces tied timestamps and false-positive staleness.
- **Simulated consistency**: In `simulated` mode, each level maps to a visibility delay. Real Cosmos sweeps require configuring the account's consistency level (not exposed by the package client).
- **Partition key**: Cosmos memories partition on hierarchical `["/user_id", "/thread_id"]`, so all agents on the same thread share one register.
- **Agent attribution**: Written via `metadata.agent_id` (and optionally a `"agent:<id>"` tag). Queryable post-hoc via get_memories filters.

## References

- Golab, Papadopoulos, Shah, Gupta. "Client-Centric Benchmarking of Eventual Consistency for Cloud Storage Systems." ICDCS 2014.
- MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- AgentMemoryToolkit: https://github.com/AzureCosmosDB/AgentMemoryToolkit
