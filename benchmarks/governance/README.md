# Shared-Memory Governance Benchmark (SMGB)

A system-agnostic benchmark for evaluating **shared** agent memory — memory
reused across agents, users, teams, tenants, and organizations — on the
**governance** axes that single-user memory benchmarks (LoCoMo, LongMemEval,
DMR) do not test.

Single-user benchmarks ask *"did the system recall the right fact?"* SMGB adds
the question that matters once memory is shared: *"did the system return the
right fact **to the right principal, at the right time, and nothing it was not
entitled to**?"*

> **Two questions, two harnesses.** SMGB scores *governance* — are shared reads
> *authorized and scoped correctly* (the **right writes for this reader**)? A
> companion multi-agent **consistency** harness (tracked separately) scores
> whether shared writes are *durable and current* under concurrency (the **right
> latest writes**, per Golab et al.). Storage-correct ≠ governance-correct; a
> full evaluation needs both.

---

## Why this exists

There is **no public benchmark** for agentic shared-memory governance. Every
in-use memory benchmark is single-user. The two 2026 papers that touch the space
each cover only a slice:

- **PiSAs** (arXiv:2607.05318) — cross-user privacy spillage, but the
  access-control slice only, and labels are **LLM-judged**.
- **Governed Shared Memory / ArgusFleet** (arXiv:2606.24535) — the full
  governance slice (leakage, stale propagation, contradiction, provenance) but
  tied to a **proprietary** service; no public dataset.

SMGB's differentiator: labels are **derived deterministically** from a scope +
policy graph (no LLM judge), so scoring is reproducible and cheap. See
[`Docs/shared_memory_governance_benchmark.md`](../../Docs/shared_memory_governance_benchmark.md)
for the full design and gap analysis.

---

## Quick start

Run from the repo root with the package importable:

```bash
# score the built-in reference runners (oracle + two naive baselines) on the seed set
python -m benchmarks.governance.run_governance --reference

# validate the dataset only (checks referential integrity + hand labels vs policy)
python -m benchmarks.governance.run_governance --validate-only

# score an external system from a run file
python -m benchmarks.governance.run_governance --run my_run.json --system my_system
```

Reference leaderboard on the seed set:

```
system        queries mean_recall leak_rate total_leaks isolation_violations stale_leaks
-----------------------------------------------------------------------------------------
oracle        14      1.000       0.000     0           0                    0
naive_shared  14      1.000       0.500     13          0                    1
naive_global  14      1.000       0.643     15          2                    1
```

The takeaway the benchmark is built to make visible: **recall alone cannot tell
a governed system from a naive one** — all three score `mean_recall = 1.000`.
The leakage, isolation, and stale-propagation axes are what separate them.

---

## The seven axes

| Axis | Question | Metric |
|------|----------|--------|
| `utility` | Does the principal get the facts they're entitled to? | `recall@k` over authorized-and-relevant targets |
| `leakage` | Are unauthorized facts withheld? | `leak_rate` (queries with any forbidden hit), `total_leaks` |
| `isolation` | Is cross-tenant memory never returned? | `isolation_violations` (must be 0) |
| `promotion` | Are promoted facts visible after `T` to the new scope, not before? | recall/leak evaluated at `as_of` before vs after promotion |
| `conflict` | In `current` mode, only the live version is retrievable? | `stale_leaks` (superseded "ghost" hits) |
| `deletion` | Are deleted facts (and copies) gone for all scope members? | leak on deletion-axis queries |
| `provenance` | Does every shared fact trace to the right author/source? | `provenance_accuracy` |

---

## How scoring works

A **run** is a plain mapping — no SDK, no model calls required to score:

```json
{ "<scenario_id>": { "<query_id>": ["<memory_id>", "..."] } }
```

For each query the scorer:

1. computes the memory's **state as of the query time** — current scope set
   (initial scope plus any `promote` targets), plus `superseded` / `deleted`
   flags — from the event timeline (`policy.memory_state`);
2. derives **authorized** (exists, not deleted, same tenant, scope overlap) and
   **allowed** (authorized and — outside `history` mode — not superseded), and
   labels everything else **forbidden** (`policy.compute_query_labels`);
3. intersects the run with those label sets to produce per-axis metrics
   (`scorer.score_scenario`).

Because the labels come from the policy layer, not a judge, the dataset is
self-checking: the loader **validates every hand annotation against the derived
labels** (`loader.validate_scenario`) and refuses a scenario whose
`must_retrieve` / `must_not_retrieve` disagree with policy.

---

## Package layout

| File | Role |
|------|------|
| `schema.py` | Dataclasses: `Scope`, `Principal`, `Provenance`, `Memory`, `Event`, `Query`, `Scenario`; `parse_time`, `QUERY_AXES`, `EVENT_TYPES`. |
| `policy.py` | Deterministic core: `memory_state`, `is_authorized`, `compute_query_labels`. All labels derive here. |
| `scorer.py` | System-agnostic scoring, the `Report` aggregates, `merge_reports`, and the reference runners. |
| `loader.py` | `load_scenarios`, `validate_scenario` / `validate_all` (referential integrity + hand labels vs policy). |
| `run_governance.py` | CLI: `--reference`, `--run`, `--validate-only`, `--json-out`, `--k`. |
| `data/seed_scenarios.jsonl` | The checked-in seed dataset (6 scenarios, all validated). |
| `data/_generate_seed.py` | Reproducible builder for the seed file. |

Tests live in [`../tests/`](../tests/):
`test_governance_policy.py`, `test_governance_scorer.py`,
`test_governance_seed.py`.

---

## Dataset format

One JSON object per line (JSONL). A scenario is fully self-contained:

```json
{
  "scenario_id": "acct_satya_steve",
  "tenants": ["contoso"],
  "scopes": [
    {"id": "user:satya", "kind": "user", "tenant": "contoso", "members": ["satya"]},
    {"id": "acct:northwind", "kind": "account", "tenant": "contoso", "members": ["satya", "steve"]}
  ],
  "principals": [
    {"id": "satya", "tenant": "contoso", "scopes": ["user:satya", "acct:northwind"]},
    {"id": "steve", "tenant": "contoso", "scopes": ["acct:northwind"]}
  ],
  "memories": [
    {"id": "m_pricing", "scope": "user:satya", "tenant": "contoso",
     "created_at": "2026-01-01T00:00:00Z", "provenance": {"author": "satya-agent"}}
  ],
  "events": [
    {"t": "2026-02-01T00:00:00Z", "type": "promote", "memory_id": "m_pricing", "to_scope": "acct:northwind"}
  ],
  "queries": [
    {"id": "q_before", "principal": "steve", "as_of": "2026-01-15T00:00:00Z",
     "axis": "promotion", "relevant": ["m_pricing"], "must_retrieve": []},
    {"id": "q_after", "principal": "steve", "as_of": "2026-02-15T00:00:00Z",
     "axis": "promotion", "relevant": ["m_pricing"], "must_retrieve": ["m_pricing"]}
  ]
}
```

`must_retrieve` / `must_not_retrieve` are **optional** hand annotations; when
present the loader checks them against the policy-derived labels, which is how
the dataset catches its own authoring bugs.

---

## Extending the benchmark

- **Add scenarios**: append lines to `seed_scenarios.jsonl` (or add a new
  `.jsonl` and point `--data` at the directory), then run `--validate-only`.
- **Score your system**: emit a run file in the format above and pass `--run`.
  A thin live adapter can map scopes onto AMT's real API — `tags` / `metadata`
  for scope, `include_superseded` for history mode, `created_after/before` for
  `as_of` — so the same scenarios can be replayed against Cosmos DB.
- **Add an axis or event type**: extend `QUERY_AXES` / `EVENT_TYPES` in
  `schema.py`, teach `policy.py` how it changes state or labels, and add a
  metric in `scorer.py`.
