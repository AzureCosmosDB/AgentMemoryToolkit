# Shared-Memory Governance Benchmark (SMGB) — Design

**Status:** Draft for team review
**Author:** Muntasir Raihan Rahman (AMT)
**Scope:** A public, system-agnostic benchmark for the *governance* of shared
agent memory. Complements the single-user accuracy benchmarks reported in
PR #31 (LongMemEval, LoCoMo, BEAM 1M). It is designed to sit alongside a
companion multi-agent **consistency** harness (tracked separately), which
measures the durability and staleness of shared *writes* rather than the
governance of shared *reads*.

Reference implementation: [`benchmarks/governance/`](../benchmarks/governance/).

---

## 1. Problem

PR #31 reports AMT scores on **single-user** memory benchmarks. Those measure
one question: *did the system recall the right fact for the one user in the
transcript?* They say nothing about what happens when memory is **shared** —
reused across agents, users, teams, tenants, and organizations — which is
precisely AMT's differentiator and the harder engineering problem.

Once memory is shared, correctness is no longer just recall. It is **governance**:

- **Authorization-filtered retrieval** — a reader receives only memories they are
  entitled to, not everything in the store.
- **Private → shared promotion** — a fact authored privately becomes visible to a
  broader scope *after* an explicit promotion, and not a moment before.
- **Tenant isolation** — memory never crosses an organization boundary, even when
  scope names or questions collide.
- **Conflict / supersession** — when a fact is updated, the current view returns
  only the live version; stale "ghosts" do not resurface.
- **Scope-aware deletion** — a deleted memory disappears for *every* member of its
  scope, with no retrievable copies.
- **Provenance** — every shared fact traces to its author, source, and time.

A system can ace every single-user benchmark and still leak a private note to a
teammate, surface a deleted workaround, or serve tenant A's revenue target to
tenant B. Nothing in the current evaluation stack would catch it.

---

## 2. Gap analysis (verified)

**No dedicated, publicly released agentic-shared-memory governance benchmark
exists.** Every benchmark in active use is single-user; only two 2026 papers
partially address the space, and neither is a usable public dataset for this
problem.

### 2.1 In-use benchmarks are single-user

| Benchmark | arXiv | Shared memory? | Governance? |
|---|---|---|---|
| LoCoMo (ACL 2024) | 2402.17753 | No — dyadic, single user | No |
| LongMemEval (ICLR 2025) | 2410.10813 | No — one user ↔ assistant | No |
| DMR (MemGPT, 2023) | 2310.08560 | No — single-user recall | No |
| BEAM | *unverified* | *unverified* | *unverified* |

> BEAM (cited in PR #31) could not be confirmed as a peer-reviewed paper during
> this review — treat as gray literature and verify the source before relying on
> it in external comparisons.

### 2.2 Two 2026 papers cover only a slice (both verified via the arXiv API)

- **PiSAs — "Benchmarking Contextual Integrity in Multi-User Agentic Systems"**
  (arXiv:2607.05318, Meta AI / Mila). Measures cross-user information spillage,
  including through shared memory. **System-agnostic**, which is the right shape.
  - *Covers:* the access-control / leakage slice.
  - *Missing:* promotion, temporal supersession / stale propagation, provenance,
    scope-aware deletion.
  - *Limitation:* labels rely on **LLM contextual-integrity judgment** — noisy,
    non-reproducible, and expensive to run.

- **"Governed Shared Memory for Multi-Agent LLM Systems"** (MemClaw service +
  **ArgusFleet** harness, arXiv:2606.24535). Formalizes the fleet-memory problem
  and exactly the four failure modes SMGB targets — *unauthorized leakage, stale
  propagation, contradiction persistence, provenance collapse.*
  - *Covers:* the full governance slice — the dimensions are right.
  - *Limitation:* it evaluates a **proprietary production service**, not a
    standalone public dataset. It cannot be pointed at an arbitrary memory system.

### 2.3 Where SMGB sits

SMGB takes PiSAs' **system-agnostic** posture and ArgusFleet's **governance
dimensions**, and removes both of their blockers:

- **No LLM judge.** Ground truth is derived *deterministically* from a scope +
  policy graph. Scoring is reproducible, free, and unambiguous.
- **No proprietary dependency.** The dataset is plain JSONL and the scorer is a
  dependency-free Python package that consumes a mapping of retrieved ids.

This is the differentiator: a **reproducible, public, policy-labeled** governance
benchmark that any memory system can be scored against.

---

## 3. Design principles

1. **System-agnostic.** A system under test produces a *run*: for each query, a
   ranked list of retrieved memory ids. The scorer needs nothing else — no model
   calls, no SDK, no network.
2. **Deterministic labels.** For any principal at any time, the *allowed* and
   *forbidden* memory sets are computed from the scope graph and an event
   timeline. Labels are never hand-assigned in a way the harness trusts blindly.
3. **Self-checking data.** Scenarios may carry hand annotations
   (`must_retrieve` / `must_not_retrieve`). The loader **validates every one
   against the policy-derived labels** and rejects any mismatch, so authoring
   bugs cannot silently corrupt the benchmark.
4. **Time is first-class.** Every query has an `as_of` timestamp; promotion,
   supersession, and deletion are timeline events. The same question asked before
   and after a promotion has different correct answers.
5. **Complementary, not redundant.** The consistency harness asks "is this read
   the *latest write*?" SMGB asks "is this read the *right write for this
   reader*?" Both are necessary; neither implies the other.

---

## 4. Model

### 4.1 Entities

- **Scope** — a visibility unit (`user`, `agent`, `session`, `project`, `group`,
  `account`, `org`, `tenant`), with an owning tenant and optional member list.
- **Principal** — a user or agent that issues reads/writes, with a tenant, roles,
  and the set of scopes it belongs to.
- **Memory** — a record with a birth time (`created_at`), an **initial** scope,
  a tenant, and provenance (author, source, confidence, `derived_from`).
- **Event** — a governance mutation on the timeline: `promote` (adds a broader
  scope), `supersede` (a newer memory replaces it), `delete`.
- **Query** — a read by a principal at `as_of`, tagged with one **axis** and an
  optional `temporal_mode` (`current` default, or `history`).

Governance state is **not** stored on the memory; it is computed as of a query
time from the event timeline, keeping the dataset a single source of truth.

### 4.2 Policy (the deterministic core)

State of a memory as of `as_of`:

- **exists** — `created_at <= as_of`.
- **scopes** — `{initial scope} ∪ {promote.to_scope : promote.t <= as_of}`.
  Promotion *adds* a broader scope; the author keeps visibility because they are a
  member of that broader scope.
- **superseded** — some `supersede` event with `t <= as_of` targeted it.
- **deleted** — some `delete` event with `t <= as_of` targeted it.

A memory is **authorized** for a principal when, *in order*: it exists, is not
deleted, is in the **same tenant** (hard isolation, checked before scope), and
shares at least one scope with the principal. It is **allowed** for a query when
it is authorized and — unless the query is `history` mode — not superseded.
Everything else in the scenario is **forbidden**.

```
must_retrieve       = relevant ∩ allowed          # the utility target
forbidden           = all_memories − allowed
cross_tenant        = forbidden from another tenant   # isolation signal
superseded_forbidden = forbidden only because stale   # conflict signal
```

Checking tenant **before** scope matters: a cross-tenant memory is never
authorized even if two tenants happen to use the same scope id.

---

## 5. Task axes and metrics

| Axis | Question | Metric |
|------|----------|--------|
| `utility` | Does the principal get the facts they're entitled to? | `recall@k` over `must_retrieve` |
| `leakage` | Are unauthorized facts withheld? | `leak_rate` (queries with ≥1 forbidden hit); `total_leaks` (item count) |
| `isolation` | Is cross-tenant memory never returned? | `isolation_violations` (target: 0) |
| `promotion` | Are promoted facts visible after `T`, not before? | recall/leak at `as_of` before vs after the promote event |
| `conflict` | In `current` mode, only the live version is retrievable? | `stale_leaks` (superseded hits) |
| `deletion` | Are deleted facts (and copies) gone for all members? | leak on deletion-axis queries |
| `provenance` | Does every shared fact trace to the right author/source? | `provenance_accuracy` |

**Headline metrics** reported per system: `mean_recall`, `leak_rate`,
`total_leaks`, `isolation_violations`, `stale_leaks`, `provenance_accuracy`,
plus a per-axis rollup.

---

## 6. Data format

One JSON object per line (JSONL); each scenario is fully self-contained. See the
package [README](../benchmarks/governance/README.md#dataset-format) for a
complete example. A scenario declares `tenants`, `scopes`, `principals`,
`memories`, `events`, and `queries`; a query names its `principal`, `as_of`,
`axis`, `relevant` ids, optional `temporal_mode`, and optional
`must_retrieve` / `must_not_retrieve` annotations.

### 6.1 Seed dataset

Six scenarios ship in `benchmarks/governance/data/seed_scenarios.jsonl` (14
queries), each anchored in a concrete situation:

| Scenario | Governance property under test |
|----------|-------------------------------|
| `acct_satya_steve` | Account-scope sharing + private notes stay private (promotion, utility, leakage) |
| `proj_scott_deploy` | Promote a deploy fix to project scope; working prefs stay private |
| `tenant_isolation_revenue` | Two tenants, identically shaped facts; neither may see the other's |
| `conflict_contact_supersession` | Current-vs-history retrieval of a superseded fact |
| `deletion_workaround` | A deleted shared memory is gone for all scope members |
| `provenance_reranker` | A shared decision traces to its author/source; derived fact keeps its parent link |

The first two encode the original whiteboard brainstorm (account- and
project-scoped promotion). The seed set is small by design — its job is to
**anchor the axes and prove the metrics discriminate**, not to be a leaderboard.

---

## 7. Reference runners and the key result

Three baseline "systems" are included so the metrics can be validated offline and
to anchor a leaderboard:

- **oracle** — returns exactly the authorized-and-relevant target set.
- **naive_shared** — returns everything in the principal's tenant, ignoring scope
  and supersession (no authorization filter).
- **naive_global** — returns everything across all tenants.

On the seed set:

```
system        queries mean_recall leak_rate total_leaks isolation_violations stale_leaks
-----------------------------------------------------------------------------------------
oracle        14      1.000       0.000     0           0                    0
naive_shared  14      1.000       0.500     13          0                    1
naive_global  14      1.000       0.643     15          2                    1
```

**This table is the whole argument for the benchmark.** All three baselines score
`mean_recall = 1.000` — a recall-only benchmark (i.e. every single-user benchmark)
would rate them identical. Only the leakage, isolation, and stale-propagation axes
reveal that `naive_shared` leaks 13 facts and `naive_global` additionally breaches
tenant isolation twice. **Recall cannot distinguish a governed memory system from
a naive one; SMGB can.**

---

## 8. Harness integration

- **Package:** `benchmarks/governance/` — `schema.py`, `policy.py`, `scorer.py`,
  `loader.py`, `run_governance.py`, `data/`.
- **CLI:** `python -m benchmarks.governance.run_governance --reference`
  (or `--validate-only`, or `--run my_run.json --system name`, with `--k` and
  `--json-out`).
- **Tests:** `benchmarks/tests/test_governance_{policy,scorer,seed}.py` — policy
  truth tables, scorer discrimination, and seed-dataset integrity. Run with
  `python -m pytest benchmarks/tests -q`; lint with `ruff check benchmarks/governance`.
- **Live adapter (future).** SMGB maps cleanly onto AMT's public API: scope encoded
  via `add_cosmos(..., tags=, metadata=)`; authorization-filtered retrieval via
  `search_cosmos(..., tags_all/tags_any/exclude_tags)`; `history` mode via
  `include_superseded`; `as_of` via `created_after/before`. A thin adapter that
  replays scenarios against Cosmos DB turns SMGB into a live governance test for
  the toolkit itself.

---

## 9. Differentiation summary

| | Single-user (LoCoMo, LongMemEval, DMR) | PiSAs (2607.05318) | ArgusFleet (2606.24535) | **SMGB** |
|---|---|---|---|---|
| Shared memory | No | Partial | Yes | **Yes** |
| Authorization / leakage | — | Yes | Yes | **Yes** |
| Promotion (private→shared) | — | No | Yes | **Yes** |
| Tenant isolation | — | Partial | Yes | **Yes** |
| Supersession / stale | — | No | Yes | **Yes** |
| Scope-aware deletion | — | No | Partial | **Yes** |
| Provenance | — | No | Yes | **Yes** |
| Labels | reference answers | **LLM-judged** | policy (proprietary) | **deterministic, public** |
| Public dataset | Yes | Yes | **No** | **Yes** |

---

## 10. Open questions for review

1. **Scenario coverage.** The seed set is six scenarios. Which real AMT
   customer/agent situations should the v1 corpus prioritize (support handoff,
   multi-tenant SaaS, project teams, org rollouts)?
2. **Scope taxonomy.** Is `user / agent / session / project / group / account /
   org / tenant` the right ladder, or do we need finer role-based scopes?
3. **Live adapter scope.** Ship the read-only Cosmos adapter in v1, or keep v1
   offline (dataset + scorer) and add the adapter in a follow-up?
4. **Provenance depth.** The schema supports `derived_from` chains (cf.
   ArgusFleet's depth-N reconstruction). Do we score multi-hop provenance in v1?
5. **Naming / publication.** If we intend to release this publicly, confirm the
   name and whether it should be positioned as an AMT artifact or a standalone
   benchmark.
