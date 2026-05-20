# Azure Cosmos DB Agent Memory Toolkit - Public Preview

![Agent Memory Toolkit overview](Overview.png)


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Azure Cosmos DB](https://img.shields.io/badge/Azure-Cosmos%20DB-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/products/cosmos-db/)
[![Follow on X](https://img.shields.io/twitter/follow/AzureCosmosDB?style=social)](https://twitter.com/AzureCosmosDB)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Azure%20Cosmos%20DB-0077B5?logo=linkedin)](https://www.linkedin.com/showcase/azure-cosmos-db/)
[![YouTube](https://img.shields.io/badge/YouTube-Azure%20Cosmos%20DB-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@AzureCosmosDB)


Agent Memory Toolkit is a Python SDK for storing, retrieving, and transforming agent memories on Azure Cosmos DB. It gives your agent both raw conversation history and higher-value derived memory — thread summaries, extracted facts, and cross-thread user profiles — all searchable semantically. The processing pipeline can run **in-process** (zero infra) or in a sibling **Azure Durable Function app** that watches the Cosmos DB change feed. Sync (`CosmosMemoryClient`) and async (`AsyncCosmosMemoryClient`) APIs are mirror-images of each other.


---

## Quickstart

### 1. Install

```bash
pip install .

# With dev/test dependencies
pip install ".[dev]"
```

### 2. Provision Azure resources

The toolkit needs a Cosmos DB account, an Azure OpenAI / AI Foundry deployment, and (optionally for the remote processor) an Azure Function app. Pick whichever path matches your situation:

**Option A — One-command provision (`azd up`).** Creates everything from scratch — Cosmos + AI Foundry + Function app (Flex Consumption, idle cost ≈ $0) + UAMI + RBAC — and writes a working `.env` to `.azure/<env>/.env`:

```bash
# Prereqs: az + azd installed; subscription with quota for gpt-4o-mini
# and text-embedding-3-large in your chosen region (default: eastus2,
# also supported: swedencentral, westus3).

az login
azd auth login

azd env new memorytoolkit-dev
# Optional: pin a region other than eastus2
# azd env set AZURE_LOCATION swedencentral

azd up
# ~10 min later: Cosmos account + AI Foundry account + 2 model deployments
# (gpt-4o-mini, text-embedding-3-large) + UAMI + RBAC + Function app
# are provisioned. Outputs are written to .azure/memorytoolkit-dev/.env
```

The Function app is always provisioned but only used when you opt into `DurableFunctionProcessor` — it sits idle (and bills nothing) for in-process workloads.

Load the generated env vars and you're ready to use the SDK:

```bash
set -a && . ./.azure/memorytoolkit-dev/.env && set +a
```

To tear everything down later: `azd down --purge` (the `--purge` flag skips Cosmos / AI Foundry soft-delete so names are immediately reusable).

**Option B — Bring your own resources.** If you already have a Cosmos DB account and an AI Foundry / Azure OpenAI deployment, copy the env template and fill in the endpoints:

```bash
cp .env.template .env
# edit COSMOS_DB_ENDPOINT, AI_FOUNDRY_ENDPOINT, AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME, AI_FOUNDRY_CHAT_DEPLOYMENT_NAME
```

You can also point `azd up` at existing resources via `azd env set USE_EXISTING_COSMOS true` / `USE_EXISTING_AI_FOUNDRY true` (full BYOR flag list in `infra/README.md`).

> For the Durable Function app counter-trigger settings, Bicep module reference, and RBAC scopes — see **[`infra/README.md`](infra/README.md)**.

### 3. Use the SDK

```python
import os, uuid
from dotenv import load_dotenv
from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()

memory = CosmosMemoryClient(
    cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
    cosmos_database=os.getenv("COSMOS_DB_DATABASE", "ai_memory"),
    cosmos_container=os.getenv("COSMOS_DB_CONTAINER", "memories"),
    ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
    embedding_deployment_name=os.getenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
    chat_deployment_name=os.getenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
    use_default_credential=True,
    # processor=InProcessProcessor()   # implicit default
)
memory.connect_cosmos()  # auto-creates database + containers if missing

USER, THREAD = "user-001", str(uuid.uuid4())

# Add raw turns to a conversation
memory.add_cosmos(user_id=USER, thread_id=THREAD, role="user", content="I love Cosmos DB.")
memory.add_cosmos(user_id=USER, thread_id=THREAD, role="assistant", content="It is fantastic.")

# Run the processing pipeline (thread summary + fact extraction + user summary)
memory.process_now(user_id=USER, thread_id=THREAD)

# Search semantically across the stored memory
hits = memory.search_cosmos(user_id=USER, query_text="Cosmos DB preferences", top=5)
for h in hits:
    print(h["memory_type"], "-", h["content"][:80])

# Retrieve the cross-thread user profile
print(memory.get_user_summary(user_id=USER))
```

> Async API is identical — just `await` each call:
> ```python
> from agent_memory_toolkit.aio import AsyncCosmosMemoryClient
> ```

### 4. Run a sample

```bash
python Samples/Quickstarts/quickstart_cosmos.py
```

See [`Samples/`](Samples/) for end-to-end scenarios (chat memory, RAG, multi-agent, customer support, remote processor).

---

## Concepts in 60 seconds

| Concept | What it is | API |
|---|---|---|
| **Turn** | One message (user or assistant) — the raw conversation atom | `add_cosmos(...)`, `add_local(...)` |
| **Thread summary** | LLM-generated, incrementally updated rollup of a single thread | `generate_thread_summary(...)` |
| **Fact** | Discrete, independently searchable assertion extracted from turns | `extract_memories(...)` |
| **Procedural** | Behavioral rule / instruction the user wants followed | `extract_memories(...)` |
| **Episodic** | Past situation → action → outcome experience (90-day TTL) | `extract_memories(...)` |
| **User summary** | Cross-thread profile of what's known about a user | `generate_user_summary(...)`, `get_user_summary(...)` |
| **Search** | Vector + full-text + filter; returns any of the above | `search_cosmos(...)` |
| **Process now** | Run the full pipeline (summary → facts → user profile) for recent turns | `process_now(...)`, `process_now_and_wait(...)` |

All memory kinds live in the same Cosmos container, partitioned by `(user_id, thread_id)`, distinguished by a `type` discriminator.

### Memory Type Taxonomy

The `extract_memories` pipeline classifies each item it pulls from the conversation into one of four buckets. Every memory carries a top-level `confidence` (0.0–1.0) so retrieval can suppress weakly-grounded extractions.

| Bucket | Meaning | Storage type | TTL |
|---|---|---|---|
| Fact | Declarative knowledge ("user prefers dark mode") | `type="fact"` | none |
| Procedural | Behavioral rule ("always confirm before deleting") | `type="procedural"` | none |
| Episodic | Past experience: situation → action → outcome | `type="episodic"` | 90 days |
| Unclassified | Item worth keeping but the LLM couldn't confidently classify | `type="fact"` + tag `sys:unclassified` | none |

#### Confidence Scale

| Range | Meaning |
|---|---|
| 0.9–1.0 | Directly stated and unambiguous |
| 0.7–0.9 | Clearly implied, no contradicting evidence |
| 0.5–0.7 | Inferred from context — plausible but not explicit |
| < 0.5 | Should be in `unclassified` instead |

Filter at retrieval time:

```python
results = memory.search_cosmos("user preferences", user_id="u1", min_confidence=0.7)
high_conf_facts = memory.get_memories(user_id="u1", memory_types=["fact"], min_confidence=0.7)
```

### Memory Reconciliation

`reconcile(user_id, n=50)` (on the public client; underlying pipeline method is `ProcessingPipeline.reconcile_memories`) collapses paraphrased duplicates and resolves semantic contradictions in a single LLM pass over the N most-recent active facts. Both outcomes soft-delete the loser with a `supersede_reason` of `"duplicate"` or `"contradiction"`. See [Docs/concepts.md](Docs/concepts.md#memory-reconciliation) for details.

> **Cost note.** Each reconciliation makes one LLM call covering up to `n` facts (default 50, hard cap 500). With auto-trigger, this fires every `FACT_EXTRACTION_EVERY_N × DEDUP_EVERY_N` turns per user, with `n` taken from `DEDUP_POOL_SIZE`. The previous cosine-cluster pre-filter was removed deliberately — it could not catch semantic contradictions like "vegetarian" vs "ribeye steak" — so the LLM is now invoked whenever there are ≥ 2 active facts. To bound LLM cost more tightly: raise `DEDUP_EVERY_N` (lower frequency — reconcile fires every Nth extraction, so a *higher* N means *less often*), lower `DEDUP_POOL_SIZE` (smaller per-call pool), or override `n` per call when invoking `reconcile()` directly.

| New `MemoryRecord` field | Meaning |
|---|---|
| `content_hash` | SHA-256 of normalized content; enables write-time exact-dedup short-circuit |
| `supersede_reason` | `"duplicate"` or `"contradiction"` (None for live records) |
| `superseded_at` | ISO timestamp when the supersede happened (None for live records) |
| `superseded_by` | Id of the record that replaced this one (existing field) |

### Auto-trigger (per-turn extraction)

By default, the **InProcess processor** runs each pipeline step independently as its own threshold trips inside `push_to_cosmos()`:

| Env var | Default | Step that fires | Async behavior |
|---|---|---|---|
| `FACT_EXTRACTION_EVERY_N` | `1` (every turn) | `process_extract_memories` | scheduled via `asyncio.create_task` |
| `DEDUP_EVERY_N` | `5` | `process_reconcile` (fires every Nth extract → effectively every `FACT_EXTRACTION_EVERY_N × DEDUP_EVERY_N` turns) | scheduled via `asyncio.create_task` |
| `DEDUP_POOL_SIZE` | `50` | pool size (`n`) passed to `process_reconcile` from the auto-trigger; hard-capped at `500` | n/a (per-call) |
| `THREAD_SUMMARY_EVERY_N` | `10` | `process_thread_summary` | scheduled via `asyncio.create_task` |
| `USER_SUMMARY_EVERY_N` | `20` | `process_user_summary` | scheduled via `asyncio.create_task` |

Each `*_EVERY_N=0` disables only that step. Dedup is gated independently of extract because cross-thread dedup is dramatically more expensive than per-thread extract (it reads every active fact for the user) — running it on every extract slammed AI Foundry. The Durable backend uses the same defaults via the change-feed function app (the function-app `azd` deploy bumps `FACT_EXTRACTION_EVERY_N` to `5` since the FA path is intended for higher-volume workloads). Calling `process_now()` is normally redundant — it remains as an explicit "process now" hook for tests, manual workflows, and operators who set every threshold to `0`.

The async client (`AsyncCosmosMemoryClient.push_to_cosmos`) does **not** await the auto-trigger; it schedules it as a background `asyncio.Task` so the write call returns as soon as the Cosmos upserts complete. Background failures are surfaced via `logger.warning` (search for `"Background auto-trigger task failed"`).

#### Backend exclusivity (`MEMORY_PROCESSOR_OWNER`)

Both the SDK auto-trigger and the function-app change-feed processor write into the same `counter` container. If you accidentally point an `InProcessProcessor` at a Cosmos container that already has a function app attached, both backends will run the pipeline on the same writes — double extraction, double dedup, double counters.

Set the env var on **both sides** to make ownership explicit:

| `MEMORY_PROCESSOR_OWNER` | SDK behavior | Function-app behavior |
|---|---|---|
| _unset_ (default) | runs auto-trigger | runs orchestrator (today's behavior) |
| `inprocess` | runs auto-trigger | change-feed trigger skips batch + logs |
| `durable` | auto-trigger logs warning + skips | runs orchestrator |

The default (unset) preserves backward compatibility. For any production deployment we recommend setting it on both sides so a misconfiguration produces a loud log line instead of silent double-work.

> **Advisory, not enforced.** `MEMORY_PROCESSOR_OWNER` is operator-configured exclusivity, not a server-side lock. Each backend reads its own env var; if the SDK is set to `inprocess` but the FA forgets to set `durable` (or vice versa), both still run. As a backstop, every counter write stamps `last_owner=<this backend>` on the doc — when the SDK observes a counter previously written by `durable` (or vice versa), it logs a one-shot `WARN` so misconfiguration surfaces in logs without spamming. Treat this as a configuration audit signal, not a hard guarantee.

---

## Two processor flavors

Pick at construction time via the `processor=` kwarg.

| | `InProcessProcessor` (default) | `DurableFunctionProcessor` |
|---|---|---|
| Infra | None — just `pip install` | Sibling Azure Function app |
| Best for | Prototypes, low TPS, single-agent | Fleet / multi-agent / high TPS |
| `process_now()` | Synchronous, returns when done | No-op (work runs async on change feed) |
| `process_now_and_wait()` | Returns immediately after flush | Polls until summary visible (RU-costly; tests/demos) |

```python
from agent_memory_toolkit import CosmosMemoryClient, DurableFunctionProcessor

memory = CosmosMemoryClient(..., processor=DurableFunctionProcessor())
```

`DurableFunctionProcessor` is a thin marker — there is no SDK→Function HTTP call. The SDK just writes turns; the deployed Function app picks them up via the Cosmos change feed. Counter-based trigger configuration and Bicep module reference live in [`infra/README.md`](infra/README.md).

---

### Architecture overview

```
+--------------------------+
|        Agent app         |
+------------+-------------+
             |
             v
+--------------------------+      +--------------------------+
| Agent Memory Toolkit     | <--> | Microsoft Foundry        |
| Python sync/async SDK    |      | LLMs + embeddings        |
+------------+-------------+      +------------+-------------+
             ^                                 ^
             |                                 |
             v                                 v
+--------------------------+      +--------------------------+
| Azure Cosmos DB          | <--> | Azure Durable Functions  |
| memories + search        |      | optional processing      |
+--------------------------+      +--------------------------+
```

---

## Public API reference

| Symbol | Module | Purpose |
|---|---|---|
| `CosmosMemoryClient` | `agent_memory_toolkit` | Sync client — local CRUD, Cosmos DB I/O, processing |
| `AsyncCosmosMemoryClient` | `agent_memory_toolkit.aio` | Async mirror |
| `MemoryProcessor` | `agent_memory_toolkit` | Protocol that any processor backend implements |
| `InProcessProcessor` | `agent_memory_toolkit` | Default backend — runs the pipeline in-process |
| `DurableFunctionProcessor` | `agent_memory_toolkit` | Marker backend — work runs in sibling Function app via change feed |
| `client.process_now()` | — | Run the pipeline for recent turns (in-process) or no-op (remote) |
| `client.process_now_and_wait()` | — | Opt-in poll until processing completes; useful for tests/demos with the remote backend |
| `MemoryRecord`, `MemoryType`, `Role` | `agent_memory_toolkit` | Pydantic models / enums |

Async equivalents (`AsyncInProcessProcessor`, `AsyncDurableFunctionProcessor`) live in `agent_memory_toolkit.aio`.

---

## Documentation

- **[Docs/concepts.md](Docs/concepts.md)** — Memory types, threads, roles, embeddings, processing pipeline
- **[Docs/design_patterns.md](Docs/design_patterns.md)** — Integration patterns for chat apps and multi-agent systems
- **[Docs/local_testing.md](Docs/local_testing.md)** — Prerequisites, environment setup, running locally, debugging
- **[Docs/azure_testing.md](Docs/azure_testing.md)** — Azure deployment, RBAC, cloud validation
- **[infra/README.md](infra/README.md)** — `azd` deployment, Bicep modules, BYOR settings, counter-trigger tuning
- **[Docs/troubleshooting.md](Docs/troubleshooting.md)** — Common issues and resolutions for setup, auth, Cosmos DB, embeddings, Durable Functions, vector search, change feed, etc.

---

## Project structure

```
agent_memory_toolkit/   Python SDK (sync + aio mirror)
  processors/           MemoryProcessor Protocol + InProcess/Durable backends
function_app/           Sibling Azure Durable Function app
infra/                  Bicep modules + main.bicep for `azd up`
azure.yaml              `azd` config — provisions Cosmos + AI Foundry + Function app
Samples/                Categorized demo notebooks + sample scripts
Docs/                   Conceptual + operational docs
tests/                  Unit + integration tests (pytest)
```

---

## Migration notes

- **`agent_memory_toolkit.processing.ProcessingClient` is removed.** Drop the import and call `client.process_now()` (or `client.process_now_and_wait()`) instead. Same for the async `AsyncProcessingClient`.
- **New `processor=` kwarg.** Defaults to `InProcessProcessor()` — existing code keeps its current behavior with no edits.
- **`adf_endpoint` / `adf_key` constructor kwargs are gone.** The SDK no longer makes HTTP calls to the Function app at runtime; the Function app reads from the Cosmos change feed.

## Trademark notice
Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
