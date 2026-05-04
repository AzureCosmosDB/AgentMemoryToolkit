# Azure Cosmos DB Agent Memory Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Azure Cosmos DB](https://img.shields.io/badge/Azure-Cosmos%20DB-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/products/cosmos-db/)
[![Follow on X](https://img.shields.io/twitter/follow/AzureCosmosDB?style=social)](https://twitter.com/AzureCosmosDB)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Azure%20Cosmos%20DB-0077B5?logo=linkedin)](https://www.linkedin.com/showcase/azure-cosmos-db/)
[![YouTube](https://img.shields.io/badge/YouTube-Azure%20Cosmos%20DB-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@AzureCosmosDB)


Agent Memory Toolkit is a Python SDK for storing, retrieving, and transforming agent memories on Azure Cosmos DB. It gives your agent both raw conversation history and higher-value derived memory — thread summaries, extracted facts, and cross-thread user profiles — all searchable semantically. The processing pipeline can run **in-process** (zero infra) or in a sibling **Azure Durable Function app** that watches the Cosmos DB change feed. Sync (`CosmosMemoryClient`) and async (`AsyncCosmosMemoryClient`) APIs are mirror-images of each other.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                  YOUR AGENTIC APP                                    │
│                   Uses CosmosMemoryClient / AsyncCosmosMemoryClient                  │
└─────────────────────────────────────────┬────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                        AGENT MEMORY TOOLKIT (Python SDK)                             │
│                                                                                      │
│  • Local in-memory CRUD                                                              │
│  • Cosmos DB storage and retrieval                                                   │
│  • Pluggable processor: in-process or remote Durable Function app                    │
└──────────────────────────────────────────┬──────────────────────────────┬────────────┘
                        │                                            │
                        │ read / write                               │ Invoke processing pipeline
                        ▼                                            ▼
┌───────────────────────────────────┐                           ┌──────────────────────────────────┐
│      AZURE COSMOS DB (NoSQL)      │                           │     AZURE DURABLE FUNCTIONS      │
│                                   │                           │                                  │
│  Stores:                          │                           │  Orchestrates memory processing: │
│  • turns                          │                           │  • thread summaries              │
│  • summaries                      │◄─── memory management ───►│  • fact extraction               │
│  • facts                          │                           │  • user summaries                │
│  • user summaries                 │                           │                                  │
│                                   │                           │ On-demand (SDK) or automatic     │
│  Supports query, vector, text     │    change feed trigger    │ (Cosmos DB change feed trigger). │
│  search over stored memories.     │───────────────────────────►│                                  │
└───────────────────────┬───────────┘                           └──────────────────┬───────────────┘
                        │             embeddings and LLM-based processing          │
                        └──────────────────────┬───────────────────────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │         MICROSOFT FOUNDRY        │
                              │                                  │
                              │  • Embeddings for search         │
                              │  • Chat/LLM generation           │
                              │                                  │
                              └──────────────────────────────────┘
```

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
memory.flush(user_id=USER, thread_id=THREAD)

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
python Samples/quickstart_cosmos.py
```

See [`Samples/`](Samples/) for end-to-end scenarios (chat memory, RAG, multi-agent, customer support, remote processor).

---

## Concepts in 60 seconds

| Concept | What it is | API |
|---|---|---|
| **Turn** | One message (user or assistant) — the raw conversation atom | `add_cosmos(...)`, `add_local(...)` |
| **Thread summary** | LLM-generated, incrementally updated rollup of a single thread | `generate_thread_summary(...)` |
| **Fact** | Discrete, independently searchable assertion extracted from turns | `extract_facts(...)` |
| **User summary** | Cross-thread profile of what's known about a user | `generate_user_summary(...)`, `get_user_summary(...)` |
| **Search** | Vector + full-text + filter; returns turns, summaries, and facts | `search_cosmos(...)` |
| **Flush** | Run the full pipeline (summary → facts → user profile) for recent turns | `flush(...)`, `flush_and_wait(...)` |

All four memory kinds live in the same Cosmos container, partitioned by `(user_id, thread_id)`, distinguished by a `memory_type` discriminator.

---

## Two processor flavors

Pick at construction time via the `processor=` kwarg.

| | `InProcessProcessor` (default) | `DurableFunctionProcessor` |
|---|---|---|
| Infra | None — just `pip install` | Sibling Azure Function app |
| Best for | Prototypes, low TPS, single-agent | Fleet / multi-agent / high TPS |
| `flush()` | Synchronous, returns when done | No-op (work runs async on change feed) |
| `flush_and_wait()` | Returns immediately after flush | Polls until summary visible (RU-costly; tests/demos) |

```python
from agent_memory_toolkit import CosmosMemoryClient, DurableFunctionProcessor

memory = CosmosMemoryClient(..., processor=DurableFunctionProcessor())
```

`DurableFunctionProcessor` is a thin marker — there is no SDK→Function HTTP call. The SDK just writes turns; the deployed Function app picks them up via the Cosmos change feed. Counter-based trigger configuration and Bicep module reference live in [`infra/README.md`](infra/README.md).

---

## Public API reference

| Symbol | Module | Purpose |
|---|---|---|
| `CosmosMemoryClient` | `agent_memory_toolkit` | Sync client — local CRUD, Cosmos DB I/O, processing |
| `AsyncCosmosMemoryClient` | `agent_memory_toolkit.aio` | Async mirror |
| `MemoryProcessor` | `agent_memory_toolkit` | Protocol that any processor backend implements |
| `InProcessProcessor` | `agent_memory_toolkit` | Default backend — runs the pipeline in-process |
| `DurableFunctionProcessor` | `agent_memory_toolkit` | Marker backend — work runs in sibling Function app via change feed |
| `client.flush()` | — | Run the pipeline for recent turns (in-process) or no-op (remote) |
| `client.flush_and_wait()` | — | Opt-in poll until processing completes; useful for tests/demos with the remote backend |
| `MemoryRecord`, `MemoryType`, `Role` | `agent_memory_toolkit` | Pydantic models / enums |

Async equivalents (`AsyncInProcessProcessor`, `AsyncDurableFunctionProcessor`) live in `agent_memory_toolkit.aio`.

---

## Documentation

- **[Docs/concepts.md](Docs/concepts.md)** — Memory types, threads, roles, embeddings, processing pipeline
- **[Docs/design_patterns.md](Docs/design_patterns.md)** — Integration patterns for chat apps and multi-agent systems
- **[Docs/local_testing.md](Docs/local_testing.md)** — Prerequisites, environment setup, running locally, debugging
- **[Docs/azure_testing.md](Docs/azure_testing.md)** — Azure deployment, RBAC, cloud validation
- **[infra/README.md](infra/README.md)** — `azd` deployment, Bicep modules, BYOR settings, counter-trigger tuning

---

## Project structure

```
agent_memory_toolkit/   Python SDK (sync + aio mirror)
  processors/           MemoryProcessor Protocol + InProcess/Durable backends
function_app/           Sibling Azure Durable Function app
infra/                  Bicep modules + main.bicep for `azd up`
azure.yaml              `azd` config — provisions Cosmos + AI Foundry + Function app
Samples/                Demo notebooks + sample scripts
Docs/                   Conceptual + operational docs
tests/                  Unit + integration tests (pytest)
```

---

## Migration notes

- **`agent_memory_toolkit.processing.ProcessingClient` is removed.** Drop the import and call `client.flush()` (or `client.flush_and_wait()`) instead. Same for the async `AsyncProcessingClient`.
- **New `processor=` kwarg.** Defaults to `InProcessProcessor()` — existing code keeps its current behavior with no edits.
- **`adf_endpoint` / `adf_key` constructor kwargs are gone.** The SDK no longer makes HTTP calls to the Function app at runtime; the Function app reads from the Cosmos change feed.
