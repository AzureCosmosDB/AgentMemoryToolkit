# Azure Cosmos DB Agent Memory Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Azure Cosmos DB](https://img.shields.io/badge/Azure-Cosmos%20DB-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/products/cosmos-db/)
[![Follow on X](https://img.shields.io/twitter/follow/AzureCosmosDB?style=social)](https://twitter.com/AzureCosmosDB)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Azure%20Cosmos%20DB-0077B5?logo=linkedin)](https://www.linkedin.com/showcase/azure-cosmos-db/)
[![YouTube](https://img.shields.io/badge/YouTube-Azure%20Cosmos%20DB-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@AzureCosmosDB)


Agent Memory Toolkit is a Python library and Azure-backed reference implementation for storing, retrieving, and transforming agent memories over time. It combines a simple SDK for local and Cosmos DB operations with Durable Functions pipelines that generate thread summaries, extract facts, and build cross-thread user profiles. The toolkit is designed for agent applications that need both raw conversation history and higher-value derived memory that can be searched semantically later. It provides matching sync (`AgentMemory`) and async (`AsyncAgentMemory`) APIs so the same memory model can be used in scripts, services, notebooks, and larger agent systems.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            YOUR AGENTIC APP                                │
│                  Uses AgentMemory / AsyncAgentMemory                       │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                   AGENT MEMORY TOOLKIT (Python SDK)                        │
│                                                                            │
│  • Local in-memory CRUD                                                    │
│  • Cosmos DB storage and retrieval                                         │
│  • Calls into Azure Durable Functions for memory processing                │
└───────────────────────┬────────────────────────────────────────────┬───────┘
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
│                                   │                           │ Reads raw memories and           │
│  Supports query, vector, text     │                           │ writes processed memories back.  │
│  search over stored memories.     │                           └──────────────────┬───────────────┘
└───────────────────────┬───────────┘                                              │
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

## Features

| Feature | Description |
|---------|-------------|
| **Local memory store** | In-memory CRUD — no Azure needed for development |
| **Cosmos DB integration** | CRUD, `push_to_cosmos()` bulk upload, semantic search, hierarchical partition key, vector + full-text indexes |
| **Thread summaries** | `generate_thread_summary()` — LLM-generated, incrementally updated, embedded and stored |
| **Fact extraction** | `extract_facts()` — discrete, independently searchable assertions from a thread |
| **User summaries** | `generate_user_summary()` — cross-thread user profile, incrementally updated |
| **Incremental updates** | Thread and user summaries use point-read + time-filtering to merge new data with existing summaries |
| **Externalized prompts** | LLM prompts live in editable Markdown files (`azure_functions/prompts/`) |
| **Entra ID auth** | `DefaultAzureCredential` everywhere — `az login`, managed identities |

---

## Project Structure

```
agent_memory_toolkit/          Python library — sync API
  memory.py                    AgentMemory orchestrator
  cosmos_memory_client.py      CosmosMemoryStore — Cosmos DB CRUD + vector search
  embeddings.py                EmbeddingsClient — Azure OpenAI embeddings
  processing.py                ProcessingClient — Durable Functions polling
  models.py                    Pydantic data models (MemoryRecord, enums)
  exceptions.py                Custom exception hierarchy
  _query_builder.py            Shared query builder (private)
  aio/                         Async API (mirrors azure.cosmos.aio convention)
    memory.py                  AsyncAgentMemory
    cosmos_memory_client.py    AsyncCosmosMemoryStore
    embeddings.py              AsyncEmbeddingsClient
    processing.py              AsyncProcessingClient
azure_functions/               Durable Functions — orchestrator, activities, HTTP trigger
  prompts/                     LLM system prompts — summarize, facts, user_summary + update variants
Samples/                       Demo notebooks — sync (Demo.ipynb) + async (Demo_async.ipynb)
Docs/                          Documentation — concepts, local testing, Azure deployment
tests/                         Unit tests (pytest) — 184 tests, 87% coverage
```

---

## Quick Start

### 1. Install

```bash
pip install .

# With dev/test dependencies
pip install ".[dev]"
```

### 2. Local-only (no Azure)

```python
import uuid
from agent_memory_toolkit import AgentMemory

memory = AgentMemory(use_default_credential=False)
thread_id = str(uuid.uuid4())
memory.add_local(user_id="user-001", role="user", thread_id=thread_id, content="Hello world")
print(memory.get_local())
```

### 3. With Cosmos DB + Azure OpenAI

```bash
cp .env.template .env   # fill in endpoint values
```

```python
import os, uuid
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from agent_memory_toolkit import AgentMemory

load_dotenv()

memory = AgentMemory(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE"),
    cosmos_container=os.getenv("COSMOS_DB_CONTAINER"),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    adf_endpoint=os.getenv("ADF_ENDPOINT", "http://localhost:7071/api"),
    adf_key=os.getenv("ADF_KEY", ""),
    use_default_credential=True,
    cosmos_credential=DefaultAzureCredential(),
)
memory.create_memory_store()
memory.connect_cosmos()

# Add directly to Cosmos
thread_id = str(uuid.uuid4())
memory.add_cosmos(user_id="user-001", role="user", thread_id=thread_id, content="Stored in Cosmos")
print(memory.get_memories(user_id="user-001", thread_id=thread_id))

# Or add locally first, then bulk-upload
memory.add_local(user_id="user-001", role="agent", thread_id=thread_id, content="Response text")
memory.push_to_cosmos()
```

### 4. Durable Function operations

These require the Azure Durable Functions host. See [local_testing.md](Docs/local_testing.md) for setup.

```python
# Thread summary (incremental — merges with existing if present)
result = memory.generate_thread_summary(user_id="user-001", thread_id=thread_id, recent_k=5)

# Fact extraction
result = memory.extract_facts(user_id="user-001", thread_id=thread_id)

# User summary (incremental — cross-thread profile)
result = memory.generate_user_summary(user_id="user-001")

# Retrieve stored user summary
summary = memory.get_user_summary(user_id="user-001")
```

> The async API (`AsyncAgentMemory`) is identical — just `await` each call. Import from the `aio` subpackage:
>
> ```python
> from agent_memory_toolkit.aio import AsyncAgentMemory
> ```

---

## Azure Resources

| Resource | Purpose |
|----------|---------|
| **Cosmos DB for NoSQL** | Memory store with hierarchical partition key, vector index, full-text index |
| **Azure OpenAI / AI Foundry** | Embedding model + chat model for summarization / fact extraction |
| **Azure Functions** | Durable Functions orchestrator and activity functions |

All services use **Entra ID** auth via `DefaultAzureCredential`.

---

## Documentation

- **[concepts.md](Docs/concepts.md)** — Memory types, threads, roles, embeddings, processing pipeline
- **[design_patterns.md](Docs/design_patterns.md)** — Integration patterns for chat apps and multi-agent systems
- **[local_testing.md](Docs/local_testing.md)** — Prerequisites, environment setup, running locally, debugging
- **[azure_testing.md](Docs/azure_testing.md)** — Azure deployment, RBAC, cloud validation

---