# Troubleshooting Agent Memory Toolkit

Use this guide when local memory works but Cosmos DB, embeddings, Durable Functions, or automatic change feed processing does not.

---

## Quick Triage

| Symptom | First checks |
|---------|--------------|
| Import errors | Install with `pip install -e ".[dev]"` and import `CosmosMemoryClient` or `AsyncCosmosMemoryClient`. |
| Missing configuration | Verify `.env`, `azure_functions/local.settings.json`, and Azure Function App settings use the same endpoint, database, and container values. |
| Cosmos 401 or 403 | Run `az login` and confirm Cosmos DB data-plane RBAC is assigned. |
| Cosmos operations fail before connecting | Call `create_memory_store()` or `connect_cosmos()` before cloud operations. |
| Search returns no vector results | Confirm embeddings are generated and `EMBEDDING_DIMENSIONS` matches the container vector policy. |
| Durable Function calls fail | Start the Functions host and check `ADF_ENDPOINT`, `ADF_KEY`, and the orchestrator route. |
| Change feed does not create summaries or facts | Confirm change feed settings, thresholds, lease container, counter container, and that inserted documents have `type: "turn"`. |

---

## 1. Environment And Imports

Install the package from the repository root:

```bash
pip install -e ".[dev]"
pip install -r azure_functions/requirements.txt
```

The public clients are:

```python
from agent_memory_toolkit import CosmosMemoryClient
from agent_memory_toolkit.aio import AsyncCosmosMemoryClient
```

If notebooks cannot import the package, run them from the repo root with paths such as `Samples/Notebooks/Demo.ipynb`, or add the repository root to `sys.path`.

---

## 2. Configuration And Authentication

For local runs, keep `.env` and `azure_functions/local.settings.json` aligned:

```env
COSMOS_DB_ENDPOINT=https://<account>.documents.azure.com:443/
COSMOS_DB_DATABASE=ai_memory
COSMOS_DB_CONTAINER=memories
COSMOS_DB_COUNTERS_CONTAINER=counter
COSMOS_DB_LEASE_CONTAINER=leases
AI_FOUNDRY_ENDPOINT=https://<project>.services.ai.azure.com/
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=1536
ADF_ENDPOINT=http://localhost:7071/api
ADF_KEY=
```

Run `az login` before using `DefaultAzureCredential`.

Required roles:

| Service | Role |
|---------|------|
| Cosmos DB | Cosmos DB Built-in Data Contributor |
| Azure OpenAI / AI Services | Cognitive Services OpenAI User |

RBAC changes can take several minutes to propagate.

---

## 3. Cosmos DB Store Creation

Run `create_memory_store()` before relying on cloud operations. It creates the database plus the `memories`, `counter`, and `leases` containers.

The memories container is created with:

- hierarchical partition key on `user_id` and `thread_id`
- vector index on `/embedding`
- full-text index on `/content`

If vector or full-text search fails after changing dimensions or indexing settings, create a fresh container with the desired configuration. Cosmos container vector policies are creation-time infrastructure choices.

Use `COSMOS_DB_THROUGHPUT_MODE=serverless` for the default setup. Use `autoscale` with `COSMOS_DB_AUTOSCALE_MAX_RU` when you need provisioned autoscale throughput.

---

## 4. Embeddings And Search

Embedding failures usually mean one of these is wrong:

- `AI_FOUNDRY_ENDPOINT`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`
- Azure OpenAI / AI Services RBAC

For hybrid search, `search_terms` is required when `hybrid_search=True`.

If search returns documents but scores look poor, check that records have an `embedding` field and that the query uses similar language to the stored memory content.

---

## 5. Durable Functions Processing

Thread summaries, fact extraction, and user summaries require the Functions host.

Start local dependencies:

```bash
azurite --silent --location /tmp/azurite --debug /tmp/azurite/debug.log
cd azure_functions
func start
```

The SDK posts to:

```text
<ADF_ENDPOINT>/orchestrators/memory_orchestrator
```

For local testing, `ADF_ENDPOINT` is usually `http://localhost:7071/api` and `ADF_KEY` is blank. For Azure, use the deployed Function App URL and set `ADF_KEY` if function-key auth is enabled.

If orchestration polling times out, check the Functions logs first. The orchestration may still be running, or an activity may be waiting on Cosmos DB or the LLM endpoint.

---

## 6. Change Feed Processing

Automatic processing requires these settings in the Functions app or `local.settings.json`:

```json
"COSMOS_DB__accountEndpoint": "https://<account>.documents.azure.com:443/",
"COSMOS_DB_COUNTERS_CONTAINER": "counter",
"COSMOS_DB_LEASE_CONTAINER": "leases",
"THREAD_SUMMARY_EVERY_N": "5",
"FACT_EXTRACTION_EVERY_N": "3",
"USER_SUMMARY_EVERY_N": "10"
```

Set a threshold to `"0"` to disable that processing type.

Only documents with `type: "turn"` increment counters. Derived memories such as `summary`, `fact`, and `user_summary` do not trigger threshold counts.

If nothing fires:

- verify the Functions host shows the Cosmos DB trigger
- confirm the `leases` container exists
- confirm the `counter` container is writable
- insert enough new turn documents to cross the configured threshold
- check for generated documents with `memory_type="summary"`, `memory_type="fact"`, or `get_user_summary(user_id=...)`

---

## 7. Async Client Notes

Use async Azure credentials with the async client:

```python
from azure.identity.aio import DefaultAzureCredential
from agent_memory_toolkit.aio import AsyncCosmosMemoryClient
```

Always `await` cloud operations and close the client when done:

```python
await memory.close()
```

In notebooks, top-level `await` is supported, so do not wrap cells with `asyncio.run()`.