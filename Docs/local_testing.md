# Testing Agent Memory Toolkit Locally

This guide covers the shortest path to running the library locally, then validating Cosmos DB and Durable Functions.

---

## Prerequisites

### Local tools

| Tool | Install command | Purpose |
|------|-----------------|---------|
| Python 3.11+ | `brew install python@3.13` | Runtime |
| Azure CLI | `brew install azure-cli` | `az login` for `DefaultAzureCredential` |
| Azure Functions Core Tools v4 | `brew tap azure/functions && brew install azure-functions-core-tools@4` | Run Functions locally |
| Azurite | `npm install -g azurite` | Local storage emulator for Functions |
| Node.js | `brew install node` | Required for Azurite |

### Python packages

```bash
pip install -e ".[dev]"
pip install -r function_app/requirements.txt
```

---

## Azure Resources Needed

You need these before Cosmos DB or LLM-backed features will work:

1. **Cosmos DB for NoSQL**
2. **Azure AI Services / Azure OpenAI** with an embedding model and a chat model

The toolkit can create the database and the `memories`, `memories_turns`, `memories_summaries`, `counter`, and `leases` containers for you with `create_memory_store()`.

For automatic change feed processing, the function stores lightweight counter documents in the `counter` container, and the change feed trigger keeps its checkpoints in the `leases` container.

### RBAC

Grant your identity:

- **Cosmos DB Built-in Data Contributor** on the Cosmos account
- **Cognitive Services OpenAI User** on the AI resource

Example Cosmos role assignment:

```bash
USER_OID=$(az ad signed-in-user show --query id -o tsv)

az cosmosdb sql role assignment create \
  --account-name <your-cosmos-account> \
  --resource-group <your-rg> \
  --role-definition-name "Cosmos DB Built-in Data Contributor" \
  --principal-id "$USER_OID" \
  --scope "/"
```

---

## Environment Configuration

Copy the template:

```bash
cp .env.template .env
```

Minimum `.env` values:

```env
COSMOS_DB_ENDPOINT=https://<your-account>.documents.azure.com:443/
COSMOS_DB_DATABASE=ai_memory
COSMOS_DB_MEMORIES_CONTAINER=memories
COSMOS_DB_COUNTERS_CONTAINER=counter
COSMOS_DB_LEASE_CONTAINER=leases
COSMOS_DB_THROUGHPUT_MODE=serverless
COSMOS_DB_AUTOSCALE_MAX_RU=1000

AI_FOUNDRY_ENDPOINT=https://<your-account>.services.ai.azure.com/
AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AI_FOUNDRY_EMBEDDING_DIMENSIONS=1536
AI_FOUNDRY_CHAT_DEPLOYMENT_NAME=gpt-5-mini
```

The Functions host reads `function_app/local.settings.json`, not `.env`. Create it from the template and mirror the same values there:

```bash
cp function_app/local.settings.json.template function_app/local.settings.json
```

`COSMOS_DB_THROUGHPUT_MODE=serverless` is the default and creates the required Cosmos containers without specifying RU/s. If you set `COSMOS_DB_THROUGHPUT_MODE=autoscale`, the toolkit provisions all five containers with the shared max RU/s value from `COSMOS_DB_AUTOSCALE_MAX_RU`. The snippets below pass these two values to the client as `cosmos_throughput_mode` and `cosmos_autoscale_max_ru`; the toolkit does not read them from the environment itself.

Optionally, set `AZURE_OPENAI_API_VERSION` to pin the Azure OpenAI REST API version that the chat and embeddings clients use (default `2024-12-01-preview`).

### Change feed settings (optional)

In `function_app/local.settings.json`, add these to enable automatic processing:

```json
"COSMOS_DB__accountEndpoint": "https://<your-account>.documents.azure.com:443/",
"COSMOS_DB_COUNTERS_CONTAINER": "counter",
"COSMOS_DB_LEASE_CONTAINER": "leases",
"MEMORY_PROCESSOR_OWNER": "durable",
"THREAD_SUMMARY_EVERY_N": "5",
"FACT_EXTRACTION_EVERY_N": "3",
"USER_SUMMARY_EVERY_N": "10"
```

The Function App skips every change feed batch unless `MEMORY_PROCESSOR_OWNER` is `durable`. Set `MEMORY_PROCESSOR_OWNER=durable` in the SDK's `.env` as well, so the SDK does not also process the same turns in-process.

Set any threshold to `"0"` to disable that processing type. `USER_SUMMARY_WAIT_SECONDS` (default `120`) and `USER_SUMMARY_WAIT_INTERVAL_SECONDS` (default `10`) set how long a user-summary run waits, polling at that interval, for extraction to save a new user's first memories before it skips that cycle. See `function_app/local.settings.json.template` for the full template.

---

## 1. Test Local-Only CRUD

No Azure resources are required for local in-memory operations.

```python
import uuid
from azure.cosmos.agent_memory import CosmosMemoryClient

memory = CosmosMemoryClient(use_default_credential=False)

THREAD_ID = str(uuid.uuid4())

memory.add_local(user_id="user-001", role="user", thread_id=THREAD_ID, content="Hello world")
memory.add_local(user_id="user-001", role="agent", thread_id=THREAD_ID, content="Hi there!")

for m in memory.get_local():
    print(f"  [{m['thread_id'][:8]}...] [{m['id'][:8]}...] role={m['role']:<6} {m['content'][:50]}")

mem_id = memory.get_local()[0]["id"]
memory.update_local(mem_id, content="Updated content")
memory.delete_local(mem_id)
print(f"Remaining: {len(memory.get_local())}")
```

`AsyncCosmosMemoryClient` works the same way for local operations (local methods are synchronous).

---

## 2. Test Cosmos DB Operations

Log in first:

```bash
az login
```

Then run a minimal smoke test:

### Sync

```python
import os, uuid
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.cosmos.agent_memory import CosmosMemoryClient

load_dotenv()

memory = CosmosMemoryClient(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE"),
    cosmos_container=os.getenv("COSMOS_DB_MEMORIES_CONTAINER"),
    cosmos_counter_container=os.getenv("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
    cosmos_lease_container=os.getenv("COSMOS_DB_LEASE_CONTAINER", "leases"),
    cosmos_throughput_mode=os.getenv("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
    cosmos_autoscale_max_ru=int(os.getenv("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_deployment_name=os.getenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
    chat_deployment_name=os.getenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
    use_default_credential=True,
    cosmos_credential=DefaultAzureCredential(),
)

memory.create_memory_store()
memory.connect_cosmos()

# Add local memories, then push them all to Cosmos
thread_id = str(uuid.uuid4())
memory.add_local(user_id="user-001", role="user", thread_id=thread_id, content="Stored in Cosmos")
memory.push_to_cosmos()

# Or add directly to Cosmos
memory.upsert_memory(user_id="user-001", role="agent", thread_id=thread_id, content="Direct Cosmos write")

# Read the thread's turns back (get_memories returns facts, episodic memories and procedures)
results = memory.get_thread(thread_id=thread_id, user_id="user-001")
for r in results:
    print(f"  [{r['thread_id'][:8]}...] [{r['id'][:8]}...] role={r['role']:<6} {r['content'][:60]}")
```

### Async

```python
import os, uuid
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.cosmos.agent_memory.aio import AsyncCosmosMemoryClient

load_dotenv()

memory = AsyncCosmosMemoryClient(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE"),
    cosmos_container=os.getenv("COSMOS_DB_MEMORIES_CONTAINER"),
    cosmos_counter_container=os.getenv("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
    cosmos_lease_container=os.getenv("COSMOS_DB_LEASE_CONTAINER", "leases"),
    cosmos_throughput_mode=os.getenv("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
    cosmos_autoscale_max_ru=int(os.getenv("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_deployment_name=os.getenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
    chat_deployment_name=os.getenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
    use_default_credential=True,
    cosmos_credential=AsyncDefaultAzureCredential(),
)

await memory.connect_cosmos(
    endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    database=os.getenv("COSMOS_DB_DATABASE"),
    container=os.getenv("COSMOS_DB_MEMORIES_CONTAINER"),
    credential=AsyncDefaultAzureCredential(),
)
await memory.create_memory_store()

thread_id = str(uuid.uuid4())
await memory.upsert_memory(user_id="user-001", role="user", thread_id=thread_id, content="Async Cosmos write")
results = await memory.get_thread(thread_id=thread_id, user_id="user-001")
for r in results:
    print(f"  [{r['thread_id'][:8]}...] [{r['id'][:8]}...] role={r['role']:<6} {r['content'][:60]}")

await memory.close()
```

`create_memory_store()` creates the database and the `memories`, `memories_turns`, `memories_summaries`, `counter`, and `leases` containers. It configures the hierarchical partition key (`user_id`, `thread_id`) for the first four and `/id` for `leases`, gives a newly created `memories_turns` container a 30-day default TTL and a newly created `counter` container a 90-day one, and applies either serverless or autoscale throughput based on `cosmos_throughput_mode` (set from `COSMOS_DB_THROUGHPUT_MODE` above).

---

## 3. Run Durable Functions Locally

### Start Azurite

```bash
azurite --silent --location /tmp/azurite --debug /tmp/azurite/debug.log
```

### Start the Functions host

```bash
cd function_app
pip install -r requirements.txt
func start
```

Expected functions:

- `on_memory_change` (Cosmos DB change feed trigger on the turns container; needs `COSMOS_DB__accountEndpoint`)
- Orchestrators: `ExtractMemoriesOrchestrator`, `ExtractEpisodesOrchestrator`, `ThreadSummaryOrchestrator`, `UserSummaryOrchestrator`, `SynthesizeProceduralOrchestrator`
- Activities: `em_Extract`, `em_Persist`, `em_AdvanceExtractWatermark`, `em_ReconcileMemories`, `ee_ExtractEpisodes`, `ts_Extract`, `ts_PersistSummary`, `us_Extract`, `us_PersistUserSummary`, `sp_SynthesizeProcedural`

The app registers no HTTP-triggered function, so there is no function key to set. Orchestrations start only from the change feed.

---

## 4. Trigger the Pipeline

The Function App starts orchestrations only from the change feed (see [section 6](#6-test-change-feed-auto-processing)). To run the same steps on demand, call the client; these calls run in-process and do not need the Functions host.

### Python client example

#### Sync

```python
result = memory.generate_thread_summary(user_id="user-001", thread_id=thread_id)
print(result)
```

#### Async

```python
result = await memory.generate_thread_summary(user_id="user-001", thread_id=thread_id)
print(result)
```

---

## 5. Validate Derived Memories

### Thread summary

```python
result = memory.generate_thread_summary(user_id="user-001", thread_id=thread_id)
print(result["content"])
```

If a summary already exists, only newer memories are processed and merged into the existing summary.

### Fact extraction

```python
counts = memory.extract_memories(user_id="user-001", thread_id=thread_id)
print(counts)

facts = memory.get_memories(user_id="user-001", thread_id=thread_id, memory_types=["fact"])
for fact in facts:
    print(f"  [{fact.get('thread_id', '')[:8]}...] user={fact.get('user_id', ''):<10} type={fact.get('type', ''):<8} {fact.get('content', '')[:80]}")
```

### User summary

```python
result = memory.generate_user_summary(user_id="user-001")
print(result["content"])
print(memory.get_user_summary(user_id="user-001"))
```

User summaries also update incrementally when one already exists.

---

## 6. Test Change Feed Auto-Processing

If you have configured the change feed settings above, you can test automatic processing.

### Prerequisites

1. Ensure `function_app/local.settings.json` has the change feed settings, including `MEMORY_PROCESSOR_OWNER` (see [Change feed settings](#change-feed-settings-optional) above), and that the SDK's `.env` sets `MEMORY_PROCESSOR_OWNER=durable` too.

2. Restart the Functions host (`func start`).

### Test steps

1. Set a low threshold for testing, e.g. `THREAD_SUMMARY_EVERY_N=3` (restart the host after changing it).
2. Write turns to Cosmos with the SDK until the threshold is crossed.
3. Watch the Functions host logs - you should see the orchestrator being started automatically, for example `change-feed: starting ThreadSummaryOrchestrator instance=thread_summary:user-001:<thread-id>:3`.

```python
import uuid

thread_id = str(uuid.uuid4())
for i in range(3):
    memory.upsert_memory(
        user_id="user-001",
        thread_id=thread_id,
        role="user",
        content=f"Turn {i+1} for change feed test",
    )
```

4. After a few seconds, the change feed trigger should pick up the new turns, increment the counter, cross the threshold, and start a thread summary orchestration.
5. Verify the summary was created:

```python
result = memory.get_thread_summary(user_id="user-001", thread_id=thread_id)
print(result)
```

> **Note:** The change feed has a small polling delay (a few seconds by default). Derived memories may not appear immediately.

---

## VS Code Debugging

1. Start Azurite.
2. Press `F5`.
3. Use the `func: host start` task.
4. Set breakpoints in the activities under `function_app/orchestrators/` or in `function_app/triggers/change_feed.py`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: azure.identity` | Run `pip install -e ".[dev]"` |
| `DefaultAzureCredential` fails | Run `az login` and confirm the active subscription |
| Cosmos 403 | Check Cosmos DB RBAC and wait for propagation |
| `func: command not found` | Install Azure Functions Core Tools v4 |
| Azurite not found | Install Azurite and Node.js |
| Functions cannot access storage | Start Azurite before `func start` |
| OpenAI 401/403 | Check `Cognitive Services OpenAI User` role |
| Change feed trigger not firing | Verify `COSMOS_DB__accountEndpoint` is set and matches your Cosmos account |
| Counter updates fail | Verify the function can write to the configured `COSMOS_DB_COUNTERS_CONTAINER` container |
| Auto-processing not starting | Check that threshold settings are > 0, `MEMORY_PROCESSOR_OWNER` is `durable`, and the Functions host shows `on_memory_change` at startup |

For full cloud deployment and validation, see `Docs/azure_testing.md`.
