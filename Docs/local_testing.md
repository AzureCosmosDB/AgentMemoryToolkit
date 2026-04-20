# Testing Agent Memory Toolkit Locally

This guide covers the shortest path to running the library locally, then validating Cosmos DB and Durable Functions.

---

## Prerequisites

### Local tools

| Tool | Install command | Purpose |
|------|-----------------|---------|
| Python 3.11+ | `brew install python@3.13` | Runtime |
| Azure CLI | `brew install azure-cli` | `az login` for `DefaultAzureCredential` |
| Azure Functions Core Tools v4 | `brew install azure-functions-core-tools@4` | Run Functions locally |
| Azurite | `npm install -g azurite` | Local storage emulator for Functions |
| Node.js | `brew install node` | Required for Azurite |

### Python packages

```bash
pip install -e ".[dev]"
pip install -r azure_functions/requirements.txt
```

---

## Azure Resources Needed

You need these before Cosmos DB or LLM-backed features will work:

1. **Cosmos DB for NoSQL**
2. **Azure AI Services / Azure OpenAI** with an embedding model and a chat model

The toolkit can create the database and both containers for you with `create_memory_store()`.

For automatic change feed processing, the function stores lightweight counter documents in a dedicated `counter` container. A `leases` container is created automatically by the Azure Functions runtime.

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
COSMOS_DB_CONTAINER=memories
COSMOS_DB_COUNTERS_CONTAINER=counter
COSMOS_DB_LEASE_CONTAINER=leases
COSMOS_DB_THROUGHPUT_MODE=serverless
COSMOS_DB_AUTOSCALE_MAX_RU=1000

AI_FOUNDRY_ENDPOINT=https://<your-project>.services.ai.azure.com/
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=1536
LLM_MODEL=gpt-5-mini

ADF_ENDPOINT=http://localhost:7071/api
ADF_KEY=
```

The Functions runtime uses `azure_functions/local.settings.json`, not `.env`, so mirror the same values there.

`COSMOS_DB_THROUGHPUT_MODE=serverless` is the default and creates the required Cosmos containers without specifying RU/s. If you set `COSMOS_DB_THROUGHPUT_MODE=autoscale`, the toolkit provisions the memories, counter, and lease containers with the shared max RU/s value from `COSMOS_DB_AUTOSCALE_MAX_RU`.

### Change feed settings (optional)

In `azure_functions/local.settings.json`, add these to enable automatic processing:

```json
"COSMOS_DB__accountEndpoint": "https://<your-account>.documents.azure.com:443/",
"COSMOS_DB_COUNTERS_CONTAINER": "counter",
"COSMOS_DB_LEASE_CONTAINER": "leases",
"COSMOS_DB_THROUGHPUT_MODE": "serverless",
"COSMOS_DB_AUTOSCALE_MAX_RU": "1000",
"THREAD_SUMMARY_EVERY_N": "5",
"FACT_EXTRACTION_EVERY_N": "3",
"USER_SUMMARY_EVERY_N": "10"
```

Set any threshold to `"0"` to disable that processing type. See `azure_functions/local.settings.json.template` for the full template.

---

## 1. Test Local-Only CRUD

No Azure resources are required for local in-memory operations.

```python
import uuid
from agent_memory_toolkit import AgentMemory

memory = AgentMemory(use_default_credential=False)

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

`AsyncAgentMemory` works the same way for local operations (local methods are synchronous).

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
from agent_memory_toolkit import AgentMemory

load_dotenv()

memory = AgentMemory(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE"),
    cosmos_container=os.getenv("COSMOS_DB_CONTAINER"),
    cosmos_counter_container=os.getenv("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
    cosmos_lease_container=os.getenv("COSMOS_DB_LEASE_CONTAINER", "leases"),
    cosmos_throughput_mode=os.getenv("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
    cosmos_autoscale_max_ru=int(os.getenv("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    adf_endpoint=os.getenv("ADF_ENDPOINT", "http://localhost:7071/api"),
    adf_key=os.getenv("ADF_KEY", ""),
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
memory.add_cosmos(user_id="user-001", role="agent", thread_id=thread_id, content="Direct Cosmos write")

# Query with filters including thread_id
results = memory.get_memories(user_id="user-001", thread_id=thread_id)
for r in results:
    print(f"  [{r['thread_id'][:8]}...] [{r['id'][:8]}...] role={r['role']:<6} {r['content'][:60]}")
```

### Async

```python
import os, uuid
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from agent_memory_toolkit.aio import AsyncAgentMemory

load_dotenv()

memory = AsyncAgentMemory(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE"),
    cosmos_container=os.getenv("COSMOS_DB_CONTAINER"),
    cosmos_counter_container=os.getenv("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
    cosmos_lease_container=os.getenv("COSMOS_DB_LEASE_CONTAINER", "leases"),
    cosmos_throughput_mode=os.getenv("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
    cosmos_autoscale_max_ru=int(os.getenv("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    adf_endpoint=os.getenv("ADF_ENDPOINT", "http://localhost:7071/api"),
    adf_key=os.getenv("ADF_KEY", ""),
    use_default_credential=True,
    cosmos_credential=AsyncDefaultAzureCredential(),
)

await memory.connect_cosmos(
    endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    database=os.getenv("COSMOS_DB_DATABASE"),
    container=os.getenv("COSMOS_DB_CONTAINER"),
    credential=AsyncDefaultAzureCredential(),
)
await memory.create_memory_store()

thread_id = str(uuid.uuid4())
await memory.add_cosmos(user_id="user-001", role="user", thread_id=thread_id, content="Async Cosmos write")
results = await memory.get_memories(user_id="user-001", thread_id=thread_id)
for r in results:
    print(f"  [{r['thread_id'][:8]}...] [{r['id'][:8]}...] role={r['role']:<6} {r['content'][:60]}")

await memory.close()
```

`create_memory_store()` creates the database and required containers, configures the hierarchical partition key (`user_id`, `thread_id`) for memories and counters, uses `/id` for the lease container, and applies either serverless or autoscale throughput based on `COSMOS_DB_THROUGHPUT_MODE`.

---

## 3. Run Durable Functions Locally

### Start Azurite

```bash
azurite --silent --location /tmp/azurite --debug /tmp/azurite/debug.log
```

### Start the Functions host

```bash
cd azure_functions
pip install -r azure_functions/requirements.txt
func start
```

Expected functions include:

- `memory_orchestrator`
- `load_memories`
- `generate_embeddings`
- `store_results`
- `generate_thread_summary`
- `extract_facts`
- `generate_user_summary`
- `http_start`
- `on_memory_change` (change feed trigger — only active when `COSMOS_DB__accountEndpoint` is set)

### Function keys

- **Local:** `func start` does not enforce `AuthLevel.FUNCTION`, so `ADF_KEY` can be empty.
- **Azure:** a function key is required.

---

## 4. Trigger the Pipeline

You can test through `curl` or through the Python client.

### `curl` example

```bash
curl -X POST "http://localhost:7071/api/orchestrators/memory_orchestrator" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test-thread-001",
    "user_id": "user-1",
    "content": "This is a test memory to embed and store.",
    "role": "user",
    "memory_type": "turn",
    "metadata": {},
    "thread_summary": false
  }'
```

Poll the returned `statusQueryGetUri` to track progress.

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
print(result.get("output", {}))
```

If a summary already exists, only newer memories are processed and merged into the existing summary.

### Fact extraction

```python
result = memory.extract_facts(user_id="user-001", thread_id=thread_id)
output = result.get("output", [])
for fact in output:
    print(f"  [{fact.get('thread_id', '')[:8]}...] user={fact.get('user_id', ''):<10} type={fact.get('type', ''):<8} {fact.get('content', '')[:80]}")
```

### User summary

```python
result = memory.generate_user_summary(user_id="user-001")
print(result.get("output", {}))
print(memory.get_user_summary(user_id="user-001"))
```

User summaries also update incrementally when one already exists.

---

## 6. Test Change Feed Auto-Processing

If you have configured the change feed settings above, you can test automatic processing.

### Prerequisites

1. Ensure `local.settings.json` has the change feed settings (see [Change feed settings](#change-feed-settings-optional) above).

2. Restart the Functions host (`func start`).

### Test steps

1. Set a low threshold for testing, e.g. `THREAD_SUMMARY_EVERY_N=3`.
2. Write turns to Cosmos (via the SDK or `curl`) until the threshold is crossed.
3. Watch the Functions host logs — you should see the orchestrator being started automatically.

```python
import uuid

thread_id = str(uuid.uuid4())
for i in range(3):
    memory.add_cosmos(
        user_id="user-001",
        thread_id=thread_id,
        role="user",
        content=f"Turn {i+1} for change feed test",
    )
```

4. After a few seconds, the change feed trigger should pick up the new turns, increment the counter, cross the threshold, and start a thread summary orchestration.
5. Verify the summary was created:

```python
result = memory.get_memories(user_id="user-001", thread_id=thread_id, memory_type="summary")
print(result)
```

> **Note:** The change feed has a small polling delay (a few seconds by default). Derived memories may not appear immediately.

---

## VS Code Debugging

1. Start Azurite.
2. Press `F5`.
3. Use the `func: host start` task.
4. Set breakpoints in `azure_functions/activities.py` or `azure_functions/function_app.py`.

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
| Function 401 in Azure | Set `ADF_KEY` or pass `?code=<key>` |
| Change feed trigger not firing | Verify `COSMOS_DB__accountEndpoint` is set and matches your Cosmos account |
| Counter updates fail | Verify the function can write to the configured `COSMOS_DB_COUNTERS_CONTAINER` container |
| Auto-processing not starting | Check that threshold settings are > 0 and the Functions host shows `on_memory_change` at startup |

For full cloud deployment and validation, see `Docs/azure_testing.md`.
