# Deploying and Testing Agent Memory Toolkit in Azure

This guide covers the minimum Azure resources, deployment steps, throughput settings, and validation order for running the toolkit in Azure.

---

## Required Azure Services

| Service | Purpose |
|---------|---------|
| **Azure Cosmos DB for NoSQL** | Persistent memory store with vector and full-text indexes |
| **Azure OpenAI / AI Services** | Embeddings and chat generation |
| **Azure Functions** | Durable Functions orchestrator and activities |
| **Azure Storage Account** | Required by Azure Functions |
| **Application Insights** | Recommended for monitoring |

---

## Prerequisites

You need:

- an Azure subscription
- `az login`
- Python 3.11+
- Azure Functions Core Tools v4
- dependencies installed:

```bash
pip install -e ".[dev]"
pip install -r function_app/requirements.txt
```

> The recommended way to provision **all** required Azure resources is `azd up` from the repo root, which uses the Bicep templates under `infra/`. See [`infra/README.md`](../infra/README.md) for details. The manual `az ...` commands below are kept as a reference for operators who can't use `azd`.

---

## 1. Create Azure Resources

Create, or reuse, the following:

1. resource group
2. storage account
3. Function App
4. Cosmos DB for NoSQL account
5. Azure OpenAI resource with:
   - one embedding model
   - one chat model

Examples:

```bash
az group create --name <resource-group> --location <location>

az storage account create \
  --name <storage-account-name> \
  --resource-group <resource-group> \
  --location <location> \
  --sku Standard_LRS

az functionapp create \
  --name <function-app-name> \
  --resource-group <resource-group> \
  --storage-account <storage-account-name> \
  --consumption-plan-location <location> \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux

az cosmosdb create \
  --name <cosmos-account-name> \
  --resource-group <resource-group>
```

The toolkit can create the database and required containers later via `create_memory_store()`.

---

## 2. Assign RBAC

Grant these roles:

- **Cosmos DB Built-in Data Contributor** on the Cosmos account
- **Cognitive Services OpenAI User** on the AI resource

Enable managed identity on the Function App and use that principal for production role assignments:

```bash
az functionapp identity assign \
  --name <function-app-name> \
  --resource-group <resource-group>
```

---

## 3. Configure Function App Settings

Set the runtime settings:

```bash
az functionapp config appsettings set \
  --name <function-app-name> \
  --resource-group <resource-group> \
  --settings \
    COSMOS_DB_ENDPOINT="https://<cosmos-account-name>.documents.azure.com:443/" \
    COSMOS_DB_DATABASE="ai_memory" \
    COSMOS_DB_MEMORIES_CONTAINER="memories" \
    COSMOS_DB_COUNTERS_CONTAINER="counter" \
    COSMOS_DB_LEASE_CONTAINER="leases" \
    COSMOS_DB_THROUGHPUT_MODE="serverless" \
    COSMOS_DB_AUTOSCALE_MAX_RU="1000" \
    AI_FOUNDRY_ENDPOINT="https://<openai-account-name>.openai.azure.com/" \
    AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large" \
    AI_FOUNDRY_EMBEDDING_DIMENSIONS="1536" \
    AI_FOUNDRY_CHAT_DEPLOYMENT_NAME="gpt-5-mini" \
    THREAD_SUMMARY_EVERY_N="10" \
    FACT_EXTRACTION_EVERY_N="1" \
    USER_SUMMARY_EVERY_N="20" \
    MEMORY_PROCESSOR_OWNER="durable"
```

`COSMOS_DB_THROUGHPUT_MODE=serverless` is the default and creates the `memories`, `memories_turns`, `memories_summaries`, `counter`, and `leases` containers without specifying RU/s. Set `COSMOS_DB_THROUGHPUT_MODE=autoscale` to apply the shared `COSMOS_DB_AUTOSCALE_MAX_RU` cap to all required containers.

`MEMORY_PROCESSOR_OWNER=durable` tells the SDK that the deployed Function App owns processing, so any `CosmosMemoryClient` pointed at the same container will skip its in-process auto-trigger and avoid double-extraction. See the README's processor-ownership table for details.

### Change feed settings (optional)

To enable automatic processing via the change feed trigger, add these settings:

```bash
az functionapp config appsettings set \
  --name <function-app-name> \
  --resource-group <resource-group> \
  --settings \
    COSMOS_DB__accountEndpoint="https://<cosmos-account-name>.documents.azure.com:443/" \
    COSMOS_DB_COUNTERS_CONTAINER="counter" \
    COSMOS_DB_LEASE_CONTAINER="leases" \
    COSMOS_DB_THROUGHPUT_MODE="serverless" \
    COSMOS_DB_AUTOSCALE_MAX_RU="1000" \
    THREAD_SUMMARY_EVERY_N="5" \
    FACT_EXTRACTION_EVERY_N="3" \
    USER_SUMMARY_EVERY_N="10"
```

Set any threshold to `"0"` to disable that processing type.

The `leases` container is provisioned by `create_memory_store()` alongside the `memories` and `counter` containers, so the Function App should be configured to use that existing lease container.

The Function App authenticates to Cosmos DB and Azure OpenAI via its managed identity — there's no shared key or function-key handoff between the SDK and the Function App.

---

## 4. Deploy the Functions Project

The recommended path is `azd up` (which builds and deploys the `function_app/` service automatically). For manual deployment:

```bash
cd function_app
func azure functionapp publish <function-app-name>
```

Verify deployment:

```bash
az functionapp function list \
  --name <function-app-name> \
  --resource-group <resource-group> \
  -o table
```

---

## 5. Configure the Python Client

Update `.env` to point at Azure instead of localhost:

```env
COSMOS_DB_ENDPOINT=https://<cosmos-account-name>.documents.azure.com:443/
COSMOS_DB_DATABASE=ai_memory
COSMOS_DB_MEMORIES_CONTAINER=memories
COSMOS_DB_COUNTERS_CONTAINER=counter
COSMOS_DB_LEASE_CONTAINER=leases
COSMOS_DB_THROUGHPUT_MODE=serverless
COSMOS_DB_AUTOSCALE_MAX_RU=1000

AI_FOUNDRY_ENDPOINT=https://<openai-account-name>.openai.azure.com/
AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AI_FOUNDRY_EMBEDDING_DIMENSIONS=1536
AI_FOUNDRY_CHAT_DEPLOYMENT_NAME=gpt-5-mini

# Tells the SDK that the deployed Function App owns auto-processing,
# so this client skips its in-process auto-trigger.
MEMORY_PROCESSOR_OWNER=durable
```

---

## 6. Create Cosmos Resources

Run once if the database and container do not already exist:

### Sync

```python
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()

memory = CosmosMemoryClient(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE", "ai_memory"),
    cosmos_container=os.getenv("COSMOS_DB_MEMORIES_CONTAINER", "memories"),
    cosmos_counter_container=os.getenv("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
    cosmos_lease_container=os.getenv("COSMOS_DB_LEASE_CONTAINER", "leases"),
    cosmos_throughput_mode=os.getenv("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
    cosmos_autoscale_max_ru=int(os.getenv("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_deployment_name=os.getenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
    chat_deployment_name=os.getenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-5-mini"),
    use_default_credential=True,
    cosmos_credential=DefaultAzureCredential(),
)

memory.create_memory_store()
memory.connect_cosmos()
```

### Async

```python
import os
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from agent_memory_toolkit.aio import AsyncCosmosMemoryClient

load_dotenv()

memory = AsyncCosmosMemoryClient(
    cosmos_endpoint=os.getenv("COSMOS_DB_ENDPOINT"),
    cosmos_database=os.getenv("COSMOS_DB_DATABASE", "ai_memory"),
    cosmos_container=os.getenv("COSMOS_DB_MEMORIES_CONTAINER", "memories"),
    cosmos_counter_container=os.getenv("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
    cosmos_lease_container=os.getenv("COSMOS_DB_LEASE_CONTAINER", "leases"),
    cosmos_throughput_mode=os.getenv("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
    cosmos_autoscale_max_ru=int(os.getenv("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
    ai_foundry_endpoint=os.getenv("AI_FOUNDRY_ENDPOINT"),
    embedding_deployment_name=os.getenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
    chat_deployment_name=os.getenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-5-mini"),
    use_default_credential=True,
    cosmos_credential=AsyncDefaultAzureCredential(),
)

await memory.connect_cosmos()
await memory.create_memory_store()
```

This provisions the `memories`, `memories_turns`, `memories_summaries`, `counter`, and `leases` containers. `serverless` is the default throughput mode; if you set `COSMOS_DB_THROUGHPUT_MODE=autoscale`, the shared `COSMOS_DB_AUTOSCALE_MAX_RU` value is applied to all five containers.

---

## 7. Validation Order

Bring the environment up in this order:

1. `az login`
2. verify Cosmos DB RBAC
3. verify Azure OpenAI RBAC
4. create Cosmos resources with `create_memory_store()`
5. test `add_cosmos()` / `push_to_cosmos()` / `get_memories()`
6. test `get_memories(user_id=..., thread_id=...)` filtering
7. test `search_cosmos()`
8. deploy the Function App (e.g., via `azd up`) so the change-feed processor is running
9. write a few turns and verify a thread `summary` memory appears
10. write more turns and verify `fact`, `procedural`, and `episodic` memories appear
11. verify a per-user `user_summary` memory appears once `USER_SUMMARY_EVERY_N` turns have accumulated for that user
12. test deduplication by writing two near-duplicate facts and confirming the dedup orchestrator merges them

This keeps failures isolated and easier to diagnose.

---

## 8. Validate Processing

### Basic Cosmos operations

```python
memory.add_cosmos(user_id="user-1", role="user", content="Hello from Azure")
print(memory.get_memories(user_id="user-1"))
```

### Semantic search

```python
print(memory.search_cosmos("hello", user_id="user-1"))
```

### Durable processing (change-feed driven)

Processing is no longer invoked directly from the SDK — write turns with `add_cosmos()` / `push_to_cosmos()` and the deployed Function App's change-feed trigger fires the `extract_memories`, `thread_summary`, and `user_summary` orchestrators per the configured thresholds.

```python
# Write enough turns to cross THREAD_SUMMARY_EVERY_N (default 10).
for i in range(10):
    memory.add_cosmos(
        user_id="user-1",
        thread_id="thread-1",
        role="user",
        content=f"Turn {i+1}",
    )

# Wait for the change-feed processor to catch up, then read derived memories.
import time; time.sleep(15)
print(memory.get_thread_summary(user_id="user-1", thread_id="thread-1"))
print(memory.get_memories(user_id="user-1", memory_types=["fact"]))
print(memory.get_user_summary(user_id="user-1"))
```

### Change feed auto-processing

If you configured the change feed settings, verify automatic processing:

```python
import uuid

# Use a threshold of 3 (THREAD_SUMMARY_EVERY_N=3) for testing
thread_id = str(uuid.uuid4())
for i in range(3):
    memory.add_cosmos(
        user_id="user-1",
        thread_id=thread_id,
        role="user",
        content=f"Turn {i+1} for change feed validation",
    )

# Wait a few seconds for the change feed to trigger, then check:
import time
time.sleep(10)
results = memory.get_thread_summary(user_id="user-1", thread_id=thread_id)
print(results)  # Should contain an auto-generated summary
```

Check the Function App logs to confirm the `on_memory_change` trigger fired and the orchestrator completed.

### Verify stored results

```python
print(memory.get_thread_summary(user_id="user-1", thread_id="thread-1"))
print(memory.get_memories(user_id="user-1", memory_types=["fact"]))
print(memory.get_user_summary(user_id="user-1"))
```

---

## Monitoring and Troubleshooting

Tail Function App logs:

```bash
az functionapp log tail \
  --name <function-app-name> \
  --resource-group <resource-group>
```

Common issues:

| Symptom | Likely Cause |
|---------|--------------|
| 401 / 403 from Cosmos DB | Missing Cosmos DB RBAC |
| 401 / 403 from Azure OpenAI | Missing OpenAI RBAC |
| Durable Function starts but fails | Missing app settings or downstream RBAC |
| `No memories found` | No turn memories exist, or all candidate turns predate the existing summary |
| Search is slow | Embedding latency, index choice, or region mismatch |
| Change feed trigger not firing | Verify `COSMOS_DB__accountEndpoint` is set and the function can write to the configured `COSMOS_DB_COUNTERS_CONTAINER` container |
| Auto-processing not starting | Check threshold settings are > 0 in Function App configuration |

Recommended checks:

- enable Application Insights
- confirm Function App managed identity roles
- confirm `MEMORY_PROCESSOR_OWNER=durable` is set on any client pointed at a container that the Function App is also processing
- confirm model deployment names are correct
