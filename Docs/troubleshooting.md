# Troubleshooting Agent Memory Toolkit

Use this guide when local memory works but Cosmos DB, embeddings, Durable Functions, or automatic change feed processing does not.

---

## Quick Triage

| Symptom | First checks |
|---------|--------------|
| Import errors | Install with `pip install -e ".[dev]"` and import `CosmosMemoryClient` or `AsyncCosmosMemoryClient`. |
| Missing configuration | Verify `.env`, `function_app/local.settings.json`, and Azure Function App settings use the same endpoint, database, container, and AI deployment values. |
| Cosmos 401 or 403 | Run `az login` and confirm Cosmos DB data-plane RBAC is assigned. |
| Cosmos operations fail before connecting | Call `create_memory_store()` or `connect_cosmos()` before cloud operations. |
| Search returns no vector results | Confirm embeddings are generated and `AI_FOUNDRY_EMBEDDING_DIMENSIONS` matches the container vector policy. |
| A fact, procedure or preference is missing from search results | It may be retired (`superseded_by` is set; pass `include_superseded=True`) or, for a procedure, still a `candidate` or past `valid_until`. See section 4. |
| A stored memory contains `[REDACTED-SECRET]` | The write guard replaced a credential it recognized and tagged the record `sys:secret`; the credential itself is not written to Cosmos DB. See [Secrets](design_patterns.md#11-secrets-and-deleting-memories). |
| Durable Functions processing fails | Start the Functions host and check `function_app/local.settings.json`, the change feed trigger, and the orchestrator logs. |
| A Durable activity fails with only `<code> (<ExcType>)` | The error message is withheld on purpose; read the `Activity ... failed` line in the Function App log. See section 5. |
| Change feed does not create summaries or facts | Confirm change feed settings, thresholds, lease container, counter container, and that inserted documents have `type: "turn"`. |

---

## 1. Environment And Imports

Install the package from the repository root:

```bash
pip install -e ".[dev]"
pip install -r function_app/requirements.txt
```

The public clients are:

```python
from azure.cosmos.agent_memory import CosmosMemoryClient
from azure.cosmos.agent_memory.aio import AsyncCosmosMemoryClient
```

If notebooks cannot import the package, run them from the repo root with paths such as `Samples/Notebooks/Demo.ipynb`, or add the repository root to `sys.path`.

---

## 2. Configuration And Authentication

For local runs, keep `.env`, `function_app/local.settings.json`, and deployed Function App settings aligned:

```env
COSMOS_DB_ENDPOINT=https://<account>.documents.azure.com:443/
COSMOS_DB__accountEndpoint=https://<account>.documents.azure.com:443/
COSMOS_DB_KEY=
COSMOS_DB_DATABASE=ai_memory
COSMOS_DB_MEMORIES_CONTAINER=memories
COSMOS_DB_COUNTERS_CONTAINER=counter
COSMOS_DB_LEASE_CONTAINER=leases

AI_FOUNDRY_ENDPOINT=https://<account>.openai.azure.com/
AI_FOUNDRY_API_KEY=
AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AI_FOUNDRY_EMBEDDING_DIMENSIONS=1536
AI_FOUNDRY_CHAT_DEPLOYMENT_NAME=<chat-deployment-name>
```

The notebooks and samples pass these values into the client like this:

| `.env` setting | Client argument |
|---|---|
| `COSMOS_DB_ENDPOINT` | `cosmos_endpoint` |
| `COSMOS_DB_DATABASE` | `cosmos_database` |
| `COSMOS_DB_MEMORIES_CONTAINER` | `cosmos_container` |
| `COSMOS_DB_COUNTERS_CONTAINER` | `cosmos_counter_container` |
| `COSMOS_DB_LEASE_CONTAINER` | `cosmos_lease_container` |
| `COSMOS_DB_KEY` | `cosmos_key` |
| `AI_FOUNDRY_ENDPOINT` | `ai_foundry_endpoint` |
| `AI_FOUNDRY_API_KEY` | `ai_foundry_api_key` |
| `AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME` | `embedding_deployment_name` |
| `AI_FOUNDRY_CHAT_DEPLOYMENT_NAME` | `chat_deployment_name` |
| `COSMOS_DB_PREFERRED_REGIONS` | `cosmos_preferred_locations` |

`COSMOS_DB_PREFERRED_REGIONS` is optional: a comma-separated list of Cosmos DB regions, closest first (for example `West US 3,West US 2`). The client reads the variable itself when the argument is omitted, and the argument overrides it. The Function App does not read it.

Run `az login` before using `DefaultAzureCredential`.

Required roles:

| Service | Role |
|---------|------|
| Cosmos DB | Cosmos DB Built-in Data Contributor |
| Azure OpenAI / AI Services | Cognitive Services OpenAI User |

RBAC changes can take several minutes to propagate.

---

## 3. Cosmos DB Store Creation

Run `create_memory_store()` before relying on cloud operations. It creates the database plus the `memories`, `memories_turns`, `memories_summaries`, `counter`, and `leases` containers.

The memories container is created with:

- hierarchical partition key on `user_id` and `thread_id`
- vector index on `/embedding`
- full-text index on `/content`

If vector or full-text search fails after changing dimensions or indexing settings, create a fresh container with the desired configuration. Cosmos container vector policies are creation-time infrastructure choices.

Pass `cosmos_throughput_mode="serverless"` (the default) when creating the client. Use `cosmos_throughput_mode="autoscale"` with `cosmos_autoscale_max_ru` when you need provisioned autoscale throughput.

---

## 4. Embeddings And Search

Embedding failures usually mean one of these is wrong:

- `AI_FOUNDRY_ENDPOINT`
- `AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME`
- `AI_FOUNDRY_EMBEDDING_DIMENSIONS`
- Azure OpenAI / AI Services RBAC

Search always uses hybrid vector/full-text ranking when keyword extraction finds terms; all-stopword queries fall back to vector-only ranking.

If search returns documents but scores look poor, check that records have an `embedding` field and that the query uses similar language to the stored memory content.

Keyword searches read across partitions. The Python SDK skips query planning for a query sent with a partition key, so `FullTextScore` ranking is ignored ([Azure/azure-sdk-for-python#42241](https://github.com/Azure/azure-sdk-for-python/issues/42241)); a keyword query therefore filters on `c.user_id` (and `c.thread_id`, when given) in the `WHERE` clause instead and sets `full_text_score_scope="Local"`. Results stay confined to the same user and thread (in `search_cosmos`, episodic and procedural memories match on the user alone), but a large container costs more RUs than a single-partition read. A vector-only query with both `user_id` and `thread_id` still reads one partition, except a `search_cosmos` that searches episodic or procedural memories, which always runs across partitions. If a custom query sent with a partition key raises a `ValidationError` that mentions `FullTextScore`, drop the partition key and filter on the partition key fields in the `WHERE` clause.

Search hides some records on purpose:

- A retired fact (`superseded_by` is set) is hidden unless you pass `include_superseded=True`. `supersede_reason` says why: `duplicate`, `combined`, `contradict`, `update` or `procedure`. The survivor of a duplicate or a combine lists the retired facts in `merged_from`.
- `search_cosmos` shows a procedure only while it is `active` and not past `valid_until`. A `candidate` stays hidden until its evidence arrives; `get_procedural_memories` and `retrieve_procedures(..., status=None)` show candidates.

If `preference_slot` or `procedure_slot` adds nothing:

- Slots need `user_id`, are skipped with `memory_id`, and need `top_k` larger than the number of slots.
- A slot with no candidate takes the next ranked result; its `retrieval["slot"]` is `backfill`.
- Only a fact whose `metadata.category` is `preference` takes the preference slot (facts must be searched), and only an active situational procedure (`applies="when"`) takes the procedure slot (it has no effect when only procedural types are searched).

---

## 5. Durable Functions Processing

Durable Functions processing requires the Functions host.

Start local dependencies:

```bash
azurite --silent --location /tmp/azurite --debug /tmp/azurite/debug.log
cd function_app
func start
```

The SDK does not post to a Function endpoint. With `DurableFunctionProcessor`, the SDK writes turns to Cosmos DB and the deployed Function App picks them up from the Cosmos DB change feed. For local testing, keep `function_app/local.settings.json` aligned with `.env` and confirm the Functions host starts the change feed trigger.

If orchestration polling times out, check the Functions logs first. The orchestration may still be running, or an activity may be waiting on Cosmos DB or the LLM endpoint.

A failed activity reaches Durable, and so the orchestration status, as `<code> (<ExcType>)` and nothing more, for example `write_conflict (CosmosAccessConditionFailedError)`, `llm (LLMError)` or `dependency_failure (APIConnectionError)`. The code is `write_conflict` for an HTTP 412, `memory_not_found` for a 404, the toolkit's own error code (`llm`, `validation`, `cosmos_operation` and so on), or `dependency_failure` for anything else. The message is withheld because an exception can quote model output or memory text, and Durable keeps errors in the orchestration history. The detail is in the Function App log, on the line `Activity <name> failed error_code=<code>: <details>`: the exception type, HTTP status, validation fields and the code locations it passed through, never its message or arguments. The failure is still raised, so the activity retry applies as before.

The toolkit's own logs follow the same rule: no exception message, memory text, search query or model output is logged. In JSON logs (`JsonFormatter`, used when stderr is not a terminal) a record with an exception has `exc_type` and `exc_detail`.

Reconcile and procedure activities stop starting model calls after `ACTIVITY_TIME_BUDGET_SECONDS` (default `240`, minimum `30`) and continue from their cursors on the next run. If one still reaches the Function App timeout, lower the setting.

An empty model reply is retried with exponential backoff (about a minute over six attempts) and then raises `LLMError`, so an activity can stay busy that long before it fails. A content-filtered, refused or tool-only reply, or one cut off at the output limit with no text, raises `LLMError` at once and is not retried. During extraction, the batch that drew such a reply is quarantined: it yields no facts, is counted in `quarantined_turn_count`, and is not tried again.

---

## 6. Change Feed Processing

Automatic processing requires these settings in the Functions app or `local.settings.json`:

```json
"COSMOS_DB__accountEndpoint": "https://<account>.documents.azure.com:443/",
"COSMOS_DB_ENDPOINT": "https://<account>.documents.azure.com:443/",
"COSMOS_DB_DATABASE": "ai_memory",
"COSMOS_DB_MEMORIES_CONTAINER": "memories",
"COSMOS_DB_TURNS_CONTAINER": "memories_turns",
"COSMOS_DB_SUMMARIES_CONTAINER": "memories_summaries",
"COSMOS_DB_COUNTERS_CONTAINER": "counter",
"COSMOS_DB_LEASE_CONTAINER": "leases",
"MEMORY_PROCESSOR_OWNER": "durable",
"AI_FOUNDRY_ENDPOINT": "https://<account>.openai.azure.com/",
"AI_FOUNDRY_CHAT_DEPLOYMENT_NAME": "gpt-4o-mini",
"AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-large",
"THREAD_SUMMARY_EVERY_N": "5",
"FACT_EXTRACTION_EVERY_N": "3",
"USER_SUMMARY_EVERY_N": "10"
```

Set a threshold to `"0"` to disable that processing type. `MEMORY_PROCESSOR_OWNER` must be `"durable"` for the Function App's change-feed trigger to actually fire - leave it unset (or set to `"inprocess"`) for SDK-only deployments.

Cosmos DB memory documents store their category in the JSON `type` field. Only documents with `type: "turn"` increment counters. Derived memories with `type: "thread_summary"`, `type: "fact"`, `type: "episodic"`, `type: "procedural"`, or `type: "user_summary"` do not trigger threshold counts.

If nothing fires:

- verify the Functions host shows the Cosmos DB trigger
- confirm the `leases` container exists
- confirm the `memories`, `memories_turns` and `memories_summaries` containers exist; a missing one fails the batch with `Topology error: container '<name>' does not exist`
- confirm the `counter` container is writable
- confirm `MEMORY_PROCESSOR_OWNER=durable` is set on the Function App
- insert enough new turn documents to cross the configured threshold
- check for generated documents where the Cosmos JSON field is `type="thread_summary"`, `type="fact"`, or `type="user_summary"`

---

## 7. Async Client Notes

Use async Azure credentials with the async client:

```python
from azure.identity.aio import DefaultAzureCredential
from azure.cosmos.agent_memory.aio import AsyncCosmosMemoryClient
```

Always `await` cloud operations and close the client when done:

```python
await memory.close()
```

In notebooks, top-level `await` is supported, so do not wrap cells with `asyncio.run()`.