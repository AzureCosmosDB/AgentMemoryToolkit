## Release History

## [Unreleased]

#### Features Added
* **New extraction prompt.** `extract_memories-v3.prompty` is now the default. Set `AMT_EXTRACT_MEMORIES_PROMPT` to use an earlier prompt.
* **Duplicates are folded and combined.** Before resolving contradictions, `reconcile()` retires a fact into another that states everything it states (`supersede_reason="duplicate"`, with `merged_from` on the survivor) and combines facts about one specific thing into one `sys:combined` fact (`supersede_reason="combined"`). It returns counts for both passes, and `DEDUP_POOL_SIZE` limits how many new facts one pass checks.
* **Procedural memory is learned from conversations.** The toolkit learns standing instructions, habits and skills from what users teach and correct, and `build_procedural_context(user_id, task=None)` returns the rules to add to a prompt.
* **Search slots.** `search_cosmos(preference_slot=True, procedure_slot=True)` reserves places in `top_k` for the user's best-matching preference and situational procedure, so closer-matching facts do not crowd them out.
* **Secrets are redacted on write.** Common credential formats (keys, tokens, connection strings, private keys, passwords in URLs) are replaced with `[REDACTED-SECRET]` before text is embedded or stored, and the record is tagged `sys:secret`.
* **Deleting a memory removes what was folded into it.** `delete_memory()` also deletes the records retired into the deleted fact or procedure.
* **Time-budgeted passes.** `PipelineService.reconcile_memories()`, `consolidate_memories()` and `synthesize_procedural()` accept a `deadline`. The Function App sets it from the new `ACTIVITY_TIME_BUDGET_SECONDS` (default 240), so a run stops before the function timeout and the next one continues.
* **Cosmos DB read regions.** `cosmos_preferred_locations` (or `COSMOS_DB_PREFERRED_REGIONS`) sets the regions the SDK reads from, closest first.

#### Breaking Changes
* **Procedural memory is reshaped** `procedure_kind` is now `instruction`, `habit` or `skill`, and a procedure's `status` follows from its evidence. `scope_type` / `scope_value`, `ProcedureScopeType` and `synthesize_procedural.prompty` are removed, `synthesize_procedural()` works incrementally, `build_procedural_context()` returns only standing rules (plus the nearest situational rule for a `task`), and `search_cosmos()` hides procedures that are not active. Delete or ignore procedural documents written by earlier betas.
* **Logs and failure messages no longer quote exception text.** JSON logs write `exc_detail` (type, error code and location) instead of `exc_msg`, so update any query or alert that reads `exc_msg`. A failed Function App activity now reports only `<code> (<ExcType>)` to Durable Functions.
* **Bicep creates `memories_summaries` with a vector and full-text index**, as `create_memory_store()` already does, so `search_summaries()` works on a deployment from the template. Cosmos DB sets a vector policy only when a container is created: an environment provisioned from an earlier template needs that container created again, and `azd up` over it can fail.

#### Bugs Fixed
* Procedure learning no longer ignores newer facts and episodes.
* Turns with the same `created_at`, as in an imported session, are processed in write order, and `recent_k` keeps the newest.
* Empty model replies are retried, and content-filtered or refused replies raise an `LLMError` that names the cause instead of being retried on every run.
* The Function App parses boolean settings like the SDK does, and a `MAX_BATCH_SIZE` below 1 falls back to 20 instead of silently skipping the backlog.

#### Other Changes
* New prompts `extract_memories-v3`, `dedup_verify` and `combine_memories`; `extract_procedure` is rewritten. Every memory-writing prompt now tells the model never to write a literal secret.
* Search results include the new fact and procedure fields and `merged_from`, and `MemoryStore.search()` gains `categories`, `vector_only`, `active_procedures_only`, `applies` and `query_embedding`.
* Bicep adds the `dedupPoolSize`, `activityTimeBudgetSeconds`, `chatCapacity` and `embeddingCapacity` parameters (azd variables `DEDUP_POOL_SIZE`, `ACTIVITY_TIME_BUDGET_SECONDS`, `AI_FOUNDRY_CHAT_CAPACITY`, `AI_FOUNDRY_EMBEDDING_CAPACITY`). New `counter` containers expire their documents 90 days after the last update; existing containers are unchanged.
* The SDK now requires `azure-cosmos>=4.16.3`.

## [0.3.0b2] (2026-08-17)

#### Features Added
* Episodic memory is now a first-class memory type. Bounded experiences are segmented from the turn stream at idle-gap, topic-drift, and max-size boundaries, each captured as an `EpisodicRecord` with a summary, timeline events, an optional outcome, and first-class `lessons`. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* `search_cosmos(include_episodes=True)` blends facts and episodes into a single ranked query sharing one `top_k` budget, and `search_episodic_memories()` searches episodes directly. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* Fact extraction now defaults to the higher-recall v2 prompt (`extract_memories-v2.prompty`); set `AMT_EXTRACT_MEMORIES_PROMPT=extract_memories.prompty` to fall back to v1. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* Procedural memory is now an atomic, retrievable skill and policy library. `ProceduralRecord` stores individual procedures (behavioral policies, workflows, decision rules, tool-usage notes, recovery strategies) with scope, activation conditions, steps, status, and source provenance. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* The Durable Functions backend now runs episodic extraction, driven by the Cosmos DB change feed on a per-thread cadence set via `EPISODE_EVAL_EVERY_N` (with `EPISODE_IDLE_GAP_SECONDS`, `EPISODE_TOPIC_DRIFT`, `EPISODE_MAX_TURNS`, and `EPISODE_MIN_TURNS` mirrored on the Functions side). See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* New delete helpers on both clients: `delete_turn()`, `delete_thread_summary()`, `delete_user_summary()`, and bulk `delete_thread()` (removes a thread's turns and, by default, its summary; distilled facts, episodes, and procedures are left intact). See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)

#### Breaking Changes
* The default fact-extraction cadence is now every 2 turns (`FACT_EXTRACTION_EVERY_N=2`) instead of every turn, across the SDK, and the Functions deploy default. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* `add_cosmos()` is renamed to `upsert_memory()` on both clients and the store; behavior is unchanged (write-or-replace by id). See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* `delete_cosmos()` is renamed to `delete_memory()`. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* Procedural memory has been reshaped: `ProceduralRecord` is now an atomic procedure rather than a single compiled system-prompt document, and the compiled prompt is produced on demand by `build_procedural_context()`. Pre-existing single-prompt procedural documents from earlier betas are not migrated. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* Write-time vector deduplication (in-place fold) has been removed, along with the `DEDUP_VECTOR_ENABLED` and similarity-threshold knobs. Fact dedup is now in-batch hash plus deterministic-id create/409; contradiction reconciliation is unchanged. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)

#### Bugs Fixed
* Episode extraction now isolates LLM failures: a transient error leaves the open segment un-stamped for retry, while a non-retryable error (content filter, context-length) quarantines the segment so it can neither wedge the thread nor grow it without bound. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* Out-of-range or non-numeric fact `salience` / `confidence` values are clamped instead of aborting the whole extraction batch and stalling the fact watermark. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* Segment time bounds and the open-episode-segment loader now order turns chronologically by parsed timestamp, so mixed UTC offsets and tied timestamps no longer invert episode bounds or destabilize the deterministic episode id. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* Per-turn extraction watermarks are stamped with a single-field conditional patch, so concurrent fact and episode writers no longer clobber each other's watermark field. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)
* `parse_llm_json` now rejects a non-object JSON root with a typed error instead of letting it surface downstream as a misclassified transient failure. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)
* Threshold environment values of `NaN` / `inf` are rejected instead of silently disabling the affected boundary. See [PR:#37](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/37)

#### Other Changes
* Fact hash-dedup no longer issues a per-extraction query to preload the user's existing fact hashes; exact duplicates are caught in-batch and by the deterministic-id create (409), reducing per-turn latency. See [PR:#38](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/38)

## [0.3.0b1] (2026-07-24)

#### Features Added
* Searchable summaries: the summaries container now carries a vector index, and `search_summaries()` / `search_cosmos(include_summaries=True)` vector-search across a user's thread + user summaries. See [PR:#34](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/34)
* Agent-sourced facts: extraction can now capture the assistant's own actions and recommendations (not just user statements), tagged `source=agent` / `sys:agent-fact` so retrieval can include or exclude them. See [PR:#31](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/31)
* Event time on write: `add_cosmos(..., created_at=...)` sets a memory's event time (falls back to ingestion time), enabling time-aware retrieval and temporal reasoning over backfilled conversations. See [PR:#31](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/31)
* Unified retrieval: `search_cosmos(include_turns=True)` blends raw conversation turns into results alongside extracted memories. See [PR:#31](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/31)
* `DEDUP_VECTOR_ENABLED` is now an environment knob (default `false` = add-only) instead of a fixed internal constant. See [PR:#31](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/31)
* Write-time in-place deduplication: near-duplicate memories fold into the existing record (same id, newer content) instead of creating a new doc. See [PR:#26](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/26)
* Reconciliation now resolves contradictions only, soft-deleting the loser with `superseded_by`; `get_memory_history()` walks that chain. See [PR:#26](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/26)

#### Bugs Fixed
* Fixed a re-extraction loop that re-extracted the whole conversation every cycle (turns were never stamped `extracted_at` when vector dedup was on). See [PR:#26](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/26)

#### Other Changes
* Extraction prompt v3: named attribution (use real names, not a generic "the user", so multi-speaker turns don't blur), stronger temporal grounding (resolve relative dates against the turn's time, never the current clock), and meaning/completeness rules. See [PR:#34](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/34)
* Extraction now sees per-turn timestamps and resolves relative dates ("3 weeks ago") to absolute dates in the fact text (time-range filtering uses the memory's `created_at`). See [PR:#31](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/31)
* Reworked the extraction prompt (anti-inference, preserve specifics, topic-grouped memories) and simplified the schema to `fact`/`episodic` with fixed fact categories. See [PR:#26](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/26)
* Token-bounded extraction batches with per-batch failure isolation; embedding inputs truncated to the model token budget. See [PR:#26](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/26)

## [0.2.0b3] (2026-07-08)

#### Features Added
* A custom user-agent can now be supplied via the new `user_agent` constructor argument on `CosmosMemoryClient` and `AsyncCosmosMemoryClient`. The toolkit's own user-agent (`azsdk-python-cosmos-agent-memory/<version>`) is always sent to Azure Cosmos DB so usage can be tracked; when a custom value is provided it is prefixed and the toolkit's user-agent is suffixed behind it (`"<custom> <toolkit>"`). See [PR:#30](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/30)
* Per-turn processing cadence can now be set in-process via the new `cadence_thresholds` constructor argument on `CosmosMemoryClient` and `AsyncCosmosMemoryClient`, instead of only through environment variables. Pass a mapping keyed by the same names as the env vars (e.g. `FACT_EXTRACTION_EVERY_N`,`DEDUP_EVERY_N`, `THREAD_SUMMARY_EVERY_N`, `USER_SUMMARY_EVERY_N`); any key not present falls back to the environment/defaults, and `None` preserves today's env-only behavior. See [PR:#29](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/29)
## [0.2.0b2] (2026-07-01)

#### Features Added
* Embeddings and chat clients can now be injected via the new `embeddings_client` and `chat_client` constructor arguments on `CosmosMemoryClient` and `AsyncCosmosMemoryClient`. See [PR:#27](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/27)

## [0.2.0b1] (2026-06-30)

#### Features Added
* Raw conversation turns can now be embedded and vector-searched. Set `enable_turn_embeddings=True` (env `ENABLE_TURN_EMBEDDINGS`) to generate an embedding when each turn is written, then call `search_turns()` (sync and async, on both the client and store) to semantically search the raw turn log. See [PR:#22](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/22/)

#### Other Changes
* The memories container's vector index type is now configurable instead of being hard-coded to `diskANN`. Set it via the `vector_index_type` argument to `create_memory_store(...)` or the `AI_FOUNDRY_EMBEDDING_VECTOR_INDEX_TYPE` environment variable. See [PR:#24](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/24)
* `ai_foundry_endpoint` now accepts a project-scoped Azure AI Foundry URL (`https://<resource>.services.ai.azure.com/api/projects/<name>`) in addition to the account-level inference endpoint. See [PR:#23](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/23)

## [0.1.0b2] (2026-06-03)

#### Bugs Fixed
* Hardened memory extraction: stops emitting phantom/synthesized facts the user never asserted, stops extracting facts from `[assistant]:` turns, stops re-processing already-extracted turns (which previously produced reversed `CONTRADICT` decisions and meta-facts like `"X is contradicted by Y"`), and stops storing near-duplicate episodic memories for the same scope. Episodic memories also now embed the actual content instead of a boilerplate `"intent recorded"` string. See [PR:#20](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/20/)
* Fixed `add_cosmos` + `process_now` silently bypassing the cadence subsystem: cadence env vars (`THREAD_SUMMARY_EVERY_N`, `FACT_EXTRACTION_EVERY_N`, `USER_SUMMARY_EVERY_N`, etc.) had no effect, and procedural / user-summary synthesis never ran. `add_cosmos` now triggers cadence on turn writes; `process_now` now runs the full 5-step pipeline on the in-process processor.See [PR:#20](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/20/)

#### Other Changes
* `ProcessThreadResult` gains `procedural` and `user_summary` fields. `extract_memories` returns a `dropped_episodic_count` for monitoring LLM-extraction quality.See [PR:#20](https://github.com/aayush3011/AgentMemoryToolkit/pull/20)


## [0.1.0b1] - 2026-06-01

Initial public preview release.

This is a **beta release**. The public surface may evolve in backward-incompatible ways before the `1.0.0` general-availability cut.
Pin a specific version when integrating.

#### Added

- Sync (`CosmosMemoryClient`) and async (`AsyncCosmosMemoryClient`) clients for storing, retrieving, and transforming agent memories backed by Azure Cosmos DB.
- Typed memory record hierarchy (Pydantic): `TurnRecord`, `FactRecord`, `EpisodicRecord`, `ProceduralRecord`, `ThreadSummaryRecord`, `UserSummaryRecord`.
- Vector + full-text + hybrid search over memories with metadata filters, tag filters, and per-type scoping.
- Built-in memory processing pipeline: fact extraction, thread/user summarization, procedural-memory synthesis, contradiction handling, and deduplication - all driven by versioned `.prompty` prompts.
- Two processor backends: `InProcessProcessor` (default, runs in your application process) and `DurableFunctionProcessor` (offloads work to a sibling Azure Function app via Cosmos DB change feed).
- One-command `azd up` deployment that provisions Cosmos DB (with vector + full-text search enabled), Azure AI Foundry (chat + embedding deployments), Azure Function app (Flex Consumption), Storage, App Insights, and the User-Assigned Managed Identity wiring all of it together.
- Focused exception hierarchy: `AgentMemoryError`, `ConfigurationError`, `ValidationError`, `CosmosNotConnectedError`, `CosmosOperationError`, `MemoryNotFoundError`, `MemoryTypeMismatchError`, `LLMError`.
- Structured JSON logging via `azure.cosmos.agent_memory.logging` (`configure_logging`, `JsonFormatter`).

#### Package layout

- Distribution name: **`azure-cosmos-agent-memory`** (PyPI)
- Import path: **`azure.cosmos.agent_memory`** 
