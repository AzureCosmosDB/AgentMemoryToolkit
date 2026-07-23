## Release History

## [0.3.0b1] (Unreleased)

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
* A custom user-agent can now be supplied via the new `user_agent` constructor
  argument on `CosmosMemoryClient` and `AsyncCosmosMemoryClient`. The toolkit's
  own user-agent (`azsdk-python-cosmos-agent-memory/<version>`) is always sent to
  Azure Cosmos DB so usage can be tracked; when a custom value is provided it is prefixed and
  the toolkit's user-agent is suffixed behind it (`"<custom> <toolkit>"`). See [PR:#30](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/30)
* Per-turn processing cadence can now be set in-process via the new
  `cadence_thresholds` constructor argument on `CosmosMemoryClient` and
  `AsyncCosmosMemoryClient`, instead of only through environment variables. Pass a
  mapping keyed by the same names as the env vars (e.g. `FACT_EXTRACTION_EVERY_N`,
  `DEDUP_EVERY_N`, `THREAD_SUMMARY_EVERY_N`, `USER_SUMMARY_EVERY_N`); any key not
  present falls back to the environment/defaults, and `None` preserves today's
  env-only behavior. See [PR:#29](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/29)
## [0.2.0b2] (2026-07-01)

#### Features Added
* Embeddings and chat clients can now be injected via the new `embeddings_client`
  and `chat_client` constructor arguments on `CosmosMemoryClient` and
  `AsyncCosmosMemoryClient`. See [PR:#27](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/27)

## [0.2.0b1] (2026-06-30)

#### Features Added
* Raw conversation turns can now be embedded and vector-searched. Set
  `enable_turn_embeddings=True` (env `ENABLE_TURN_EMBEDDINGS`) to generate an
  embedding when each turn is written, then call `search_turns()` (sync and
  async, on both the client and store) to semantically search the raw turn
  log. See [PR:#22](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/22/)

#### Other Changes
* The memories container's vector index type is now configurable instead of being
  hard-coded to `diskANN`. Set it via the `vector_index_type` argument to
  `create_memory_store(...)` or the `AI_FOUNDRY_EMBEDDING_VECTOR_INDEX_TYPE`
  environment variable. See [PR:#24](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/24)
* `ai_foundry_endpoint` now accepts a project-scoped Azure AI Foundry URL
  (`https://<resource>.services.ai.azure.com/api/projects/<name>`) in addition
  to the account-level inference endpoint. See [PR:#23](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/23)

## [0.1.0b2] (2026-06-03)

#### Bugs Fixed
* Hardened memory extraction: stops emitting phantom/synthesized facts the user never asserted, stops extracting facts from `[assistant]:` turns, stops re-processing already-extracted turns (which previously produced reversed `CONTRADICT` decisions and meta-facts like `"X is contradicted by Y"`), and stops storing near-duplicate episodic memories for the same scope. Episodic memories also now embed the actual content instead of a boilerplate `"intent recorded"` string. See [PR:#20](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/20/)
* Fixed `add_cosmos` + `process_now` silently bypassing the cadence subsystem: cadence env vars (`THREAD_SUMMARY_EVERY_N`, `FACT_EXTRACTION_EVERY_N`, `USER_SUMMARY_EVERY_N`, etc.) had no effect, and procedural / user-summary synthesis never ran. `add_cosmos` now triggers cadence on turn writes; `process_now` now runs the full 5-step pipeline on the in-process processor.See [PR:#20](https://github.com/AzureCosmosDB/AgentMemoryToolkit/pull/20/)

#### Other Changes
* `ProcessThreadResult` gains `procedural` and `user_summary` fields. `extract_memories` returns a `dropped_episodic_count` for monitoring LLM-extraction quality.See [PR:#20](https://github.com/aayush3011/AgentMemoryToolkit/pull/20)


## [0.1.0b1] — 2026-06-01


Initial public preview release.

This is a **beta release**. The public surface may evolve in
backward-incompatible ways before the `1.0.0` general-availability cut.
Pin a specific version when integrating.

#### Added

- Sync (`CosmosMemoryClient`) and async (`AsyncCosmosMemoryClient`) clients
  for storing, retrieving, and transforming agent memories backed by Azure
  Cosmos DB.
- Typed memory record hierarchy (Pydantic): `TurnRecord`, `FactRecord`,
  `EpisodicRecord`, `ProceduralRecord`, `ThreadSummaryRecord`,
  `UserSummaryRecord`.
- Vector + full-text + hybrid search over memories with metadata filters,
  tag filters, and per-type scoping.
- Built-in memory processing pipeline: fact extraction, thread/user
  summarization, procedural-memory synthesis, contradiction handling, and
  deduplication — all driven by versioned `.prompty` prompts.
- Two processor backends: `InProcessProcessor` (default, runs in your
  application process) and `DurableFunctionProcessor` (offloads work to a
  sibling Azure Function app via Cosmos DB change feed).
- One-command `azd up` deployment that provisions Cosmos DB (with vector +
  full-text search enabled), Azure AI Foundry (chat + embedding
  deployments), Azure Function app (Flex Consumption), Storage, App
  Insights, and the User-Assigned Managed Identity wiring all of it
  together.
- Focused exception hierarchy: `AgentMemoryError`, `ConfigurationError`,
  `ValidationError`, `CosmosNotConnectedError`, `CosmosOperationError`,
  `MemoryNotFoundError`, `MemoryTypeMismatchError`, `LLMError`.
- Structured JSON logging via `azure.cosmos.agent_memory.logging`
  (`configure_logging`, `JsonFormatter`).

#### Package layout

- Distribution name: **`azure-cosmos-agent-memory`** (PyPI)
- Import path: **`azure.cosmos.agent_memory`** 
