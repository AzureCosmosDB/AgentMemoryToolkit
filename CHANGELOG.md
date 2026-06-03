## Release History

### 0.1.0b2 (2026-06-03)

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
