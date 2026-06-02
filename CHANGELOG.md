# Changelog

All notable changes to `azure-cosmos-agent-memory` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) versioning.

## [0.1.0b1] — 2026-06-01

Initial public preview release.

This is a **beta release**. The public surface may evolve in
backward-incompatible ways before the `1.0.0` general-availability cut.
Pin a specific version when integrating.

### Added

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

### Package layout

- Distribution name: **`azure-cosmos-agent-memory`** (PyPI)
- Import path: **`azure.cosmos.agent_memory`** 

[0.1.0b1]: https://github.com/AzureCosmosDB/AgentMemoryToolkit/releases/tag/v0.1.0b1
