# Public API

## Architecture

`CosmosMemoryClient` and `AsyncCosmosMemoryClient` are thin orchestrators. They keep local-buffer state and Cosmos connection lifecycle, then delegate persistence to `MemoryStore` / `AsyncMemoryStore` and higher-level behavior to:

- `ChatClient` / `EmbeddingsClient` (sync) and `AsyncEmbeddingsClient` (async) — Azure OpenAI wrappers.
- `RetrievalService` / `AsyncRetrievalService` for filtering, vector search, and episodic context.
- `PipelineService` for extraction, summaries, procedural synthesis, and reconciliation.
- `InProcessProcessor` / `AsyncInProcessProcessor` / `DurableFunctionProcessor` for immediate or change-feed-driven processing.
- `auto_trigger.maybe_trigger_steps` (sync) and `aio.auto_trigger.maybe_trigger_steps` (async) for threshold-driven step firing after each `push_to_cosmos`.

## CosmosMemoryClient (sync)

### Connection

- `__init__(cosmos_endpoint=None, cosmos_credential=None, cosmos_key=None, cosmos_database=None, cosmos_container=None, cosmos_counter_container=None, cosmos_lease_container=None, cosmos_throughput_mode=None, cosmos_autoscale_max_ru=None, ai_foundry_endpoint=None, ai_foundry_credential=None, ai_foundry_api_key=None, embedding_deployment_name='text-embedding-3-large', embedding_dimensions=None, chat_deployment_name='gpt-4o-mini', use_default_credential=True, processor=None) -> None` — configure local state, model clients, optional Cosmos auto-connect, and optional processing backend.
- `close() -> None` — close Cosmos/model clients and owned credentials.
- `connect_cosmos(endpoint=None, credential=None, key=None, database=None, container=None) -> None` — connect to an existing memory container.
- `create_memory_store(database=None, container=None, counter_container=None, lease_container=None, endpoint=None, credential=None, key=None, embedding_dimensions=None, embedding_data_type=None, distance_function=None, full_text_language=None, throughput_mode=None, autoscale_max_ru=None) -> None` — create/connect the memory, counter, and lease containers.

### Memory CRUD

- `add_local(user_id, role, content, memory_type='turn', agent_id=None, metadata=None, thread_id=None, tags=None, ttl=None, salience=None) -> None` — append a memory to the local buffer.
- `get_local(memory_id=None, user_id=None, role=None, memory_types=None) -> list[dict]` — filter local buffered memories.
- `update_local(memory_id, content=None, role=None, memory_type=None, metadata=None) -> None` — update a local buffered memory.
- `delete_local(memory_id) -> None` — remove a local buffered memory.
- `add_cosmos(user_id, role, content, memory_type='turn', metadata=None, thread_id=None, tags=None, ttl=None, salience=None, embedding=None, embed=None) -> str` — upsert one memory to Cosmos and return its id.
- `push_to_cosmos(batch_size=25) -> None` — flush local buffered memories to Cosmos.
- `get_memories(memory_id=None, user_id=None, thread_id=None, role=None, memory_types=None, recent_k=None, tags=None, any_tags=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None) -> list[dict]` — retrieve memories with filters.
- `update_cosmos(memory_id, content=None, role=None, memory_type=None, metadata=None) -> None` — update a Cosmos memory.
- `delete_cosmos(memory_id, thread_id, user_id) -> None` — delete a Cosmos memory.
- `get_thread(thread_id, user_id=None, memory_types=None, recent_k=None, tags=None, exclude_tags=None, include_superseded=False) -> list[dict]` — retrieve a thread oldest-first.
- `get_user_summary(user_id) -> Optional[dict]` — retrieve the active user-summary document.

### Retrieval

- `search_cosmos(search_terms, memory_id=None, user_id=None, role=None, memory_types=None, thread_id=None, hybrid_search=False, top_k=5, tags=None, any_tags=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None) -> list[dict]` — vector or hybrid search memories.
- `get_procedural_prompt(user_id) -> Optional[str]` — read the active procedural prompt.
- `get_procedural_history(user_id, limit=10) -> list[dict]` — read procedural prompt history.
- `get_procedural_memories(user_id, priority=None, category=None, min_salience=None, include_superseded=False) -> list[dict]` — retrieve procedural memory documents.
- `search_episodic_memories(user_id, search_terms, top_k=5, min_salience=None, include_superseded=False) -> list[dict]` — search episodic memories.
- `build_procedural_context(user_id) -> str` — format procedural context for prompts.
- `build_episodic_context(user_id, query, top_k=3) -> str` — format relevant episodic context.

### Processing

- `extract_memories(user_id, thread_id, recent_k=None) -> dict[str, int]` — extract facts/episodic memories from a thread.
- `synthesize_procedural(user_id, *, force=False) -> dict` — synthesize the procedural prompt.
- `generate_thread_summary(user_id, thread_id, recent_k=None, **kwargs) -> dict` — generate and persist a thread summary.
- `generate_user_summary(user_id, thread_ids=None, recent_k=None, **kwargs) -> dict` — generate and persist a user summary.
- `reconcile(user_id, n=None) -> dict[str, int]` — reconcile duplicate or contradictory facts.
- `process_now(*, user_id, thread_id) -> ProcessThreadResult` — run the configured processor immediately.
- `process_now_and_wait(*, user_id, thread_id, timeout=30.0) -> bool` — process and wait for a summary.

### Tagging

- `add_tags(memory_id, user_id, thread_id, tags) -> None` — add tags to a memory.
- `remove_tags(memory_id, user_id, thread_id, tags) -> None` — remove tags from a memory.

## AsyncCosmosMemoryClient

Local-buffer methods remain synchronous in-memory operations; Cosmos, retrieval, and processing methods are `async` and must be awaited.

### Connection

- `__init__(cosmos_endpoint=None, cosmos_credential=None, cosmos_key=None, cosmos_database=None, cosmos_container=None, cosmos_counter_container=None, cosmos_lease_container=None, cosmos_throughput_mode=None, cosmos_autoscale_max_ru=None, ai_foundry_endpoint=None, ai_foundry_credential=None, ai_foundry_api_key=None, embedding_deployment_name='text-embedding-3-large', embedding_dimensions=None, chat_deployment_name='gpt-4o-mini', use_default_credential=True, processor=None) -> None` — configure async local state, model clients, and optional processing backend.
- `async close() -> None` — close async/sync resources and owned credentials.
- `async connect_cosmos(endpoint=None, credential=None, key=None, database=None, container=None) -> None` — connect to an existing memory container.
- `async create_memory_store(database=None, container=None, counter_container=None, lease_container=None, endpoint=None, credential=None, key=None, embedding_dimensions=None, embedding_data_type=None, distance_function=None, full_text_language=None, throughput_mode=None, autoscale_max_ru=None) -> None` — create/connect memory, counter, and lease containers.

### Memory CRUD

- `add_local(user_id, role, content, memory_type='turn', agent_id=None, metadata=None, thread_id=None, tags=None, ttl=None, salience=None) -> None` — append a memory to the local buffer.
- `get_local(memory_id=None, user_id=None, role=None, memory_types=None) -> list[dict]` — filter local buffered memories.
- `update_local(memory_id, content=None, role=None, memory_type=None, metadata=None) -> None` — update a local buffered memory.
- `delete_local(memory_id) -> None` — remove a local buffered memory.
- `async add_cosmos(user_id, role, content, memory_type='turn', metadata=None, thread_id=None, tags=None, ttl=None, salience=None, embedding=None, embed=None) -> str` — upsert one memory to Cosmos and return its id.
- `async push_to_cosmos(batch_size=25) -> None` — flush local buffered memories to Cosmos.
- `async get_memories(memory_id=None, user_id=None, thread_id=None, role=None, memory_types=None, recent_k=None, tags=None, any_tags=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None) -> list[dict]` — retrieve memories with filters.
- `async update_cosmos(memory_id, content=None, role=None, memory_type=None, metadata=None) -> None` — update a Cosmos memory.
- `async delete_cosmos(memory_id, thread_id, user_id) -> None` — delete a Cosmos memory.
- `async get_thread(thread_id, user_id=None, memory_types=None, recent_k=None, tags=None, exclude_tags=None, include_superseded=False) -> list[dict]` — retrieve a thread oldest-first.
- `async get_user_summary(user_id) -> Optional[dict]` — retrieve the active user-summary document.

### Retrieval

- `async search_cosmos(search_terms, memory_id=None, user_id=None, role=None, memory_types=None, thread_id=None, hybrid_search=False, top_k=5, tags=None, any_tags=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None) -> list[dict]` — vector or hybrid search memories.
- `async get_procedural_prompt(user_id) -> Optional[str]` — read the active procedural prompt.
- `async get_procedural_history(user_id, limit=10) -> list[dict]` — read procedural prompt history.
- `async get_procedural_memories(user_id, priority=None, category=None, min_salience=None, include_superseded=False) -> list[dict]` — retrieve procedural memory documents.
- `async search_episodic_memories(user_id, search_terms, top_k=5, min_salience=None, include_superseded=False) -> list[dict]` — search episodic memories.
- `async build_procedural_context(user_id) -> str` — format procedural context for prompts.
- `async build_episodic_context(user_id, query, top_k=3) -> str` — format relevant episodic context.

### Processing

- `async extract_memories(user_id, thread_id, recent_k=None) -> dict[str, int]` — extract facts/episodic memories from a thread.
- `async synthesize_procedural(user_id, *, force=False) -> dict` — synthesize the procedural prompt.
- `async generate_thread_summary(user_id, thread_id, recent_k=None, **kwargs) -> dict` — generate and persist a thread summary.
- `async generate_user_summary(user_id, thread_ids=None, recent_k=None, **kwargs) -> dict` — generate and persist a user summary.
- `async reconcile(user_id, n=None) -> dict[str, int]` — reconcile duplicate or contradictory facts.
- `async process_now(*, user_id, thread_id) -> ProcessThreadResult` — run the configured processor immediately.
- `async process_now_and_wait(*, user_id, thread_id, timeout=30.0) -> bool` — process and wait for a summary.

### Tagging

- `async add_tags(memory_id, user_id, thread_id, tags) -> None` — add tags to a memory.
- `async remove_tags(memory_id, user_id, thread_id, tags) -> None` — remove tags from a memory.

## Extension Points

Sync extension protocols live in `agent_memory_toolkit.services`; async variants live in `agent_memory_toolkit.aio.services`.

- `MemoryStoreProtocol` (`agent_memory_toolkit.services`): persistence primitives (`query`, `read_item`, `add_cosmos`, `mark_superseded`) consumed by the pipeline.

Concrete service classes are exported from their respective packages:

- Sync: `RetrievalService`, `PipelineService` from `agent_memory_toolkit.services` (sub-modules `retrieval`, `pipeline`).
- Async: `AsyncRetrievalService` from `agent_memory_toolkit.aio.services` (sub-module `retrieval`). The pipeline is sync-only and is driven through `asyncio.to_thread` by the async client.
- Threshold-driven auto-trigger: `maybe_trigger_steps` from `agent_memory_toolkit.auto_trigger` (sync) and `agent_memory_toolkit.aio.auto_trigger` (async).
