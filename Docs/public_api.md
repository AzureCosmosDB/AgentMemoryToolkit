# Public API

## Architecture

`CosmosMemoryClient` and `AsyncCosmosMemoryClient` are thin orchestrators. They keep local-buffer state and Cosmos connection lifecycle, then delegate persistence to `MemoryStore` / `AsyncMemoryStore` and higher-level behavior to:

- `ChatClient` / `EmbeddingsClient` (sync) and `AsyncEmbeddingsClient` (async) - Azure OpenAI wrappers.
- `RetrievalService` / `AsyncRetrievalService` for filtering, vector search, and episodic context.
- `PipelineService` for extraction, summaries, procedural synthesis, and reconciliation.
- `InProcessProcessor` / `AsyncInProcessProcessor` / `DurableFunctionProcessor` for immediate or change-feed-driven processing.
- `auto_trigger.maybe_trigger_steps` (sync) and `aio.auto_trigger.maybe_trigger_steps` (async) for threshold-driven step firing after each `push_to_cosmos`.

## CosmosMemoryClient (sync)

### Connection

- `__init__(cosmos_endpoint=None, cosmos_credential=None, cosmos_key=None, cosmos_database=None, cosmos_container=None, cosmos_turns_container='memories_turns', cosmos_summaries_container='memories_summaries', cosmos_counter_container=None, cosmos_lease_container=None, cosmos_throughput_mode=None, cosmos_autoscale_max_ru=None, ai_foundry_endpoint=None, ai_foundry_credential=None, ai_foundry_api_key=None, embedding_deployment_name='text-embedding-3-large', embedding_dimensions=None, chat_deployment_name='gpt-4o-mini', use_default_credential=True, enable_turn_embeddings=None, user_agent=None, embeddings_client=None, chat_client=None, processor=None, transcript_metadata_keys=None, cadence_thresholds=None, cosmos_preferred_locations=None) -> None` - configure local state, model clients, optional Cosmos auto-connect, and optional processing backend. The SDK uses a hard 3-container topology: turns in `memories_turns`, facts/episodic/procedural in `memories`, and summaries in `memories_summaries` (or the names you pass). `enable_turn_embeddings` (default `False`, env `ENABLE_TURN_EMBEDDINGS`) embeds raw turns on write so they can be vector-searched via `search_turns()`; the turns container is always provisioned with a vector index, so toggling this never requires recreating it. `cosmos_preferred_locations` (a list or a comma-separated string, env `COSMOS_DB_PREFERRED_REGIONS`) orders the Cosmos DB regions the client reads from, closest first; unset keeps the account's default order.
- `close() -> None` - close Cosmos/model clients and owned credentials.
- `connect_cosmos(endpoint=None, credential=None, key=None, database=None, container=None, turns_container=None, summaries_container=None) -> None` - connect to existing memory, turns, and summaries containers.
- `create_memory_store(database=None, container=None, turns_container=None, summaries_container=None, counter_container=None, lease_container=None, endpoint=None, credential=None, key=None, embedding_dimensions=None, embedding_data_type=None, distance_function=None, full_text_language=None, throughput_mode=None, autoscale_max_ru=None) -> None` - create/connect the memory, turns, summaries, counter, and lease containers.
- `validate_topology() -> None` - read metadata for all three memory containers and raise `RuntimeError` if any is missing or unreachable; call after connecting to catch infrastructure/config drift before writes.

### Memory CRUD

- `add_local(user_id, role, content, memory_type='turn', agent_id=None, metadata=None, thread_id=None, tags=None, ttl=None, salience=None) -> None` - append a memory to the local buffer.
- `get_local(memory_id=None, user_id=None, role=None, memory_types=None) -> list[dict]` - filter local buffered memories.
- `update_local(memory_id, content=None, role=None, memory_type=None, metadata=None) -> None` - update a local buffered memory.
- `delete_local(memory_id) -> None` - remove a local buffered memory.
- `upsert_memory(user_id, role, content, memory_type='turn', metadata=None, thread_id=None, tags=None, ttl=None, salience=None, embedding=None, embed=None) -> str` - upsert one memory to Cosmos and return its id.
- `push_to_cosmos(batch_size=25) -> None` - flush local buffered memories to Cosmos.
- `get_memories(memory_id=None, user_id=None, thread_id=None, role=None, memory_types=None, recent_k=None, tags_all=None, tags_any=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None, created_after=None, created_before=None) -> list[dict]` - retrieve memories from the MEMORIES container. `memory_types` defaults to `["fact", "episodic", "procedural"]` and must be a subset of those three.
- `update_cosmos(memory_id, *, user_id, thread_id, memory_type, content=None, role=None, metadata=None) -> None` - point-update a memory in the container that holds `memory_type`. The `type` field itself is never mutated. A vector is rebuilt when the text it encodes changes or the secret guard redacts it (one embedding call); if embedding fails or is not configured, the memory is stored without a vector.
- `delete_memory(memory_id, *, user_id, thread_id, memory_type) -> None` - delete a memory from the container that holds `memory_type`. Deleting a `fact` or `procedural` memory first deletes the records folded into it: duplicates retired into a fact (`supersede_reason="duplicate"`), the sources of a combined fact (`"combined"`), the directives and corrections retired into a procedure (`"procedure"`), and anything folded into those. They are found across the user's partitions and deleted one at a time, each under its own `thread_id`; if one delete fails the memory is kept and a retry resumes. Records replaced for another reason (`"update"`, `"contradict"`) are earlier versions and stay.
- `get_thread(thread_id, user_id=None, recent_k=None, tags_all=None, tags_any=None, exclude_tags=None, include_superseded=False, created_after=None, created_before=None) -> list[dict]` - retrieve turns from the TURNS container oldest-first.
- `get_thread_summary(user_id, thread_id, recent_k=None) -> list[dict]` - retrieve thread summary documents from the SUMMARIES container for a single `(user_id, thread_id)` partition.
- `get_user_summary(user_id) -> Optional[dict]` - retrieve the active user-summary document.

### Retrieval

- `search_cosmos(search_terms, memory_id=None, user_id=None, role=None, memory_types=None, thread_id=None, top_k=5, tags_all=None, tags_any=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None, created_after=None, created_before=None, include_episodes=False, include_turns=False, turn_top_k=None, include_summaries=False, summary_top_k=None, preference_slot=False, procedure_slot=False) -> list[dict]` - hybrid vector/full-text search memories, falling back to vector-only for all-stopword queries. `include_summaries` / `include_turns` append summaries / turns after the memories (capped by `summary_top_k` / `turn_top_k`, default `top_k`), deduped by content. `preference_slot` / `procedure_slot` reserve result slots within `top_k`. Procedures are returned only while `active` and not past `valid_until`. See [Search scoping, procedures and slots](#search-scoping-procedures-and-slots).
- `search_turns(search_terms, user_id, thread_id=None, role=None, top_k=5, tags_all=None, tags_any=None, exclude_tags=None, created_after=None, created_before=None) -> list[dict]` - hybrid vector/full-text search the raw conversation log instead of facts/episodic/procedural (requires turn embeddings; see `enable_turn_embeddings`). `user_id` is required and always filters the results (`c.user_id`); a keyword query runs across partitions, a vector-only query with a `thread_id` reads one.
- `get_procedural_prompt(user_id) -> Optional[str]` - the user's standing instructions (the same text as `build_procedural_context(user_id)`), or `None` when there are none.
- `get_procedural_history(user_id, limit=10) -> list[dict]` - list up to `limit` of the user's procedures, active ones first, then by `version`.
- `get_procedural_memories(user_id, priority=None, category=None, min_salience=None, include_superseded=False) -> list[dict]` - retrieve the user's procedural memory documents, candidates included.
- `retrieve_procedures(user_id, search_terms, top_k=5, *, procedure_kind=None, status='active', include_superseded=False) -> list[dict]` - rank the user's procedures by relevance to a task or situation. `procedure_kind` narrows to `instruction`, `habit` or `skill`; `status=None` also ranks candidates. It does not apply `valid_until`.
- `search_episodic_memories(user_id, search_terms, top_k=5, min_salience=None, include_superseded=False) -> list[dict]` - search episodic memories.
- `build_procedural_context(user_id, task=None) -> str` - the user's standing instructions for prompt injection: the active, unexpired procedures that apply to nearly every interaction (`applies="always"`), at most 10 rules and 2,048 bytes, taught before corrected before repeated before endorsed before practiced, then newest first, under a `# Standing instructions from the user` header; `""` when there are none. With `task`, it also adds the nearest active, unexpired situational procedure (`applies="when"`; there is no similarity cutoff) under `## Apply only when the situation matches`, as `When <conditions>: <rule>`.
- `build_episodic_context(user_id, query, top_k=3) -> str` - format relevant episodic context.

Episodic retrieval has two scoping modes. `search_episodic(...)` on the store is
episode-only and, when called with `thread_id`, hard-filters to that thread.
`search_cosmos(..., include_episodes=True, thread_id=...)` treats episodic
memory as user-scoped and can recall episodes from other threads for the same
user. Use the episode-only surface for thread-local recall; use
`search_cosmos(include_episodes=True)` when cross-thread episodic recall should
compete with facts in one ranked result set.

#### Search scoping, procedures and slots

- **Partitions.** A keyword query ranks with `FullTextScore`, which is ignored in a query the Python SDK sends with a partition key, because it skips query planning ([Azure/azure-sdk-for-python#42241](https://github.com/Azure/azure-sdk-for-python/issues/42241)). `search_cosmos`, `search_turns`, `search_summaries`, `search_episodic_memories` and `retrieve_procedures` therefore run a keyword query across partitions, with `c.user_id` (and `c.thread_id`, when given) in the `WHERE` clause, and set `full_text_score_scope="Local"` so BM25 statistics come from the partition ranges the query targets. The filters still confine the results to the same user and thread; ranking now follows the question. Expect more RUs than a single-partition read on a large container. A vector-only query (all-stopword terms) with both `user_id` and `thread_id` still reads one partition.
- **Procedures.** `search_cosmos` hides procedures that are not `active` (a `candidate`, for one) or are past `valid_until` (an ISO date; no value means no end). `get_procedural_memories` lists them and `retrieve_procedures(..., status=None)` ranks candidates.
- **Slots** (opt-in). `preference_slot=True` reserves one of the `top_k` results for the user's best-matching preference fact (`metadata.category="preference"`) that the ranked results do not already show. `procedure_slot=True` reserves one for the user's best-matching active situational procedure (`applies="when"`); procedures then leave the ranked results. Slot candidates are ranked by vector similarity alone and use the search's own filters, except that the procedure slot ignores `role`, `tags_all`, `tags_any`, `min_salience` and `min_confidence`, which no procedure can meet (a procedure has no salience or confidence, its role is `system` and its tags are the toolkit's own). A slot with no candidate takes the next ranked result, so `top_k` stays filled. A slotted result carries `retrieval={"rank": <1-based position>, "slot": "preference" | "procedure" | "backfill"}`; ranked results carry none. Slots need `user_id` and are skipped with `memory_id` or when `top_k` would leave no ranked result. The preference slot applies only when facts are searched, and `procedure_slot` has no effect when only procedural types are searched. Episodes never take a slot, and summaries and turns blended in with `include_summaries` / `include_turns` still follow the `top_k` results.

### Processing

- `extract_memories(user_id, thread_id, recent_k=None) -> dict[str, int]` - extract facts/episodic memories from a thread.
- `extract_episodes(user_id, thread_id, *, flush=False) -> dict[str, int]` - segment the thread's open turn stream into episodes. Automatic processing uses `flush=False`, so the trailing open segment closes lazily on a later boundary. Call with `flush=True` on session close when the in-process backend owns processing and you need to drain the tail immediately.
- `synthesize_procedural(user_id, *, force=False) -> dict` - learn procedures (instructions, habits and skills; see [Procedural memory](#procedural-memory)) from the behavioral facts and episodes written since the user's last run. A `procedure_cursor` records what was read, so each source is read once. The result has `status` and `procedures_created`: `status` is `synthesized`, `unchanged` (nothing new, no model call), `skipped` (every model answer failed or was unusable) or `deferred` (a retryable model error, or a transient embedding or Cosmos DB error while writing a procedure, stopped the run and left the cursor, so the next run retries). A `synthesized` result also has `procedures_updated`, `procedures_reinforced`, `procedures_skipped` and `facts_retired`. `force` is accepted and has no effect. It needs the in-process processor: with `DurableFunctionProcessor` it raises `NotImplementedError`, because the Function App runs it after each reconcile.
- `generate_thread_summary(user_id, thread_id, recent_k=None, **kwargs) -> dict` - generate and persist a thread summary.
- `generate_user_summary(user_id, thread_ids=None, recent_k=None, **kwargs) -> dict` - generate and persist a user summary.
- `reconcile(user_id, n=None) -> dict[str, int]` - fold duplicate facts, then resolve contradictions. `n` defaults to `DEDUP_POOL_SIZE` (at most 500) and caps both the facts the dedup pass checks and the contradiction pool. The dedup pass compares each fact written since the last pass with its nearest active neighbours and retires one into another (`supersede_reason="duplicate"`) only when the two state the same claim, or one contains the other, and nothing it states would be lost. It also combines facts about one specific thing that each hold details the others lack into one new fact (tagged `sys:combined`), retiring its sources with `supersede_reason="combined"`. The contradiction pass then retires the loser of each contradiction (`supersede_reason="contradict"`). Returns `kept` (facts left active in the contradiction pass), `merged` (duplicates retired), `contradicted`, and the dedup pass's `checked` (facts checked), `compared` (pairs judged), `vetoed` (folds blocked because a number, name or negation would be lost), `combined` (facts retired into a combined fact), `synthesized` (combined facts written), `combine_groups` (groups tried) and `combine_failed` (groups declined or not written).
- `process_now(*, user_id, thread_id) -> ProcessThreadResult` - run the configured processor immediately.
- `process_now_and_wait(*, user_id, thread_id, timeout=30.0) -> bool` - process and wait for a summary.

The clients run each pass to completion. `PipelineService.reconcile_memories(user_id, n=50, *, memory_type="fact", deadline=None)`, `consolidate_memories(...)` (the dedup pass alone) and `synthesize_procedural(user_id, *, force=False, deadline=None)` also accept `deadline`, a `time.monotonic()` value for callers with a time limit, such as a Durable activity. Once it has passed they start no new model call: reconcile applies the folds it found, moves its dedup cursor past the facts it finished and skips the contradiction pass; `synthesize_procedural` keeps what it wrote and its cursor. The next run continues from there. The Function App sets `deadline` from `ACTIVITY_TIME_BUDGET_SECONDS`.

### Tagging

- `add_tags(memory_id, user_id, thread_id, memory_type, tags) -> None` - add tags to a memory. `memory_type` must be one of `fact`, `episodic`, `procedural`.
- `remove_tags(memory_id, user_id, thread_id, memory_type, tags) -> None` - remove tags from a memory. `memory_type` must be one of `fact`, `episodic`, `procedural`.
- `list_tags(user_id, *, thread_id=None, prefix=None, include_sys=False) -> list[str]` - list sorted, deduped tags for a user; omits `sys:*` by default.

## AsyncCosmosMemoryClient

Local-buffer methods remain synchronous in-memory operations; Cosmos, retrieval, and processing methods are `async` and must be awaited.

### Connection

- `__init__(cosmos_endpoint=None, cosmos_credential=None, cosmos_key=None, cosmos_database=None, cosmos_container=None, cosmos_turns_container='memories_turns', cosmos_summaries_container='memories_summaries', cosmos_counter_container=None, cosmos_lease_container=None, cosmos_throughput_mode=None, cosmos_autoscale_max_ru=None, ai_foundry_endpoint=None, ai_foundry_credential=None, ai_foundry_api_key=None, embedding_deployment_name='text-embedding-3-large', embedding_dimensions=None, chat_deployment_name='gpt-4o-mini', use_default_credential=True, enable_turn_embeddings=None, user_agent=None, embeddings_client=None, chat_client=None, processor=None, transcript_metadata_keys=None, cadence_thresholds=None, cosmos_preferred_locations=None) -> None` - configure async local state, model clients, and optional processing backend. The async SDK uses the same hard 3-container topology as the sync client. `enable_turn_embeddings` (default `False`, env `ENABLE_TURN_EMBEDDINGS`) embeds raw turns on write so they can be vector-searched via `search_turns()`. `cosmos_preferred_locations` (a list or a comma-separated string, env `COSMOS_DB_PREFERRED_REGIONS`) orders the Cosmos DB regions the client reads from, closest first; unset keeps the account's default order.
- `async close() -> None` - close async/sync resources and owned credentials.
- `async connect_cosmos(endpoint=None, credential=None, key=None, database=None, container=None, turns_container=None, summaries_container=None) -> None` - connect to existing memory, turns, and summaries containers.
- `async create_memory_store(database=None, container=None, turns_container=None, summaries_container=None, counter_container=None, lease_container=None, endpoint=None, credential=None, key=None, embedding_dimensions=None, embedding_data_type=None, distance_function=None, full_text_language=None, throughput_mode=None, autoscale_max_ru=None) -> None` - create/connect memory, turns, summaries, counter, and lease containers.
- `async validate_topology() -> None` - read metadata for all three memory containers and raise `RuntimeError` if any is missing or unreachable; call after connecting to catch infrastructure/config drift before writes.

### Memory CRUD

- `add_local(user_id, role, content, memory_type='turn', agent_id=None, metadata=None, thread_id=None, tags=None, ttl=None, salience=None) -> None` - append a memory to the local buffer.
- `get_local(memory_id=None, user_id=None, role=None, memory_types=None) -> list[dict]` - filter local buffered memories.
- `update_local(memory_id, content=None, role=None, memory_type=None, metadata=None) -> None` - update a local buffered memory.
- `delete_local(memory_id) -> None` - remove a local buffered memory.
- `async upsert_memory(user_id, role, content, memory_type='turn', metadata=None, thread_id=None, tags=None, ttl=None, salience=None, embedding=None, embed=None) -> str` - upsert one memory to Cosmos and return its id.
- `async push_to_cosmos(batch_size=25) -> None` - flush local buffered memories to Cosmos.
- `async get_memories(memory_id=None, user_id=None, thread_id=None, role=None, memory_types=None, recent_k=None, tags_all=None, tags_any=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None, created_after=None, created_before=None) -> list[dict]` - retrieve memories from the MEMORIES container. `memory_types` defaults to `["fact", "episodic", "procedural"]` and must be a subset of those three.
- `async update_cosmos(memory_id, *, user_id, thread_id, memory_type, content=None, role=None, metadata=None) -> None` - point-update a memory in the container that holds `memory_type`. The `type` field itself is never mutated. A vector is rebuilt when the text it encodes changes or the secret guard redacts it (one embedding call); if embedding fails or is not configured, the memory is stored without a vector.
- `async delete_memory(memory_id, *, user_id, thread_id, memory_type) -> None` - delete a memory from the container that holds `memory_type`. Cascades like the sync method: deleting a `fact` or `procedural` memory first deletes the records folded into it (`supersede_reason` `duplicate`, `combined` or `procedure`, and anything folded into those); `update` and `contradict` versions stay.
- `async get_thread(thread_id, user_id=None, recent_k=None, tags_all=None, tags_any=None, exclude_tags=None, include_superseded=False, created_after=None, created_before=None) -> list[dict]` - retrieve turns from the TURNS container oldest-first.
- `async get_thread_summary(user_id, thread_id, recent_k=None) -> list[dict]` - retrieve thread summary documents from the SUMMARIES container for a single `(user_id, thread_id)` partition.
- `async get_user_summary(user_id) -> Optional[dict]` - retrieve the active user-summary document.

### Retrieval

- `async search_cosmos(search_terms, memory_id=None, user_id=None, role=None, memory_types=None, thread_id=None, top_k=5, tags_all=None, tags_any=None, exclude_tags=None, include_superseded=False, min_salience=None, min_confidence=None, created_after=None, created_before=None, include_episodes=False, include_turns=False, turn_top_k=None, include_summaries=False, summary_top_k=None, preference_slot=False, procedure_slot=False) -> list[dict]` - hybrid vector/full-text search memories, falling back to vector-only for all-stopword queries. Blending, slots, procedure visibility and partition scoping match the sync method (see [Search scoping, procedures and slots](#search-scoping-procedures-and-slots)).
- `async search_turns(search_terms, user_id, thread_id=None, role=None, top_k=5, tags_all=None, tags_any=None, exclude_tags=None, created_after=None, created_before=None) -> list[dict]` - hybrid vector/full-text search the raw conversation log instead of facts/episodic/procedural (requires turn embeddings; see `enable_turn_embeddings`). `user_id` is required and always filters the results (`c.user_id`); a keyword query runs across partitions, a vector-only query with a `thread_id` reads one.
- `async get_procedural_prompt(user_id) -> Optional[str]` - the user's standing instructions (the same text as `build_procedural_context(user_id)`), or `None` when there are none.
- `async get_procedural_history(user_id, limit=10) -> list[dict]` - list up to `limit` of the user's procedures, active ones first, then by `version`.
- `async get_procedural_memories(user_id, priority=None, category=None, min_salience=None, include_superseded=False) -> list[dict]` - retrieve the user's procedural memory documents, candidates included.
- `async retrieve_procedures(user_id, search_terms, top_k=5, *, procedure_kind=None, status='active', include_superseded=False) -> list[dict]` - rank the user's procedures by relevance to a task or situation. `procedure_kind` narrows to `instruction`, `habit` or `skill`; `status=None` also ranks candidates. It does not apply `valid_until`.
- `async search_episodic_memories(user_id, search_terms, top_k=5, min_salience=None, include_superseded=False) -> list[dict]` - search episodic memories.
- `async build_procedural_context(user_id, task=None) -> str` - the user's standing instructions for prompt injection, with the optional nearest situational procedure for `task`; same output as the sync method.
- `async build_episodic_context(user_id, query, top_k=3) -> str` - format relevant episodic context.

Episodic retrieval has the same scoping behavior as the sync API. The store's
episode-only `search_episodic(...)` hard-filters to `thread_id` when provided,
while `search_cosmos(..., include_episodes=True, thread_id=...)` treats episodic
memory as user-scoped and may return episodes from other threads for that user.

### Processing

- `async extract_memories(user_id, thread_id, recent_k=None) -> dict[str, int]` - extract facts/episodic memories from a thread.
- `async extract_episodes(user_id, thread_id, *, flush=False) -> dict[str, int]` - segment the thread's open turn stream into episodes. Automatic processing uses `flush=False`, so call with `flush=True` on session close when the in-process backend owns processing and you need to drain the tail immediately.
- `async synthesize_procedural(user_id, *, force=False) -> dict` - async counterpart of `synthesize_procedural`: same cursor, result keys and `NotImplementedError` with the Durable processor.
- `async generate_thread_summary(user_id, thread_id, recent_k=None, **kwargs) -> dict` - generate and persist a thread summary.
- `async generate_user_summary(user_id, thread_ids=None, recent_k=None, **kwargs) -> dict` - generate and persist a user summary.
- `async reconcile(user_id, n=None) -> dict[str, int]` - async counterpart of `reconcile`: folds duplicate facts (and combines facts about one specific thing), then resolves contradictions, and returns the same keys (`kept`, `merged`, `contradicted`, `checked`, `compared`, `vetoed`, `combined`, `synthesized`, `combine_groups`, `combine_failed`).
- `async process_now(*, user_id, thread_id) -> ProcessThreadResult` - run the configured processor immediately.
- `async process_now_and_wait(*, user_id, thread_id, timeout=30.0) -> bool` - process and wait for a summary.

`AsyncPipelineService.reconcile_memories`, `consolidate_memories` and `synthesize_procedural` accept the same `deadline` as the sync pipeline.

### Tagging

- `async add_tags(memory_id, user_id, thread_id, memory_type, tags) -> None` - add tags to a memory. `memory_type` must be one of `fact`, `episodic`, `procedural`.
- `async remove_tags(memory_id, user_id, thread_id, memory_type, tags) -> None` - remove tags from a memory. `memory_type` must be one of `fact`, `episodic`, `procedural`.
- `async list_tags(user_id, *, thread_id=None, prefix=None, include_sys=False) -> list[str]` - list sorted, deduped tags for a user; omits `sys:*` by default.

## Fact fields

`extract_memories` uses `extract_memories-v3.prompty` by default; set `AMT_EXTRACT_MEMORIES_PROMPT` to `extract_memories-v2.prompty` or `extract_memories.prompty` to select an earlier prompt. Facts it extracts carry these optional fields beside `content`, `confidence`, `salience` and `metadata.category`:

| Field | Meaning |
|---|---|
| `assertion` | How the conversation put the fact forward: `stated`, `decided`, `suggested`, `observed`, `directive` (the user told the assistant how to act in future situations) or `correction` (the user corrected how the assistant did something). An agent-sourced fact (`sys:agent-fact`, `metadata.source="agent"`) cannot be a `directive` or `correction` and is recorded as `stated`. |
| `subject` | Who or what the fact is about; `user` and `assistant` are normalized. |
| `attribute` | The single-valued property the fact gives the current value of, in a few lowercase words (for example `home city`); otherwise empty. It is recorded for retrieval and for later extractions to reuse; the toolkit does not retire facts by attribute. |
| `entities` | Up to 12 names the fact mentions. |
| `source_turn_ids` | The turns the fact came from. The v3 prompt cites turns per fact; earlier prompts record every identified turn of the batch. |
| `merged_from` | IDs of the facts folded into this one by dedup, or the sources of a combined fact. |

Facts from earlier prompts have no `assertion`, `subject`, `attribute` or `entities`. The v3 prompt also reads, as context only, up to 6 earlier turns of the thread (within 1,500 tokens), the thread summary when it was written before the new turns, and the 15 active facts nearest the new turns; facts still come only from the new turns. The context costs one embedding and one vector query per batch, plus one turns query and one summary read per window, and a lookup that fails only drops its context.

`supersede_reason` is `update`, `contradict`, `duplicate` (folded into another fact), `combined` (folded into a combined fact) or `procedure` (a directive or correction retired into the procedure that holds it). Search hides superseded records unless `include_superseded=True`.

## Procedural memory

A procedural memory (type `procedural`, stored under the sentinel thread id `__procedural__`) says how the assistant should act for one user. `procedure_kind` records how it was learned: `instruction` (the user taught how to act from now on), `habit` (the user corrected the assistant, or chose the same way repeatedly) or `skill` (a method of at least two `steps` that worked in practice or that the user endorsed). `applies` is `always` for a standing rule or `when` for a situational one with `activation_conditions`. `evidence` lists the support seen (`taught`, `corrected`, `repeated`, `practiced`, `endorsed`), with `support_dates`, `user_support_count` and `practiced_dates`. `status` follows from the evidence and is never set by the model:

| Kind | Active once |
|---|---|
| `instruction` | the user taught or corrected it: a supporting fact the user stated, recorded as a `directive` or `correction` |
| `habit` | the user corrected the assistant, or facts the user stated (or episodes with a successful or partially successful outcome) support it on two different dates; facts the assistant stated and failed episodes do not count toward those dates |
| `skill` | two successful practices (episodes with a successful or partially successful outcome) on different dates, or the user's endorsement |

Anything else stays a `candidate`, which search, `retrieve_procedures` (default `status="active"`) and `build_procedural_context` do not show. Teaching evidence counts only from a fact the user stated that extraction recorded as a `directive` or `correction` (a fact extracted before assertions existed also counts), so a plain preference or a request about one task stays a candidate. `valid_until` (an ISO date) ends a procedure's validity, `version` rises with every revision, `source_kind` and `source_authority` are derived from the strongest evidence, and the ID derives from the user, kind and name, so a repeated lesson reinforces one record.

`synthesize_procedural` reads the user's active preference and requirement facts, facts with salience of at least 0.8, `directive` and `correction` facts, and episodes with lessons, written after the `procedure_cursor` document (under `[user_id, "__procedural__"]`): at most 50 facts and 50 episodes per run, shown to the model 10 at a time with the user's nearest existing procedures so it revises or reinforces one instead of writing a paraphrase. A revision that drops a number or a negation records nothing, unless it is the user's correction. When an active situational procedure holds a directive or correction fact it cites, that fact is retired into the procedure (`supersede_reason="procedure"`), so the rule is stored and retrieved once. A standing rule leaves its facts in place, because it reaches the agent through `build_procedural_context`. Procedures written under the earlier schema (kinds such as `behavioral_policy` or `workflow`, no `applies` or evidence) are not included by `build_procedural_context`, and `ProceduralRecord` rejects the old kinds.

## Secret redaction

Every write through the store or the pipeline passes a secret guard before the record is embedded or stored: `upsert_memory`, `push_to_cosmos`, `update_cosmos`, `add_tags`, and the facts, episodes, summaries and procedures the pipeline writes. It recognizes 12 credential formats: private-key blocks, Azure account keys and Cosmos DB connection strings, SAS signatures and connection strings, JWTs, Bearer tokens, Basic authorization values, OpenAI keys, GitHub tokens, AWS access key IDs, and passwords inside URLs. Every non-structural string of the record (content, metadata, titles and so on) is scanned and each match is replaced with `[REDACTED-SECRET]`; ids, partition keys, timestamps and similar fields are left alone, and a tag that holds a credential is dropped. A redacted record gets the `sys:secret` tag, `metadata.classification="secret"` and `metadata.secret_patterns` (the names of the formats found). Its `content_hash` is recomputed, and a vector built from the original text is dropped and rebuilt by embedding the redacted text, so no raw credential reaches the embedding endpoint and the record stays in vector search. `update_cosmos` also re-embeds a record whose text it changes, but never gives a record a vector it did not have. If the embedding call fails, the store has no embeddings client, or `upsert_memory` was called with `embed=False`, the record is written without a vector. The log records the id and the pattern names, never the text.

Credentials without a fixed format, such as `password: hunter2` typed into a chat, are not matched. Every prompt that writes memories tells the model never to store a literal secret. A pattern can also redact legitimate text that looks like a credential.

## Topology validation

Use `validate_topology()` (sync) or `await validate_topology()` (async) after `connect_cosmos()` or `create_memory_store()` to verify that `memories`, `memories_turns`, and `memories_summaries` all exist and are readable. The method raises `RuntimeError` with redeploy guidance on the first missing or unreachable container.

## Extension Points

Sync extension protocols live in `azure.cosmos.agent_memory.services`; async variants live in `azure.cosmos.agent_memory.aio.services`.

- `MemoryStoreProtocol` (`azure.cosmos.agent_memory.services`): persistence primitives (`query`, `read_item`, `upsert_memory`, `mark_superseded`) consumed by the pipeline.

Concrete service classes are exported from their respective packages:

- Sync: `RetrievalService`, `PipelineService` from `azure.cosmos.agent_memory.services` (sub-modules `retrieval`, `pipeline`).
- Async: `AsyncRetrievalService` and `AsyncPipelineService` from `azure.cosmos.agent_memory.aio.services` (sub-modules `retrieval`, `pipeline`). The async pipeline is a fully-native asyncio implementation - not an `asyncio.to_thread` shim over the sync pipeline.
- Threshold-driven auto-trigger: `maybe_trigger_steps` from `azure.cosmos.agent_memory.auto_trigger` (sync) and `azure.cosmos.agent_memory.aio.auto_trigger` (async).
