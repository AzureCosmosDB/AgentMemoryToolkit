# Design Patterns

This guide shows when and how to use the toolkit's main operations in real applications. All examples use the async API (`AsyncCosmosMemoryClient`); the sync API (`CosmosMemoryClient`) has the same method signatures without `await`.

---

## 1. Storing Conversation Turns (CRUD)

### When to write memories

Write a turn memory every time a user or agent message is produced. If the application runs locally first and syncs later, use the local + bulk-upload pattern.

```python
from azure.cosmos.agent_memory.aio import AsyncCosmosMemoryClient

mem = AsyncCosmosMemoryClient(
    cosmos_endpoint=COSMOS_DB_ENDPOINT,
    cosmos_database="ai_memory",
    cosmos_container="memories",
    ai_foundry_endpoint=AI_FOUNDRY_ENDPOINT,
    embedding_deployment_name="text-embedding-3-large",
    chat_deployment_name="gpt-4o-mini",
    use_default_credential=True,
)
await mem.connect_cosmos()

THREAD_ID = "thread-abc-123"

# Store user message
await mem.upsert_memory(
    user_id="user-1", thread_id=THREAD_ID,
    role="user", content="I need to migrate our PostgreSQL database to Cosmos DB.",
)

# Store agent response
await mem.upsert_memory(
    user_id="user-1", thread_id=THREAD_ID,
    role="agent", content="I can help with that. What's your current schema look like?",
)

# Store a tool call result with metadata
await mem.upsert_memory(
    user_id="user-1", thread_id=THREAD_ID,
    role="tool",
    content='{"tables": 12, "foreign_keys": 3}',
    metadata={"tool_name": "schema_inspector", "tool_call_id": "call_xyz789"},
)
```

### Local-first with bulk upload

Useful when collecting a batch of turns before committing to Cosmos.

```python
mem.add_local(user_id="user-1", thread_id=THREAD_ID, role="user", content="...")
mem.add_local(user_id="user-1", thread_id=THREAD_ID, role="agent", content="...")

# Push everything to Cosmos at once
await mem.push_to_cosmos()
```

### Updating and deleting

```python
# Update content of an existing memory
await mem.update_cosmos(memory_id="<id>", user_id="user-1", thread_id=THREAD_ID, memory_type="fact", content="Corrected message text")

# Delete a memory (requires the partition keys and the memory_type)
await mem.delete_memory(memory_id="<id>", user_id="user-1", thread_id=THREAD_ID, memory_type="fact")
```

Deleting a fact or procedure also deletes the records that were retired into it (duplicates, the sources of a combined fact, and directives or corrections held by a procedure), so deleted information does not survive in history. See [Section 11](#11-secrets-and-deleting-memories).

---

## 2. Generating a Thread Summary

### When to call

- **End of conversation** - after the user closes a session or a support ticket is resolved.
- **Long-running thread** - when a thread exceeds a token budget (e.g. > 50 turns) and you need a compact representation for context.
- **Periodic background job** - on a schedule to keep summaries up to date for active threads.
- **Automatic (change feed)** - set `THREAD_SUMMARY_EVERY_N` and the change feed trigger handles it. See [Section 8](#8-automatic-processing-with-change-feed).

Summaries are incremental: if one already exists for the thread, only newer turns are merged in.

### How to call

```python
result = await mem.generate_thread_summary(
    user_id="user-1",
    thread_id=THREAD_ID,
    recent_k=100,        # optional: limit to the most recent 100 turns
)
print(result["content"])  # the summary text; result is the stored summary document
```

The summary is stored automatically in Cosmos with id `summary_user-1_thread-abc-123` and `type="thread_summary"`.

---

## 3. Extracting Facts

### When to call

- **After each meaningful exchange** - extract facts from the latest turns so they are available for retrieval immediately.
- **End of conversation** - capture all discrete preferences, decisions, and requirements from the thread.
- **Before a planning step** - in multi-agent workflows, extract facts before handing context to a planner agent.
- **Automatic (change feed)** - set `FACT_EXTRACTION_EVERY_N` and the change feed trigger handles it. See [Section 8](#8-automatic-processing-with-change-feed).

Each fact is stored as its own document with its own embedding, making it ideal for fine-grained semantic search.

### How to call

```python
result = await mem.extract_memories(
    user_id="user-1",
    thread_id=THREAD_ID,
    recent_k=50,
)
print(result["fact_count"])  # facts created by this call
```

### What a fact carries

Facts extracted with the default prompt (`extract_memories-v3.prompty`) record how the conversation put each fact forward (`assertion`: `stated`, `decided`, `suggested`, `observed`, `directive` or `correction`), what it is about (`subject`, `attribute`, `entities`) and the turns it came from (`source_turn_ids`). Search returns these fields, so retrieval can tell a commitment from an option:

```python
facts = await mem.search_cosmos(search_terms="database migration", user_id="user-1", top_k=10)
decided = [f["content"] for f in facts if f.get("assertion") == "decided"]
options = [f["content"] for f in facts if f.get("assertion") == "suggested"]
```

Facts from earlier prompts, and facts you write by hand, have no `assertion`, `subject`, `attribute` or `entities`, so read them with `.get()`. A `directive` or `correction` is a user's standing instruction to the assistant; when an active situational procedure takes it over, it leaves the fact pool (see [Section 9](#9-procedures-and-standing-instructions)).

### Context and prompt selection

With the default prompt, each batch of turns is extracted with read-only context: up to 6 earlier turns of the thread, the thread summary when it was written before the new turns, and the user's 15 nearest stored facts. Replies such as "yes, do that" become complete facts, and facts that are already stored are not restated. Facts still come only from the new turns. Context is bounded and best-effort, so there is nothing to arrange; it costs one embedding and one vector query per batch. Set `AMT_EXTRACT_MEMORIES_PROMPT=extract_memories-v2.prompty` (or `extract_memories.prompty`) to use an earlier prompt, which reads no context and records none of the fields above except `source_turn_ids`.

---

## 4. Generating a User Summary

### When to call

- **Cross-session onboarding** - at the start of a new thread, generate (or update) the user summary so the agent has context from all prior conversations.
- **After a thread summary is created** - chain it: summarize the thread, then update the user summary.
- **On a schedule** - for users with many threads, run periodically to keep the profile current.
- **Automatic (change feed)** - set `USER_SUMMARY_EVERY_N` and the change feed trigger handles it. See [Section 8](#8-automatic-processing-with-change-feed).

User summaries are also incremental. The pipeline merges only new thread data into the existing profile.

### How to call

```python
result = await mem.generate_user_summary(
    user_id="user-1",
    recent_k=50,  # optional: use at most the newest 50 items per thread
)
```

The summary draws on the user's facts, episodes, procedures and thread summaries across all threads. It is stored with id `user_summary_user-1` and `thread_id="__user_summary__"`.

---

## 5. Retrieving Memories

### Get an entire thread

```python
turns = await mem.get_thread(thread_id=THREAD_ID, user_id="user-1", recent_k=20)
```

### Semantic search

`search_cosmos` ranks a user's facts by default, by vector similarity and full-text relevance. Use `memory_types` to search other types, and `include_episodes`, `include_summaries` or `include_turns` (turn embeddings must be enabled) to bring in episodes, summaries and raw turns.

```python
# Facts, ranked by vector similarity and full-text relevance
facts = await mem.search_cosmos(
    search_terms="database migration requirements",
    user_id="user-1",
    memory_types=["fact"],
    top_k=10,
)

# Facts and episodes ranked together, with matching summaries added after them
results = await mem.search_cosmos(
    search_terms="PostgreSQL to Cosmos DB",
    user_id="user-1",
    top_k=5,
    include_episodes=True,
    include_summaries=True,
)
```

### Search slots: preferences and procedures

A ranked search fills `top_k` with the closest facts, which can crowd out a standing preference or a relevant procedure. Reserve a place for each **inside** `top_k`:

```python
results = await mem.search_cosmos(
    search_terms="Plan a weekend in Lisbon",
    user_id="user-1",
    top_k=5,
    preference_slot=True,   # the best-matching preference fact not already shown
    procedure_slot=True,    # the best-matching active situational procedure
)
for r in results:
    slot = (r.get("retrieval") or {}).get("slot", "ranked")
    print(slot, "-", r["content"][:80])
```

With `top_k=5` and both slots you get 3 ranked results, 1 preference and 1 procedure. A slot with no candidate takes the next ranked result (`slot == "backfill"`), so you still get 5. Slotted results carry `retrieval={"rank": ..., "slot": ...}`. Slots need `user_id`, are skipped when `memory_id` is set, and never take episodes; summaries and turns blended in with `include_summaries` / `include_turns` still follow the `top_k` results. Use slots when a prompt should always include the user's preferences and a matching procedure; use `build_procedural_context` ([Section 9](#9-procedures-and-standing-instructions)) for the rules that apply to every interaction.

### Retrieve the user summary

```python
profile = await mem.get_user_summary(user_id="user-1")
```

### Query with filters

`get_memories` reads a user's facts, episodes and procedures. Turns come from `get_thread` and summaries from `get_thread_summary` and `get_user_summary`.

```python
# A thread's summary, as a list (empty when none exists)
summaries = await mem.get_thread_summary(user_id="user-1", thread_id=THREAD_ID)

# All facts
facts = await mem.get_memories(user_id="user-1", memory_types=["fact"])

# Facts and episodes extracted from one thread
thread_memories = await mem.get_memories(user_id="user-1", thread_id=THREAD_ID)
```

---

## 6. End-to-End: Chat Application

A typical chat application lifecycle looks like this:

```
New session starts
  │
  ├─ Retrieve user summary          (get_user_summary)
  ├─ Retrieve standing instructions (build_procedural_context)
  ├─ Semantic search for prior facts (search_cosmos, memory_types=["fact"])
  │
  │  ┌── Conversation loop ──┐
  │  │ Store each turn        │  (upsert_memory)
  │  │ Optionally extract     │  (extract_memories - every N turns or on key exchanges)
  │  └────────────────────────┘
  │
  ├─ Summarize the thread            (generate_thread_summary)
  ├─ Extract remaining facts         (extract_memories)
  └─ Update user summary             (generate_user_summary)
```

### Minimal example

```python
# --- Session start ---
profile = await mem.get_user_summary(user_id="user-1")
standing = await mem.build_procedural_context("user-1")  # "" when the user has no standing instructions
relevant = await mem.search_cosmos("topic of interest", user_id="user-1", memory_types=["fact"], top_k=5)

# Build system prompt with profile, standing instructions and relevant facts
system_prompt = build_prompt(profile, standing, relevant)

# --- Conversation loop ---
while not done:
    user_msg = get_user_input()
    await mem.upsert_memory(user_id="user-1", thread_id=THREAD_ID, role="user", content=user_msg)

    agent_reply = call_llm(system_prompt, user_msg)
    await mem.upsert_memory(user_id="user-1", thread_id=THREAD_ID, role="agent", content=agent_reply)

# --- Session end ---
await mem.generate_thread_summary(user_id="user-1", thread_id=THREAD_ID)
await mem.extract_memories(user_id="user-1", thread_id=THREAD_ID)
await mem.generate_user_summary(user_id="user-1")
```

---

## 7. End-to-End: Multi-Agent Application

In a multi-agent system, different agents share the same memory store but may read and write different memory types.

```
                  ┌───────────────┐
                  │  Orchestrator │
                  └───────┬───────┘
            ┌─────────────┼───────────┐
            ▼            ▼            ▼
      ┌───────────┐ ┌─────────┐ ┌──────────┐
      │ Research  │ │ Planner │ │ Executor │
      │ Agent     │ │ Agent   │ │ Agent    │
      └───────────┘ └─────────┘ └──────────┘
            │            │            │
            └────────────┼────────────┘
                         ▼
                    Cosmos DB
                   (shared memory)
```

### Pattern: shared context via facts and summaries

```python
# Research agent stores findings as turns
await mem.upsert_memory(
    user_id="user-1", thread_id="research-thread",
    role="agent", metadata={"agent_id": "research-agent"},
    content="Found that the source DB has 12 tables with 3 foreign key chains.",
)

# After research is complete, extract facts for other agents to consume
await mem.extract_memories(user_id="user-1", thread_id="research-thread")

# Planner agent retrieves relevant facts before generating a plan
facts = await mem.search_cosmos(
    search_terms="source database schema foreign keys",
    user_id="user-1",
    memory_types=["fact"],
    top_k=10,
)

# Planner writes its plan as a turn in its own thread
await mem.upsert_memory(
    user_id="user-1", thread_id="planning-thread",
    role="agent", metadata={"agent_id": "planner-agent"},
    content=plan_text,
)
```

### Pattern: per-agent threads, cross-agent retrieval

Each agent writes to its own `thread_id`. Other agents discover relevant context through `search_cosmos` across all threads for the user. At the end, `generate_user_summary` produces a unified profile from all agent threads.

```python
# After all agents finish
await mem.generate_user_summary(user_id="user-1")
```

---

## 8. Automatic Processing with Change Feed

Instead of calling `generate_thread_summary()`, `extract_memories()`, or `generate_user_summary()` explicitly, you can let the Cosmos DB change feed trigger fire them automatically in the background.

### How it works

When a new turn is written to the `memories_turns` container, the change feed trigger:

1. Increments a counter document in the dedicated `counter` container for each relevant scope.
2. Checks whether the counter has crossed a configured threshold.
3. Starts the appropriate Durable Functions orchestration if the threshold is crossed.

### Configuration

Set these application settings (in `local.settings.json` locally or Function App settings in Azure):

| Setting | Scope | Effect |
|---------|-------|--------|
| `THREAD_SUMMARY_EVERY_N=5` | Per `(user_id, thread_id)` | Summarize the thread every 5 turns |
| `FACT_EXTRACTION_EVERY_N=3` | Per `(user_id, thread_id)` | Extract facts every 3 turns |
| `USER_SUMMARY_EVERY_N=10` | Per `user_id` | Update user profile every 10 turns across all threads |

Set any value to `0` to disable that processing type. The defaults are `10`, `2` and `20` (see [Threshold settings](concepts.md#threshold-settings)). Reconcile (`DEDUP_EVERY_N`) and procedure learning (`PROCEDURAL_SYNTHESIS_AUTO`) follow extraction automatically; see [Section 9](#9-procedures-and-standing-instructions) and [Section 10](#10-keeping-the-fact-pool-clean). The Function App also needs `MEMORY_PROCESSOR_OWNER=durable` (the Bicep template and `local.settings.json.template` set it); without it the trigger skips every batch.

### Required infrastructure

The change feed trigger watches the `memories_turns` container, and the Function App writes derived memories to `memories` and `memories_summaries`. It needs two additional Cosmos DB containers:

- **`counter`** - stores lightweight per-thread and per-user message counters used for threshold checks (90-day default TTL when the toolkit creates the container)
- **`leases`** - auto-created by the Azure Functions runtime for change feed checkpointing

The `COSMOS_DB__accountEndpoint` setting must also be configured for the identity-based change feed binding.

### When to use automatic vs. on-demand

| Approach | Best for |
|----------|----------|
| **On-demand** | Full control, testing, one-off processing, chaining operations |
| **Automatic** | Always-on background processing, fire-and-forget, production workloads |

Both approaches run the same pipeline, so the output is identical.

---

## 9. Procedures and Standing Instructions

Procedural memory records how the assistant should act for one user: instructions the user taught, habits learned from corrections or repeated choices, and skills that worked in practice. Procedures are learned from evidence, so a passing preference or a request about one task does not become a rule. See [Procedural](concepts.md#procedural) for the evidence each kind needs.

### How a procedure appears

1. The user teaches or corrects the assistant: "Always cite your sources." Extraction records the fact with `assertion="directive"` (or `"correction"`).
2. The procedure pass reads new behavioral facts and episode lessons and adds, revises or reinforces procedures. An `instruction` the user taught or corrected is `active` at once. A `habit` needs a correction by the user, or support on two different dates from facts the user stated or episodes that went well (a successful or partially successful outcome), and a `skill` needs two such episodes on different dates or the user's endorsement; until then it stays a `candidate`, which search and prompt injection do not show.
3. The pass runs after each reconcile (every `FACT_EXTRACTION_EVERY_N × DEDUP_EVERY_N` turns by default) unless `PROCEDURAL_SYNTHESIS_AUTO=false`. To run it now with the in-process backend:

```python
result = await mem.synthesize_procedural("user-1")
print(result["status"])  # "synthesized", "unchanged" (nothing new, no model call), "skipped" or "deferred"
```

With `DurableFunctionProcessor` the Function App runs the pass and this call raises `NotImplementedError`.

### Put standing instructions in the prompt

```python
standing = await mem.build_procedural_context("user-1")
# "# Standing instructions from the user\n- Reply in Spanish.\n- ..." - or "" when there are none
```

Only active, unexpired procedures that apply to nearly every interaction (`applies="always"`) are included, at most 10 rules and 2,048 bytes, taught rules first. Inject the block once, at the top of the system prompt for the session. For a task, pass `task=` to add the nearest situational rule, rendered as "When <conditions>: <rule>":

```python
context = await mem.build_procedural_context("user-1", task="Review this pull request")
```

Situational procedures can also arrive as a search slot ([Section 5](#search-slots-preferences-and-procedures)), or be looked up directly:

```python
rules = await mem.retrieve_procedures("user-1", "reviewing a pull request", top_k=3)
for rule in rules:
    print(rule["name"], rule["procedure_kind"], rule["applies"], rule["status"])
```

`retrieve_procedures` returns active procedures by default and can narrow by `procedure_kind`; pass `status=None` to see candidates too, for example to review what the assistant is still unsure about. When an active situational procedure cites a user's `directive` or `correction` fact, that fact is retired into the procedure (`supersede_reason="procedure"`), so the rule is retrieved once, as the procedure.

---

## 10. Keeping the Fact Pool Clean

Extraction only adds facts, so a long-running user accumulates restatements, overlapping details and, sometimes, contradictions. `reconcile` cleans up behind it in two passes: a duplicate pass retires a fact only when another states everything it states (and combines facts about one specific thing that each hold details the others lack), then a contradiction pass retires the loser of each pair of opposing claims. The duplicate pass never retires a contradiction, and the contradiction pass reads its pool after the duplicate pass has finished.

### When to call

- **Automatic** - `DEDUP_EVERY_N` (default 5) fires reconcile every Nth extraction, so every 10 turns with the default `FACT_EXTRACTION_EVERY_N=2`, in-process or through the change feed.
- **After a bulk import or backfill** - so the first searches see clean facts.
- **On a schedule** - for users whose facts accumulate faster than the automatic cadence cleans them up.

### How to call

```python
result = await mem.reconcile("user-1")  # n defaults to DEDUP_POOL_SIZE (50, at most 500)
print(result["merged"], result["combined"], result["contradicted"])
```

`merged` is the number of duplicates retired, `combined` the facts retired into combined facts (`synthesized` combined facts were written), and `contradicted` the contradiction losers; `checked`, `compared`, `vetoed`, `combine_groups` and `combine_failed` describe the duplicate pass. A pass checks at most `n` new facts, oldest first, and remembers where it stopped, so after a large import call `reconcile` again until `checked` is below `n`.

### What changes in your data

- Retired facts stay in the container with a `supersede_reason` (`duplicate`, `combined`, `contradict`); search hides them, `include_superseded=True` shows them, and `get_memory_history(memory_id, user_id)` walks a fact's chain.
- A surviving duplicate keeps its wording and gains `merged_from`; a combined fact is tagged `sys:combined` and lists its sources in `merged_from`.
- A user's statement is never retired into an agent-sourced fact, and agent-sourced facts are not part of the contradiction pass. A combined fact (`sys:combined`) is part of it: when a newer fact contradicts one, the facts it combined are made active again first, so a later pass retires only the one that conflicts.

### Cost and control

A pass makes one duplicate check per new fact that has close neighbours, a few calls per combine group, and one contradiction call. Raise `DEDUP_EVERY_N` to run reconcile less often, or lower `DEDUP_POOL_SIZE` to check fewer facts per pass. In the Function App, `ACTIVITY_TIME_BUDGET_SECONDS` stops a pass from starting model calls near the function timeout and the next pass continues from its cursor.

---

## 11. Secrets and Deleting Memories

### Secrets

Every write is scanned for credentials with a fixed format (private keys, Azure account keys and connection strings, SAS signatures, JWTs, Bearer and Basic authorization values, OpenAI, GitHub and AWS keys, and passwords inside URLs). A match is replaced with `[REDACTED-SECRET]` before the text is embedded or stored, raw turns included, so a credential a user pastes into a chat is not stored by the toolkit or sent to the embedding endpoint. The record is tagged `sys:secret` and carries `metadata.classification="secret"`:

```python
flagged_turns = await mem.get_thread(thread_id=THREAD_ID, user_id="user-1", tags_any=["sys:secret"])
flagged_memories = await mem.get_memories(user_id="user-1", tags_any=["sys:secret"])
```

Credentials without a fixed format, such as `password: hunter2`, cannot be recognized by a pattern: do not rely on the guard for them. The prompts that write memories tell the model never to store a literal secret, but treat that as a second line of defense, and keep raw secrets out of the conversation where you can.

### Deleting

```python
await mem.delete_memory(memory_id="<fact id>", user_id="user-1", thread_id=THREAD_ID, memory_type="fact")
```

- Deleting a fact or procedure first deletes the records retired into it (`duplicate`, `combined` and `procedure`) and anything retired into those, each under its own `thread_id`, so what the user deleted is gone from history too. Records replaced by a contradiction or an update stay, because they are earlier versions rather than copies.
- To remove a piece of information, delete the fact that is active now. If it was folded into another fact, search returns the survivor; deleting a retired record removes only that record.
- The cascade is sequential and best-effort: if deleting a folded record fails, the memory is kept and a retry resumes where it stopped.
- `delete_thread` removes a thread's turns and summary only; the facts, episodes and procedures distilled from it are user-scoped and stay.

---

## Quick Reference

| Operation | Method | When |
|-----------|--------|------|
| Store a turn | `upsert_memory` / `add_local` | Every user or agent message |
| Bulk upload | `push_to_cosmos` | After collecting local turns |
| Update a memory | `update_cosmos` | Correct or annotate an existing record |
| Delete a memory | `delete_memory` | Remove incorrect or sensitive data; a fact or procedure takes the records retired into it along |
| Get a thread | `get_thread` | Load recent conversation context |
| Semantic search | `search_cosmos` | Find relevant facts for a prompt; add episodes, summaries or turns with `include_*` |
| Search with slots | `search_cosmos(preference_slot=True, procedure_slot=True)` | Always include a matching preference and procedure within `top_k` |
| Summarize a thread | `generate_thread_summary` | End of conversation, periodically, or automatic via change feed |
| Extract facts | `extract_memories` | After key exchanges, end of conversation, or automatic via change feed |
| Learn procedures | `synthesize_procedural` | After reconcile (automatic by default), or on demand with the in-process backend |
| Standing instructions | `build_procedural_context` | Start of a session; pass `task=` to add the nearest situational rule |
| Look up procedures | `retrieve_procedures` | Find procedures for a task; `status=None` also lists candidates |
| Clean up facts | `reconcile` | Fold duplicates, combine facts, resolve contradictions: automatic, or after a bulk import |
| Summarize a user | `generate_user_summary` | Cross-session profiling, after thread summaries, or automatic via change feed |
| Get user summary | `get_user_summary` | Start of a new session |
