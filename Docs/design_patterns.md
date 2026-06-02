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
await mem.add_cosmos(
    user_id="user-1", thread_id=THREAD_ID,
    role="user", content="I need to migrate our PostgreSQL database to Cosmos DB.",
)

# Store agent response
await mem.add_cosmos(
    user_id="user-1", thread_id=THREAD_ID,
    role="agent", content="I can help with that. What's your current schema look like?",
)

# Store a tool call result with metadata
await mem.add_cosmos(
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
await mem.update_cosmos(memory_id="<id>", content="Corrected message text")

# Delete a memory (requires all partition key values)
await mem.delete_cosmos(memory_id="<id>", user_id="user-1", thread_id=THREAD_ID)
```

---

## 2. Generating a Thread Summary

### When to call

- **End of conversation** — after the user closes a session or a support ticket is resolved.
- **Long-running thread** — when a thread exceeds a token budget (e.g. > 50 turns) and you need a compact representation for context.
- **Periodic background job** — on a schedule to keep summaries up to date for active threads.
- **Automatic (change feed)** — set `THREAD_SUMMARY_EVERY_N` and the change feed trigger handles it. See [Section 8](#8-automatic-processing-with-change-feed).

Summaries are incremental: if one already exists for the thread, only newer turns are merged in.

### How to call

```python
result = await mem.generate_thread_summary(
    user_id="user-1",
    thread_id=THREAD_ID,
    recent_k=100,        # optional: limit to the most recent 100 turns
)
print(result["output"])  # orchestration result with the summary
```

The summary is stored automatically in Cosmos with id `summary_user-1_thread-abc-123` and `type="summary"`.

---

## 3. Extracting Facts

### When to call

- **After each meaningful exchange** — extract facts from the latest turns so they are available for retrieval immediately.
- **End of conversation** — capture all discrete preferences, decisions, and requirements from the thread.
- **Before a planning step** — in multi-agent workflows, extract facts before handing context to a planner agent.
- **Automatic (change feed)** — set `FACT_EXTRACTION_EVERY_N` and the change feed trigger handles it. See [Section 8](#8-automatic-processing-with-change-feed).

Each fact is stored as its own document with its own embedding, making it ideal for fine-grained semantic search.

### How to call

```python
result = await mem.extract_facts(
    user_id="user-1",
    thread_id=THREAD_ID,
    recent_k=50,
)
```

---

## 4. Generating a User Summary

### When to call

- **Cross-session onboarding** — at the start of a new thread, generate (or update) the user summary so the agent has context from all prior conversations.
- **After a thread summary is created** — chain it: summarize the thread, then update the user summary.
- **On a schedule** — for users with many threads, run periodically to keep the profile current.
- **Automatic (change feed)** — set `USER_SUMMARY_EVERY_N` and the change feed trigger handles it. See [Section 8](#8-automatic-processing-with-change-feed).

User summaries are also incremental. The pipeline merges only new thread data into the existing profile.

### How to call

```python
result = await mem.generate_user_summary(
    user_id="user-1",
    thread_ids=["thread-abc-123", "thread-def-456"],  # optional: specific threads
    recent_k=50,
)
```

The summary is stored with id `user_summary_user-1` and `thread_id="__user_summary__"`.

---

## 5. Retrieving Memories

### Get an entire thread

```python
turns = await mem.get_thread(thread_id=THREAD_ID, user_id="user-1", recent_k=20)
```

### Semantic search

Search across all memories (or filter by type) to find the most relevant context for a prompt.

```python
# Vector search for relevant facts
facts = await mem.search_cosmos(
    search_terms="database migration requirements",
    user_id="user-1",
    memory_types=["fact"],
    top_k=10,
)

# Hybrid search (vector + full-text) across all memory types
results = await mem.search_cosmos(
    search_terms="PostgreSQL to Cosmos DB",
    user_id="user-1",
    hybrid_search=True,
    top_k=5,
)
```

### Retrieve the user summary

```python
profile = await mem.get_user_summary(user_id="user-1")
```

### Query with filters

```python
# All summaries for a user
summaries = await mem.get_memories(user_id="user-1", memory_types=["summary"])

# All facts
facts = await mem.get_memories(user_id="user-1", memory_types=["fact"])

# Filter by thread_id
thread_turns = await mem.get_memories(user_id="user-1", thread_id=THREAD_ID)
```

---

## 6. End-to-End: Chat Application

A typical chat application lifecycle looks like this:

```
New session starts
  │
  ├─ Retrieve user summary          (get_user_summary)
  ├─ Semantic search for prior facts (search_cosmos, memory_types=["fact"])
  │
  │  ┌── Conversation loop ──┐
  │  │ Store each turn        │  (add_cosmos)
  │  │ Optionally extract     │  (extract_facts — every N turns or on key exchanges)
  │  └────────────────────────┘
  │
  ├─ Summarize the thread            (generate_thread_summary)
  ├─ Extract remaining facts         (extract_facts)
  └─ Update user summary             (generate_user_summary)
```

### Minimal example

```python
# --- Session start ---
profile = await mem.get_user_summary(user_id="user-1")
relevant = await mem.search_cosmos("topic of interest", user_id="user-1", memory_types=["fact"], top_k=5)

# Build system prompt with profile and relevant facts
system_prompt = build_prompt(profile, relevant)

# --- Conversation loop ---
while not done:
    user_msg = get_user_input()
    await mem.add_cosmos(user_id="user-1", thread_id=THREAD_ID, role="user", content=user_msg)

    agent_reply = call_llm(system_prompt, user_msg)
    await mem.add_cosmos(user_id="user-1", thread_id=THREAD_ID, role="agent", content=agent_reply)

# --- Session end ---
await mem.generate_thread_summary(user_id="user-1", thread_id=THREAD_ID)
await mem.extract_facts(user_id="user-1", thread_id=THREAD_ID)
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
await mem.add_cosmos(
    user_id="user-1", thread_id="research-thread",
    role="agent", agent_id="research-agent",
    content="Found that the source DB has 12 tables with 3 foreign key chains.",
)

# After research is complete, extract facts for other agents to consume
await mem.extract_facts(user_id="user-1", thread_id="research-thread")

# Planner agent retrieves relevant facts before generating a plan
facts = await mem.search_cosmos(
    search_terms="source database schema foreign keys",
    user_id="user-1",
    memory_types=["fact"],
    top_k=10,
)

# Planner writes its plan as a turn in its own thread
await mem.add_cosmos(
    user_id="user-1", thread_id="planning-thread",
    role="agent", agent_id="planner-agent",
    content=plan_text,
)
```

### Pattern: per-agent threads, cross-agent retrieval

Each agent writes to its own `thread_id`. Other agents discover relevant context through `search_cosmos` across all threads for the user. At the end, `generate_user_summary` produces a unified profile from all agent threads.

```python
# After all agents finish
await mem.generate_user_summary(
    user_id="user-1",
    thread_ids=["research-thread", "planning-thread", "execution-thread"],
)
```

---

## 8. Automatic Processing with Change Feed

Instead of calling `generate_thread_summary()`, `extract_facts()`, or `generate_user_summary()` explicitly, you can let the Cosmos DB change feed trigger fire them automatically in the background.

### How it works

When a new turn is written to the `memories` container, the change feed trigger:

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

Set any value to `0` to disable that processing type. All three default to `0` (disabled).

### Required infrastructure

The change feed trigger needs two additional Cosmos DB containers beyond the existing `memories` container:

- **`counter`** — stores lightweight per-thread and per-user message counters used for threshold checks
- **`leases`** — auto-created by the Azure Functions runtime for change feed checkpointing

The `COSMOS_DB__accountEndpoint` setting must also be configured for the identity-based change feed binding.

### When to use automatic vs. on-demand

| Approach | Best for |
|----------|----------|
| **On-demand** | Full control, testing, one-off processing, chaining operations |
| **Automatic** | Always-on background processing, fire-and-forget, production workloads |

Both approaches use the same orchestrator and activities, so the output is identical.

---

## Quick Reference

| Operation | Method | When |
|-----------|--------|------|
| Store a turn | `add_cosmos` / `add_local` | Every user or agent message |
| Bulk upload | `push_to_cosmos` | After collecting local turns |
| Update a memory | `update_cosmos` | Correct or annotate an existing record |
| Delete a memory | `delete_cosmos` | Remove incorrect or sensitive data |
| Get a thread | `get_thread` | Load recent conversation context |
| Semantic search | `search_cosmos` | Find relevant facts or summaries for a prompt |
| Summarize a thread | `generate_thread_summary` | End of conversation, periodically, or automatic via change feed |
| Extract facts | `extract_facts` | After key exchanges, end of conversation, or automatic via change feed |
| Summarize a user | `generate_user_summary` | Cross-session profiling, after thread summaries, or automatic via change feed |
| Get user summary | `get_user_summary` | Start of a new session |
