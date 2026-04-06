# Design Patterns

This guide shows when and how to use the toolkit's main operations in real applications. All examples use the async API (`AsyncAgentMemory`); the sync API (`AgentMemory`) has the same method signatures without `await`.

---

## 1. Storing Conversation Turns (CRUD)

### When to write memories

Write a turn memory every time a user or agent message is produced. If the application runs locally first and syncs later, use the local + bulk-upload pattern.

```python
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from agent_memory_toolkit.aio import AsyncAgentMemory

mem = AsyncAgentMemory(
    cosmos_endpoint=COSMOS_ENDPOINT,
    cosmos_database="memory",
    cosmos_container="memories",
    ai_foundry_endpoint=AOAI_ENDPOINT,
    embedding_model="text-embedding-3-large",
    adf_endpoint=ADF_ENDPOINT,
    adf_key=ADF_KEY,
    use_default_credential=True,
    cosmos_credential=AsyncDefaultAzureCredential(),
)
await mem.connect_cosmos(
    endpoint=COSMOS_ENDPOINT,
    database="memory",
    container="memories",
    credential=AsyncDefaultAzureCredential(),
)

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
    memory_type="fact",
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
summaries = await mem.get_memories(user_id="user-1", memory_type="summary")

# All facts
facts = await mem.get_memories(user_id="user-1", memory_type="fact")

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
  ├─ Semantic search for prior facts (search_cosmos, type="fact")
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
relevant = await mem.search_cosmos("topic of interest", user_id="user-1", memory_type="fact", top_k=5)

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
    memory_type="fact",
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

## Quick Reference

| Operation | Method | When |
|-----------|--------|------|
| Store a turn | `add_cosmos` / `add_local` | Every user or agent message |
| Bulk upload | `push_to_cosmos` | After collecting local turns |
| Update a memory | `update_cosmos` | Correct or annotate an existing record |
| Delete a memory | `delete_cosmos` | Remove incorrect or sensitive data |
| Get a thread | `get_thread` | Load recent conversation context |
| Semantic search | `search_cosmos` | Find relevant facts or summaries for a prompt |
| Summarize a thread | `generate_thread_summary` | End of conversation or periodically |
| Extract facts | `extract_facts` | After key exchanges or end of conversation |
| Summarize a user | `generate_user_summary` | Cross-session profiling, after thread summaries |
| Get user summary | `get_user_summary` | Start of a new session |
