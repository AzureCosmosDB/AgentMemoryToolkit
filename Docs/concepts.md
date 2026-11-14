# Concepts

Agent Memory Toolkit stores conversation data as memory documents and builds higher-level artifacts on top of those documents for retrieval and reuse.

---

## Memory Schema

Every memory uses the same base shape:

| Field        | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| `id`         | Unique identifier                                                             |
| `user_id`    | User the memory belongs to                                                    |
| `thread_id`  | Conversation thread                                                           |
| `role`       | `user`, `agent`, `tool`, or `system`                                          |
| `type`       | `turn`, `thread_summary`, `fact`, `episodic`, `procedural`, or `user_summary` |
| `content`    | Main text payload                                                             |
| `embedding`  | Vector used for semantic search                                               |
| `metadata`   | Extra context (e.g. `tool_name`, `tool_call_id`, edit reasons)                |
| `created_at` | ISO 8601 timestamp                                                            |

---

## Memory Types

### Turn

**Type:** `turn`

Turn memories are raw conversation records. They are created by `add_local()`, `upsert_memory()`, and `push_to_cosmos()` (which bulk-uploads local memories to Cosmos DB). They act as the source material for summaries and facts.

**Use for:** full conversation history and short-term context.

### Summary

**Type:** `thread_summary`

A summary compresses one thread into a compact record of the main topic, key decisions, open issues, and next steps. It is created by `generate_thread_summary()` and stored with deterministic ID `summary_{user_id}_{thread_id}`.

If a summary already exists, the pipeline reads it, loads only newer memories, and merges them into an updated summary instead of rebuilding from scratch.

**Use for:** long conversations that need a compact recap.

### Fact

**Type:** `fact`

Facts are discrete assertions extracted from a thread, such as preferences, requirements, or confirmed decisions. `extract_memories()` stores each fact as its own document with its own embedding. It also stores substantive information the agent provided or did (`source=agent`, tag `sys:agent-fact`), so retrieval can include or exclude it.

Facts extracted with the default prompt (`extract_memories-v3.prompty`) record, beside `content`, `salience`, `confidence` and `metadata.category` (`preference`, `requirement`, `biographical` or `other`):

| Field             | Meaning                                                                                                                    |
|-------------------|----------------------------------------------------------------------------------------------------------------------------|
| `assertion`       | How the conversation put the fact forward (values below)                                                                   |
| `subject`         | Who or what the fact is mainly about; `user` and `assistant` are normalized                                                |
| `attribute`       | The single-valued property the fact gives a value of, in a few lowercase words (`home city`, `ci system`); empty otherwise |
| `entities`        | The people, organizations, places, projects and tools the fact names (at most 12)                                          |
| `source_turn_ids` | The turns the fact cites                                                                                                   |
| `merged_from`     | Ids of the facts retired into this one (see [Memory Reconciliation](#memory-reconciliation))                               |

`assertion` is one of:

- `stated` - a claim about how things are: the user, their world, their history, or content they share. The default, also for what the agent provided.
- `decided` - a choice, plan or commitment the user or their team adopted.
- `suggested` - a recommendation or proposal that was not adopted.
- `observed` - a result found or verified while working, or an action the agent completed.
- `directive` - the user told the assistant how to act in future, recurring situations ("always ...", "from now on ...").
- `correction` - the user corrected how the assistant did something, for next time.

Only the user can direct or correct the assistant, so an agent-sourced fact is never a `directive` or `correction`, and neither is a fact that does not cite a turn a user wrote (text from a tool or the agent cannot teach, whatever the model labels it): it is recorded as `stated`. `attribute` and `entities` are recorded for retrieval and for later extractions to reuse the same names; the toolkit does not retire facts by attribute. Facts from earlier prompts and facts written by hand have no `assertion`, `subject`, `attribute` or `entities`.

**Use for:** fine-grained semantic retrieval across threads.

### Episodic

**Type:** `episodic`

An episode is one bounded experience, segmented from a thread's turn stream at an idle gap, a topic shift or a size cap (the `EPISODE_*` settings): a title, a summary, timeline events, an optional outcome and `lessons`. It is created by `extract_episodes()`, which the auto-trigger evaluates every `EPISODE_EVAL_EVERY_N` turns; it expires after 90 days and is retrieved with `search_cosmos(include_episodes=True)` or `search_episodic_memories()`. Procedure learning reads episode lessons, and counts an episode with a successful or partially successful outcome as practice.

**Use for:** recalling what happened and what was learned in past sessions.

### Procedural

**Type:** `procedural`

A procedure says how the assistant should act for one user. It is a small, independently retrievable record, not one compiled prompt: it is stored under the sentinel thread id `__procedural__`, embedded from its `retrieval_text`, and created by `synthesize_procedural()`, which learns procedures from evidence rather than from a single message. A procedure's `procedure_kind` says how it was learned, and its `status` follows from the `evidence` it has gathered:

| `procedure_kind` | What it is                                                                   | Evidence                                                                                                       | Becomes `active`                                                                                                                                 |
|------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `instruction`    | The user told the assistant how to act in future ("always ...", "never ...") | `taught`, or `corrected` when the user corrects the rule                                                       | once the user has taught or corrected it                                                                                                         |
| `habit`          | A consistent way of acting the assistant should keep for this user           | `corrected` (the user corrected or redirected the assistant) or `repeated` (the user chose the same way again) | after a correction by the user, or when facts the user stated (or successful or partially successful episodes) support it on two different dates |
| `skill`          | A multi-step method for a recurring kind of task (at least two `steps`)      | `practiced` (it worked: a successful or partially successful episode) or `endorsed` (the user confirmed it)    | after two successful practices on different dates, or the user's endorsement                                                                     |

`taught` and `corrected` count only from a fact the user stated that extraction recorded as a `directive` or `correction` (a fact extracted before assertions were recorded also counts), so a plain preference, a request about one task, or anything the agent said cannot teach. A habit's two dates count only facts the user stated and episodes with a successful or partially successful outcome: facts the agent stated, and failed episodes, do not. Anything that has not met its kind's bar is a `candidate`: search, `retrieve_procedures()` (default `status="active"`) and `build_procedural_context()` do not show it, and an active procedure never goes back to a candidate.

A procedure `applies` either `always` (nearly every interaction: response language, tone, format, a tool convention) or `when` its `activation_conditions` match. It also records an optional `valid_until` date (after which search and prompt injection no longer show it), its support (`evidence`, `support_dates`, `user_support_count`, `practiced_dates`) and a `version` that rises with every revision. Its id is derived from the user, kind and name, so a lesson that comes up again reinforces one record instead of adding a paraphrase.

**Standing instructions.** `build_procedural_context(user_id, task=None)` compiles the user's active, unexpired `always` procedures into a block for the system prompt, or `""` when there are none. At most 10 rules and 2,048 bytes are included: taught rules first, then corrected, repeated, endorsed and practiced, newest first within each. Each rule appears as its summary; skill steps are not included. With a `task`, the nearest active situational (`when`) procedure by vector similarity is added, rendered with its conditions; if that lookup fails, the standing rules are returned without it:

```
# Standing instructions from the user
- Reply in Spanish.
- Keep answers under 150 words.

## Apply only when the situation matches
- When reviewing code for the user: Include a suggested fix with every review comment.
```

**How procedures are learned.** `synthesize_procedural()` runs after each reconcile (when `PROCEDURAL_SYNTHESIS_AUTO` is true) or on demand with the in-process backend. It reads the user's active preference and requirement facts, facts with salience of at least 0.8, `directive` and `correction` facts, and episodes with lessons, written since its cursor (the `procedure_cursor` document under `[user_id, "__procedural__"]`). A run reads at most 50 facts and 50 episodes and shows them to the model 10 at a time, each batch with the user's nearest existing procedures, so the model adds a procedure, revises one in place or records new evidence for one instead of writing a paraphrase. The cursor moves after each batch, so a run that stops early resumes where it stopped, and a run with nothing new makes no model call. A retryable model error, or a transient embedding or Cosmos DB error while writing a procedure, ends the run without moving the cursor past that batch, so the next run retries it. A revision that drops a number or a negation is discarded unless it records the user's correction.

**Retiring taught facts.** When an active situational procedure cites a user's `directive` or `correction` fact, the fact is retired into the procedure (`supersede_reason="procedure"`), so the rule is stored and retrieved once. A standing rule leaves its facts searchable, because it reaches the agent through `build_procedural_context()`. Only facts extracted with an `assertion` are retired.

**Reversed rules.** A rule is not retired because the user later says the opposite. The learning pass revises a rule in place only when the model judges the new statement to correct it; otherwise both rules stay active, and of two with the same standing the more recently updated comes first. Delete a rule you want gone.

**Use for:** keeping standing instructions and learned methods out of the fact pool and in the prompt.

### User Summary

**Type:** `user_summary`

A user summary is a cross-thread profile for one user. It captures stable context such as preferences, current work, environment details, and constraints. It is created by `generate_user_summary()` and stored with ID `user_summary_{user_id}` and `thread_id="__user_summary__"`.

Like thread summaries, user summaries update incrementally by merging the existing profile with only the new memories.

**Use for:** cross-session personalization and onboarding context.

---

## Short-Term vs. Long-Term Memory

|                 | Short-Term                                             | Long-Term                                                                                                                         |
|-----------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **What**        | Turn messages                                          | Summaries, facts, episodes, procedures, user summaries                                                                            |
| **Granularity** | Per message                                            | Per thread, fact, episode, procedure, or user                                                                                     |
| **Created by**  | `add_local()` / `upsert_memory()` / `push_to_cosmos()` | `generate_thread_summary()` / `extract_memories()` / `extract_episodes()` / `synthesize_procedural()` / `generate_user_summary()` |
| **Purpose**     | Replay recent context                                  | Compact recall, semantic retrieval and standing instructions                                                                      |

Common pattern: keep turns during an active conversation, then generate summaries, facts or procedures when the thread gets long or is complete.

---

## Threads and Roles

A thread is the unit of conversation. `get_thread()` returns the memories for one `thread_id`, optionally limited to the most recent `k` entries. `get_memories()` also supports a `thread_id` filter to retrieve memories from a specific thread.

| Role     | Meaning                                                           |
|----------|-------------------------------------------------------------------|
| `user`   | Human message                                                     |
| `agent`  | Assistant message                                                 |
| `tool`   | Tool output (metadata can include `tool_name` and `tool_call_id`) |
| `system` | Generated artifacts such as summaries and facts                   |

---

## Embeddings and Search

Memories stored in Cosmos DB include embeddings generated by Microsoft AI Foundry (e.g. `text-embedding-3-large`). This enables semantic search: a query is embedded, then Cosmos returns the closest matching memories via vector distance. `search_cosmos` also supports hybrid search (vector + full-text ranking via RRF).

Facts work especially well for vector search because each fact is stored as a small, self-contained document.

By default raw conversation turns are *not* embedded - only derived memories (facts, episodic, procedural, summaries) carry vectors. Set `enable_turn_embeddings=True` (env `ENABLE_TURN_EMBEDDINGS`) to also embed turns on write, then call `search_turns()` to vector-search the raw conversation log. The turns container is always provisioned with a `quantizedFlat` vector index, so this flag only toggles embedding generation and can be turned on or off at any time without recreating the container.

### Unified retrieval (`include_turns`)

`search_cosmos(..., include_turns=True)` returns extracted memories plus raw conversation turns in one call - useful for recovering detail that extraction dropped. Up to `turn_top_k` turns (default `top_k`) are **appended after** the memory hits, so memory results keep priority; the two sets are not score-fused. A turn is skipped only when its content is an **exact** string match of a returned memory - paraphrased overlaps are not de-duplicated. The turn search requires `enable_turn_embeddings` and is best-effort: if it fails, the memory results are returned unchanged.

### Result slots (`preference_slot`, `procedure_slot`)

A ranked search can fill `top_k` with topically close facts and miss a standing preference or a relevant procedure. Two opt-in flags reserve places **inside** `top_k` (they do not add results):

- `preference_slot=True` - the user's best-matching preference fact (`metadata.category="preference"`) that the ranked results do not already show.
- `procedure_slot=True` - the user's best-matching active situational procedure (`applies="when"`). Procedures then leave the ranked results and come only through the slot.

Slot candidates are ranked by vector similarity alone and use the search's own filters (user, thread, tags, salience, dates), so a slot can only return what the search itself could. The procedure slot ignores the filters no procedure can meet - `role`, `tags_all`, `tags_any`, `min_salience` and `min_confidence`, because a procedure has no salience or confidence, its `role` is `system` and its tags are the toolkit's own - and still applies the user, date, `exclude_tags` and `include_superseded` filters (a procedure matches on the user alone, whatever the `thread_id`). A slot with no candidate takes the next ranked result, so `top_k` stays filled: with `top_k=5` and both slots, a search returns 3 ranked results, 1 preference and 1 procedure. Each slotted result carries `retrieval={"rank": <1-based position>, "slot": "preference" | "procedure" | "backfill"}`; ranked results carry none.

Slots need `user_id` and are skipped when `memory_id` is set or when they would leave no ranked result (`top_k` must exceed the number of slots). The preference slot applies only when facts are searched, and `procedure_slot` has no effect when only procedural types are searched, because those searches already rank procedures. Episodes never take a slot, and summaries and turns blended in with `include_summaries` / `include_turns` still follow the `top_k` results. A failed slot query is logged and the slot falls back to the next ranked result.

### Procedures in search results

`search_cosmos` hides procedures that are not `active` or are past `valid_until`, so a candidate or an expired rule is never returned as a memory. `retrieve_procedures()` ranks a user's procedures for a task: `status="active"` by default (`status=None` includes candidates), optionally narrowed by `procedure_kind`; it does not check `valid_until`.

### Keyword search and partition scope

A keyword (hybrid) query ranks with `FullTextScore`, which is ignored in a query the Python SDK sends with a partition key, because it skips query planning ([Azure/azure-sdk-for-python#42241](https://github.com/Azure/azure-sdk-for-python/issues/42241)), so every question would get the same fixed order. A search that has keywords therefore runs across partitions with `c.user_id` (and `c.thread_id`, when given) in the `WHERE` clause, which confines it to the same records, and sets `full_text_score_scope="Local"` so BM25 statistics come from the partition ranges the query targets. This applies to `search_cosmos`, `search_turns`, `search_summaries`, `search_episodic_memories` and `retrieve_procedures`; expect more RUs than a single-partition read on a large container. A query whose terms are all stopwords has no keywords and ranks by vector similarity alone; with both `user_id` and `thread_id` it still reads one partition, except a `search_cosmos` that searches episodic or procedural memories, which always runs across partitions (those memories match on the user alone, whatever the `thread_id`). `MemoryStore.query()` raises `ValidationError` for a `FullTextScore` query sent with a partition key.

---

## Processing Pipeline

Derived memories are generated by the processing pipeline, which runs in-process or as Azure Durable Functions:

```
read existing doc if present -> query source memories -> call LLM -> embed output -> upsert to Cosmos DB
```

Thread summaries and user summaries support:

- deterministic IDs
- incremental updates
- `recent_k` limits to restrict how much history is processed (the newest turns are kept; turns are ordered by event time, `created_at`, then by write order, so turns that share a `created_at`, as in an imported session, keep the order they were written in)

Prompts live in `azure/cosmos/agent_memory/prompts/` as `.prompty` files, separate from pipeline code. Fact extraction uses `extract_memories-v3.prompty`; set `AMT_EXTRACT_MEMORIES_PROMPT` to `extract_memories-v2.prompty` or `extract_memories.prompty` to select an earlier prompt (an unknown value falls back to v3).

### Fact extraction and context

`extract_memories()` reads a thread's turns that are not yet stamped `extracted_at` (see [Extraction gating](#extraction-gating-extracted_at--recent_k)), packs them into token-bounded batches and calls the extraction prompt once per batch, so a single poisoned turn fails only its own batch.

The v3 prompt labels each new turn `Turn <id>` and also receives read-only context, so it can resolve references such as "yes, do that" or "the second option", keep the rationale a new turn points at, and skip what is already stored:

| Context        | What it holds                                                                                                                                                                      | Bound                                                               |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| Earlier turns  | The turns just before the batch (the thread's turns before the window, then the window's earlier batches), shown without turn ids                                                  | Up to 6 turns within 1,500 tokens, each clipped to 1,200 characters |
| Thread summary | Its overview and decisions, only when it summarizes nothing from the new turns                                                                                                     | 1,500 characters                                                    |
| Stored facts   | The user's active facts nearest the batch by vector similarity, written no later than the batch, each with its date, `user` or `agent`, assertion, subject, attribute and entities | 15 facts                                                            |

Facts still come only from the new turns, and each cites the turns it came from in `source_turn_ids` (a citation that does not resolve to a turn in the batch is dropped). Context is best-effort: a failed read costs only that part, and if the model call fails for a reason a retry would not fix (a content filter, an over-long prompt) the batch is retried once without context. It costs one embedding and one vector query per batch, plus one turns query and one summary read per extraction window. Turns that share one `created_at`, as in an imported session, are cut arbitrarily at the earlier-turns limit. The v1 and v2 prompts see only the transcript, and a fact they extract records every identified turn of its batch in `source_turn_ids`.

An empty model reply is retried with backoff and raises `LLMError` if it stays empty. A content-filtered reply, a refusal, a tool-only reply or a reply cut off at the output limit with no text is not retried: the batch is quarantined (counted in `quarantined_turn_count`) instead of being tried again on every run.

---

## Secret Redaction

Every write through the store or the pipeline passes a guard before the record is embedded or stored: `upsert_memory()`, `push_to_cosmos()`, `update_cosmos()`, and the facts, episodes, summaries and procedures the pipeline writes. Raw turns pass it too, so a turn that contains a credential is stored with the marker, not the credential.

The guard recognizes 12 formats whose shape cannot be mistaken for anything else: private-key blocks, Azure account keys and connection strings, SAS signatures and connection strings, JWTs, Bearer tokens, Basic authorization values, OpenAI keys, GitHub tokens, AWS access key ids, and passwords inside URLs. A URL password that holds a raw `/`, `@` or whitespace is not matched in full, so those characters must be percent-encoded, as URLs require. Every non-structural string of a record is scanned (ids, partition keys, timestamps and similar fields are not) and each match becomes `[REDACTED-SECRET]`. The record is tagged `sys:secret` and gets `metadata.classification="secret"` and `metadata.secret_patterns` (the format names). Derived state follows the text: `content_hash` is recomputed, and a vector built from the original text is dropped and rebuilt by embedding the redacted text, so no raw credential reaches the embedding endpoint and the record stays searchable. An update that changes the text of a record that has a vector re-embeds it as well; if that embedding fails or the store has no embeddings client, the update keeps the record's previous vector instead, dropping it only if the text it was built from held a credential. If `upsert_memory(embed=False)` switched embedding off, or the embedding fails or there is no client, `upsert_memory` has no vector to fall back on and writes the record without one. A tag that holds a credential is dropped rather than redacted, including one passed to `add_tags`. Logs name the record and the formats, never the text.

Credentials without a fixed format, such as `password: hunter2` typed into a chat, cannot be told from ordinary text by a pattern and are not matched. Every prompt that writes memory text (extraction, episodes, summaries, procedures, duplicate checks and combining) tells the model never to store a literal secret. A pattern can also redact legitimate text that looks like a credential.

---

## Memory Reconciliation

Two passes keep the fact pool clean and convergent, and they run in order inside `reconcile_memories()` (the clients' `reconcile()`): a **duplicate pass** that retires duplicate facts, and combines facts about one specific thing, without losing any detail, and an **LLM contradiction pass** that resolves opposing claims. The contradiction pass reads its pool after the duplicate pass has retired facts, so the two never act on the same stale list. Neither runs at write time: extraction only creates facts, and reconcile cleans up behind it.

### Duplicate pass (lossless dedup)

`consolidate_memories(user_id, n)` is the duplicate pass on its own; `reconcile_memories()` runs it first. It works through the facts written since the user's last pass:

1. **Seeds.** Up to `n` active facts written after the user's dedup cursor, oldest first. The cursor is a `dedup_cursor_<user_id>_fact` document under the sentinel thread id `__dedup__`; it records a position in Cosmos DB's write timestamp (`_ts`), so a fact is a seed when it is written and again whenever it is rewritten (for example, a survivor that gains `merged_from`, or a fact whose tags change). Older facts are also compared whenever a new fact lands near them.
2. **Candidates.** The seed's 5 nearest active facts for the same user, across threads, whose cosine similarity is at least 0.85.
3. **Verdicts.** One `dedup_verify` call labels every candidate (`same_claim`, `seed_contains_candidate`, `candidate_contains_seed`, `partial_overlap`, `update`, `inconsistent`, `related` or `unrelated`) and lists the details each side states that the other lacks.
4. **Folds.** A fact is retired into another only when the two state the same claim or the other contains it, the model lists nothing the retired fact would lose, and a deterministic check agrees that the survivor keeps the retired fact's numbers, names and negation. The survivor keeps its content verbatim, takes the union of both facts' tags and source turns and the higher salience and confidence, and lists the retired fact in `merged_from`; the retired fact gets `superseded_by` and `supersede_reason="duplicate"`.

Guards: a fact that has absorbed others (it has `merged_from`) is never retired, so merges do not chain; a user's statement is never retired into an agent-sourced fact; a fact is not folded into one with a different `assertion` (a suggestion and the later decision are a transition, not a duplicate), and a `directive` or `correction` is never folded into a fact that has no assertion; and conflicting pairs (`update`, `inconsistent`) are never retired here, they are left to the contradiction pass. Folds are not transactional: the survivor is written first (with an ETag check), then each loser is retired, so a crash leaves a duplicate that a later pass folds again, and a fact that changed since the pass read it stays as it is.

The cursor moves past the facts a pass finished. A retryable model error, or a `deadline` (below), ends the pass early: it applies the folds it found, and the next pass resumes after the last fact it checked. A fact whose check cannot be parsed, or fails for a reason a retry would not fix, is kept.

### Combining facts

The same pass also combines facts about one specific thing that each hold details the others lack, such as several facts about one dinner or one trip, so a search shows one complete fact instead of several overlapping ones:

- A fact is also compared with neighbours down to cosine 0.65 that it could be combined with: the same speaker (user or agent) and the same `assertion`, neither a `directive` or `correction`, and neither already a combined fact. Folds still need 0.85.
- Pairs judged the same claim, contained or overlapping form groups of up to 4 facts.
- `combine_memories.prompty` picks the members about one specific thing (at least two) and writes one fact that states every detail of each, keeping each member's status and date. `dedup_verify` must then find that it contains each member with nothing missing, a numbers / names / negation check must pass, and it must stay within 800 characters. One retry lists what was missing; after that the members stay as they are.
- The combined fact is created first (tag `sys:combined`, `merged_from` naming its members, the newest member's date, thread, category and speaker, the members' tags and the highest salience and confidence), then each member is retired into it with `supersede_reason="combined"`. Its id derives from the user and the members, so a retry after a crash, or a concurrent pass, finds it instead of making a second one.

A combined fact is never combined again, and the duplicate pass never retires it (the contradiction pass can, see below). Combining adds model calls to every pass.

### LLM contradiction reconcile

`reconcile_memories(user_id, n=50, *, memory_type="fact", deadline=None)` runs the duplicate pass, then loads up to `n` active (non-superseded) facts, most recent first, and asks the dedup prompt to identify **contradicted pairs** - opposing claims about the same subject (e.g. "deadline March 1" vs "March 15"). Agent-sourced facts (`sys:agent-fact`) are left out: they record what the agent did or recommended, so a later user statement does not supersede them. A combined fact (`sys:combined`) takes part, so a newer fact can correct it, and a fact combined earlier in the same run can retire an older one. A combined fact that loses holds details the correction may not touch, so it is retired only after the facts retired into it as `combined` are made active again (each is rewritten under an ETag check; one that changed meanwhile is left as it is). A later pass then compares each of them with the winner and retires only the one it contradicts. If a member cannot be restored, or the combined fact changed meanwhile, the pair is left for a later pass. The facts made active again are not counted in the result. Each loser is soft-deleted with `supersede_reason="contradict"` and `superseded_by` set to the winner (more recent wins, higher confidence as tiebreaker). Chained contradictions are guarded, so a fact already superseded in this pass can't be used to tombstone a third.

`episodic` and `procedural` types are no-ops (episodic has no contradiction semantics). The pass returns a dict:

| Key              | Meaning                                                                |
|------------------|------------------------------------------------------------------------|
| `kept`           | Facts left active in the contradiction pass                            |
| `merged`         | Duplicates retired into another fact                                   |
| `contradicted`   | Losers retired as contradicted                                         |
| `checked`        | New facts the duplicate pass checked                                   |
| `compared`       | Fact pairs the model judged                                            |
| `vetoed`         | Folds blocked because a number, name or negation would be lost         |
| `combined`       | Facts retired into a combined fact                                     |
| `synthesized`    | Combined facts written                                                 |
| `combine_groups` | Groups the combine step tried                                          |
| `combine_failed` | Groups it declined or could not write, whose members stay as they were |

### Time budget (`deadline`)

`PipelineService.reconcile_memories()`, `consolidate_memories()` and `synthesize_procedural()` accept a `deadline`, a `time.monotonic()` value for callers with a time limit. Once it has passed they start no new model call: the duplicate pass applies the folds it found and moves its cursor past the facts it checked, reconcile skips the contradiction pass, and procedure learning keeps what it wrote; the next run continues from the cursors. When combining, the duplicate checks stop halfway to the deadline so the combined facts have time to be written. The Function App sets it from `ACTIVITY_TIME_BUDGET_SECONDS`; the clients' `reconcile()` and the in-process auto-trigger set none.

### Loser preservation

Retired facts stay in the container with their `supersede_reason` (`contradict`, `duplicate`, `combined`, or `procedure` for a directive or correction retired into a procedure), `superseded_at`, and `superseded_by` fields populated. Default reads (`get_memories`, `search_cosmos`) filter them out via `superseded_by IS NULL`. To inspect the audit trail, opt out of the filter at the query level (`include_superseded=True`), or walk a fact's chain with `get_memory_history()`. Facts belong to a user, not a thread: a fact retired into a fact from another thread no longer shows in a read filtered to its own thread.

### Deleting a memory

`delete_memory()` removes one document. For a fact or procedure it first deletes the records retired into it (`supersede_reason` of `duplicate`, `combined` or `procedure`) and, transitively, anything retired into those, because they hold the same information as the memory they were folded into: what the user deleted is gone. The records are looked up across the user's partitions and deleted deepest first, each under its own `thread_id`, before the memory itself. The delete is sequential and best-effort: if one fails, the memory is kept and a retry resumes. Records retired for another reason (`contradict`, `update`) are earlier versions, not copies, and stay. Deleting a retired record deletes only that record, so to remove information delete the fact that is active now (the survivor or the combined fact). `delete_thread()` deletes only a thread's turns and summary.

### Exact duplicates at write time

Each fact written by `extract_memories` carries a `content_hash` (SHA-256 of normalized content, truncated to 32 hex chars; lowercase, whitespace-collapsed) and a deterministic id derived from the user, the thread and that hash. A repeat inside one extraction is skipped and counted in `exact_dedup_skipped`; a repeat of an already stored fact in the same thread collides on its id and is skipped when the create returns 409. There is no per-extraction query for existing hashes. The same content in another thread is a separate document, and the duplicate pass folds it.

### Extraction gating (`extracted_at` + `recent_k`)

Extraction only ever reads turns not yet stamped `extracted_at`. The in-process `extract_memories()` stamps every turn it consumed once the extracted memories are durably written, so this gate is its authoritative "what's left to extract" signal: a failed or lagging extract leaves its turns unstamped, so the full backlog is retried on the next run and no turns are skipped. The Function App does not stamp turns (a rewritten turn would reach the change feed again); it relies on the watermark below.

- **In-process SDK auto-trigger** - calls extraction with `recent_k=None`, so it drains the entire unextracted backlog each run and relies purely on the `extracted_at` gate.
- **Change-feed / Durable Function App** - **sizes** `recent_k` from a per-thread **watermark** (`last_extract_count` on the counter doc): `recent_k = current_count − last_extract_count` (with `last_extract_count` treated as `0` before the first successful extract); turns stamped `extracted_at` are still skipped. The watermark advances **only after a successful extract**, and it moves past the turns of a batch the model deferred after a retryable error (`deferred_turn_count`), so a later run does not retry them. The window is deliberately **not** capped by `DEDUP_POOL_SIZE` (that knob governs the reconcile pool, not the extraction window) - capping would extract only the newest N and silently strand the oldest backlog turns.

> **Caveat (rare, change-feed path):** the counter increment is best-effort - under sustained optimistic-concurrency contention it can drop an increment rather than block the user's write path (see `increment_counter_sync`). A dropped increment leaves `current_count` lagging the true turn count, which can under-size the watermark-derived `recent_k`. The Function App backend avoids stranding turns by raising to force change-feed redelivery; the in-process path is unaffected because it uses `recent_k=None` and the `extracted_at` gate, not the counter, to decide what to extract.

### Tunable

These settings are operator-configurable (the Function App reads the same names from its App Settings):

- `DEDUP_EVERY_N` (default `5`) - how often reconcile runs in the auto-trigger path (every Nth **extract**, not every Nth turn). Set to `0` to disable.
- `DEDUP_POOL_SIZE` (default `50`, hard cap `500`) - the `n` passed to `reconcile_memories`: how many new facts one duplicate pass checks (oldest first; the rest wait for the next pass) and how many facts the contradiction pass reads; also overridable per call. Larger values check more facts per pass and give the LLM a wider view, at higher token cost.
- `PROCEDURAL_SYNTHESIS_AUTO` (default `true`) - whether procedure learning follows each reconcile.
- `ACTIVITY_TIME_BUDGET_SECONDS` (Function App only; default `240`, minimum `30`) - how long one reconcile or procedure activity keeps starting model calls. Keep it below the app's `functionTimeout`.

The similarity floors (0.85 to fold, 0.65 to combine), the neighbour count (5) and the combine limits (4 facts, 800 characters) are fixed in `thresholds.py`, not settings. Cost: a pass makes one `dedup_verify` call per new fact that has close neighbours, up to four calls per combine group (write, check, and one retry of each), and one contradiction call.

> **Indexing note.** The contradiction pass orders its pool query by `created_at` (matching the prompt's "more recent first" tiebreaker), and the extraction prompt's earlier-turns query does too. Cosmos's default indexing policy includes every property, so this works out of the box. If you customize the indexing policy to reduce write RU, ensure `/created_at/?` remains indexed or the contradiction query will fail with a 400 (`Order-by over a non-indexed path`) and the earlier-turns context will be dropped. The duplicate pass orders by `_ts`, which is always indexed, and ranks neighbours with `VectorDistance`, which uses the vector index the toolkit creates on `/embedding`.

---

## Automatic Processing (Change Feed)

In addition to on-demand processing via the SDK, the toolkit includes a Cosmos DB change feed trigger that **automatically** starts processing orchestrations when enough new turns have been written.

```
memories_turns container
      │  (change feed)
      ▼
on_memory_change trigger
      │
      ├── count turns per (user_id, thread_id)
      │   └── crosses threshold? ──► start thread_summary / extract_memories / extract_episodes
      │
      └── count turns per user_id
          └── crosses threshold? ──► start user_summary
```

### How it works

1. The change feed trigger watches the `memories_turns` container for new documents.
2. Only documents with `type == "turn"` are counted.
3. Documents in the dedicated `counter` container track how many turns have been seen per scope using ETag-based optimistic concurrency.
4. When a counter crosses a configured threshold, the corresponding Durable Functions orchestration is started automatically.

### Threshold settings

| Setting                   | Scope                                                                           | Default |
|---------------------------|---------------------------------------------------------------------------------|---------|
| `THREAD_SUMMARY_EVERY_N`  | Per `(user_id, thread_id)`                                                      | `10`    |
| `FACT_EXTRACTION_EVERY_N` | Per `(user_id, thread_id)`                                                      | `2`     |
| `DEDUP_EVERY_N`           | Per `(user_id, thread_id)`; reconcile every `FACT_EXTRACTION_EVERY_N × N` turns | `5`     |
| `EPISODE_EVAL_EVERY_N`    | Per `(user_id, thread_id)`; checks for an episode boundary every N turns        | `4`     |
| `USER_SUMMARY_EVERY_N`    | Per `user_id` (across all threads)                                              | `20`    |

These defaults are shared by both backends (`function_app/shared/config.py` imports the same constants from `azure.cosmos.agent_memory.thresholds`), so the InProcess and Durable processors fire on the same turn boundaries unless overridden. Set any value to `0` to disable that processing type. For example, setting `THREAD_SUMMARY_EVERY_N=5` generates a thread summary every 5 new turns in each thread.

An extraction orchestration runs extract and persist, then, when the reconcile threshold is crossed, a reconcile activity, and then, when `PROCEDURAL_SYNTHESIS_AUTO` is true, procedure learning as a sub-orchestration. The reconcile and procedure activities stop starting model calls after `ACTIVITY_TIME_BUDGET_SECONDS` (default `240`), keep what they finished and continue from their cursors on the next run.

A failed activity reports only `<code> (<ExcType>)` to Durable Functions, never the original exception text, which can quote memory text or model output: the code is `write_conflict` for HTTP 412, `memory_not_found` for 404, the toolkit's own `error_code` when there is one, and `dependency_failure` otherwise. The Function App log keeps the exception's type, status and code locations, also without its message. The activity is retried (3 attempts, 2 s first interval).

### Required containers

| Container            | Partition Key                           | Purpose                                                                                         |
|----------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------|
| `memories`           | `/user_id`, `/thread_id` (hierarchical) | Durable derived memories (`fact`, `episodic`, `procedural`)                                     |
| `memories_turns`     | `/user_id`, `/thread_id` (hierarchical) | Raw conversation turns (`turn`) - append-only, TTL-pruned                                       |
| `memories_summaries` | `/user_id`, `/thread_id` (hierarchical) | Thread + user summaries (`thread_summary`, `user_summary`)                                      |
| `counter`            | `/user_id`, `/thread_id` (hierarchical) | Message count tracking for automatic processing; 90-day default TTL when the toolkit creates it |
| `leases`             | `/id`                                   | Change feed checkpointing container created by `create_memory_store()`                          |

The 90-day TTL is the `counter` container's default, so it applies to every document in it - thread counters, user counters and the extraction watermark kept on a thread counter - and each expires 90 days after its last update. An existing `counter` container keeps its current TTL setting: `create_container_if_not_exists` does not change it. The `memories` container also holds the cursor documents that resume the duplicate pass (`dedup_cursor`, under `__dedup__`), procedure learning (`procedure_cursor`, under `__procedural__`) and episode extraction (`episode_cursor`, one per thread); they are not memories and `get_memories()` does not return them.

### Throughput configuration

The toolkit provisions all required Cosmos containers under one shared throughput mode:

- `serverless` is the default. The toolkit creates the `memories`, `memories_turns`, `memories_summaries`, `counter`, and `leases` containers without specifying RU/s.
- `autoscale` applies the shared `autoscale_max_ru` cap (default `1000`; the samples read it from `COSMOS_DB_AUTOSCALE_MAX_RU`) to all five containers.

This keeps the change feed dependencies aligned with the main memory store instead of letting the Functions trigger create the lease container independently.

### Push vs. pull

| Mode                 | Trigger                                      | Use case                                                                    |
|----------------------|----------------------------------------------|-----------------------------------------------------------------------------|
| **On-demand (pull)** | SDK call (`generate_thread_summary()`, etc.) | Explicit control over when processing happens                               |
| **Automatic (push)** | Change feed trigger                          | Fire-and-forget - processing happens in the background as turns are written |

Both modes run the same pipeline code (`PipelineService`; the Function App's activities call it), so prompts, incremental update logic, and stored outputs are identical.

---

## Local vs. Cloud Storage

| Backend               | Use Case                                   | Persistence          |
|-----------------------|--------------------------------------------|----------------------|
| **Local (in-memory)** | Development and quick testing              | Lost on process exit |
| **Cosmos DB**         | Production, shared access, semantic search | Durable              |

Local storage is enough for CRUD testing. Cosmos DB is required for persistence, vector search, and the processing pipeline.

Cosmos DB uses a hierarchical partition key (`user_id`, `thread_id`) for efficient queries scoped to a user or thread.
