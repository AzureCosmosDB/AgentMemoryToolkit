# Durable orchestrator activity chains

Extraction is split from persistence so Durable retries after a Cosmos write failure do not re-run the LLM activity.

```text
ExtractMemoriesOrchestrator
  em_Extract  (load recent turns + LLM + parse; read-only context lookups, no writes)
  em_Persist  (embeddings + deterministic create_item; 409 = already persisted)
  em_AdvanceExtractWatermark (when the change feed passes a count; runs only after a successful persist)
  em_ReconcileMemories (optional; single activity for GA; time-budgeted dedup, combine and contradiction passes)
  SynthesizeProceduralOrchestrator (best-effort sub-orchestration after reconcile when PROCEDURAL_SYNTHESIS_AUTO is on)

ExtractEpisodesOrchestrator
  ee_ExtractEpisodes

ThreadSummaryOrchestrator
  ts_Extract
  ts_PersistSummary

UserSummaryOrchestrator
  us_Extract
  us_PersistUserSummary

SynthesizeProceduralOrchestrator
  sp_SynthesizeProcedural (single activity for GA; cursor-based and time-budgeted)
```

A batch whose model call fails with a retryable error yields no facts and is counted in `deferred_turn_count`. `em_AdvanceExtractWatermark` still moves past its turns, so the Function App does not retry them (the in-process backend does).

Fact IDs are deterministic from user, thread, and normalized content, and episode IDs from user, thread, the segment's first and last turn and the episode's position in it. Thread and user summaries keep their deterministic summary IDs. Procedure IDs derive from the user, kind and name, and a combined fact's ID from the user and its source IDs, so a retried activity finds the record it already wrote (409) instead of writing a second one.

`UserSummaryOrchestrator` can start before extraction has persisted anything for a new user. `us_Extract` then returns `no_memories_yet` instead of failing, and the orchestrator polls on a durable timer (App Settings `USER_SUMMARY_WAIT_SECONDS`, default `120`, every `USER_SUMMARY_WAIT_INTERVAL_SECONDS`, default `10`) before it skips that cadence; the next user-summary threshold tries again. `USER_SUMMARY_WAIT_SECONDS=0` makes it a single attempt.

## Reconcile activity

`em_ReconcileMemories` calls `PipelineService.reconcile_memories` for `fact` and for `episodic` (a no-op) with `n = DEDUP_POOL_SIZE` and one deadline, `time.monotonic() + ACTIVITY_TIME_BUDGET_SECONDS` (default 240 s, minimum 30). Keep the budget below the app's `functionTimeout`. Each call runs two passes in order:

1. **Dedup pass** (`consolidate_memories`). It takes up to `n` active facts written after the user's dedup cursor, oldest first, and compares each with its nearest active neighbours in one `dedup_verify` call. A fact is folded into another (`supersede_reason="duplicate"`) only when nothing it states would be lost. Facts about one specific thing that each hold details the others lack also become one new fact (tagged `sys:combined`) and their sources are retired as `combined`. The cursor is the `dedup_cursor_<user_id>_fact` document under the thread id `__dedup__`.
2. **Contradiction pass.** It re-queries the `n` most recent active facts, after the folds (so a fact combined in this run is in the pool), and retires the loser of each contradiction (`contradict`). Agent-sourced facts (`sys:agent-fact`) are left out of it: a later user statement does not supersede what the agent did or recommended. A combined fact (`sys:combined`) that loses first has the facts retired into it as `combined` written back as active, each under an ETag check, and is then retired, so a later pass retires only the source the winner contradicts. A fact that changed meanwhile, or a write that fails, leaves the pair for a later pass.

Once the deadline has passed no new model call starts. The dedup pass applies the folds it found and moves its cursor past the facts it finished, and the contradiction pass is skipped. When combining, checking stops halfway to the deadline so the groups have time to be written, and the cursor stops before the first fact of a group that was not tried, so the next run checks it again. A retryable model error also ends the dedup pass early, with the same cursor handling.

Folds and combines are not transactional; each is ordered so a crash leaves a state the next run finishes. A fold writes the survivor first (it lists the retired facts in `merged_from`) and then retires each loser. A combined fact is created first and its sources are retired after. The cursor moves only after the writes it covers.

## Procedure activity

`sp_SynthesizeProcedural` calls `PipelineService.synthesize_procedural` with the same time budget. The pass reads the facts and episodes written after the user's `procedure_cursor` (document `procedure_cursor` under the thread id `__procedural__`), up to 50 of each, and sends them to the model 10 sources at a time, so a run makes at most 10 sequential model calls. The cursor moves after each batch. At the deadline the run stops starting calls and the next run continues. A retryable model error, or a transient embedding or Cosmos failure while writing a procedure, stops the run (`deferred`) without moving the cursor past that batch, and the result counts what the run wrote before it stopped; any other failure skips that batch. A procedure is created with `create_item` (409 = already there) and revised with an ETag-guarded replace; a directive or correction fact retired into it (`supersede_reason="procedure"`) follows after the procedure is written. The activity returns only `status`, `procedures_created` and `facts_retired` to the Durable history.

## Failure reporting and redaction

Every activity carries `@redact_errors` (`function_app/shared/error_redaction.py`), directly under `@bp.activity_trigger`; a unit test fails if one does not. A failing activity raises `ActivityFailedError("<code> (<ExcType>)")` instead of the original exception, because Durable keeps the error in the orchestration history and an exception can quote model output or memory text. The code is `write_conflict` for an HTTP 412 and `memory_not_found` for a 404 anywhere along the `__cause__` chain, otherwise the toolkit's error code, otherwise `dependency_failure`. The Function App log gets `Activity <name> failed error_code=<code>: <details>` with the exception's status, validation fields and code locations (`describe_exception`), never its message, arguments or local variables. The failure is still raised, so `default_retry_options` (3 attempts, 2 s first interval) applies.

Fact documents are redacted for credentials before they leave `em_Extract`. A summary body is redacted when `ts_PersistSummary` or `us_PersistUserSummary` writes it, so it crosses the Durable history before that.
