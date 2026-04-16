"""Azure Durable Functions app for Agent Memory operations.

Activities:
  - load_memories: Fetch memories from Cosmos DB by thread_id
  - generate_embeddings: Embed text via Azure AI Foundry
  - store_results: Upsert a memory document into Cosmos DB
  - generate_thread_summary: Generate or incrementally update a thread summary
  - extract_facts: Extract facts from memories using an AI Foundry LLM
  - generate_user_summary: Generate a cross-thread user profile

Change Feed Trigger:
  - on_memory_change: Watches the memories container for new turn documents,
    manages per-scope counters, and starts processing orchestrations based
    on configurable thresholds.

The orchestrator chains these activities in sequence.

Change Feed Configuration
-------------------------
The change feed trigger automatically processes new turn documents and starts
orchestrations when configurable message count thresholds are crossed.

Required application settings:

  COSMOS_DB__accountEndpoint
      The Cosmos DB account endpoint URL used by the change feed trigger
      binding (identity-based connection) and all Cosmos container clients.

  COSMOS_DB_DATABASE
      The Cosmos DB database name used by the trigger and container clients.

  COSMOS_DB_CONTAINER
      The memories container watched by the change feed trigger.

  COSMOS_DB_LEASE_CONTAINER
      The lease container used by the trigger for checkpointing.

Processing threshold settings (set to ``"0"`` to disable):

  THREAD_SUMMARY_EVERY_N
      Trigger a thread summary every N turns per ``(user_id, thread_id)`` pair.

  FACT_EXTRACTION_EVERY_N
      Trigger fact extraction every N turns per ``(user_id, thread_id)`` pair.

  USER_SUMMARY_EVERY_N
      Trigger a user summary every N turns per ``user_id`` across all threads.

Required Cosmos DB containers:

  - ``memories`` – existing container for memory documents
  - ``counter``  – dedicated container for change feed counter documents
                   (configurable via ``COSMOS_DB_COUNTERS_CONTAINER``)
  - ``leases``   – auto-created by the trigger for change feed checkpointing
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

import azure.durable_functions as df
import azure.functions as func
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from activities import _get_cosmos_counter_container
from activities import bp as activities_bp

logger = logging.getLogger(__name__)

df_app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)
df_app.register_functions(activities_bp)


# =====================================================================
# Shared helpers for change feed trigger
# =====================================================================

USER_COUNTER_THREAD_ID = "__counters__"
CHANGE_FEED_DATABASE = os.environ.get("COSMOS_DB_DATABASE", "ai_memory")
CHANGE_FEED_CONTAINER = os.environ.get("COSMOS_DB_CONTAINER", "memories")
CHANGE_FEED_LEASE_CONTAINER = os.environ.get("COSMOS_DB_LEASE_CONTAINER", "leases")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_threshold(name: str) -> int:
    """Parse an integer threshold from an environment variable.

    Returns 0 (disabled) if the variable is missing, empty, or not a valid
    integer, and logs a warning so misconfigurations are visible.
    """
    raw = os.environ.get(name, "0")
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for %s=%r, defaulting to 0 (disabled)", name, raw,
        )
        return 0


async def increment_counter_by(
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
    *,
    batch_max_lsn: int | None = None,
) -> tuple[int, int]:
    """Atomically increment a counter document by *count* using ETag concurrency.

    Returns ``(old_count, new_count)``.  Creates the counter document if it
    does not exist.  Retries up to 3 times on ETag conflicts (HTTP 412).

    If *batch_max_lsn* is provided, the counter stores it alongside the
    pre-increment count.  On a change-feed retry (same batch replayed),
    the function detects the duplicate via LSN comparison and returns the
    cached ``(pre_batch_count, current_count)`` **without** writing,
    preserving threshold-crossing semantics for the caller.

    .. note::

       LSN-based replay detection is perfect for **thread-scoped** counters
       (single logical partition → monotonic LSNs).  For **user-scoped**
       counters that aggregate across partitions, it handles the common
       single-partition-range retry but may not detect cross-partition
       interleaving.  Deterministic orchestration instance IDs provide
       an additional safety net against duplicate orchestration starts.
    """
    container = await _get_cosmos_counter_container()
    max_retries = 3
    partition_key = [user_id, thread_id]

    for attempt in range(max_retries):
        # ---- Read current counter (or default to 0) ----
        old_count = 0
        etag = None
        existing_doc = None
        try:
            doc = await container.read_item(item=counter_id, partition_key=partition_key)
            old_count = doc.get("count", 0)
            etag = doc.get("_etag")
            existing_doc = doc
        except CosmosResourceNotFoundError:
            pass  # first time — will create
        except CosmosHttpResponseError:
            raise

        # ---- Replay detection via LSN ----
        if (
            batch_max_lsn is not None
            and existing_doc is not None
            and existing_doc.get("last_batch_lsn") == batch_max_lsn
        ):
            replay_old = existing_doc.get("last_batch_old_count", old_count)
            logger.info(
                "Counter replay detected counter_id=%s lsn=%s, returning cached result",
                counter_id, batch_max_lsn,
            )
            return (replay_old, old_count)

        new_count = old_count + count
        new_doc = {
            "id": counter_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "count": new_count,
            "last_batch_lsn": batch_max_lsn,
            "last_batch_old_count": old_count,
            "created_at": existing_doc.get("created_at", _utc_now_iso()) if existing_doc else _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

        # ---- Upsert with ETag if we had an existing doc ----
        try:
            if etag is not None:
                await container.upsert_item(
                    body=new_doc,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            else:
                # First-time creation: use create_item to avoid last-writer-wins
                # race when multiple Function instances see 404 concurrently.
                try:
                    await container.create_item(body=new_doc)
                except CosmosHttpResponseError as create_exc:
                    if create_exc.status_code == 409 and attempt < max_retries - 1:
                        # Another instance created it first — retry with read-modify-write
                        logger.warning(
                            "Counter create conflict counter_id=%s attempt=%d/%d, retrying",
                            counter_id, attempt + 1, max_retries,
                        )
                        continue
                    if create_exc.status_code == 409:
                        logger.warning(
                            "Counter create conflict exhausted retries counter_id=%s, re-raising for batch retry",
                            counter_id,
                        )
                        raise
                    raise
            return (old_count, new_count)
        except CosmosHttpResponseError as exc:
            if exc.status_code == 412 and attempt < max_retries - 1:
                logger.warning(
                    "Counter ETag conflict counter_id=%s attempt=%d/%d, retrying",
                    counter_id, attempt + 1, max_retries,
                )
                continue
            if exc.status_code == 412:
                logger.warning(
                    "Counter ETag conflict exhausted retries counter_id=%s, skipping",
                    counter_id,
                )
                return (old_count, old_count)  # skip this increment
            raise

    return (old_count, old_count)  # should not be reached


def crosses_threshold(old_count: int, new_count: int, n: int) -> bool:
    """Return True if any multiple of *n* was crossed in the range (old, new].

    Raises:
        ValueError: If ``n`` is not a positive integer threshold.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    return old_count // n != new_count // n


# =====================================================================
# Orchestrator
# =====================================================================

@df_app.orchestration_trigger(context_name="context")
def memory_orchestrator(context: df.DurableOrchestrationContext):
    """Orchestrate a full memory-processing pipeline.

    Input payload::

        {
            "thread_id": "...",
            "user_id": "...",
            "content": "...",           # new content to embed & store
            "role": "user",             # optional, default "user"
            "memory_type": "turn",       # optional
            "metadata": {},             # optional
            "thread_summary": true,     # optional – trigger thread summary
            "thread_summary_only": false, # optional – skip embed/store
            "extract_facts": true,      # optional – trigger fact extraction
            "extract_facts_only": false,# optional – skip embed/store
            "user_summary": true,       # optional – trigger user summary
            "user_summary_only": false, # optional – skip embed/store
            "thread_ids": null,         # optional – limit user summary to
                                        #   specific threads
            "recent_k": null            # optional – limit memories to
                                        #   the most recent k for summary/facts
        }
    """
    payload = context.get_input()
    thread_summary_only = payload.get("thread_summary_only", False)
    extract_facts_only = payload.get("extract_facts_only", False)
    user_summary_only = payload.get("user_summary_only", False)

    if thread_summary_only:
        # --- Thread-summary-only path ---
        summary_doc = yield context.call_activity(
            "generate_thread_summary",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )
        return summary_doc

    if extract_facts_only:
        # --- Extract-facts-only path ---
        facts_doc = yield context.call_activity(
            "extract_facts",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )
        return facts_doc

    if user_summary_only:
        # --- User-summary-only path ---
        user_summary_doc = yield context.call_activity(
            "generate_user_summary",
            {
                "user_id": payload["user_id"],
                "thread_ids": payload.get("thread_ids"),
                "recent_k": payload.get("recent_k"),
            },
        )
        return user_summary_doc

    # --- Full pipeline path ---

    # 1. Load existing memories for the thread
    memories = yield context.call_activity(
        "load_memories",
        {"thread_id": payload.get("thread_id")},
    )

    # 2. Generate embeddings for the new content
    embedding = yield context.call_activity(
        "generate_embeddings",
        {"text": payload.get("content")},
    )

    # 3. Store the new memory (with its embedding) in Cosmos DB
    store_input = {
        "user_id": payload.get("user_id"),
        "thread_id": payload.get("thread_id"),
        "role": payload.get("role", "user"),
        "content": payload.get("content"),
        "memory_type": payload.get("memory_type", "turn"),
        "metadata": payload.get("metadata", {}),
        "embedding": embedding,
    }
    yield context.call_activity("store_results", store_input)

    # 4. Optionally summarize the thread
    summary = None
    if payload.get("thread_summary"):
        summary = yield context.call_activity(
            "generate_thread_summary",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )

    # 5. Optionally extract facts from the thread
    facts = None
    if payload.get("extract_facts"):
        facts_doc = yield context.call_activity(
            "extract_facts",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )
        facts = facts_doc

    # 6. Optionally generate a user summary
    user_summary = None
    if payload.get("user_summary"):
        user_summary = yield context.call_activity(
            "generate_user_summary",
            {
                "user_id": payload["user_id"],
                "thread_ids": payload.get("thread_ids"),
                "recent_k": payload.get("recent_k"),
            },
        )

    return {
        "thread_id": payload.get("thread_id"),
        "memories_loaded": len(memories) if memories else 0,
        "stored": True,
        "summary": summary,
        "facts": facts,
        "user_summary": user_summary,
    }


# =====================================================================
# HTTP starter
# =====================================================================

@df_app.route(route="orchestrators/{functionName}")
@df_app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client):
    """HTTP trigger that starts the durable orchestration."""
    function_name = req.route_params.get("functionName", "memory_orchestrator")
    payload = req.get_json()
    instance_id = await client.start_new(function_name, client_input=payload)

    return client.create_check_status_response(req, instance_id)


# =====================================================================
# Cosmos DB Change Feed Trigger
# =====================================================================


async def process_changefeed_batch(documents: list[dict], starter) -> None:
    """Core logic for processing a change feed batch.

    Filters to ``type == "turn"`` documents, groups them by scope, increments
    counters, and starts orchestrations when configurable thresholds are crossed.

    Extracted from the trigger function so it can be tested without the
    Durable Functions middleware.
    """
    n_thread = _parse_threshold("THREAD_SUMMARY_EVERY_N")
    n_facts = _parse_threshold("FACT_EXTRACTION_EVERY_N")
    n_user = _parse_threshold("USER_SUMMARY_EVERY_N")

    if n_thread == 0 and n_facts == 0 and n_user == 0:
        return  # all processing disabled

    # ---- Step 1: Filter to turns, group by scope ----
    thread_counts: dict[tuple[str, str], int] = defaultdict(int)
    user_counts: dict[str, int] = defaultdict(int)
    thread_max_lsn: dict[tuple[str, str], int] = {}
    user_max_lsn: dict[str, int] = {}

    for doc in documents:
        # Counter writes land in the separate counter container, so only raw
        # conversation turns from the memories container affect thresholds.
        if doc.get("type") != "turn":
            continue

        user_id = doc.get("user_id")
        thread_id = doc.get("thread_id")
        if not user_id or not thread_id:
            logger.warning("on_memory_change: turn doc missing user_id or thread_id, skipping")
            continue

        thread_counts[(user_id, thread_id)] += 1
        user_counts[user_id] += 1

        # Track max _lsn per scope for replay detection
        lsn = doc.get("_lsn")
        if lsn is not None:
            key = (user_id, thread_id)
            thread_max_lsn[key] = max(thread_max_lsn.get(key, 0), lsn)
            user_max_lsn[user_id] = max(user_max_lsn.get(user_id, 0), lsn)

    thread_counters_enabled = (n_thread > 0 or n_facts > 0)
    user_counters_enabled = (n_user > 0)
    enabled_thread_groups = len(thread_counts) if thread_counters_enabled else 0
    enabled_user_groups = len(user_counts) if user_counters_enabled else 0

    if enabled_thread_groups == 0 and enabled_user_groups == 0:
        return  # no turn documents in this batch for enabled counter processing

    logger.info(
        "on_memory_change: processing batch thread_groups=%d user_groups=%d",
        enabled_thread_groups, enabled_user_groups,
    )

    # ---- Step 2: Thread-scoped counters and threshold checks ----
    orchestration_errors: list[Exception] = []

    if thread_counters_enabled:
        for (user_id, thread_id), batch_count in thread_counts.items():
            lsn = thread_max_lsn.get((user_id, thread_id))
            old_count, new_count = await increment_counter_by(
                f"thread_counter_{user_id}_{thread_id}", user_id, thread_id, batch_count,
                batch_max_lsn=lsn,
            )

            if n_thread > 0 and crosses_threshold(old_count, new_count, n_thread):
                bucket = new_count // n_thread
                instance_id = f"ts_{user_id}_{thread_id}_{bucket}"
                logger.info(
                    "on_memory_change: triggering thread_summary user_id=%s thread_id=%s count=%d instance=%s",
                    user_id, thread_id, new_count, instance_id,
                )
                try:
                    await starter.start_new(
                        "memory_orchestrator",
                        instance_id=instance_id,
                        client_input={
                            "thread_summary_only": True,
                            "user_id": user_id,
                            "thread_id": thread_id,
                        },
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to start thread_summary orchestration user_id=%s thread_id=%s",
                        user_id, thread_id,
                    )
                    orchestration_errors.append(exc)

            if n_facts > 0 and crosses_threshold(old_count, new_count, n_facts):
                bucket = new_count // n_facts
                instance_id = f"ef_{user_id}_{thread_id}_{bucket}"
                logger.info(
                    "on_memory_change: triggering extract_facts user_id=%s thread_id=%s count=%d instance=%s",
                    user_id, thread_id, new_count, instance_id,
                )
                try:
                    await starter.start_new(
                        "memory_orchestrator",
                        instance_id=instance_id,
                        client_input={
                            "extract_facts_only": True,
                            "user_id": user_id,
                            "thread_id": thread_id,
                        },
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to start extract_facts orchestration user_id=%s thread_id=%s",
                        user_id, thread_id,
                    )
                    orchestration_errors.append(exc)

    # ---- Step 3: User-scoped counters and threshold checks ----
    if user_counters_enabled:
        for user_id, batch_count in user_counts.items():
            lsn = user_max_lsn.get(user_id)
            old_count, new_count = await increment_counter_by(
                f"user_counter_{user_id}", user_id, USER_COUNTER_THREAD_ID, batch_count,
                batch_max_lsn=lsn,
            )

            if crosses_threshold(old_count, new_count, n_user):
                bucket = new_count // n_user
                instance_id = f"us_{user_id}_{bucket}"
                logger.info(
                    "on_memory_change: triggering user_summary user_id=%s count=%d instance=%s",
                    user_id, new_count, instance_id,
                )
                try:
                    await starter.start_new(
                        "memory_orchestrator",
                        instance_id=instance_id,
                        client_input={
                            "user_summary_only": True,
                            "user_id": user_id,
                        },
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to start user_summary orchestration user_id=%s",
                        user_id,
                    )
                    orchestration_errors.append(exc)

    # Re-raise so the change feed batch retries and thresholds re-fire
    if orchestration_errors:
        raise RuntimeError(
            f"Failed to start {len(orchestration_errors)} orchestration(s); "
            "raising to retry the change feed batch"
        ) from orchestration_errors[0]


@df_app.cosmos_db_trigger(
    arg_name="documents",
    connection="COSMOS_DB",
    database_name=CHANGE_FEED_DATABASE,
    container_name=CHANGE_FEED_CONTAINER,
    lease_container_name=CHANGE_FEED_LEASE_CONTAINER,
    create_lease_container_if_not_exists=True,
)
@df_app.durable_client_input(client_name="starter")
async def on_memory_change(documents: func.DocumentList, starter) -> None:
    """Change feed trigger entry point — delegates to :func:`process_changefeed_batch`."""
    docs = [dict(doc) for doc in documents]
    await process_changefeed_batch(docs, starter)
