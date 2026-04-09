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

  COSMOS_DB_CONNECTION__accountEndpoint
      The Cosmos DB account endpoint URL used by the change feed trigger
      binding (identity-based connection).

  COSMOS_DB_COUNTERS_CONTAINER
      Name of the Cosmos DB container for message counters (default: ``"counters"``).
      Must be in the same database with partition key ``/user_id``.

Processing threshold settings (set to ``"0"`` to disable):

  THREAD_SUMMARY_EVERY_N
      Trigger a thread summary every N turns per ``(user_id, thread_id)`` pair.

  FACT_EXTRACTION_EVERY_N
      Trigger fact extraction every N turns per ``(user_id, thread_id)`` pair.

  USER_SUMMARY_EVERY_N
      Trigger a user summary every N turns per ``user_id`` across all threads.

Required Cosmos DB containers:

  - ``memories`` – existing container for memory documents
  - ``counters`` – new container for message counters (partition key: ``/user_id``)
  - ``leases``   – auto-created by the trigger for change feed checkpointing
"""

import logging
import os
from collections import defaultdict

import azure.functions as func
import azure.durable_functions as df
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential

from activities import bp as activities_bp

logger = logging.getLogger(__name__)

df_app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)
df_app.register_functions(activities_bp)


# =====================================================================
# Shared helpers for change feed trigger
# =====================================================================

_counters_container = None
_credential = None


def _get_credential():
    """Return a shared ``DefaultAzureCredential``."""
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


def _get_counters_container():
    """Return the Cosmos DB counters container client, connecting on first call."""
    global _counters_container
    if _counters_container is None:
        endpoint = os.environ["COSMOS_DB_ENDPOINT"]
        database = os.environ["COSMOS_DB_DATABASE"]
        container_name = os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counters")
        logger.info(
            "Connecting to counters container endpoint=%s database=%s container=%s",
            f"...{endpoint[-8:]}", database, container_name,
        )
        client = CosmosClient(endpoint, credential=_get_credential())
        db = client.get_database_client(database)
        _counters_container = db.get_container_client(container_name)
    return _counters_container


def increment_counter_by(counter_id: str, user_id: str, count: int) -> tuple[int, int]:
    """Atomically increment a counter document by *count* using ETag concurrency.

    Returns ``(old_count, new_count)``.  Creates the counter document if it
    does not exist.  Retries up to 3 times on ETag conflicts (HTTP 412).
    """
    container = _get_counters_container()
    max_retries = 3

    for attempt in range(max_retries):
        # ---- Read current counter (or default to 0) ----
        old_count = 0
        etag = None
        try:
            doc = container.read_item(item=counter_id, partition_key=user_id)
            old_count = doc.get("count", 0)
            etag = doc.get("_etag")
        except CosmosResourceNotFoundError:
            pass  # first time — will create
        except CosmosHttpResponseError:
            raise

        new_count = old_count + count
        new_doc = {
            "id": counter_id,
            "user_id": user_id,
            "count": new_count,
        }

        # ---- Upsert with ETag if we had an existing doc ----
        try:
            if etag is not None:
                container.upsert_item(
                    body=new_doc,
                    etag=etag,
                    match_condition="IfMatch",
                )
            else:
                container.upsert_item(body=new_doc)
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
    """Return True if any multiple of *n* was crossed in the range (old, new]."""
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
    n_thread = int(os.environ.get("THREAD_SUMMARY_EVERY_N", "0"))
    n_facts = int(os.environ.get("FACT_EXTRACTION_EVERY_N", "0"))
    n_user = int(os.environ.get("USER_SUMMARY_EVERY_N", "0"))

    if n_thread == 0 and n_facts == 0 and n_user == 0:
        return  # all processing disabled

    # ---- Step 1: Filter to turns, group by scope ----
    thread_counts: dict[tuple[str, str], int] = defaultdict(int)
    user_counts: dict[str, int] = defaultdict(int)

    for doc in documents:
        if doc.get("type") != "turn":
            continue

        user_id = doc.get("user_id")
        thread_id = doc.get("thread_id")
        if not user_id or not thread_id:
            logger.warning("on_memory_change: turn doc missing user_id or thread_id, skipping")
            continue

        thread_counts[(user_id, thread_id)] += 1
        user_counts[user_id] += 1

    if not thread_counts and not user_counts:
        return  # no turn documents in this batch

    logger.info(
        "on_memory_change: processing batch thread_groups=%d user_groups=%d",
        len(thread_counts), len(user_counts),
    )

    # ---- Step 2: Thread-scoped counters and threshold checks ----
    for (user_id, thread_id), batch_count in thread_counts.items():
        old_count, new_count = increment_counter_by(
            f"thread_counter_{user_id}_{thread_id}", user_id, batch_count,
        )

        if n_thread > 0 and crosses_threshold(old_count, new_count, n_thread):
            logger.info(
                "on_memory_change: triggering thread_summary user_id=%s thread_id=%s count=%d",
                user_id, thread_id, new_count,
            )
            await starter.start_new(
                "memory_orchestrator",
                client_input={
                    "thread_summary_only": True,
                    "user_id": user_id,
                    "thread_id": thread_id,
                },
            )

        if n_facts > 0 and crosses_threshold(old_count, new_count, n_facts):
            logger.info(
                "on_memory_change: triggering extract_facts user_id=%s thread_id=%s count=%d",
                user_id, thread_id, new_count,
            )
            await starter.start_new(
                "memory_orchestrator",
                client_input={
                    "extract_facts_only": True,
                    "user_id": user_id,
                    "thread_id": thread_id,
                },
            )

    # ---- Step 3: User-scoped counters and threshold checks ----
    for user_id, batch_count in user_counts.items():
        old_count, new_count = increment_counter_by(
            f"user_counter_{user_id}", user_id, batch_count,
        )

        if n_user > 0 and crosses_threshold(old_count, new_count, n_user):
            logger.info(
                "on_memory_change: triggering user_summary user_id=%s count=%d",
                user_id, new_count,
            )
            await starter.start_new(
                "memory_orchestrator",
                client_input={
                    "user_summary_only": True,
                    "user_id": user_id,
                },
            )


@df_app.cosmos_db_trigger(
    arg_name="documents",
    connection="COSMOS_DB_CONNECTION",
    database_name="ai_memory",
    container_name="memories",
    lease_container_name="leases",
    create_lease_container_if_not_exists=True,
)
@df_app.durable_client_input(client_name="starter")
async def on_memory_change(documents: func.DocumentList, starter) -> None:
    """Change feed trigger entry point — delegates to :func:`process_changefeed_batch`."""
    docs = [dict(doc) for doc in documents]
    await process_changefeed_batch(docs, starter)
