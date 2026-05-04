"""Memory-extraction orchestrator + activities.

Chain: ``ExtractMemories`` → ``DeduplicateFacts`` → ``PersistMemories``.

A salience pre-filter (``SALIENCE_THRESHOLD``) is applied in
``PersistMemories`` when the threshold is > 0 (spec §11.2).
"""

from __future__ import annotations

import logging

import azure.durable_functions as df

from shared import config
from shared.pipeline_factory import get_pipeline
from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@bp.orchestration_trigger(context_name="context")
def ExtractMemoriesOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    max_batch = config.get_max_batch_size()

    retry = default_retry_options()

    extracted = yield context.call_activity_with_retry(
        "em_ExtractMemories", retry,
        {"user_id": user_id, "thread_id": thread_id, "limit": max_batch},
    )

    dedup = yield context.call_activity_with_retry(
        "em_DeduplicateFacts", retry,
        {"user_id": user_id},
    )

    persisted = yield context.call_activity_with_retry(
        "em_PersistMemories", retry,
        {"user_id": user_id, "thread_id": thread_id, "extracted": extracted, "dedup": dedup},
    )

    return {
        "persisted": True,
        "extracted": extracted,
        "dedup": dedup,
        "kept": persisted.get("kept") if isinstance(persisted, dict) else None,
        "filtered": persisted.get("filtered") if isinstance(persisted, dict) else None,
    }


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@bp.activity_trigger(input_name="payload")
def em_ExtractMemories(payload: dict) -> dict:
    """Run the LLM extraction step.

    Returns the per-type counts produced by ``pipeline.extract_memories``
    (``{"facts": N, "procedural": N, "episodic": N}``-shaped). Salience-based
    filtering is delegated to the pipeline since it owns the schema.

    The pipeline loads recent turns internally, so we do NOT pre-load them in
    a separate activity (which would duplicate the query, waste RUs, and open
    a TOCTOU window between the load and the LLM call).
    """
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    limit = payload.get("limit")
    pipeline = get_pipeline()
    counts = pipeline.extract_memories(
        user_id=user_id, thread_id=thread_id, recent_k=limit,
    )
    logger.info(
        "ExtractMemories user=%s thread=%s counts=%s", user_id, thread_id, counts,
    )
    return counts or {}


@bp.activity_trigger(input_name="payload")
def em_DeduplicateFacts(payload: dict) -> dict:
    user_id = payload["user_id"]
    pipeline = get_pipeline()
    return pipeline.deduplicate_facts(user_id=user_id) or {}


@bp.activity_trigger(input_name="payload")
def em_PersistMemories(payload: dict) -> dict:
    """Apply the configured salience pre-filter and report persisted counts.

    ``ProcessingPipeline.extract_memories`` has already written the memories
    to Cosmos DB. When ``SALIENCE_THRESHOLD > 0`` we sweep the just-extracted
    memories for the thread and tombstone any whose salience is below the
    threshold. Doing this here (rather than inside the pipeline) keeps the
    library's default behaviour conservative — operators opt into the filter
    via configuration.
    """
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    extracted = payload.get("extracted") or {}
    threshold = config.get_salience_threshold()

    if threshold <= 0:
        return {"kept": extracted, "filtered": 0, "threshold": threshold}

    pipeline = get_pipeline()
    container = pipeline._container
    query = (
        "SELECT * FROM c "
        "WHERE c.user_id = @user_id "
        "AND c.thread_id = @thread_id "
        "AND c.type IN ('fact', 'procedural', 'episodic') "
        "AND IS_DEFINED(c.salience) AND c.salience < @threshold"
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@thread_id", "value": thread_id},
        {"name": "@threshold", "value": threshold},
    ]
    low_salience = list(container.query_items(query=query, parameters=parameters))
    filtered = 0
    for doc in low_salience:
        try:
            container.delete_item(
                item=doc["id"], partition_key=[user_id, thread_id],
            )
            filtered += 1
        except Exception:
            logger.exception("Failed to drop low-salience memory id=%s", doc.get("id"))

    return {"kept": extracted, "filtered": filtered, "threshold": threshold}
