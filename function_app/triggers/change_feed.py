"""Cosmos DB change-feed trigger.

Filters to ``type == "turn"`` documents, increments per-thread and per-user
counters, then starts the matching orchestrator with a deterministic instance
ID whenever a configured threshold is crossed. Spec §8.3 / §8.4 / §8.5.

The trigger is registered as a :class:`df.Blueprint` and wired into the
top-level :class:`df.DFApp` from ``function_app.py``. The pure-logic entry
point :func:`process_changefeed_batch` accepts plain ``dict`` documents and a
mockable ``starter`` so it can be exercised by unit tests without the
Functions middleware.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import azure.durable_functions as df
import azure.functions as func
from shared import config
from shared.cosmos_clients import get_counter_container_async
from shared.counters import (
    crosses_threshold,
    increment_counter_by,
    thread_counter_id,
    user_counter_id,
)

logger = logging.getLogger(__name__)

bp = df.Blueprint()

# Module-level one-shot guard so the per-batch INFO log doesn't spam when
# the SDK owns processing (potentially many batches per minute). Function
# instances are short-lived, so worst case is one INFO per cold start.
_warned_owner_skip: bool = False


# ---------------------------------------------------------------------------
# Pure-logic entry point (exposed for unit tests)
# ---------------------------------------------------------------------------


async def process_changefeed_batch(
    documents: list[dict],
    starter: Any,
    *,
    counter_container: Any | None = None,
) -> None:
    """Process a change-feed batch.

    Args:
        documents: Raw change-feed docs (already converted to ``dict``).
        starter: A :class:`df.DurableOrchestrationClient`-shaped object with
            an awaitable ``start_new(orchestration_name, instance_id, client_input)``.
        counter_container: Optional counter container for testing. When omitted
            the cached async container client is used.
    """
    # Owner exclusivity: when MEMORY_PROCESSOR_OWNER=inprocess is set the SDK
    # client owns auto-trigger for this Cosmos container; the FA must stay
    # silent or both backends would double-extract / double-dedup against
    # the same writes. The default (unset) preserves today's behavior so
    # existing deployments don't need to change anything.
    from agent_memory_toolkit.thresholds import (
        PROCESSOR_OWNER_INPROCESS,
        get_processor_owner,
    )

    if get_processor_owner() == PROCESSOR_OWNER_INPROCESS:
        global _warned_owner_skip
        if not _warned_owner_skip:
            _warned_owner_skip = True
            logger.warning(
                "MEMORY_PROCESSOR_OWNER=inprocess; change-feed trigger skipping batch "
                "(SDK owns auto-trigger for this container). Further skipped batches "
                "will be logged at DEBUG level."
            )
        else:
            logger.debug("Change-feed batch skipped: MEMORY_PROCESSOR_OWNER=inprocess")
        return

    n_thread = config.get_thread_summary_every_n()
    n_facts = config.get_fact_extraction_every_n()
    n_user = config.get_user_summary_every_n()
    n_dedup = config.get_dedup_every_n()

    # Reconcile fires every (n_facts * n_dedup) turns, matching the SDK
    # auto-trigger contract. Disabled when either knob is 0.
    n_dedup_turns = n_facts * n_dedup if (n_facts > 0 and n_dedup > 0) else 0

    if n_thread == 0 and n_facts == 0 and n_user == 0:
        return  # all orchestrators disabled

    # ---- Step 1: Filter to turns + group by scope ----
    thread_counts: dict[tuple[str, str], int] = defaultdict(int)
    user_counts: dict[str, int] = defaultdict(int)
    thread_max_lsn: dict[tuple[str, str], int] = {}
    user_max_lsn: dict[str, int] = {}

    # Track which threads contributed to each user counter so the user-summary
    # orchestrator can scope its query to those threads (avoids a full
    # cross-partition scan on the user's whole memory set).
    user_thread_ids: dict[str, set[str]] = defaultdict(set)

    for doc in documents:
        if doc.get("type") != "turn":
            continue
        user_id = doc.get("user_id")
        thread_id = doc.get("thread_id")
        if not user_id or not thread_id:
            logger.warning("change-feed: turn doc missing user_id/thread_id, skipping")
            continue

        thread_counts[(user_id, thread_id)] += 1
        user_counts[user_id] += 1
        user_thread_ids[user_id].add(thread_id)

        lsn = doc.get("_lsn")
        if lsn is not None:
            try:
                lsn_int = int(lsn)
            except (TypeError, ValueError):
                continue
            tkey = (user_id, thread_id)
            thread_max_lsn[tkey] = max(thread_max_lsn.get(tkey, 0), lsn_int)
            user_max_lsn[user_id] = max(user_max_lsn.get(user_id, 0), lsn_int)

    thread_enabled = n_thread > 0 or n_facts > 0
    user_enabled = n_user > 0

    if not thread_counts and not user_counts:
        return
    if (not thread_enabled or not thread_counts) and (not user_enabled or not user_counts):
        return

    if counter_container is None:
        counter_container = await get_counter_container_async()

    orchestration_errors: list[Exception] = []

    # ---- Step 2: Thread-scoped counters ----
    if thread_enabled:
        for (user_id, thread_id), batch_count in thread_counts.items():
            cid = thread_counter_id(user_id, thread_id)
            lsn = thread_max_lsn.get((user_id, thread_id))
            old_count, new_count = await increment_counter_by(
                counter_container,
                cid,
                user_id,
                thread_id,
                batch_count,
                batch_max_lsn=lsn,
            )

            if n_thread > 0 and crosses_threshold(old_count, new_count, n_thread):
                instance_id = f"thread_summary:{user_id}:{thread_id}:{new_count}"
                await _safe_start(
                    starter,
                    "ThreadSummaryOrchestrator",
                    instance_id,
                    {"user_id": user_id, "thread_id": thread_id, "count": new_count},
                    orchestration_errors,
                )

            if n_facts > 0 and crosses_threshold(old_count, new_count, n_facts):
                instance_id = f"extract:{user_id}:{thread_id}:{new_count}"
                should_reconcile = bool(
                    n_dedup_turns > 0 and crosses_threshold(old_count, new_count, n_dedup_turns)
                )
                await _safe_start(
                    starter,
                    "ExtractMemoriesOrchestrator",
                    instance_id,
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "count": new_count,
                        "reconcile": should_reconcile,
                    },
                    orchestration_errors,
                )

    # ---- Step 3: User-scoped counters ----
    if user_enabled:
        for user_id, batch_count in user_counts.items():
            cid = user_counter_id(user_id)
            lsn = user_max_lsn.get(user_id)
            old_count, new_count = await increment_counter_by(
                counter_container,
                cid,
                user_id,
                config.USER_COUNTER_THREAD_ID,
                batch_count,
                batch_max_lsn=lsn,
            )
            if crosses_threshold(old_count, new_count, n_user):
                instance_id = f"user_summary:{user_id}:{new_count}"
                await _safe_start(
                    starter,
                    "UserSummaryOrchestrator",
                    instance_id,
                    {
                        "user_id": user_id,
                        "count": new_count,
                        "thread_ids": sorted(user_thread_ids.get(user_id, set())),
                    },
                    orchestration_errors,
                )

    # Re-raise so the change-feed batch is retried & thresholds re-fire.
    if orchestration_errors:
        raise RuntimeError(f"Failed to start {len(orchestration_errors)} orchestration(s)") from orchestration_errors[0]


async def _safe_start(
    starter: Any,
    orchestration_name: str,
    instance_id: str,
    client_input: dict,
    errors: list[Exception],
) -> None:
    logger.info(
        "change-feed: starting %s instance=%s",
        orchestration_name,
        instance_id,
    )
    try:
        await starter.start_new(
            orchestration_name,
            instance_id=instance_id,
            client_input=client_input,
        )
    except Exception as exc:
        # Duplicate-instance-id is the idempotency mechanism for change-feed
        # retries: it shows up as a 409 Conflict from the Durable client.
        # The Durable Python SDK currently raises bare ``Exception`` for
        # everything (no typed class for OrchestrationAlreadyExists), so
        # we keep two signals — status_code first, then the canonical
        # message string. We always log ``type(exc).__name__`` so any future
        # SDK switch to a typed exception shows up immediately in App
        # Insights instead of silently breaking string matching.
        exc_type = type(exc).__name__
        status_code = getattr(exc, "status_code", None)
        if status_code == 409:
            logger.info(
                "change-feed: instance %s already exists (409 Conflict, exc_type=%s), skipping",
                instance_id,
                exc_type,
            )
            return
        msg = str(exc).lower()
        if "already exists" in msg and "instance" in msg:
            logger.info(
                "change-feed: instance %s already exists (message match, exc_type=%s), skipping",
                instance_id,
                exc_type,
            )
            return
        logger.exception(
            "change-feed: failed to start %s (exc_type=%s)",
            instance_id,
            exc_type,
        )
        errors.append(exc)


# ---------------------------------------------------------------------------
# Trigger registration
# ---------------------------------------------------------------------------


@bp.cosmos_db_trigger(
    arg_name="documents",
    connection="COSMOS_DB",
    database_name=config.CHANGE_FEED_DATABASE,
    container_name=config.CHANGE_FEED_CONTAINER,
    lease_container_name=config.CHANGE_FEED_LEASE_CONTAINER,
    create_lease_container_if_not_exists=True,
)
@bp.durable_client_input(client_name="starter")
async def on_memory_change(documents: func.DocumentList, starter) -> None:
    """Change-feed trigger entry point — delegates to :func:`process_changefeed_batch`."""
    docs = [dict(doc) for doc in documents]
    await process_changefeed_batch(docs, starter)
