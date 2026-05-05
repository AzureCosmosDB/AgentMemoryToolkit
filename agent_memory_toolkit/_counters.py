"""Shared counter helpers used by SDK clients to drive auto-trigger thresholds.

The Function App's change-feed processor (see ``function_app/shared/counters.py``)
uses the same counter container and document shape, so InProcess and Durable
backends can be swapped without losing per-thread / per-user counts.

Counter document shape::

    # thread-scoped — id = "thread:{user_id}:{thread_id}", PK = [user_id, thread_id]
    { "id": ..., "user_id": ..., "thread_id": ..., "count": int,
      "last_batch_lsn": int|None, "last_batch_old_count": int }

    # user-scoped — id = "user:{user_id}", PK = [user_id, "__counters__"]
    { "id": ..., "user_id": ..., "thread_id": "__counters__",
      "count": int, "last_batch_lsn": int|None, "last_batch_old_count": int }

Unlike the FA-side helper, the SDK clients drive these counters without LSN
replay protection because each ``push_to_cosmos()`` call is its own atomic
boundary — there is no change-feed redelivery to defend against. We still
pass ``batch_max_lsn=None`` so the cached doc shape stays compatible with
the FA-side dedup logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

logger = logging.getLogger(__name__)

USER_COUNTER_THREAD_ID = "__counters__"
MAX_RETRIES = 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def thread_counter_id(user_id: str, thread_id: str) -> str:
    return f"thread:{user_id}:{thread_id}"


def user_counter_id(user_id: str) -> str:
    return f"user:{user_id}"


def crosses_threshold(old_count: int, new_count: int, n: int) -> bool:
    """Return ``True`` if any multiple of *n* lies in the half-open range ``(old, new]``.

    Mirrors :func:`function_app.shared.counters.crosses_threshold` exactly so the
    InProcess and Durable backends fire on the same turn boundaries.

    Raises:
        ValueError: if ``n <= 0``. Callers should gate on ``n > 0`` instead of
            relying on a "disabled" sentinel here.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    return old_count // n != new_count // n


# ---------------------------------------------------------------------------
# Sync increment
# ---------------------------------------------------------------------------


def increment_counter_sync(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
) -> tuple[int, int]:
    """Atomically increment ``counter_id`` by *count* and return ``(old, new)``.

    Uses ETag-based optimistic concurrency, retrying up to ``MAX_RETRIES``
    times on HTTP 412. Uses ``create_item`` for the first-write path,
    retrying on HTTP 409 in case multiple SDK clients raced to seed the
    counter.

    Returns ``(0, 0)`` and logs a warning if the container is unreachable —
    auto-trigger failures must never block the user's primary write path.
    """
    partition_key = [user_id, thread_id]

    for attempt in range(MAX_RETRIES):
        old_count = 0
        etag: str | None = None
        existing_doc: dict | None = None
        try:
            existing_doc = container.read_item(item=counter_id, partition_key=partition_key)
            old_count = existing_doc.get("count", 0)
            etag = existing_doc.get("_etag")
        except CosmosResourceNotFoundError:
            pass

        new_count = old_count + count
        new_doc = {
            "id": counter_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "count": new_count,
            "last_batch_lsn": None,
            "last_batch_old_count": old_count,
            "created_at": existing_doc.get("created_at", _utc_now_iso()) if existing_doc else _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

        try:
            if etag is not None:
                container.upsert_item(
                    body=new_doc,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            else:
                try:
                    container.create_item(body=new_doc)
                except CosmosHttpResponseError as create_exc:
                    if create_exc.status_code == 409 and attempt < MAX_RETRIES - 1:
                        continue
                    raise
            return (old_count, new_count)
        except CosmosHttpResponseError as exc:
            if exc.status_code == 412 and attempt < MAX_RETRIES - 1:
                continue
            logger.warning(
                "Counter increment failed counter_id=%s status=%s — auto-trigger skipped",
                counter_id,
                exc.status_code,
            )
            return (0, 0)

    return (0, 0)


# ---------------------------------------------------------------------------
# Async increment
# ---------------------------------------------------------------------------


async def increment_counter_async(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
) -> tuple[int, int]:
    """Async version of :func:`increment_counter_sync`."""
    partition_key = [user_id, thread_id]

    for attempt in range(MAX_RETRIES):
        old_count = 0
        etag: str | None = None
        existing_doc: dict | None = None
        try:
            existing_doc = await container.read_item(item=counter_id, partition_key=partition_key)
            old_count = existing_doc.get("count", 0)
            etag = existing_doc.get("_etag")
        except CosmosResourceNotFoundError:
            pass

        new_count = old_count + count
        new_doc = {
            "id": counter_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "count": new_count,
            "last_batch_lsn": None,
            "last_batch_old_count": old_count,
            "created_at": existing_doc.get("created_at", _utc_now_iso()) if existing_doc else _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }

        try:
            if etag is not None:
                await container.upsert_item(
                    body=new_doc,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            else:
                try:
                    await container.create_item(body=new_doc)
                except CosmosHttpResponseError as create_exc:
                    if create_exc.status_code == 409 and attempt < MAX_RETRIES - 1:
                        continue
                    raise
            return (old_count, new_count)
        except CosmosHttpResponseError as exc:
            if exc.status_code == 412 and attempt < MAX_RETRIES - 1:
                continue
            logger.warning(
                "Counter increment failed counter_id=%s status=%s — auto-trigger skipped",
                counter_id,
                exc.status_code,
            )
            return (0, 0)

    return (0, 0)


__all__ = [
    "USER_COUNTER_THREAD_ID",
    "thread_counter_id",
    "user_counter_id",
    "crosses_threshold",
    "increment_counter_sync",
    "increment_counter_async",
]
