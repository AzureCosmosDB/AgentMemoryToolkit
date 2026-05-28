"""Shared counter helpers used by SDK clients to drive auto-trigger thresholds.

The Function App's change-feed processor (see ``function_app/shared/counters.py``)
uses the same counter container and document shape, so InProcess and Durable
backends can be swapped without losing per-thread / per-user counts.

Counter document shape::

    # thread-scoped — id = "thread:{user_id}:{thread_id}", PK = [user_id, thread_id]
    { "id": ..., "user_id": ..., "thread_id": ..., "count": int,
      "last_batch_lsn": int|None, "last_batch_old_count": int,
      "last_failure_at": str|None, "last_failure_reason": str|None,
      "last_owner": str|None }

    # user-scoped — id = "user:{user_id}", PK = [user_id, "__counters__"]
    { ... same fields ... }

Unlike the FA-side helper, the SDK clients drive these counters without LSN
replay protection because each ``push_to_cosmos()`` call is its own atomic
boundary — there is no change-feed redelivery to defend against. We
**preserve any existing ``last_batch_lsn``** the FA may have written so the
FA's monotonicity assumption stays valid even on shared deployments.

The ``last_owner`` field is informational only: it records which backend
last wrote the counter (``"inprocess"`` or ``"durable"``) so operators can
detect double-write configurations after the fact. It is not a lock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from agent_memory_toolkit.logging import get_logger

logger = get_logger(__name__)

USER_COUNTER_THREAD_ID = "__counters__"
MAX_RETRIES = 3

# Module-level dedup set so we only log "double-write detected" WARNs once
# per (counter_id, observer_owner) per process. On busy clients this would
# otherwise fire on every push.
_warned_owner_mismatch: set[tuple[str, str]] = set()


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


def _maybe_warn_owner_mismatch(counter_id: str, existing_owner: Optional[str], observer_owner: Optional[str]) -> None:
    """One-shot WARN when the previous writer disagrees with the current one.

    Operators run with either ``MEMORY_PROCESSOR_OWNER=inprocess`` or
    ``=durable``. If the same counter doc keeps getting touched by both
    backends, that's a misconfiguration — both pipelines are extracting
    against the same memories container.
    """
    if not existing_owner or not observer_owner or existing_owner == observer_owner:
        return
    key = (counter_id, observer_owner)
    if key in _warned_owner_mismatch:
        return
    _warned_owner_mismatch.add(key)
    logger.warning(
        "Counter doc %s was last written by owner=%r but this process is "
        "owner=%r — both backends appear to be processing the same container. "
        "Set MEMORY_PROCESSOR_OWNER consistently across all clients and the "
        "Function App to avoid double-extraction. Further mismatches for this "
        "counter will be logged at DEBUG level.",
        counter_id,
        existing_owner,
        observer_owner,
    )


def increment_counter_sync(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
    *,
    owner: Optional[str] = None,
) -> tuple[int, int]:
    """Atomically increment ``counter_id`` by *count* and return ``(old, new)``.

    Uses ETag-based optimistic concurrency, retrying up to ``MAX_RETRIES``
    times on HTTP 412. Uses ``create_item`` for the first-write path,
    retrying on HTTP 409 in case multiple SDK clients raced to seed the
    counter.

    Preserves any existing ``last_batch_lsn`` written by the FA-side
    increment helper so the FA's change-feed replay-dedup logic stays valid
    on shared deployments. Stamps ``last_owner`` (advisory) when *owner*
    is provided.

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

        if existing_doc is not None and owner is not None:
            _maybe_warn_owner_mismatch(counter_id, existing_doc.get("last_owner"), owner)

        new_count = old_count + count
        new_doc = _build_counter_doc(
            counter_id=counter_id,
            user_id=user_id,
            thread_id=thread_id,
            new_count=new_count,
            old_count=old_count,
            existing=existing_doc,
            owner=owner,
        )

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


async def increment_counter_async(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
    *,
    owner: Optional[str] = None,
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

        if existing_doc is not None and owner is not None:
            _maybe_warn_owner_mismatch(counter_id, existing_doc.get("last_owner"), owner)

        new_count = old_count + count
        new_doc = _build_counter_doc(
            counter_id=counter_id,
            user_id=user_id,
            thread_id=thread_id,
            new_count=new_count,
            old_count=old_count,
            existing=existing_doc,
            owner=owner,
        )

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


def _build_counter_doc(
    *,
    counter_id: str,
    user_id: str,
    thread_id: str,
    new_count: int,
    old_count: int,
    existing: Optional[dict],
    owner: Optional[str] = None,
) -> dict:
    """Construct the counter doc, preserving FA-managed fields when present."""
    now = _utc_now_iso()
    doc: dict[str, Any] = {
        "id": counter_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "count": new_count,
        "last_batch_old_count": old_count,
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
    }
    # Preserve FA-managed LSN replay-dedup fields if present; only initialize
    # to None when seeding the document for the first time. Mutating an
    # existing FA-written LSN to None would invalidate the FA's monotonicity
    # assumption on the next change-feed redelivery. ``last_batch_old_count``
    # is paired with ``last_batch_lsn`` for cached-result replay; if we
    # overwrote it with the SDK's local ``old_count``, a redelivered batch
    # would replay an inconsistent ``(old, new)`` and re-fire orchestrators.
    if existing is not None and "last_batch_lsn" in existing:
        doc["last_batch_lsn"] = existing.get("last_batch_lsn")
        doc["last_batch_old_count"] = existing.get("last_batch_old_count", old_count)
    else:
        doc["last_batch_lsn"] = None
    # Carry over auto-trigger failure breadcrumbs so they aren't blown away
    # by a successful write. ``stamp_failure_*`` helpers refresh them on
    # failure; operators can monitor ``last_failure_at`` directly.
    if existing is not None:
        if "last_failure_at" in existing:
            doc["last_failure_at"] = existing.get("last_failure_at")
        if "last_failure_reason" in existing:
            doc["last_failure_reason"] = existing.get("last_failure_reason")
    # Stamp the writing backend (advisory). When owner is None (legacy
    # caller), preserve whatever the previous writer recorded.
    if owner is not None:
        doc["last_owner"] = owner
    elif existing is not None and "last_owner" in existing:
        doc["last_owner"] = existing.get("last_owner")
    return doc


def stamp_failure_sync(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    reason: str,
) -> None:
    """Best-effort stamp ``last_failure_at`` / ``last_failure_reason`` on the counter doc.

    Uses Cosmos ``patch_item`` so only the failure fields are touched —
    concurrent counter increments cannot lose updates here. Failures are
    logged and swallowed; we never want failure-stamping itself to break
    the user's write path.
    """
    partition_key = [user_id, thread_id]
    patch_ops = [
        {"op": "add", "path": "/last_failure_at", "value": _utc_now_iso()},
        {"op": "add", "path": "/last_failure_reason", "value": (reason or "")[:500]},
    ]
    try:
        container.patch_item(
            item=counter_id,
            partition_key=partition_key,
            patch_operations=patch_ops,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("stamp_failure_sync failed counter_id=%s: %s", counter_id, exc)


async def stamp_failure_async(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    reason: str,
) -> None:
    """Async version of :func:`stamp_failure_sync`."""
    partition_key = [user_id, thread_id]
    patch_ops = [
        {"op": "add", "path": "/last_failure_at", "value": _utc_now_iso()},
        {"op": "add", "path": "/last_failure_reason", "value": (reason or "")[:500]},
    ]
    try:
        await container.patch_item(
            item=counter_id,
            partition_key=partition_key,
            patch_operations=patch_ops,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("stamp_failure_async failed counter_id=%s: %s", counter_id, exc)


__all__ = [
    "USER_COUNTER_THREAD_ID",
    "thread_counter_id",
    "user_counter_id",
    "crosses_threshold",
    "increment_counter_sync",
    "increment_counter_async",
    "stamp_failure_sync",
    "stamp_failure_async",
]
