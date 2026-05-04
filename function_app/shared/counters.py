"""Per-scope counter helpers (ETag concurrency + LSN replay dedup).

Carried over and adapted from the original ``main`` branch
``azure_functions/function_app.py``. The container client is injected (rather
than fetched via a module-level singleton) so unit tests can mock it cleanly.

Counter document shape::

    # thread-scoped — id = "thread:{user_id}:{thread_id}", PK = [user_id, thread_id]
    { "id": ..., "user_id": ..., "thread_id": ..., "count": int,
      "last_batch_lsn": int|None, "last_batch_old_count": int }

    # user-scoped — id = "user:{user_id}", PK = [user_id, "__counters__"]
    { "id": ..., "user_id": ..., "thread_id": "__counters__",
      "count": int, "last_batch_lsn": int|None, "last_batch_old_count": int }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def thread_counter_id(user_id: str, thread_id: str) -> str:
    return f"thread:{user_id}:{thread_id}"


def user_counter_id(user_id: str) -> str:
    return f"user:{user_id}"


async def increment_counter_by(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
    *,
    batch_max_lsn: int | None = None,
) -> tuple[int, int]:
    """Atomically increment ``counter_id`` by *count* and return ``(old, new)``.

    * Uses ETag-based optimistic concurrency, retrying up to ``MAX_RETRIES``
      times on HTTP 412.
    * Uses ``create_item`` for the first-write path, retrying on HTTP 409 in
      case multiple Function workers raced to seed the counter.
    * If ``batch_max_lsn`` matches the LSN persisted on the existing doc, this
      is treated as a change-feed replay and we return the cached
      ``(pre_batch_count, current_count)`` **without writing** so the
      threshold-crossing semantics are preserved without double-counting.
    """
    partition_key = [user_id, thread_id]

    for attempt in range(MAX_RETRIES):
        # ---- Read current counter (or default to 0) ----
        old_count = 0
        etag: str | None = None
        existing_doc: dict | None = None
        try:
            existing_doc = await container.read_item(
                item=counter_id, partition_key=partition_key
            )
            old_count = existing_doc.get("count", 0)
            etag = existing_doc.get("_etag")
        except CosmosResourceNotFoundError:
            pass  # first time — will create

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
            "created_at": existing_doc.get("created_at", _utc_now_iso())
            if existing_doc
            else _utc_now_iso(),
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
                        logger.warning(
                            "Counter create conflict counter_id=%s attempt=%d/%d, retrying",
                            counter_id, attempt + 1, MAX_RETRIES,
                        )
                        continue
                    if create_exc.status_code == 409:
                        logger.warning(
                            "Counter create conflict exhausted retries counter_id=%s, re-raising",
                            counter_id,
                        )
                        raise
                    raise
            return (old_count, new_count)
        except CosmosHttpResponseError as exc:
            if exc.status_code == 412 and attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Counter ETag conflict counter_id=%s attempt=%d/%d, retrying",
                    counter_id, attempt + 1, MAX_RETRIES,
                )
                continue
            if exc.status_code == 412:
                # Exhausted ETag retries — RAISE so the change-feed batch retries
                # via at-least-once redelivery. The last_batch_lsn replay-protection
                # ensures the next attempt won't double-increment. Silently
                # returning (old, old) here would advance the lease without ever
                # firing the orchestrator that the increment was supposed to
                # trigger, causing permanent threshold-miss bugs.
                logger.warning(
                    "Counter ETag conflict exhausted retries counter_id=%s, "
                    "raising to force change-feed batch retry",
                    counter_id,
                )
                raise
            raise

    # Should never reach here — the loop either returns or raises every iteration.
    raise RuntimeError(
        f"increment_counter_by({counter_id}) exhausted MAX_RETRIES without resolution"
    )


def crosses_threshold(old_count: int, new_count: int, n: int) -> bool:
    """Return ``True`` if any multiple of *n* lies in the half-open range ``(old, new]``.

    Examples (with N=4)::

        crosses_threshold(0, 4, 4) is True   # crossed at 4
        crosses_threshold(3, 4, 4) is True   # crossed at 4
        crosses_threshold(3, 5, 4) is True   # crossed at 4
        crosses_threshold(4, 8, 4) is True   # crossed at 8
        crosses_threshold(6, 9, 4) is True   # crossed at 8
        crosses_threshold(0, 3, 4) is False
        crosses_threshold(4, 7, 4) is False
        crosses_threshold(5, 5, 4) is False  # no progress

    Raises:
        ValueError: if ``n <= 0``. Callers should gate on ``n > 0`` instead of
            relying on a "disabled" sentinel here — keeping this strict makes
            misuse loud.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    return old_count // n != new_count // n
