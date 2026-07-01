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

# One-shot mismatch-warn dedup. Mirrors the SDK-side pattern in
# ``azure/cosmos/agent_memory/_counters.py`` so the FA also surfaces double-write
# misconfigurations without spamming logs (key = (counter_id, my_owner)).
_warned_owner_mismatch: set[tuple[str, str]] = set()


def _maybe_warn_owner_mismatch(
    counter_id: str,
    existing_owner: str | None,
    my_owner: str | None,
) -> None:
    """Log a one-shot WARN when the counter's previous writer differs from us.

    Advisory-only — the FA still runs the orchestration. ``MEMORY_PROCESSOR_OWNER``
    is operator-configured exclusivity, not a server-side lock; this just
    surfaces accidental double-deployment so it shows up in App Insights.
    """
    if not my_owner or not existing_owner or existing_owner == my_owner:
        return
    key = (counter_id, my_owner)
    if key in _warned_owner_mismatch:
        return
    _warned_owner_mismatch.add(key)
    logger.warning(
        "Owner mismatch on counter %s: existing last_owner=%r, this process owner=%r. "
        "Both backends appear to be writing the same counter. Ensure MEMORY_PROCESSOR_OWNER "
        "is set consistently on both the SDK client and the function app, otherwise extracts "
        "and dedups will run twice. (One-shot WARN per counter+owner pair per process.)",
        counter_id,
        existing_owner,
        my_owner,
    )


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
    owner: str | None = "durable",
) -> tuple[int, int]:
    """Atomically increment ``counter_id`` by *count* and return ``(old, new)``.

    * Uses ETag-based optimistic concurrency, retrying up to ``MAX_RETRIES``
      times on HTTP 412.
    * Uses ``create_item`` for the first-write path, retrying on HTTP 409 in
      case multiple Function workers raced to seed the counter.
    * If ``batch_max_lsn`` is *less than or equal to* the LSN persisted on the
      existing doc, this is treated as a change-feed replay (immediate or
      after a lease re-balance / host crash where checkpoints regressed) and
      we return without writing. For the equal case we return the cached
      ``(pre_batch_count, current_count)`` so threshold-crossing semantics
      hold; for the strict-less case we return ``(current, current)`` (no
      crossing) because some other batch already advanced the counter past
      this one.
    * Preserves SDK-written failure breadcrumbs (``last_failure_at`` /
      ``last_failure_reason``) so monitors don't flap when the FA writes
      after an SDK failure stamp.
    * Stamps ``last_owner=owner`` (advisory) so operators can detect
      double-write configurations across SDK and FA.
    """
    partition_key = [user_id, thread_id]

    for attempt in range(MAX_RETRIES):
        # ---- Read current counter (or default to 0) ----
        old_count = 0
        etag: str | None = None
        existing_doc: dict | None = None
        try:
            existing_doc = await container.read_item(item=counter_id, partition_key=partition_key)
            old_count = existing_doc.get("count", 0)
            etag = existing_doc.get("_etag")
        except CosmosResourceNotFoundError:
            pass  # first time — will create

        # Owner-mismatch detection (advisory only).
        if existing_doc is not None:
            _maybe_warn_owner_mismatch(
                counter_id,
                existing_doc.get("last_owner"),
                owner,
            )

        # ---- Replay detection via LSN ----
        # Use ``>=`` not ``==`` so out-of-order redeliveries (lease
        # re-balance, host crash → checkpoint regression where another
        # batch landed in between) also short-circuit. For the exact
        # match we replay the cached result; for the strict-greater case
        # we return (current, current) — no threshold crossing — because
        # the batch's effect is already absorbed in a later state we have
        # no cached pre-batch value for.
        if (
            batch_max_lsn is not None
            and existing_doc is not None
            and existing_doc.get("last_batch_lsn") is not None
            and existing_doc["last_batch_lsn"] >= batch_max_lsn
        ):
            stored_lsn = existing_doc["last_batch_lsn"]
            if stored_lsn == batch_max_lsn:
                replay_old = existing_doc.get("last_batch_old_count", old_count)
                logger.info(
                    "Counter replay detected counter_id=%s lsn=%s, returning cached result",
                    counter_id,
                    batch_max_lsn,
                )
                return (replay_old, old_count)
            logger.info(
                "Counter out-of-order replay counter_id=%s redelivered_lsn=%s stored_lsn=%s; no-op",
                counter_id,
                batch_max_lsn,
                stored_lsn,
            )
            return (old_count, old_count)

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
        # Preserve SDK-written failure breadcrumbs so the FA doesn't blow
        # them away on the next successful increment. Operators alerting on
        # ``last_failure_at`` would otherwise see the field flap based on
        # which backend wrote last.
        if existing_doc is not None:
            if "last_failure_at" in existing_doc:
                new_doc["last_failure_at"] = existing_doc.get("last_failure_at")
            if "last_failure_reason" in existing_doc:
                new_doc["last_failure_reason"] = existing_doc.get("last_failure_reason")
            if "last_extract_count" in existing_doc:
                new_doc["last_extract_count"] = existing_doc.get("last_extract_count")
        # Stamp the writing backend (advisory only — not enforced server-side).
        if owner is not None:
            new_doc["last_owner"] = owner
        elif existing_doc is not None and "last_owner" in existing_doc:
            new_doc["last_owner"] = existing_doc.get("last_owner")

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
                            counter_id,
                            attempt + 1,
                            MAX_RETRIES,
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
                    counter_id,
                    attempt + 1,
                    MAX_RETRIES,
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
                    "Counter ETag conflict exhausted retries counter_id=%s, raising to force change-feed batch retry",
                    counter_id,
                )
                raise
            raise

    # Should never reach here — the loop either returns or raises every iteration.
    raise RuntimeError(f"increment_counter_by({counter_id}) exhausted MAX_RETRIES without resolution")


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


async def read_extract_watermark(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
) -> int | None:
    """Return the count value at the last successful extract, or ``None``.

    Lets recent_k cover every turn since the previous extract succeeded so
    turns are never skipped when extraction lags or transiently fails.
    Best-effort: returns ``None`` on any read error so callers fall back to a
    batch-based recent_k.
    """
    try:
        doc = await container.read_item(item=counter_id, partition_key=[user_id, thread_id])
        value = doc.get("last_extract_count")
        return int(value) if value is not None else None
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("read_extract_watermark failed counter_id=%s: %s", counter_id, exc)
        return None


async def advance_extract_watermark(
    container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    count: int,
) -> None:
    """Stamp ``last_extract_count=count`` after a successful extract."""
    patch_ops = [{"op": "add", "path": "/last_extract_count", "value": int(count)}]
    try:
        await container.patch_item(item=counter_id, partition_key=[user_id, thread_id], patch_operations=patch_ops)
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("advance_extract_watermark failed counter_id=%s: %s", counter_id, exc)
