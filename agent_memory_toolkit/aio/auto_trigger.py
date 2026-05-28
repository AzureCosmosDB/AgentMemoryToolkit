"""Async per-step threshold auto-trigger.

Mirrors :mod:`agent_memory_toolkit.auto_trigger` for the async client.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from agent_memory_toolkit import _counters
from agent_memory_toolkit import thresholds as default_thresholds
from agent_memory_toolkit.aio.processors import AsyncInProcessProcessor
from agent_memory_toolkit.auto_trigger import _threshold_int, _threshold_value
from agent_memory_toolkit.logging import get_logger

logger = get_logger(__name__)

_WARNED_OWNER_SKIP: set[int] = set()


async def _call_async_compatible(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _should_trigger(processor: Any, thresholds: Any) -> bool:
    if not isinstance(processor, AsyncInProcessProcessor):
        return False
    owner = _threshold_value(thresholds, "get_processor_owner", "MEMORY_PROCESSOR_OWNER")
    if owner == default_thresholds.PROCESSOR_OWNER_DURABLE:
        key = id(processor)
        if key not in _WARNED_OWNER_SKIP:
            _WARNED_OWNER_SKIP.add(key)
            logger.warning("MEMORY_PROCESSOR_OWNER=durable is set; async SDK auto-trigger will not run.")
        return False
    return True


async def maybe_trigger_steps(
    processor: Any,
    counter_container: Any,
    turn_counts: dict[tuple[str, str], int],
    *,
    thresholds: Any = default_thresholds,
) -> None:
    """Async counterpart of :func:`agent_memory_toolkit.auto_trigger.maybe_trigger_steps`."""
    if counter_container is None or not turn_counts:
        return
    if not _should_trigger(processor, thresholds):
        return

    n_facts = _threshold_int(thresholds, "get_fact_extraction_every_n", "FACT_EXTRACTION_EVERY_N")
    n_summary = _threshold_int(thresholds, "get_thread_summary_every_n", "THREAD_SUMMARY_EVERY_N")
    n_user = _threshold_int(thresholds, "get_user_summary_every_n", "USER_SUMMARY_EVERY_N")
    n_dedup = _threshold_int(thresholds, "get_dedup_every_n", "DEDUP_EVERY_N")
    if n_facts == 0 and n_summary == 0 and n_user == 0:
        return

    n_dedup_turns = n_facts * n_dedup if n_facts > 0 and n_dedup > 0 else 0
    user_batch_counts = await _trigger_thread_steps(
        processor,
        counter_container,
        turn_counts,
        n_facts=n_facts,
        n_summary=n_summary,
        n_dedup_turns=n_dedup_turns,
        thresholds=thresholds,
    )
    await _trigger_user_steps(processor, counter_container, user_batch_counts, n_user=n_user)


async def _trigger_thread_steps(
    processor: AsyncInProcessProcessor,
    counter_container: Any,
    turn_counts: dict[tuple[str, str], int],
    *,
    n_facts: int,
    n_summary: int,
    n_dedup_turns: int,
    thresholds: Any = None,
) -> dict[str, int]:
    user_batch_counts: dict[str, int] = {}
    for (user_id, thread_id), batch_count in turn_counts.items():
        if batch_count <= 0:
            continue
        user_batch_counts[user_id] = user_batch_counts.get(user_id, 0) + batch_count
        counter_id = _counters.thread_counter_id(user_id, thread_id)
        try:
            old_count, new_count = await _counters.increment_counter_async(
                counter_container,
                counter_id,
                user_id,
                thread_id,
                batch_count,
                owner=default_thresholds.PROCESSOR_OWNER_INPROCESS,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Async counter increment failed for %s/%s: %s", user_id, thread_id, exc)
            continue
        await _fire_thread_steps(
            processor,
            counter_container,
            counter_id,
            user_id,
            thread_id,
            fire_extract=n_facts > 0 and _counters.crosses_threshold(old_count, new_count, n_facts),
            fire_summary=n_summary > 0 and _counters.crosses_threshold(old_count, new_count, n_summary),
            fire_dedup=n_dedup_turns > 0 and _counters.crosses_threshold(old_count, new_count, n_dedup_turns),
            thresholds=thresholds,
        )
    return user_batch_counts


async def _fire_thread_steps(
    processor: AsyncInProcessProcessor,
    counter_container: Any,
    counter_id: str,
    user_id: str,
    thread_id: str,
    *,
    fire_extract: bool,
    fire_summary: bool,
    fire_dedup: bool,
    thresholds: Any = None,
) -> None:
    fire_procedural = fire_dedup and bool(
        _threshold_value(
            thresholds,
            "get_procedural_synthesis_auto",
            "PROCEDURAL_SYNTHESIS_AUTO",
            default=True,
        )
    )
    calls = (
        (
            fire_extract,
            "process_extract_memories",
            processor.process_extract_memories,
            {"user_id": user_id, "thread_id": thread_id},
        ),
        (fire_dedup, "process_reconcile", processor.process_reconcile, {"user_id": user_id}),
        (
            fire_procedural,
            "synthesize_procedural",
            processor.synthesize_procedural,
            {"user_id": user_id},
        ),
        (
            fire_summary,
            "process_thread_summary",
            processor.process_thread_summary,
            {"user_id": user_id, "thread_id": thread_id},
        ),
    )
    for enabled, label, method, kwargs in calls:
        if not enabled:
            continue
        try:
            await _call_async_compatible(method, **kwargs)
        except Exception as exc:
            logger.warning("Async auto-trigger %s failed for %s/%s: %s", label, user_id, thread_id, exc)
            await _counters.stamp_failure_async(counter_container, counter_id, user_id, thread_id, f"{label}: {exc!r}")


async def _trigger_user_steps(
    processor: AsyncInProcessProcessor,
    counter_container: Any,
    user_batch_counts: dict[str, int],
    *,
    n_user: int,
) -> None:
    if n_user <= 0:
        return
    for user_id, batch_count in user_batch_counts.items():
        counter_id = _counters.user_counter_id(user_id)
        try:
            old_count, new_count = await _counters.increment_counter_async(
                counter_container,
                counter_id,
                user_id,
                _counters.USER_COUNTER_THREAD_ID,
                batch_count,
                owner=default_thresholds.PROCESSOR_OWNER_INPROCESS,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Async user counter increment failed for %s: %s", user_id, exc)
            continue
        if not _counters.crosses_threshold(old_count, new_count, n_user):
            continue
        try:
            await _call_async_compatible(processor.process_user_summary, user_id=user_id)
        except Exception as exc:
            logger.warning("Async auto-trigger process_user_summary failed for %s: %s", user_id, exc)
            await _counters.stamp_failure_async(
                counter_container,
                counter_id,
                user_id,
                _counters.USER_COUNTER_THREAD_ID,
                f"process_user_summary: {exc!r}",
            )


__all__ = ["maybe_trigger_steps"]
