"""Per-step threshold auto-trigger logic for the in-process SDK client.

The toolkit's slim ``CosmosMemoryClient`` schedules extract / summary /
reconcile / user-summary work after each ``push_to_cosmos`` by tracking
per-thread and per-user counters in a Cosmos container, then firing the
corresponding pipeline steps when a threshold is crossed.

This module owns that orchestration. Both the sync and async clients
import :func:`maybe_trigger_steps`; the async variant lives in
:mod:`agent_memory_toolkit.aio.auto_trigger`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_memory_toolkit import _counters
from agent_memory_toolkit import thresholds as default_thresholds
from agent_memory_toolkit.logging import get_logger
from agent_memory_toolkit.processors import InProcessProcessor

logger = get_logger(__name__)

_WARNED_OWNER_SKIP: set[int] = set()


def _threshold_value(source: Any, getter_name: str, mapping_key: str, default: Any = None) -> Any:
    """Read a threshold from ``source`` falling back to module defaults.

    ``source`` may be a module exposing ``get_*`` callables (the default
    :mod:`agent_memory_toolkit.thresholds` shape), a plain mapping keyed
    by env-style names (``FACT_EXTRACTION_EVERY_N``), or an object with
    matching attributes.
    """
    if source is None:
        source = default_thresholds

    getter = getattr(source, getter_name, None)
    if callable(getter):
        return getter()

    short_key = getter_name.removeprefix("get_")
    if isinstance(source, Mapping):
        for key in (mapping_key, short_key, short_key.upper()):
            if key in source:
                return source[key]
    else:
        for key in (mapping_key, short_key, short_key.upper()):
            if hasattr(source, key):
                return getattr(source, key)

    default_getter = getattr(default_thresholds, getter_name, None)
    if callable(default_getter):
        return default_getter()
    return default


def _threshold_int(source: Any, getter_name: str, mapping_key: str) -> int:
    return int(_threshold_value(source, getter_name, mapping_key))


def _should_trigger(processor: Any, thresholds: Any) -> bool:
    """Return True if the auto-trigger should run for this processor/owner."""
    if not isinstance(processor, InProcessProcessor):
        return False
    owner = _threshold_value(thresholds, "get_processor_owner", "MEMORY_PROCESSOR_OWNER")
    if owner == default_thresholds.PROCESSOR_OWNER_DURABLE:
        key = id(processor)
        if key not in _WARNED_OWNER_SKIP:
            _WARNED_OWNER_SKIP.add(key)
            logger.warning("MEMORY_PROCESSOR_OWNER=durable is set; SDK auto-trigger will not run.")
        return False
    return True


def maybe_trigger_steps(
    processor: Any,
    counter_container: Any,
    turn_counts: dict[tuple[str, str], int],
    *,
    thresholds: Any = default_thresholds,
) -> None:
    """Increment counters and fire per-step pipeline work when thresholds cross.

    Parameters
    ----------
    processor:
        The :class:`InProcessProcessor` that will run pipeline steps. The
        trigger is skipped for any other processor type (e.g. durable).
    counter_container:
        The Cosmos counter container. ``None`` disables the trigger.
    turn_counts:
        Map of ``(user_id, thread_id) -> turns_pushed_in_this_batch``.
    thresholds:
        Optional override for the thresholds module (used in tests).
    """
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
    user_batch_counts = _trigger_thread_steps(
        processor,
        counter_container,
        turn_counts,
        n_facts=n_facts,
        n_summary=n_summary,
        n_dedup_turns=n_dedup_turns,
        thresholds=thresholds,
    )
    _trigger_user_steps(processor, counter_container, user_batch_counts, n_user=n_user)


def _trigger_thread_steps(
    processor: InProcessProcessor,
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
            old_count, new_count = _counters.increment_counter_sync(
                counter_container,
                counter_id,
                user_id,
                thread_id,
                batch_count,
                owner=default_thresholds.PROCESSOR_OWNER_INPROCESS,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Counter increment failed for %s/%s: %s", user_id, thread_id, exc)
            continue
        _fire_thread_steps(
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


def _fire_thread_steps(
    processor: InProcessProcessor,
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
    for enabled, label, call in (
        (
            fire_extract,
            "process_extract_memories",
            lambda: processor.process_extract_memories(user_id=user_id, thread_id=thread_id),
        ),
        (fire_dedup, "process_reconcile", lambda: processor.process_reconcile(user_id=user_id)),
        (
            fire_procedural,
            "synthesize_procedural",
            lambda: processor.synthesize_procedural(user_id=user_id),
        ),
        (
            fire_summary,
            "process_thread_summary",
            lambda: processor.process_thread_summary(user_id=user_id, thread_id=thread_id),
        ),
    ):
        if not enabled:
            continue
        try:
            call()
        except Exception as exc:
            logger.warning("Auto-trigger %s failed for %s/%s: %s", label, user_id, thread_id, exc)
            _counters.stamp_failure_sync(counter_container, counter_id, user_id, thread_id, f"{label}: {exc!r}")


def _trigger_user_steps(
    processor: InProcessProcessor,
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
            old_count, new_count = _counters.increment_counter_sync(
                counter_container,
                counter_id,
                user_id,
                _counters.USER_COUNTER_THREAD_ID,
                batch_count,
                owner=default_thresholds.PROCESSOR_OWNER_INPROCESS,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("User counter increment failed for %s: %s", user_id, exc)
            continue
        if not _counters.crosses_threshold(old_count, new_count, n_user):
            continue
        try:
            processor.process_user_summary(user_id=user_id)
        except Exception as exc:
            logger.warning("Auto-trigger process_user_summary failed for %s: %s", user_id, exc)
            _counters.stamp_failure_sync(
                counter_container,
                counter_id,
                user_id,
                _counters.USER_COUNTER_THREAD_ID,
                f"process_user_summary: {exc!r}",
            )


__all__ = ["maybe_trigger_steps"]
