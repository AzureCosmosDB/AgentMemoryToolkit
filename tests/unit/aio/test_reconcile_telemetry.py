"""Async mirror of the ``reconcile.outcome`` telemetry tests.

Verifies that ``AsyncPipelineService.reconcile_memories`` emits the same structured log line as the sync pipeline:
the pass's counts, its timing and the prompt lineage, whether or not the pass checked anything.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.services._dedup import dedup_result

ASYNC_LOGGER_NAME = "azure.cosmos.agent_memory.pipeline.aio"
NO_DEDUP = dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)


def _make_async_pipeline() -> AsyncPipelineService:
    p = AsyncPipelineService.__new__(AsyncPipelineService)
    p.consolidate_memories = AsyncMock(return_value=dict(NO_DEDUP))
    p._container = MagicMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    return p


def _outcome_records(caplog) -> list:
    return [r for r in caplog.records if r.name == ASYNC_LOGGER_NAME and r.getMessage() == "reconcile.outcome"]


@pytest.mark.asyncio
async def test_async_reconcile_emits_outcome_log_line_with_every_count_the_pass_returned(caplog):
    p = _make_async_pipeline()
    counts = dedup_result(
        checked=4,
        retired_seeds=1,
        merged=1,
        compared=3,
        vetoed=2,
        conflicts=2,
        combined=2,
        synthesized=1,
        combine_groups=2,
        combine_failed=1,
    )
    p.consolidate_memories.return_value = dict(counts)

    with caplog.at_level(logging.INFO, logger=ASYNC_LOGGER_NAME):
        result = await p.reconcile_memories("u1")

    records = _outcome_records(caplog)
    assert len(records) == 1
    rec = records[0]
    assert rec.operation == "reconcile_memories"
    assert rec.user_id == "u1"
    assert (rec.facts_checked, rec.candidates_considered, rec.pairs_compared) == (4, 4, 3)
    for key in ("kept", "merged", "conflicts", "vetoed", "combined", "synthesized", "combine_groups", "combine_failed"):
        assert getattr(rec, key) == result[key], key
    assert not hasattr(rec, "contradicted")
    assert isinstance(rec.duration_ms, float)
    assert rec.duration_ms >= 0.0
    assert rec.prompt_id == "dedup_verify.prompty"
    assert rec.prompt_version == "v1"


@pytest.mark.asyncio
async def test_async_reconcile_emits_outcome_log_line_when_nothing_was_checked(caplog):
    p = _make_async_pipeline()

    with caplog.at_level(logging.INFO, logger=ASYNC_LOGGER_NAME):
        result = await p.reconcile_memories("u1")

    assert result == NO_DEDUP
    records = _outcome_records(caplog)
    assert len(records) == 1
    rec = records[0]
    assert (rec.candidates_considered, rec.facts_checked, rec.kept, rec.merged, rec.conflicts) == (0, 0, 0, 0, 0)
    assert rec.user_id == "u1"
    assert rec.operation == "reconcile_memories"
    assert isinstance(rec.duration_ms, float)
    assert rec.prompt_id == "dedup_verify.prompty"
