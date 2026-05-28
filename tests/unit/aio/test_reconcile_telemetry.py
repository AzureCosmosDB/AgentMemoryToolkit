"""Async mirror of the ``reconcile.outcome`` telemetry tests.

Verifies that ``AsyncPipelineService.reconcile_memories`` emits the same
structured log line on both the successful and the empty-pool exit paths.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_toolkit.aio.services.pipeline import AsyncPipelineService

ASYNC_LOGGER_NAME = "agent_memory_toolkit.pipeline.aio"


def _make_async_pipeline() -> AsyncPipelineService:
    p = AsyncPipelineService.__new__(AsyncPipelineService)
    p._embeddings = MagicMock()
    p._embeddings.generate = AsyncMock(return_value=[0.1] * 8)
    p._upsert_memory = AsyncMock(side_effect=lambda doc: doc)
    p._mark_superseded = AsyncMock(return_value=True)
    p._container = MagicMock()
    p._container.query_items = AsyncMock(return_value=[])
    p._chat = MagicMock()
    return p


def _fact(fid: str, content: str, **extra) -> dict:
    return {
        "id": fid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "content": content,
        "confidence": extra.get("confidence", 0.8),
        "salience": extra.get("salience", 0.5),
        "created_at": extra.get("created_at", "2024-01-01T00:00:00+00:00"),
        "_etag": extra.get("etag", "etag-1"),
    }


def _outcome_records(caplog) -> list:
    return [r for r in caplog.records if r.name == ASYNC_LOGGER_NAME and r.getMessage() == "reconcile.outcome"]


@pytest.mark.asyncio
async def test_async_reconcile_emits_outcome_log_line_on_success(caplog):
    p = _make_async_pipeline()
    facts = [
        _fact("f1", "User likes aisle seats", confidence=0.9, salience=0.7),
        _fact("f2", "User prefers aisle seats on flights", confidence=0.85, salience=0.65),
    ]
    p._container.query_items = AsyncMock(return_value=facts)
    p._run_prompty = AsyncMock(
        return_value=json.dumps(
            {
                "duplicate_groups": [
                    {
                        "merged_content": "User prefers aisle seats on flights",
                        "source_ids": ["f1", "f2"],
                        "confidence": 0.9,
                        "salience": 0.7,
                    }
                ],
                "contradicted_pairs": [],
                "kept_ids": [],
            }
        )
    )

    with caplog.at_level(logging.INFO, logger=ASYNC_LOGGER_NAME):
        result = await p.reconcile_memories("u1")

    records = _outcome_records(caplog)
    assert len(records) == 1
    rec = records[0]
    assert rec.operation == "reconcile_memories"
    assert rec.user_id == "u1"
    assert isinstance(rec.kept, int)
    assert isinstance(rec.merged, int)
    assert isinstance(rec.contradicted, int)
    assert rec.kept == result["kept"]
    assert rec.merged == result["merged"]
    assert rec.contradicted == result["contradicted"]
    assert rec.candidates_considered == len(facts)
    assert isinstance(rec.duration_ms, float)
    assert rec.duration_ms > 0.0
    assert rec.prompt_id == "dedup.prompty"
    assert rec.prompt_version == "v1"


@pytest.mark.asyncio
async def test_async_reconcile_emits_outcome_log_line_on_zero_candidates(caplog):
    p = _make_async_pipeline()
    p._container.query_items = AsyncMock(return_value=[])
    p._run_prompty = AsyncMock()

    with caplog.at_level(logging.INFO, logger=ASYNC_LOGGER_NAME):
        result = await p.reconcile_memories("u1")

    p._run_prompty.assert_not_called()
    assert result == {"kept": 0, "merged": 0, "contradicted": 0}

    records = _outcome_records(caplog)
    assert len(records) == 1
    rec = records[0]
    assert rec.candidates_considered == 0
    assert rec.kept == 0
    assert rec.merged == 0
    assert rec.contradicted == 0
    assert rec.user_id == "u1"
    assert rec.operation == "reconcile_memories"
    assert isinstance(rec.duration_ms, float)
    assert rec.duration_ms > 0.0
    assert rec.prompt_id == "dedup.prompty"
    assert rec.prompt_version == "v1"
