from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.services.pipeline import PipelineService

TENANT = "acme"


def _user_summary_doc() -> dict:
    return {
        "id": "user_summary_u1",
        "user_id": "u1",
        "thread_id": "__user_summary__",
        "role": "system",
        "type": "user_summary",
        "content": "User likes Python.",
        "salience": 1.0,
        "tags": ["sys:user-summary"],
        "prompt_id": "user_summary.prompty",
        "prompt_version": "v1",
        "metadata": {
            "structured_summary": {
                "key_facts": ["User likes Python."],
                "topics": ["Python", "data science"],
            },
            "source_thread_count": 1,
            "source_memory_count": 1,
            "thread_ids": ["t1"],
            "recent_k": None,
            "incremental_update": False,
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def test_persist_user_summary_emits_topic_tags_from_summary_topics():
    pipeline = PipelineService.__new__(PipelineService)
    pipeline._tenant_id = TENANT
    pipeline._embeddings = MagicMock()
    pipeline._embeddings.generate.return_value = [0.1, 0.2]
    pipeline._upsert_summary = MagicMock(side_effect=lambda doc: doc)

    result = pipeline.persist_user_summary("u1", _user_summary_doc())

    assert "sys:user-summary" in result["tags"]
    assert "topic:python" in result["tags"]
    assert "topic:data-science" in result["tags"]
    assert result["tenant_id"] == TENANT
    pipeline._upsert_summary.assert_called_once()


async def test_async_persist_user_summary_emits_topic_tags_from_summary_topics():
    pipeline = AsyncPipelineService.__new__(AsyncPipelineService)
    pipeline._tenant_id = TENANT
    pipeline._embeddings = MagicMock()
    pipeline._embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    pipeline._upsert_summary = AsyncMock(side_effect=lambda doc: doc)

    result = await pipeline.persist_user_summary("u1", _user_summary_doc())

    assert "sys:user-summary" in result["tags"]
    assert "topic:python" in result["tags"]
    assert "topic:data-science" in result["tags"]
    assert result["tenant_id"] == TENANT
    pipeline._upsert_summary.assert_awaited_once()


def test_persist_user_summary_refuses_a_summary_of_another_tenant():
    pipeline = PipelineService.__new__(PipelineService)
    pipeline._tenant_id = TENANT
    pipeline._upsert_summary = MagicMock()

    with pytest.raises(ValidationError, match="different tenant"):
        pipeline.persist_user_summary("u1", {**_user_summary_doc(), "tenant_id": "other"})

    pipeline._upsert_summary.assert_not_called()


async def test_async_persist_user_summary_refuses_a_summary_of_another_tenant():
    pipeline = AsyncPipelineService.__new__(AsyncPipelineService)
    pipeline._tenant_id = TENANT
    pipeline._upsert_summary = AsyncMock()

    with pytest.raises(ValidationError, match="different tenant"):
        await pipeline.persist_user_summary("u1", {**_user_summary_doc(), "tenant_id": "other"})

    pipeline._upsert_summary.assert_not_awaited()
