"""No literal credential reaches the embedding provider, and a vector matches the text stored beside it.

Every write path redacts at write time, but several paths embedded first: the provider received the
raw credential, and the write-time guard then dropped the vector, leaving the record unsearchable.
"""

from __future__ import annotations

from typing import Any

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from tests.unit.services.test_extract_dry import (
    _async_containers_for_store,
    _AsyncChat,
    _AsyncEmbeddings,
    _AsyncStore,
    _containers_for_store,
    _Store,
    _SyncChat,
    _SyncEmbeddings,
)

# A GitHub token, a format the write guard redacts; built here so the source holds no token literal.
TOKEN = "ghp_" + "A1b2" * 9
SECRET = f"token {TOKEN}"


def _embedded(embeddings: _SyncEmbeddings) -> list[str]:
    return [text for call in embeddings.calls for text in call]


def _sync_service():
    memories, summaries, embeddings = _Store([]), _Store([]), _SyncEmbeddings()
    service = PipelineService(
        memories, _SyncChat([]), embeddings, containers=_containers_for_store(memories, summaries_store=summaries)
    )
    return service, summaries, embeddings


def _async_service():
    memories, summaries, embeddings = _AsyncStore([]), _AsyncStore([]), _AsyncEmbeddings()
    service = AsyncPipelineService(
        memories,
        _AsyncChat([]),
        embeddings,
        containers=_async_containers_for_store(memories, summaries_store=summaries),
    )
    return service, summaries, embeddings


def _summary(doc_type: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "structured_summary": {"overview": f"Deployed staging with {SECRET}.", "topics": ["deploy"]}
    }
    if doc_type == "user_summary":
        metadata["thread_ids"] = ["t1"]
    return {
        "user_id": "u1",
        "role": "system",
        "type": doc_type,
        "content": f"Deployed staging with {SECRET}.",
        "metadata": metadata,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }


def test_embedding_inputs_never_carry_a_literal_secret():
    service, _, embeddings = _sync_service()

    service._embed_one(f"deploy with {SECRET}")
    service._embed_batch([f"login {SECRET}", "clean text"])

    assert not any(TOKEN in text for text in _embedded(embeddings))
    assert "clean text" in _embedded(embeddings)


async def test_async_embedding_inputs_never_carry_a_literal_secret():
    service, _, embeddings = _async_service()

    await service._embed_one(f"deploy with {SECRET}")
    await service._embed_batch([f"login {SECRET}", "clean text"])

    assert not any(TOKEN in text for text in _embedded(embeddings))
    assert "clean text" in _embedded(embeddings)


def test_a_thread_summary_is_redacted_before_it_is_embedded():
    service, summaries, embeddings = _sync_service()

    service.persist_thread_summary("u1", "t1", _summary("thread_summary"))

    assert _embedded(embeddings) and not any(TOKEN in text for text in _embedded(embeddings))
    stored = summaries.docs[0]
    assert TOKEN not in repr(stored), "structured_summary carries the same text as content"
    assert stored["embedding"] == [1.0]


async def test_async_a_thread_summary_is_redacted_before_it_is_embedded():
    service, summaries, embeddings = _async_service()

    await service.persist_thread_summary("u1", "t1", _summary("thread_summary"))

    assert _embedded(embeddings) and not any(TOKEN in text for text in _embedded(embeddings))
    stored = summaries.docs[0]
    assert TOKEN not in repr(stored)
    assert stored["embedding"] == [1.0]


def test_a_user_summary_is_redacted_before_it_is_embedded():
    service, summaries, embeddings = _sync_service()

    service.persist_user_summary("u1", _summary("user_summary"))

    assert not any(TOKEN in text for text in _embedded(embeddings))
    stored = summaries.docs[0]
    assert TOKEN not in repr(stored)
    assert stored["embedding"] == [1.0]


async def test_async_a_user_summary_is_redacted_before_it_is_embedded():
    service, summaries, embeddings = _async_service()

    await service.persist_user_summary("u1", _summary("user_summary"))

    assert not any(TOKEN in text for text in _embedded(embeddings))
    stored = summaries.docs[0]
    assert TOKEN not in repr(stored)
    assert stored["embedding"] == [1.0]
