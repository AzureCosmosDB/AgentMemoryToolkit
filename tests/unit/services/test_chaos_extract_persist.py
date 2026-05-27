from __future__ import annotations

import json
from typing import Any

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError

from agent_memory_toolkit.aio.services.pipeline import AsyncPipelineService
from agent_memory_toolkit.services.pipeline import PipelineService


class _FlakyContainer:
    def __init__(self):
        self.docs: dict[str, dict[str, Any]] = {}
        self.attempts = 0
        self.failed_once = False

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        item_id = body["id"]
        if item_id in self.docs:
            raise CosmosResourceExistsError(message="exists")
        if self.attempts == 2 and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("transient persist failure")
        self.docs[item_id] = dict(body)
        return dict(body)


class _AsyncFlakyContainer(_FlakyContainer):
    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return super().create_item(body=body)


class _Store:
    def __init__(self, container: _FlakyContainer, docs: list[dict[str, Any]]):
        self.container = container
        self.docs = [dict(doc) for doc in docs]

    def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        del sql, partition_key, cross_partition
        params = {p["name"]: p["value"] for p in (parameters or [])}
        docs = [dict(doc) for doc in self.docs]
        if "@user_id" in params:
            docs = [doc for doc in docs if doc.get("user_id") == params["@user_id"]]
        if "@thread_id" in params:
            docs = [doc for doc in docs if doc.get("thread_id") == params["@thread_id"]]
        return docs

    def read_item(self, item_id: str, partition_key: Any):
        del partition_key
        return self.container.docs[item_id]

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        self.container.docs[record["id"]] = dict(record)
        return record

    def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        del old_doc, superseder_id, reason
        return True


class _AsyncStore(_Store):
    async def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        return super().query(sql, parameters=parameters, partition_key=partition_key, cross_partition=cross_partition)

    async def read_item(self, item_id: str, partition_key: Any):
        return super().read_item(item_id, partition_key)

    async def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        return super().add_cosmos(record)

    async def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        return super().mark_superseded(old_doc, superseder_id, reason=reason)


class _Chat:
    def __init__(self):
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del messages, opts
        self.calls += 1
        return json.dumps(
            {
                "facts": [
                    {"text": "The user prefers dark mode.", "action": "ADD", "category": "preference"},
                    {"text": "The user prefers concise answers.", "action": "ADD", "category": "preference"},
                ]
            }
        )


class _AsyncChat(_Chat):
    async def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        return super().generate(messages, **opts)


class _Embeddings:
    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts, start=1)]

    def generate(self, text: str) -> list[float]:
        del text
        return [0.0]


class _AsyncEmbeddings(_Embeddings):
    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        return super().generate_batch(texts)

    async def generate(self, text: str) -> list[float]:
        return [0.0]


def _turn() -> dict[str, Any]:
    return {
        "id": "turn-1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "I prefer dark mode and concise answers.",
        "created_at": "2025-01-01T00:00:00+00:00",
    }


def test_persist_retry_reuses_extract_output_without_second_llm_call() -> None:
    container = _FlakyContainer()
    chat = _Chat()
    service = PipelineService(_Store(container, [_turn()]), chat, _Embeddings())

    extracted = service.extract_memories_dry("u1", "t1")
    with pytest.raises(RuntimeError, match="transient"):
        service.persist_extracted_memories("u1", extracted)
    result = service.persist_extracted_memories("u1", extracted)

    assert chat.calls == 1
    assert result["fact_count"] == 1
    assert len(container.docs) == 2


@pytest.mark.asyncio
async def test_async_persist_retry_reuses_extract_output_without_second_llm_call() -> None:
    container = _AsyncFlakyContainer()
    chat = _AsyncChat()
    service = AsyncPipelineService(_AsyncStore(container, [_turn()]), chat, _AsyncEmbeddings())

    extracted = await service.extract_memories_dry("u1", "t1")
    with pytest.raises(RuntimeError, match="transient"):
        await service.persist_extracted_memories("u1", extracted)
    result = await service.persist_extracted_memories("u1", extracted)

    assert chat.calls == 1
    assert result["fact_count"] == 1
    assert len(container.docs) == 2
