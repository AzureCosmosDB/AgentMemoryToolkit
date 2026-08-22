from __future__ import annotations

from typing import Any

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService, _AsyncStoreContainerAdapter
from tests.unit.services.test_extract_dry import _AsyncChat, _AsyncEmbeddings, _AsyncStore


class _AsyncPatchTurnsContainer:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = [dict(doc) for doc in docs]
        self.patch_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []

    async def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [dict(doc) for doc in self.docs]

    async def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        del partition_key
        for doc in self.docs:
            if doc.get("id") == item:
                return dict(doc)
        raise KeyError(item)

    async def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        self.upsert_calls.append(body)
        for index, doc in enumerate(self.docs):
            if doc.get("id") == body.get("id"):
                self.docs[index] = body
                return body
        self.docs.append(body)
        return body

    async def patch_item(
        self, *, item: str, partition_key: Any, patch_operations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.patch_calls.append(
            {"item": item, "partition_key": partition_key, "patch_operations": [dict(op) for op in patch_operations]}
        )
        for doc in self.docs:
            if doc.get("id") == item:
                for operation in patch_operations:
                    assert operation["op"] == "set"
                    assert operation["path"].startswith("/")
                    doc[operation["path"][1:]] = operation["value"]
                return dict(doc)
        raise KeyError(item)


class _AsyncNoPatchTurnsContainer(_AsyncPatchTurnsContainer):
    patch_item = None


class _AsyncContainerBackedStore(_AsyncStore):
    def __init__(self, container: Any):
        super().__init__([])
        self._containers = {ContainerKey.TURNS: container}


def _turn_doc(**overrides: Any) -> dict[str, Any]:
    doc = {
        "id": "turn-1",
        "user_id": "u1",
        "thread_id": "t1",
        "type": "turn",
        "content": "hello",
        "created_at": "2025-01-01T00:00:00+00:00",
    }
    doc.update(overrides)
    return doc


def _service(turns_container: Any) -> AsyncPipelineService:
    memories_store = _AsyncStore([])
    summaries_store = _AsyncStore([])
    turns_adapter = _AsyncStoreContainerAdapter(_AsyncContainerBackedStore(turns_container), ContainerKey.TURNS)
    return AsyncPipelineService(
        memories_store,
        _AsyncChat([]),
        _AsyncEmbeddings(),
        containers={
            ContainerKey.TURNS: turns_adapter,
            ContainerKey.MEMORIES: _AsyncStoreContainerAdapter(memories_store, ContainerKey.MEMORIES),
            ContainerKey.SUMMARIES: _AsyncStoreContainerAdapter(summaries_store, ContainerKey.SUMMARIES),
        },
    )


@pytest.mark.asyncio
async def test_mark_turns_extracted_patches_extracted_at() -> None:
    turns_container = _AsyncPatchTurnsContainer([_turn_doc()])
    service = _service(turns_container)

    marked = await service._mark_turns_extracted([_turn_doc()])

    assert marked == 1
    assert turns_container.patch_calls == [
        {
            "item": "turn-1",
            "partition_key": ["default", "user:u1", "t1"],
            "patch_operations": [
                {"op": "set", "path": "/extracted_at", "value": turns_container.docs[0]["extracted_at"]}
            ],
        }
    ]
    assert turns_container.docs[0]["extracted_at"]
    assert turns_container.upsert_calls == []


@pytest.mark.asyncio
async def test_mark_turns_extracted_falls_back_to_read_modify_upsert() -> None:
    turns_container = _AsyncNoPatchTurnsContainer([_turn_doc()])
    service = _service(turns_container)

    marked = await service._mark_turns_extracted([_turn_doc()])

    assert marked == 1
    assert turns_container.docs[0]["extracted_at"]
    assert turns_container.upsert_calls == [turns_container.docs[0]]
