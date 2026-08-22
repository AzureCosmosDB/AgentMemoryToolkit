from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import ValidationError


class AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _fact(**overrides):
    doc = {
        "id": "fact_123",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": "Use the launch checklist.",
        "content_hash": "a" * 32,
        "metadata": {"category": "preference"},
        "created_at": "2026-01-01T00:00:00+00:00",
        "tags": ["sys:fact"],
        "tenant_id": "acme",
        "scope_type": "user",
        "scope_id": "alice",
        "scope_key": "user:alice",
        "acl": {"read": ["user:alice"], "write": ["user:alice"], "annotate": [], "forget": ["user:alice"]},
        "provenance": {"created_by": "user:alice"},
    }
    doc.update(overrides)
    return doc


def _containers(*, memories=None):
    return {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }


async def test_async_promote_allows_share_and_stamps_target_lineage():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator(
        [_fact(provenance={"created_by": "user:alice", "source_ids": ["prior"]})]
    )
    memories.upsert_item = AsyncMock(side_effect=lambda body: body)
    store = AsyncMemoryStore(containers=_containers(memories=memories))
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", roles=["team:eng:member"])

    promoted = await store.promote("fact_123", "user:alice", "team:eng", ctx)

    assert promoted["scope_key"] == "team:eng"
    assert promoted["acl"]["read"] == ["team:eng"]
    assert promoted["provenance"]["source_ids"] == ["prior", "fact_123"]
    assert promoted["metadata"]["curation_status"] == "approved"
    assert promoted["status"] == "approved"


async def test_async_promote_refuses_without_share_or_assign():
    memories = MagicMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))
    ctx = SecurityContext(tenant_id="acme", principal="user:alice")

    with pytest.raises(ValidationError, match="write permission"):
        await store.promote("fact_123", "user:alice", "team:eng", ctx)

    memories.query_items.assert_not_called()
    memories.upsert_item.assert_not_called()
