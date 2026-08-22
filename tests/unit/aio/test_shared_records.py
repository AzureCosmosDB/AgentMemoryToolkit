from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import MemoryConflictError, SharedRecordReadOnlyError, ValidationError


class FakePreconditionFailed(Exception):
    status_code = 412


class FakeAsyncSharedContainer:
    def __init__(self) -> None:
        self.docs: dict[tuple[str, str, str, str], dict] = {}
        self.version = 0
        self.conflict_once = False
        self.replace_calls = 0

    def _pk_tuple(self, body: dict) -> tuple[str, str, str, str]:
        return (body["tenant_id"], body["scope_key"], body["thread_id"], body["id"])

    def _etag(self) -> str:
        self.version += 1
        return f'"v{self.version}"'

    async def upsert_item(self, *, body: dict) -> dict:
        stored = copy.deepcopy(body)
        stored["_etag"] = self._etag()
        self.docs[self._pk_tuple(stored)] = stored
        return copy.deepcopy(stored)

    async def read_item(self, *, item: str, partition_key: list[str]) -> dict:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        key = (partition_key[0], partition_key[1], partition_key[2], item)
        if key not in self.docs:
            raise CosmosResourceNotFoundError(message="404")
        return copy.deepcopy(self.docs[key])

    async def replace_item(self, *, item: str, body: dict, match_condition=None, etag: str | None = None) -> dict:
        del item, match_condition
        self.replace_calls += 1
        key = self._pk_tuple(body)
        current = self.docs[key]
        if self.conflict_once:
            self.conflict_once = False
            current["data"]["external"] = True
            current["_etag"] = self._etag()
            raise FakePreconditionFailed()
        if etag != current.get("_etag"):
            raise FakePreconditionFailed()
        stored = copy.deepcopy(body)
        stored["_etag"] = self._etag()
        self.docs[key] = stored
        return copy.deepcopy(stored)

    def query_items(self, **kwargs):
        del kwargs

        async def _empty():
            if False:
                yield None

        return _empty()


def _store(memories: FakeAsyncSharedContainer | None = None) -> AsyncMemoryStore:
    return AsyncMemoryStore(
        containers={
            ContainerKey.TURNS: MagicMock(),
            ContainerKey.MEMORIES: memories or FakeAsyncSharedContainer(),
            ContainerKey.SUMMARIES: MagicMock(),
        }
    )


def _ctx(*roles: str) -> SecurityContext:
    return SecurityContext(tenant_id="acme", principal="user:alice", roles=list(roles))


@pytest.mark.asyncio
async def test_async_shared_record_stale_compare_and_swap_rejected() -> None:
    store = _store()
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    first = await store.put_shared_record(ctx=ctx, scope_key="team:eng", key="plan", data={"n": 0})
    second = await store.get_shared_record(ctx=ctx, scope_key="team:eng", key="plan")

    await store.compare_and_swap_shared_record(
        ctx=ctx,
        scope_key="team:eng",
        key="plan",
        record={**first, "data": {"n": 1}},
        etag=first["_etag"],
    )

    with pytest.raises(MemoryConflictError):
        await store.compare_and_swap_shared_record(
            ctx=ctx,
            scope_key="team:eng",
            key="plan",
            record={**second, "data": {"n": 2}},
            etag=second["_etag"],
        )


@pytest.mark.asyncio
async def test_async_shared_record_read_only_allows_reads_refuses_writes() -> None:
    store = _store()
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    await store.put_shared_record(ctx=ctx, scope_key="team:eng", key="config", data={"mode": "safe"}, read_only=True)

    assert (await store.get_shared_record(ctx=ctx, scope_key="team:eng", key="config"))["data"] == {"mode": "safe"}

    with pytest.raises(SharedRecordReadOnlyError):
        await store.update_shared_record(
            ctx=ctx,
            scope_key="team:eng",
            key="config",
            mutator=lambda doc: {**doc, "data": {"mode": "unsafe"}},
        )


@pytest.mark.asyncio
async def test_async_shared_record_update_requires_write_permission() -> None:
    store = _store()
    writer = _ctx("team:eng:writer", "team:eng:reader")
    reader = _ctx("team:eng:reader")
    await store.put_shared_record(ctx=writer, scope_key="team:eng", key="queue", data={"items": []})

    with pytest.raises(ValidationError, match="write permission denied"):
        await store.update_shared_record(ctx=reader, scope_key="team:eng", key="queue", mutator=lambda doc: doc)


@pytest.mark.asyncio
async def test_async_shared_record_mutator_retry_converges_after_conflict() -> None:
    memories = FakeAsyncSharedContainer()
    store = _store(memories)
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    await store.put_shared_record(ctx=ctx, scope_key="team:eng", key="queue", data={"items": []})
    memories.conflict_once = True

    async def append_task(doc: dict) -> dict:
        items = list(doc["data"].get("items", []))
        items.append("task-1")
        doc["data"] = {**doc["data"], "items": items}
        return doc

    updated = await store.update_shared_record(
        ctx=ctx,
        scope_key="team:eng",
        key="queue",
        mutator=append_task,
        max_retries=3,
    )

    assert updated["data"]["external"] is True
    assert updated["data"]["items"] == ["task-1"]
    assert memories.replace_calls == 2


@pytest.mark.asyncio
async def test_async_bound_client_uses_bound_scope_for_shared_records() -> None:
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    memories = FakeAsyncSharedContainer()
    client._memories_container_client = memories
    client._turns_container_client = MagicMock()
    client._summaries_container_client = MagicMock()
    client._store = _store(memories)
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    bound = client.session(ctx, write_scope="team:eng")

    await bound.put_shared_record(key="handoff", data={"owner": "planner"})

    assert (await bound.get_shared_record(key="handoff"))["scope_key"] == "team:eng"
