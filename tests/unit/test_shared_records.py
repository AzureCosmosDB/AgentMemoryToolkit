from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.exceptions import MemoryConflictError, SharedRecordReadOnlyError, ValidationError
from azure.cosmos.agent_memory.store import MemoryStore


class FakePreconditionFailed(Exception):
    status_code = 412


class FakeSharedContainer:
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

    def upsert_item(self, *, body: dict) -> dict:
        stored = copy.deepcopy(body)
        stored["_etag"] = self._etag()
        self.docs[self._pk_tuple(stored)] = stored
        return copy.deepcopy(stored)

    def read_item(self, *, item: str, partition_key: list[str]) -> dict:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        key = (partition_key[0], partition_key[1], partition_key[2], item)
        if key not in self.docs:
            raise CosmosResourceNotFoundError(message="404")
        return copy.deepcopy(self.docs[key])

    def replace_item(self, *, item: str, body: dict, match_condition=None, etag: str | None = None) -> dict:
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
        return []


def _store(memories: FakeSharedContainer | None = None) -> MemoryStore:
    return MemoryStore(
        containers={
            ContainerKey.TURNS: MagicMock(),
            ContainerKey.MEMORIES: memories or FakeSharedContainer(),
            ContainerKey.SUMMARIES: MagicMock(),
        }
    )


def _ctx(*roles: str) -> SecurityContext:
    return SecurityContext(tenant_id="acme", principal="user:alice", roles=list(roles))


def test_shared_record_stale_compare_and_swap_rejected() -> None:
    memories = FakeSharedContainer()
    store = _store(memories)
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    first = store.put_shared_record(ctx=ctx, scope_key="team:eng", key="plan", data={"n": 0})
    second = store.get_shared_record(ctx=ctx, scope_key="team:eng", key="plan")

    store.compare_and_swap_shared_record(
        ctx=ctx,
        scope_key="team:eng",
        key="plan",
        record={**first, "data": {"n": 1}},
        etag=first["_etag"],
    )

    with pytest.raises(MemoryConflictError, match="modified by another writer"):
        store.compare_and_swap_shared_record(
            ctx=ctx,
            scope_key="team:eng",
            key="plan",
            record={**second, "data": {"n": 2}},
            etag=second["_etag"],
        )


def test_shared_record_read_only_allows_reads_refuses_writes() -> None:
    store = _store()
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    store.put_shared_record(ctx=ctx, scope_key="team:eng", key="config", data={"mode": "safe"}, read_only=True)

    assert store.get_shared_record(ctx=ctx, scope_key="team:eng", key="config")["data"] == {"mode": "safe"}

    with pytest.raises(SharedRecordReadOnlyError):
        store.update_shared_record(
            ctx=ctx,
            scope_key="team:eng",
            key="config",
            mutator=lambda doc: {**doc, "data": {"mode": "unsafe"}},
        )


def test_shared_record_update_requires_write_permission() -> None:
    store = _store()
    writer = _ctx("team:eng:writer", "team:eng:reader")
    reader = _ctx("team:eng:reader")
    store.put_shared_record(ctx=writer, scope_key="team:eng", key="queue", data={"items": []})

    with pytest.raises(ValidationError, match="write permission denied"):
        store.update_shared_record(
            ctx=reader,
            scope_key="team:eng",
            key="queue",
            mutator=lambda doc: doc,
        )


def test_shared_record_mutator_retry_converges_after_conflict() -> None:
    memories = FakeSharedContainer()
    store = _store(memories)
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    store.put_shared_record(ctx=ctx, scope_key="team:eng", key="queue", data={"items": []})
    memories.conflict_once = True

    def append_task(doc: dict) -> dict:
        items = list(doc["data"].get("items", []))
        items.append("task-1")
        doc["data"] = {**doc["data"], "items": items}
        return doc

    updated = store.update_shared_record(
        ctx=ctx,
        scope_key="team:eng",
        key="queue",
        mutator=append_task,
        max_retries=3,
    )

    assert updated["data"]["external"] is True
    assert updated["data"]["items"] == ["task-1"]
    assert memories.replace_calls == 2


def test_bound_client_uses_bound_scope_for_shared_records() -> None:
    client = CosmosMemoryClient(use_default_credential=False)
    memories = FakeSharedContainer()
    client._memories_container_client = memories
    client._turns_container_client = MagicMock()
    client._summaries_container_client = MagicMock()
    client._store = _store(memories)
    ctx = _ctx("team:eng:writer", "team:eng:reader")
    bound = client.session(ctx, write_scope="team:eng")

    bound.put_shared_record(key="handoff", data={"owner": "planner"})

    assert bound.get_shared_record(key="handoff")["scope_key"] == "team:eng"
