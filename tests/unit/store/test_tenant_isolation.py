"""A store stays inside its tenant.

Every query a store sends filters on ``c.tenant_id``, a point operation addresses the partition ``[tenant, user]``, and
a document it writes carries the tenant; one that names another tenant is refused. Each case runs against the sync and
the async store, and the tenant is not the default one, so a hard-coded default cannot pass.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import MemoryNotFoundError, ValidationError
from azure.cosmos.agent_memory.models import MemoryRecord
from azure.cosmos.agent_memory.store import MemoryStore

TENANT = "acme"
OTHER_TENANT = "globex"
USER = "u1"
TURNS, MEMORIES, SUMMARIES = ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES
NO_KEYWORDS = "hello"
KEYWORDS = "hotels in lisbon"


class _Rows:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _Container:
    """An in-memory container that records what it is sent and answers like the SDK's, sync or async."""

    def __init__(self, is_async: bool, docs=()):
        self._is_async = is_async
        self.docs = {self._key(doc): copy.deepcopy(doc) for doc in docs}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _key(doc):
        return (doc["tenant_id"], doc["user_id"]), doc["id"]

    def _reply(self, value=None, error=None):
        if not self._is_async:
            if error is not None:
                raise error
            return value

        async def reply():
            if error is not None:
                raise error
            return value

        return reply()

    def _missing(self):
        return self._reply(error=CosmosResourceNotFoundError(status_code=404, message="not found"))

    def sent(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]

    def query_items(self, **kwargs):
        self.calls.append(("query_items", kwargs))
        return _Rows([]) if self._is_async else []

    def read_item(self, *, item, partition_key):
        self.calls.append(("read_item", {"item": item, "partition_key": partition_key}))
        doc = self.docs.get((tuple(partition_key), item))
        return self._missing() if doc is None else self._reply(copy.deepcopy(doc))

    def upsert_item(self, *, body, **kwargs):
        self.calls.append(("upsert_item", {"body": body, **kwargs}))
        self.docs[self._key(body)] = copy.deepcopy(body)
        return self._reply(body)

    def replace_item(self, *, item, body, **kwargs):
        self.calls.append(("replace_item", {"item": item, "body": body, **kwargs}))
        if self._key(body) not in self.docs:
            return self._missing()
        self.docs[self._key(body)] = copy.deepcopy(body)
        return self._reply(body)

    def delete_item(self, *, item, partition_key, **kwargs):
        self.calls.append(("delete_item", {"item": item, "partition_key": partition_key, **kwargs}))
        if self.docs.pop((tuple(partition_key), item), None) is None:
            return self._missing()
        return self._reply()


class _Embeddings:
    def __init__(self, is_async: bool):
        self._is_async = is_async
        self.generated: list[str] = []

    def _reply(self, value):
        if not self._is_async:
            return value

        async def reply():
            return value

        return reply()

    def generate(self, text):
        self.generated.append(text)
        return self._reply([0.1, 0.2])

    def generate_batch(self, texts):
        self.generated.extend(texts)
        return self._reply([[0.1, 0.2] for _ in texts])


def _record(**fields):
    return {
        "id": "m1",
        "user_id": USER,
        "thread_id": "t1",
        "type": "fact",
        "content": "I like tea",
        "tags": ["a"],
        **fields,
    }


def _fact(tenant=TENANT, **fields):
    return _record(tenant_id=tenant, **fields)


def _build(kind, *, tenant=TENANT, docs=None):
    is_async = kind == "async"
    containers = {key: _Container(is_async, (docs or {}).get(key, ())) for key in ContainerKey}
    store_class = AsyncMemoryStore if is_async else MemoryStore
    store = store_class(containers=containers, embeddings_client=_Embeddings(is_async), tenant_id=tenant)
    return store, containers


async def _call(store, method, *args, **kwargs):
    result = getattr(store, method)(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.fixture(params=["sync", "async"])
def kind(request):
    return request.param


def test_a_store_is_bound_to_one_tenant(kind):
    store, _ = _build(kind)

    assert store.tenant_id == TENANT
    with pytest.raises(AttributeError):
        store.tenant_id = OTHER_TENANT


def test_a_store_without_a_tenant_uses_the_default_one(kind):
    store_class = AsyncMemoryStore if kind == "async" else MemoryStore

    assert store_class(containers={key: object() for key in ContainerKey}).tenant_id == "default"


def test_a_store_rejects_an_invalid_tenant(kind):
    store_class = AsyncMemoryStore if kind == "async" else MemoryStore

    with pytest.raises(ValidationError):
        store_class(containers={key: object() for key in ContainerKey}, tenant_id="acme/emea")


# (method, args, kwargs, container, names the owner, reads the owner's partition)
READS = [
    pytest.param("get_memories", (), {"user_id": USER}, MEMORIES, True, True, id="get_memories"),
    pytest.param("get_memories", (), {}, MEMORIES, False, False, id="get_memories-whole-tenant"),
    pytest.param("get_thread", ("t1",), {"user_id": USER}, TURNS, True, True, id="get_thread"),
    pytest.param("get_thread", ("t1",), {}, TURNS, False, False, id="get_thread-whole-tenant"),
    pytest.param("get_thread_summary", (USER, "t1"), {}, SUMMARIES, True, True, id="get_thread_summary"),
    pytest.param("get_episodes", (USER,), {}, MEMORIES, True, True, id="get_episodes"),
    pytest.param("list_tags", (USER,), {}, MEMORIES, True, True, id="list_tags"),
    pytest.param("get_procedural_prompt", (USER,), {}, MEMORIES, True, True, id="get_procedural_prompt"),
    pytest.param("get_procedural_history", (USER,), {}, MEMORIES, True, True, id="get_procedural_history"),
    pytest.param("get_procedural_memories", (USER,), {}, MEMORIES, True, True, id="get_procedural_memories"),
    pytest.param("get_memory_history", ("m1", USER), {}, MEMORIES, True, True, id="get_memory_history"),
    pytest.param("search", (NO_KEYWORDS,), {"user_id": USER}, MEMORIES, True, True, id="search"),
    pytest.param("search", (KEYWORDS,), {"user_id": USER}, MEMORIES, True, False, id="search-keywords"),
    pytest.param("search", (NO_KEYWORDS,), {}, MEMORIES, False, False, id="search-whole-tenant"),
    pytest.param("search_turns", (NO_KEYWORDS,), {"user_id": USER}, TURNS, True, True, id="search_turns"),
    pytest.param("search_turns", (KEYWORDS,), {"user_id": USER}, TURNS, True, False, id="search_turns-keywords"),
    pytest.param("search_summaries", (NO_KEYWORDS,), {"user_id": USER}, SUMMARIES, True, True, id="search_summaries"),
    pytest.param(
        "search_summaries", (KEYWORDS,), {"user_id": USER}, SUMMARIES, True, False, id="search_summaries-keywords"
    ),
    pytest.param("search_episodic", (USER, NO_KEYWORDS), {}, MEMORIES, True, True, id="search_episodic"),
    pytest.param("search_episodic", (USER, KEYWORDS), {}, MEMORIES, True, False, id="search_episodic-keywords"),
    pytest.param("retrieve_procedures", (USER, NO_KEYWORDS), {}, MEMORIES, True, True, id="retrieve_procedures"),
    pytest.param("retrieve_procedures", (USER, KEYWORDS), {}, MEMORIES, True, False, id="retrieve_procedures-keywords"),
    pytest.param("build_episodic_context", (USER, NO_KEYWORDS), {}, MEMORIES, True, True, id="build_episodic_context"),
]


@pytest.mark.parametrize(("method", "args", "kwargs", "container", "names_owner", "in_partition"), READS)
async def test_every_read_filters_on_the_tenant(kind, method, args, kwargs, container, names_owner, in_partition):
    store, containers = _build(kind)

    await _call(store, method, *args, **kwargs)

    for key, target in containers.items():
        assert bool(target.sent("query_items")) == (key == container), "a read queries only its own container"
    (query,) = containers[container].sent("query_items")
    params = {p["name"]: p["value"] for p in query["parameters"]}
    where = query["query"].split(" WHERE ", 1)[1]
    assert "c.tenant_id = @tenant_id" in where
    assert params["@tenant_id"] == TENANT
    if names_owner:
        assert "c.user_id = @user_id" in where
        assert params["@user_id"] == USER
    else:
        assert "c.user_id" not in where
    if in_partition:
        assert query["partition_key"] == [TENANT, USER]
    else:
        assert "partition_key" not in query


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "container"),
    [
        pytest.param("get_thread", ("t1",), {"user_id": USER}, TURNS, id="get_thread"),
        pytest.param("get_memory_history", ("m1", USER), {}, MEMORIES, id="get_memory_history"),
        pytest.param("search", (KEYWORDS,), {"user_id": USER}, MEMORIES, id="search"),
        pytest.param("search_turns", (KEYWORDS,), {"user_id": USER}, TURNS, id="search_turns"),
        pytest.param("search_summaries", (KEYWORDS,), {"user_id": USER}, SUMMARIES, id="search_summaries"),
        pytest.param("search_episodic", (USER, KEYWORDS), {}, MEMORIES, id="search_episodic"),
        pytest.param("retrieve_procedures", (USER, KEYWORDS), {}, MEMORIES, id="retrieve_procedures"),
    ],
)
async def test_projected_reads_return_the_tenant_of_each_record(kind, method, args, kwargs, container):
    store, containers = _build(kind)

    await _call(store, method, *args, **kwargs)

    (query,) = containers[container].sent("query_items")
    assert "c.tenant_id" in query["query"].split(" FROM ")[0]


POINT_OPERATIONS = [
    pytest.param("update", ("m1",), {"user_id": USER, "memory_type": "fact", "content": "I like coffee"}, id="update"),
    pytest.param("delete", ("m1",), {"user_id": USER, "memory_type": "fact"}, id="delete"),
    pytest.param("add_tags", ("m1", USER, "fact", ["b"]), {}, id="add_tags"),
    pytest.param("remove_tags", ("m1", USER, "fact", ["a"]), {}, id="remove_tags"),
]


@pytest.mark.parametrize(("method", "args", "kwargs"), POINT_OPERATIONS)
async def test_point_operations_address_the_tenants_partition(kind, method, args, kwargs):
    store, containers = _build(kind, docs={MEMORIES: [_fact()]})

    await _call(store, method, *args, **kwargs)

    memories = containers[MEMORIES]
    addressed = [*memories.sent("read_item"), *memories.sent("delete_item")]
    assert addressed and all(call["partition_key"] == [TENANT, USER] for call in addressed)
    assert memories.sent("replace_item") or memories.sent("delete_item")


async def test_the_user_summary_is_read_from_the_tenants_partition(kind):
    summary = {"id": "user_summary_u1", "tenant_id": TENANT, "user_id": USER, "thread_id": "__user_summary__"}
    store, containers = _build(kind, docs={SUMMARIES: [summary]})

    assert (await _call(store, "get_user_summary", USER))["id"] == "user_summary_u1"

    assert containers[SUMMARIES].sent("read_item") == [{"item": "user_summary_u1", "partition_key": [TENANT, USER]}]


@pytest.mark.parametrize(("method", "args", "kwargs"), POINT_OPERATIONS)
async def test_another_tenants_record_with_the_same_owner_and_id_is_not_found(kind, method, args, kwargs):
    store, containers = _build(kind, docs={MEMORIES: [_fact(OTHER_TENANT)]})

    with pytest.raises(MemoryNotFoundError):
        await _call(store, method, *args, **kwargs)

    assert not containers[MEMORIES].sent("replace_item") and not containers[MEMORIES].sent("delete_item")
    assert list(containers[MEMORIES].docs) == [((OTHER_TENANT, USER), "m1")]


async def test_another_tenants_user_summary_is_not_found(kind):
    summary = {"id": "user_summary_u1", "tenant_id": OTHER_TENANT, "user_id": USER, "thread_id": "__user_summary__"}
    store, _ = _build(kind, docs={SUMMARIES: [summary]})

    assert await _call(store, "get_user_summary", USER) is None


@pytest.mark.parametrize(("memory_type", "container"), [("turn", TURNS), ("fact", MEMORIES), ("episodic", MEMORIES)])
async def test_add_writes_the_tenant_into_the_document(kind, memory_type, container):
    store, containers = _build(kind)

    await _call(store, "add", user_id=USER, role="user", content="hello", memory_type=memory_type, thread_id="t1")

    (written,) = containers[container].sent("upsert_item")
    assert written["body"]["tenant_id"] == TENANT
    assert written["body"]["user_id"] == USER


@pytest.mark.parametrize(
    ("memory_type", "container"),
    [("turn", TURNS), ("fact", MEMORIES), ("thread_summary", SUMMARIES), ("user_summary", SUMMARIES)],
)
async def test_upsert_memory_stamps_the_tenant_and_leaves_the_callers_record_alone(kind, memory_type, container):
    store, containers = _build(kind)
    record = _record(type=memory_type)

    await _call(store, "upsert_memory", record)

    (written,) = containers[container].sent("upsert_item")
    assert written["body"]["tenant_id"] == TENANT
    assert "tenant_id" not in record


async def test_a_record_may_name_the_tenant_of_the_store(kind):
    store, containers = _build(kind)

    await _call(store, "upsert_memory", _fact())

    assert len(containers[MEMORIES].sent("upsert_item")) == 1


async def test_upsert_memory_refuses_a_record_of_another_tenant(kind):
    store, containers = _build(kind)

    with pytest.raises(ValidationError):
        await _call(store, "upsert_memory", _fact(OTHER_TENANT))

    assert not containers[MEMORIES].sent("upsert_item")


async def test_a_built_record_is_stored_in_the_tenant_of_the_store_unless_it_names_another(kind):
    store, containers = _build(kind)
    fact = {
        "memory_type": "fact",
        "id": "fact_" + "a" * 32,
        "user_id": USER,
        "content": "I like tea",
        "content_hash": "a" * 32,
        "prompt_id": "extract_memories.prompty",
        "metadata": {"category": "other"},
    }

    await _call(store, "upsert_memory", MemoryRecord(**fact).to_cosmos_dict())
    with pytest.raises(ValidationError, match="different tenant"):
        await _call(store, "upsert_memory", MemoryRecord(**fact, tenant_id=OTHER_TENANT).to_cosmos_dict())

    (written,) = containers[MEMORIES].sent("upsert_item")
    assert written["body"]["tenant_id"] == TENANT


async def test_push_stamps_every_record_with_the_tenant(kind):
    store, containers = _build(kind)

    await _call(store, "push", [_record(id="t1", type="turn"), _record(id="f1")])

    assert [call["body"]["tenant_id"] for call in containers[TURNS].sent("upsert_item")] == [TENANT]
    assert [call["body"]["tenant_id"] for call in containers[MEMORIES].sent("upsert_item")] == [TENANT]


async def test_push_refuses_a_batch_with_a_record_of_another_tenant_before_writing_any_of_it(kind):
    store, containers = _build(kind)

    with pytest.raises(ValidationError):
        await _call(store, "push", [_fact(id="f1"), _fact(OTHER_TENANT, id="f2")])

    assert not containers[MEMORIES].sent("upsert_item")


INVALID_OWNERS = ["", "a/b", "x#y", "tab\there", "o" * 257]


@pytest.mark.parametrize("owner", INVALID_OWNERS)
async def test_add_refuses_an_invalid_owner_before_embedding_or_writing(kind, owner):
    store, containers = _build(kind)

    with pytest.raises(ValidationError, match="user_id"):
        await _call(store, "add", user_id=owner, role="system", content="I like tea", memory_type="fact")

    assert store._embeddings_client.generated == []
    assert not containers[MEMORIES].sent("upsert_item")


async def test_add_refuses_a_thread_ending_in_whitespace_before_embedding_or_writing(kind):
    store, containers = _build(kind)

    with pytest.raises(ValidationError, match="thread_id"):
        await _call(store, "add", user_id=USER, role="user", content="x", memory_type="fact", thread_id="t1 ")

    assert store._embeddings_client.generated == []
    assert not containers[MEMORIES].sent("upsert_item")


@pytest.mark.parametrize("owner", [None, *INVALID_OWNERS])
async def test_upsert_memory_refuses_a_record_without_a_valid_owner(kind, owner):
    store, containers = _build(kind)

    with pytest.raises(ValidationError, match="user_id"):
        await _call(store, "upsert_memory", _fact(user_id=owner, embedding=[0.1]))

    assert store._embeddings_client.generated == []
    assert not containers[MEMORIES].sent("upsert_item")


async def test_push_refuses_a_batch_with_a_record_without_a_valid_owner_before_embedding_or_writing_any_of_it(kind):
    store, containers = _build(kind)

    with pytest.raises(ValidationError, match="user_id"):
        await _call(store, "push", [_fact(id="f1"), _fact(id="f2", user_id="a/b")])

    assert store._embeddings_client.generated == []
    assert not containers[MEMORIES].sent("upsert_item")


@pytest.mark.parametrize("owner", INVALID_OWNERS)
async def test_a_read_for_an_invalid_owner_is_refused_rather_than_widened_to_the_tenant(kind, owner):
    store, containers = _build(kind)

    with pytest.raises(ValidationError, match="user_id"):
        await _call(store, "get_memories", user_id=owner)

    assert not containers[MEMORIES].sent("query_items")


async def test_mark_superseded_keeps_the_tenant_and_refuses_another(kind):
    store, containers = _build(kind, docs={MEMORIES: [_fact(), _fact(OTHER_TENANT)]})

    assert await _call(store, "mark_superseded", _fact(_etag="e1"), "f2", reason="duplicate") is True
    with pytest.raises(ValidationError):
        await _call(store, "mark_superseded", _fact(OTHER_TENANT, _etag="e1"), "f2", reason="duplicate")

    (replaced,) = containers[MEMORIES].sent("replace_item")
    assert replaced["body"]["tenant_id"] == TENANT
    assert containers[MEMORIES].docs[(OTHER_TENANT, USER), "m1"].get("superseded_by") is None
