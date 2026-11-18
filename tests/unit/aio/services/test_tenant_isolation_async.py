"""An async pipeline is bound to one tenant.

Every query it sends filters on the tenant and names the owner's partition, ``[tenant_id, user_id]``; every point
operation names that partition; every document it writes carries the tenant. The containers here are in-memory doubles
that keep documents under ``[tenant_id, user_id]`` as a container with a hierarchical partition key does, answer a
query from the partition it names and from its ``@tenant_id``, ``@user_id``, ``@thread_id`` and type conditions, and
record every call. A second tenant holds records with the same user and thread ids, so a query, read or write that
forgot the tenant would find them, or miss its own.
"""

from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace
from typing import Any, Iterable, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import compute_content_hash
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.exceptions import ConfigurationError, ValidationError
from azure.cosmos.agent_memory.services._dedup import Verdict

TENANT = "acme"
OTHER = "globex"
USER = "u1"
OWNER = [TENANT, USER]
OWNER_FILTER = "c.tenant_id = @tenant_id AND c.user_id = @user_id"

_PARAMETER_FIELDS = {
    "@tenant_id": "tenant_id",
    "@user_id": "user_id",
    "@thread_id": "thread_id",
    "@memory_type": "type",
    "@fact_type": "type",
}


class _Query(NamedTuple):
    sql: str
    params: dict[str, Any]
    partition_key: Any


def _matches(doc: dict[str, Any], sql: str, params: dict[str, Any]) -> bool:
    if any(name in params and doc.get(field) != params[name] for name, field in _PARAMETER_FIELDS.items()):
        return False
    if (found := re.search(r"c\.type = '(\w+)'", sql)) and doc.get("type") != found[1]:
        return False
    if (found := re.search(r"c\.type IN \(([^)]*)\)", sql)) and doc.get("type") not in re.findall(r"'(\w+)'", found[1]):
        return False
    if "NOT IS_DEFINED(c.extracted_at)" in sql and doc.get("extracted_at"):
        return False
    if "@last_at" in params and (doc["created_at"], doc["id"]) <= (params["@last_at"], params["@last_id"]):
        return False
    return True


class _Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = iter(rows)

    def __aiter__(self) -> _Rows:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._rows)
        except StopIteration:
            raise StopAsyncIteration


class _Container:
    """Documents by ``(tenant_id, user_id, id)``; a point operation finds one only under the partition it names."""

    def __init__(self, docs: Iterable[dict[str, Any]] = ()) -> None:
        self.docs: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.queries: list[_Query] = []
        self.reads: list[tuple[str, list[str]]] = []
        for doc in docs:
            self.put(doc)

    def put(self, doc: dict[str, Any]) -> dict[str, Any]:
        self.docs[(doc["tenant_id"], doc["user_id"], doc["id"])] = copy.deepcopy(doc)
        return doc

    def query_items(self, *, query: str, parameters: Any = None, partition_key: Any = None) -> _Rows:
        params = {p["name"]: p["value"] for p in parameters or []}
        self.queries.append(_Query(query, params, partition_key))
        rows = [
            copy.deepcopy(doc)
            for (tenant, user, _), doc in self.docs.items()
            if (partition_key is None or [tenant, user] == list(partition_key)) and _matches(doc, query, params)
        ]
        return _Rows(rows)

    async def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        self.reads.append((item, list(partition_key)))
        found = self.docs.get((*partition_key, item))
        if found is None:
            raise CosmosResourceNotFoundError(message=item)
        return copy.deepcopy(found)

    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        if (body["tenant_id"], body["user_id"], body["id"]) in self.docs:
            raise CosmosResourceExistsError(message=body["id"])
        return self.put(body)

    async def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return self.put(body)

    async def replace_item(self, *, item: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if (body["tenant_id"], body["user_id"], item) not in self.docs:
            raise CosmosResourceNotFoundError(message=item)
        return self.put(body)

    async def patch_item(self, *, item: str, partition_key: Any, patch_operations: list[dict[str, Any]]) -> dict:
        found = self.docs.get((*partition_key, item))
        if found is None:
            raise CosmosResourceNotFoundError(message=item)
        for operation in patch_operations:
            found[operation["path"].removeprefix("/")] = operation["value"]
        return copy.deepcopy(found)


class _Embeddings:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def _service(
    tenant: str = TENANT,
    *,
    memories: _Container | None = None,
    turns: _Container | None = None,
    summaries: _Container | None = None,
    embeddings: _Embeddings | None = None,
    replies: Iterable[dict[str, Any]] = (),
) -> AsyncPipelineService:
    containers = {
        ContainerKey.MEMORIES: memories if memories is not None else _Container(),
        ContainerKey.TURNS: turns if turns is not None else _Container(),
        ContainerKey.SUMMARIES: summaries if summaries is not None else _Container(),
    }
    store = MagicMock()
    store.mark_superseded = AsyncMock(return_value=True)
    service = AsyncPipelineService(
        store, MagicMock(), embeddings or _Embeddings(), containers=containers, tenant_id=tenant
    )
    service._run_prompty = AsyncMock(side_effect=[json.dumps(reply) for reply in replies])
    return service


def _turn(turn_id: str, tenant: str = TENANT, *, minute: int = 0) -> dict[str, Any]:
    return {
        "id": turn_id,
        "tenant_id": tenant,
        "user_id": USER,
        "thread_id": "t1",
        "type": "turn",
        "role": "user",
        "content": f"{tenant} turn {turn_id}: I like tea.",
        "created_at": f"2025-01-01T00:{minute:02d}:00+00:00",
    }


def _memory(memory_id: str, tenant: str = TENANT, kind: str = "fact", **fields: Any) -> dict[str, Any]:
    return {
        "id": memory_id,
        "tenant_id": tenant,
        "user_id": USER,
        "thread_id": "t1",
        "type": kind,
        "role": "system",
        "content": f"{tenant} {kind} {memory_id}",
        "created_at": "2025-01-01T00:00:00+00:00",
        **fields,
    }


def _fact(fact_id: str, tenant: str = TENANT) -> dict[str, Any]:
    """A fact as extraction builds it, ready to be persisted."""
    return _memory(
        fact_id,
        tenant,
        content_hash=compute_content_hash(f"{tenant} fact {fact_id}"),
        prompt_id="extract_memories-v3.prompty",
        prompt_version="v3",
        metadata={"category": "preference"},
    )


def _assert_owner_scoped(queries: list[_Query]) -> None:
    assert queries
    for sql, params, partition_key in queries:
        assert OWNER_FILTER in sql
        assert (params["@tenant_id"], params["@user_id"], partition_key) == (TENANT, USER, OWNER)


@pytest.mark.parametrize("tenant", ["", "  ", "acme/prod", "acme?", "a" * 300])
def test_a_pipeline_refuses_a_tenant_that_cannot_key_a_record(tenant: str) -> None:
    containers = {key: MagicMock() for key in ContainerKey}

    with pytest.raises(ValidationError):
        AsyncPipelineService(MagicMock(), MagicMock(), _Embeddings(), containers=containers, tenant_id=tenant)


def test_a_pipeline_works_in_the_tenant_of_its_store_unless_given_one() -> None:
    containers = {key: MagicMock() for key in ContainerKey}

    def build(store: Any, **options: Any) -> AsyncPipelineService:
        return AsyncPipelineService(store, MagicMock(), _Embeddings(), containers=containers, **options)

    store = SimpleNamespace(tenant_id=TENANT)
    assert build(store).tenant_id == TENANT
    assert build(store, tenant_id=TENANT).tenant_id == TENANT
    with pytest.raises(ConfigurationError, match=f"'{OTHER}' differs from the tenant of the store, '{TENANT}'"):
        build(store, tenant_id=OTHER)
    # A store that reports no tenant is not checked: the pipeline works in the default tenant unless given one.
    assert build(SimpleNamespace()).tenant_id == "default"
    assert build(SimpleNamespace(), tenant_id=TENANT).tenant_id == TENANT


# --- extraction -----------------------------------------------------------------------------------------------


async def test_extraction_reads_marks_and_writes_only_its_tenants_records(monkeypatch) -> None:
    monkeypatch.delenv("AMT_EXTRACT_MEMORIES_PROMPT", raising=False)
    turns = _Container([_turn("turn-a"), _turn("turn-g", OTHER)])
    memories, summaries = _Container(), _Container()
    fact = {"text": "The user likes tea.", "category": "preference", "confidence": 0.9, "salience": 0.8, "tags": []}
    service = _service(memories=memories, turns=turns, summaries=summaries, replies=[{"facts": [fact]}])

    counts = await service.extract_memories(USER, "t1")

    assert counts["fact_count"] == 1
    [stored] = memories.docs.values()
    assert (stored["tenant_id"], stored["user_id"], stored["thread_id"]) == (TENANT, USER, "t1")
    assert turns.docs[(TENANT, USER, "turn-a")]["extracted_at"]
    assert "extracted_at" not in turns.docs[(OTHER, USER, "turn-g")]
    _assert_owner_scoped(turns.queries)
    _assert_owner_scoped(memories.queries)
    assert len(turns.queries) == 2, "the unextracted turns, then the turns before them"
    assert summaries.reads == [("summary_u1_t1", OWNER)]


async def test_the_same_fact_is_stored_once_per_owner_and_thread() -> None:
    memories = _Container()
    fact = {"text": "The user likes tea.", "category": "preference", "confidence": 0.9, "salience": 0.8, "tags": []}
    service = _service(memories=memories, replies=[{"facts": [fact]}] * 4)

    async def extract_and_persist(user_id: str, thread_id: str) -> int:
        turn = {**_turn("turn-a"), "user_id": user_id, "thread_id": thread_id}
        extracted = await service.extract_memories_durable(user_id, thread_id, turns=[turn])
        return (await service.persist_extracted_memories(user_id, extracted))["fact_count"]

    created = [
        await extract_and_persist("u1", "t1"),
        await extract_and_persist("u1", "t1"),
        await extract_and_persist("u1", "t2"),
        await extract_and_persist("u2", "t1"),
    ]

    assert created == [1, 0, 1, 1], "only the same fact from the same thread of the same owner is a replay"
    ids = {(doc["user_id"], doc["thread_id"]): doc["id"] for doc in memories.docs.values()}
    assert sorted(ids) == [("u1", "t1"), ("u1", "t2"), ("u2", "t1")]
    assert len(set(ids.values())) == 3, "the owner is part of the id, not only of the partition"


async def test_persisting_extracted_memories_stamps_the_tenant_and_refuses_another_tenants_records() -> None:
    memories = _Container()
    service = _service(memories=memories)
    fact = {key: value for key, value in _fact("fact_1").items() if key != "tenant_id"}

    assert (await service.persist_extracted_memories(USER, {"facts": [fact]}))["fact_count"] == 1
    assert [doc["tenant_id"] for doc in memories.docs.values()] == [TENANT]

    with pytest.raises(ValidationError):
        await service.persist_extracted_memories(USER, {"facts": [_fact("fact_2", OTHER)]})
    assert list(memories.docs) == [(TENANT, USER, "fact_1")]


async def test_a_document_written_straight_to_a_container_carries_the_tenant_or_is_refused() -> None:
    memories = _Container()
    service = _service(memories=memories)
    bare = {"id": "episode-1", "user_id": USER, "thread_id": "t1", "type": "episodic", "content": "Tea."}

    await service._create_memory(bare)

    assert list(memories.docs) == [(TENANT, USER, "episode-1")]
    with pytest.raises(ValidationError):
        await service._create_memory({**bare, "id": "episode-2", "tenant_id": OTHER})
    assert list(memories.docs) == [(TENANT, USER, "episode-1")]


# --- episodes -------------------------------------------------------------------------------------------------


async def test_episodes_and_their_cursor_live_in_the_owners_partition_of_each_tenant(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    turns = _Container([_turn("turn-a", minute=1), _turn("turn-a2", minute=2), _turn("turn-g", OTHER, minute=3)])
    memories = _Container()
    episode = {"title": "Tea", "summary": "The user talked about tea.", "events": [], "participants": ["user"]}
    acme = _service(memories=memories, turns=turns, replies=[{"episodes": [episode]}])
    globex = _service(OTHER, memories=memories, turns=turns, replies=[{"episodes": [episode]}])

    assert await acme.extract_episodes(USER, "t1", flush=True) == {"episodes": 1}
    assert await acme.extract_episodes(USER, "t1", flush=True) == {"episodes": 0}, "its cursor is found again"
    assert await globex.extract_episodes(USER, "t1", flush=True) == {"episodes": 1}, "and not shared with globex"

    cursors = [doc for doc in memories.docs.values() if doc["type"] == "episode_cursor"]
    assert sorted((doc["tenant_id"], doc["last_episode_id"]) for doc in cursors) == [
        (TENANT, "turn-a2"),
        (OTHER, "turn-g"),
    ]
    episodes = [doc for doc in memories.docs.values() if doc["type"] == "episodic"]
    assert sorted(doc["tenant_id"] for doc in episodes) == [TENANT, OTHER]
    assert all((doc["tenant_id"], doc["user_id"], doc["id"]) in memories.docs for doc in [*cursors, *episodes])
    acme_queries = [query for query in turns.queries if query.params["@tenant_id"] == TENANT]
    _assert_owner_scoped(acme_queries)
    assert ("episode_cursor_u1_t1", OWNER) in memories.reads


# --- summaries ------------------------------------------------------------------------------------------------


async def test_a_thread_summary_is_read_and_written_in_the_owners_partition() -> None:
    turns = _Container([_turn("turn-a"), _turn("turn-g", OTHER)])
    summaries = _Container()
    reply = {"overview": "Tea.", "topics": []}
    acme = _service(turns=turns, summaries=summaries, replies=[reply])
    globex = _service(OTHER, turns=turns, summaries=summaries, replies=[reply])

    first = await acme.generate_thread_summary(USER, "t1")
    second = await globex.generate_thread_summary(USER, "t1")

    assert (first["tenant_id"], second["tenant_id"]) == (TENANT, OTHER)
    assert first["id"] == second["id"] == "summary_u1_t1"
    assert set(summaries.docs) == {(TENANT, USER, "summary_u1_t1"), (OTHER, USER, "summary_u1_t1")}
    assert summaries.reads == [("summary_u1_t1", OWNER), ("summary_u1_t1", [OTHER, USER])]
    assert [(query.params["@tenant_id"], query.partition_key) for query in turns.queries] == [
        (TENANT, OWNER),
        (OTHER, [OTHER, USER]),
    ]
    assert all(OWNER_FILTER in query.sql for query in turns.queries)
    assert "acme turn" in acme._run_prompty.call_args.kwargs["inputs"]["transcript"]
    assert "globex turn" not in acme._run_prompty.call_args.kwargs["inputs"]["transcript"]


async def test_a_user_summary_is_built_from_and_written_to_the_owners_partition() -> None:
    memories = _Container([_memory("f-a"), _memory("f-g", OTHER)])
    summaries = _Container([_memory("summary_u1_t1", kind="thread_summary"), _memory("s-g", OTHER, "thread_summary")])
    service = _service(memories=memories, summaries=summaries, replies=[{"key_facts": ["Likes tea."], "topics": []}])

    stored = await service.generate_user_summary(USER)

    assert (stored["tenant_id"], stored["user_id"], stored["thread_id"]) == (TENANT, USER, "__user_summary__")
    assert (TENANT, USER, "user_summary_u1") in summaries.docs and len(summaries.docs) == 3
    assert summaries.reads == [("user_summary_u1", OWNER)]
    _assert_owner_scoped(memories.queries)
    _assert_owner_scoped(summaries.queries)
    transcript = service._run_prompty.call_args.kwargs["inputs"]["transcript"]
    assert "acme fact f-a" in transcript and "acme thread_summary summary_u1_t1" in transcript
    assert "globex" not in transcript


@pytest.mark.parametrize(
    ("persist", "summary_id", "kind"),
    [
        pytest.param(
            lambda service, doc: service.persist_thread_summary(USER, "t1", doc),
            "summary_u1_t1",
            "thread_summary",
            id="thread-summary",
        ),
        pytest.param(
            lambda service, doc: service.persist_user_summary(USER, doc),
            "user_summary_u1",
            "user_summary",
            id="user-summary",
        ),
    ],
)
async def test_a_summary_naming_another_tenant_is_refused(persist, summary_id: str, kind: str) -> None:
    summaries = _Container()
    service = _service(summaries=summaries)
    doc = _memory(summary_id, OTHER, kind, metadata={"thread_ids": ["t1"]})

    with pytest.raises(ValidationError):
        await persist(service, doc)

    assert summaries.docs == {}
    stored = await persist(service, {key: value for key, value in doc.items() if key != "tenant_id"})
    assert stored["tenant_id"] == TENANT and list(summaries.docs) == [(TENANT, USER, summary_id)]


async def _persist_summary(
    service: AsyncPipelineService, kind: str, doc: dict[str, Any], user_id: str = USER, thread_id: str = "t1"
) -> dict[str, Any]:
    if kind == "thread_summary":
        return await service.persist_thread_summary(user_id, thread_id, doc)
    return await service.persist_user_summary(user_id, doc)


@pytest.mark.parametrize(
    ("kind", "summary_id", "thread_id"),
    [("thread_summary", "summary_u1_t1", "t1"), ("user_summary", "user_summary_u1", "__user_summary__")],
)
async def test_a_summary_without_an_owner_is_stored_for_the_owner_it_is_persisted_for(
    kind: str, summary_id: str, thread_id: str
) -> None:
    summaries = _Container()
    service = _service(summaries=summaries)
    doc = {"type": kind, "content": "Tea.", "metadata": {"thread_ids": ["t1"]}}
    original = copy.deepcopy(doc)

    stored = await _persist_summary(service, kind, doc)

    assert (stored["tenant_id"], stored["user_id"], stored["thread_id"], stored["id"]) == (
        TENANT,
        USER,
        thread_id,
        summary_id,
    )
    assert list(summaries.docs) == [(TENANT, USER, summary_id)]
    assert doc == original, "the caller's dict is not rewritten"


VALID_IDS = {"user_id": USER, "thread_id": "t1"}


@pytest.mark.parametrize(
    ("kind", "user_id", "thread_id", "fields", "problem"),
    [
        # The document names no owner, so the tenant is what is refused.
        pytest.param("thread_summary", USER, "t1", {"tenant_id": OTHER}, "different tenant", id="thread-tenant"),
        pytest.param("user_summary", USER, "t1", {"tenant_id": OTHER}, "different tenant", id="user-tenant"),
        # The document names a valid owner and thread of its own, so what is refused is the argument.
        pytest.param("thread_summary", "bad/id", "t1", VALID_IDS, "user_id", id="thread-owner"),
        pytest.param("thread_summary", "u1 ", "t1", VALID_IDS, "user_id", id="thread-owner-space"),
        pytest.param("thread_summary", USER, "t1 ", VALID_IDS, "thread_id", id="thread-thread-space"),
        pytest.param("user_summary", "bad/id", "t1", VALID_IDS, "user_id", id="user-owner"),
        pytest.param("user_summary", "u1 ", "t1", VALID_IDS, "user_id", id="user-owner-space"),
    ],
)
async def test_a_summary_that_cannot_be_stored_is_refused_before_its_content_is_embedded(
    kind: str, user_id: str, thread_id: str, fields: dict[str, Any], problem: str
) -> None:
    summaries, embeddings = _Container(), _Embeddings()
    service = _service(summaries=summaries, embeddings=embeddings)
    doc = {"type": kind, "content": "Tea.", "metadata": {"thread_ids": ["t1"]}, **fields}

    with pytest.raises(ValidationError, match=problem):
        await _persist_summary(service, kind, doc, user_id, thread_id)

    assert embeddings.calls == [] and summaries.docs == {}


# --- reconcile ------------------------------------------------------------------------------------------------


async def test_a_conflict_link_is_written_with_the_tenant_to_the_owners_partition() -> None:
    memories = _Container()
    service = _service(memories=memories)
    older = _memory("f1", created_at="2026-01-01")
    newer = _memory("f2", created_at="2026-02-01")
    verdict = Verdict(candidate=older, similarity=0.9, relation="update", seed_missing=(), candidate_missing=())

    assert await service._link_conflicts(USER, [(newer, [verdict])], {}) == 1

    [(tenant, user, link_id)] = memories.docs
    link = memories.docs[(tenant, user, link_id)]
    assert (tenant, user) == (TENANT, USER)
    assert (link["type"], link["tenant_id"], link["thread_id"]) == ("conflict_link", TENANT, "__conflicts__")
