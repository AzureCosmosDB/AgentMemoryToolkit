from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._partitioning import DEFAULT_TENANT_ID
from azure.cosmos.agent_memory.exceptions import ConfigurationError, ValidationError
from azure.cosmos.agent_memory.services.pipeline import PipelineService, _StoreContainerAdapter


class FakeLLMService:
    """Test helper exposing chat_client + embeddings_client pair.

    Mirrors the responses+call-tracking shape the original LLMService fake
    used so the existing assertions keep working with minimal churn.
    """

    def __init__(self, responses: list[dict[str, Any] | str]):
        self.responses = list(responses)
        self.chat_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.embed_calls: list[list[str]] = []
        self.embed_one_calls: list[str] = []

        outer = self

        class _Chat:
            def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
                outer.chat_calls.append((messages, opts))
                response = outer.responses.pop(0)
                if isinstance(response, str):
                    return response
                return json.dumps(response)

        class _Embed:
            def generate_batch(self, texts: list[str]) -> list[list[float]]:
                outer.embed_calls.append(list(texts))
                return [[float(i)] for i, _ in enumerate(texts, start=1)]

            def generate(self, text: str) -> list[float]:
                outer.embed_one_calls.append(text)
                return [0.42]

        self.chat_client = _Chat()
        self.embeddings_client = _Embed()


class FakeStore:
    """A container. Every query and point read is recorded (``queries``, ``reads``) with the partition it named; a
    query that names the tenant only answers with that tenant's records."""

    def __init__(self, docs: list[dict[str, Any]] | None = None):
        self.docs = [dict(doc) for doc in (docs or [])]
        self.upserts: list[dict[str, Any]] = []
        self.supersede_calls: list[tuple[str, str, str]] = []
        self.queries: list[tuple[str, dict[str, Any], Any]] = []
        self.reads: list[tuple[str, Any]] = []

    def query(
        self,
        sql: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: Any = None,
        cross_partition: bool = False,
    ) -> list[dict[str, Any]]:
        del cross_partition
        params = {p["name"]: p["value"] for p in (parameters or [])}
        self.queries.append((sql, params, partition_key))
        docs = [dict(doc) for doc in self.docs]
        if "@tenant_id" in params:
            docs = [doc for doc in docs if doc.get("tenant_id", params["@tenant_id"]) == params["@tenant_id"]]
        if "@user_id" in params:
            docs = [doc for doc in docs if doc.get("user_id") == params["@user_id"]]
        if "@thread_id" in params:
            docs = [doc for doc in docs if doc.get("thread_id") == params["@thread_id"]]
        if "@type" in params:
            docs = [doc for doc in docs if doc.get("type") == params["@type"]]
        if "@memory_type" in params:
            docs = [doc for doc in docs if doc.get("type") == params["@memory_type"]]
        if "c.type IN" in sql:
            types = {value for name, value in params.items() if name.startswith("@mtype")}
            docs = [doc for doc in docs if doc.get("type") in types]
        if "c.type = 'fact'" in sql:
            docs = [doc for doc in docs if doc.get("type") == "fact"]
        if "c.type != 'thread_summary'" in sql:
            docs = [doc for doc in docs if doc.get("type") != "thread_summary"]
        if "c.type != 'user_summary'" in sql:
            docs = [doc for doc in docs if doc.get("type") != "user_summary"]
        if "@id" in params:
            docs = [doc for doc in docs if doc.get("id") == params["@id"]]
        if "@category" in params:
            docs = [doc for doc in docs if doc.get("metadata", {}).get("category") == params["@category"]]
        if "@predicate" in params:
            docs = [doc for doc in docs if doc.get("metadata", {}).get("predicate") == params["@predicate"]]
        if "c.status='active'" in sql:
            docs = [doc for doc in docs if doc.get("status") == "active"]
        if "superseded_by" in sql:
            docs = [doc for doc in docs if not doc.get("superseded_by")]
        if "IS_DEFINED(c.lessons)" in sql:
            docs = [
                doc
                for doc in docs
                if isinstance(doc.get("lessons"), list)
                and any(isinstance(lesson, str) and lesson.strip() for lesson in doc.get("lessons", []))
            ]
        if "source_memory_ids" not in sql and "ORDER BY c.created_at DESC" in sql:
            docs.sort(key=lambda doc: doc.get("created_at", ""), reverse=True)
        elif "ORDER BY c.version DESC" in sql:
            docs.sort(key=lambda doc: int(doc.get("version") or 0), reverse=True)
        elif "ORDER BY c._ts DESC" in sql:
            docs.sort(key=lambda doc: int(doc.get("_ts") or 0), reverse=True)
        return docs

    def read_item(self, item_id: str, partition_key: Any) -> dict[str, Any]:
        self.reads.append((item_id, partition_key))
        for doc in self.docs:
            if doc.get("id") == item_id and doc.get("tenant_id", partition_key[0]) == partition_key[0]:
                return dict(doc)
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        raise CosmosResourceNotFoundError(message=f"not found: {item_id}")

    def upsert_memory(self, record: dict[str, Any]) -> dict[str, Any]:
        body = dict(record)
        self.upserts.append(body)
        self.docs = [doc for doc in self.docs if doc.get("id") != body.get("id")]
        self.docs.append(body)
        return body

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        if any(doc.get("id") == body.get("id") for doc in self.docs):
            raise CosmosResourceExistsError(message="conflict")
        created = dict(body)
        self.upserts.append(created)
        self.docs.append(created)
        return created

    def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        self.supersede_calls.append((old_doc["id"], superseder_id, reason))
        for doc in self.docs:
            if doc.get("id") == old_doc["id"]:
                doc["superseded_by"] = superseder_id
                doc["supersede_reason"] = reason
                doc["superseded_at"] = "2025-01-02T00:00:00+00:00"
                return True
        return False


def _containers_for_store(
    memories_store: FakeStore,
    *,
    turns_store: FakeStore | None = None,
    summaries_store: FakeStore | None = None,
) -> dict[ContainerKey, _StoreContainerAdapter]:
    turns_store = turns_store or FakeStore()
    summaries_store = summaries_store or FakeStore()
    return {
        ContainerKey.TURNS: _StoreContainerAdapter(turns_store, ContainerKey.TURNS),
        ContainerKey.MEMORIES: _StoreContainerAdapter(memories_store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _StoreContainerAdapter(summaries_store, ContainerKey.SUMMARIES),
    }


def _pipeline(
    memories_store: FakeStore,
    llm: FakeLLMService,
    *,
    turns_store: FakeStore | None = None,
    summaries_store: FakeStore | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> PipelineService:
    return PipelineService(
        memories_store,
        llm.chat_client,
        llm.embeddings_client,
        containers=_containers_for_store(memories_store, turns_store=turns_store, summaries_store=summaries_store),
        tenant_id=tenant_id,
    )


def _turn(content: str = "I prefer dark mode.") -> dict[str, Any]:
    return {
        "id": "turn1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": content,
        "created_at": "2025-01-01T00:00:00+00:00",
    }


def _fact(fid: str, content: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": fid,
        "user_id": "u1",
        "thread_id": extra.get("thread_id", "t1"),
        "role": "system",
        "type": "fact",
        "content": content,
        "confidence": extra.get("confidence", 0.8),
        "salience": extra.get("salience", 0.6),
        "metadata": extra.get("metadata", {"category": "preference"}),
        "tags": extra.get("tags", ["sys:fact"]),
        "created_at": extra.get("created_at", "2025-01-01T00:00:00+00:00"),
        "_etag": extra.get("etag", f"etag-{fid}"),
        "_ts": extra.get("ts", 1),
    }


def test_extract_memories_happy_path_writes_fact_only() -> None:
    store = FakeStore()
    turns_store = FakeStore([_turn("I prefer dark mode and learned CI needs retries.")])
    llm = FakeLLMService(
        [
            {
                "facts": [
                    {
                        "text": "The user prefers dark mode.",
                        "action": "ADD",
                        "category": "preference",
                        "confidence": 0.9,
                        "salience": 0.7,
                        "tags": ["ui"],
                    }
                ],
                "episodic": [],
            }
        ]
    )

    result = _pipeline(store, llm, turns_store=turns_store).extract_memories("u1", "t1")

    assert result["fact_count"] == 1
    assert result["episodic_count"] == 0
    assert result["updated_count"] == 0
    assert [doc["type"] for doc in store.upserts] == ["fact"]
    assert set(store.upserts[0]["tags"]) == {"sys:fact", "sys:auto-extracted", "topic:ui"}
    assert llm.chat_calls
    assert llm.embed_calls == [["The user prefers dark mode."]]


def test_extract_memories_creates_new_fact_without_superseding() -> None:
    # Extract does not detect updates or conflicts - it just adds new facts.
    # Reconcile links facts that conflict later, not at extract time.
    old = _fact("old_fact", "The user prefers light mode.")
    store = FakeStore([old])
    turns_store = FakeStore([_turn("Actually, I prefer dark mode now.")])
    llm = FakeLLMService(
        [
            {
                "facts": [
                    {
                        "text": "The user prefers dark mode.",
                        "category": "preference",
                    }
                ]
            }
        ]
    )

    result = _pipeline(store, llm, turns_store=turns_store).extract_memories("u1", "t1")

    assert result["fact_count"] == 1
    assert result["contradicted_count"] == 0
    assert store.supersede_calls == []
    old_doc = next(doc for doc in store.docs if doc["id"] == "old_fact")
    assert "superseded_by" not in old_doc


def test_reconcile_memories_links_facts_that_conflict_and_retires_neither() -> None:
    store = FakeStore(
        [
            _fact("f1", "User likes coffee", ts=1),
            _fact("f2", "User prefers coffee", ts=2),
            {**_fact("f3", "User is vegetarian", ts=3), "embedding": [1.0, 0.0, 0.0]},
            {**_fact("f4", "User eats steak", ts=4), "embedding": [0.9, 0.3, 0.0]},
        ]
    )
    llm = FakeLLMService(
        [
            {
                "candidates": [
                    {
                        "label": "C1",
                        "relation": "inconsistent",
                        "seed_details_missing_from_candidate": [],
                        "candidate_details_missing_from_seed": [],
                    }
                ]
            }
        ]
    )

    result = _pipeline(store, llm).reconcile_memories("u1", n=4)

    # f1/f2 have no embeddings, so the dedup pass has no neighbours to compare them with; f3 and f4 are compared
    # once, and found to conflict: both stay active and one link names them.
    assert (result["kept"], result["merged"], result["conflicts"], result["checked"], result["compared"]) == (
        4,
        0,
        1,
        4,
        1,
    )
    assert len(llm.chat_calls) == 1 and store.supersede_calls == []
    assert not [doc for doc in store.docs if doc.get("superseded_by")]
    [link] = [doc for doc in store.docs if doc.get("type") == "conflict_link"]
    assert ({link["first"], link["later"]}, link["kind"]) == ({"f3", "f4"}, "inconsistent")


def test_thread_summary_persists_to_summaries_container() -> None:
    turns_container = MagicMock()
    memories_container = MagicMock()
    summaries_container = MagicMock()
    summaries_container.upsert_item.side_effect = lambda body: body
    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    embeddings = FakeLLMService([]).embeddings_client
    service = PipelineService(FakeStore(), object(), embeddings, containers=containers, tenant_id="acme")
    summary_doc = {
        "id": "summary_u1_t1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "system",
        "type": "thread_summary",
        "content": "Thread discussed retries.",
        "salience": 1.0,
        "tags": ["sys:summary"],
        "prompt_id": "summarize.prompty",
        "prompt_version": "v1",
        "metadata": {"structured_summary": {"overview": "Thread discussed retries."}},
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }

    stored = service.persist_thread_summary("u1", "t1", summary_doc)

    assert stored["type"] == "thread_summary"
    summaries_container.upsert_item.assert_called_once()
    written = summaries_container.upsert_item.call_args.kwargs["body"]
    assert (written["id"], written["tenant_id"]) == ("summary_u1_t1", "acme")
    memories_container.method_calls == []
    turns_container.method_calls == []

    with pytest.raises(ValidationError, match="different tenant"):
        service.persist_thread_summary("u1", "t1", {**summary_doc, "tenant_id": "other"})
    summaries_container.upsert_item.assert_called_once()


def _pipeline_over(store: Any, **options: Any) -> PipelineService:
    return PipelineService(store, MagicMock(), MagicMock(), containers=_containers_for_store(FakeStore()), **options)


def test_a_pipeline_works_in_the_tenant_of_its_store_unless_given_one() -> None:
    store = SimpleNamespace(tenant_id="acme")

    assert _pipeline_over(store).tenant_id == "acme"
    assert _pipeline_over(store, tenant_id="acme").tenant_id == "acme"
    with pytest.raises(ConfigurationError, match="'globex' differs from the tenant of the store, 'acme'"):
        _pipeline_over(store, tenant_id="globex")


def test_a_pipeline_over_a_store_that_reports_no_tenant_works_in_the_default_tenant() -> None:
    assert _pipeline_over(SimpleNamespace()).tenant_id == DEFAULT_TENANT_ID
    assert _pipeline_over(SimpleNamespace(), tenant_id="acme").tenant_id == "acme"


def _persist_summary(
    service: PipelineService, kind: str, doc: dict[str, Any], user_id: str = "u1", thread_id: str = "t1"
) -> dict[str, Any]:
    if kind == "thread_summary":
        return service.persist_thread_summary(user_id, thread_id, doc)
    return service.persist_user_summary(user_id, doc)


@pytest.mark.parametrize(
    ("kind", "summary_id", "thread_id"),
    [("thread_summary", "summary_u1_t1", "t1"), ("user_summary", "user_summary_u1", "__user_summary__")],
)
def test_a_summary_without_an_owner_is_stored_for_the_owner_it_is_persisted_for(
    kind: str, summary_id: str, thread_id: str
) -> None:
    summaries_store = FakeStore()
    service = _pipeline(FakeStore(), FakeLLMService([]), summaries_store=summaries_store, tenant_id="acme")
    doc = {"type": kind, "content": "Thread discussed retries.", "metadata": {"thread_ids": ["t1"]}}
    original = copy.deepcopy(doc)

    stored = _persist_summary(service, kind, doc)

    assert (stored["tenant_id"], stored["user_id"], stored["thread_id"], stored["id"]) == (
        "acme",
        "u1",
        thread_id,
        summary_id,
    )
    assert summaries_store.upserts == [stored]
    assert doc == original, "the caller's dict is not rewritten"


VALID_IDS = {"user_id": "u1", "thread_id": "t1"}


@pytest.mark.parametrize(
    ("kind", "user_id", "thread_id", "fields", "problem"),
    [
        # The document names no owner, so the tenant is what is refused.
        pytest.param("thread_summary", "u1", "t1", {"tenant_id": "other"}, "different tenant", id="thread-tenant"),
        pytest.param("user_summary", "u1", "t1", {"tenant_id": "other"}, "different tenant", id="user-tenant"),
        # The document names a valid owner and thread of its own, so what is refused is the argument.
        pytest.param("thread_summary", "bad/id", "t1", VALID_IDS, "user_id", id="thread-owner"),
        pytest.param("thread_summary", "u1 ", "t1", VALID_IDS, "user_id", id="thread-owner-space"),
        pytest.param("thread_summary", "u1", "t1 ", VALID_IDS, "thread_id", id="thread-thread-space"),
        pytest.param("user_summary", "bad/id", "t1", VALID_IDS, "user_id", id="user-owner"),
        pytest.param("user_summary", "u1 ", "t1", VALID_IDS, "user_id", id="user-owner-space"),
    ],
)
def test_a_summary_that_cannot_be_stored_is_refused_before_its_content_is_embedded(
    kind: str, user_id: str, thread_id: str, fields: dict[str, Any], problem: str
) -> None:
    llm, summaries_store = FakeLLMService([]), FakeStore()
    service = _pipeline(FakeStore(), llm, summaries_store=summaries_store, tenant_id="acme")
    doc = {"type": kind, "content": "Thread discussed retries.", "metadata": {"thread_ids": ["t1"]}, **fields}

    with pytest.raises(ValidationError, match=problem):
        _persist_summary(service, kind, doc, user_id, thread_id)

    assert llm.embed_one_calls == [] and summaries_store.upserts == []


def test_generate_thread_and_user_summary_basic_shape() -> None:
    store = FakeStore()
    turns_store = FakeStore([_turn("We decided to add retries.")])
    summaries_store = FakeStore()
    llm = FakeLLMService(
        [
            {"overview": "Thread discussed retries.", "topics": ["retries"], "decisions": ["Add retries"]},
            {"key_facts": ["The project needs retries."], "open_questions": []},
        ]
    )
    service = _pipeline(store, llm, turns_store=turns_store, summaries_store=summaries_store)

    thread_summary = service.generate_thread_summary("u1", "t1")
    user_summary = service.generate_user_summary("u1", thread_ids=["t1"])

    assert thread_summary["id"] == "summary_u1_t1"
    assert thread_summary["type"] == "thread_summary"
    assert thread_summary["content"] == "Thread discussed retries."
    assert "topic:retries" in thread_summary["tags"]
    assert user_summary["id"] == "user_summary_u1"
    assert user_summary["type"] == "user_summary"
    assert user_summary["content"] == "The project needs retries."
    assert user_summary["metadata"]["thread_ids"] == ["t1"]
    assert llm.embed_one_calls == ["Thread discussed retries.", "The project needs retries."]


def test_summaries_are_read_and_written_in_the_owners_partition_of_the_tenant() -> None:
    owner = [{"tenant_id": "acme"}, {"tenant_id": "other"}]
    turns_store = FakeStore(
        [
            {**_turn("We decided to add retries."), **owner[0]},
            {**_turn("Another tenant's plan."), **owner[1]},
        ]
    )
    memories = FakeStore([{**_fact("f1", "The project needs retries."), **owner[0]}])
    memories.docs.append({**_fact("f2", "Another tenant's fact."), **owner[1]})
    summaries_store = FakeStore([{"id": "summary_u1_t1", "type": "thread_summary", "tenant_id": "other"}])
    llm = FakeLLMService(
        [
            {"overview": "Thread discussed retries.", "topics": ["retries"]},
            {"key_facts": ["The project needs retries."], "open_questions": []},
        ]
    )
    service = _pipeline(memories, llm, turns_store=turns_store, summaries_store=summaries_store, tenant_id="acme")

    thread_summary = service.generate_thread_summary("u1", "t1")
    user_summary = service.generate_user_summary("u1")

    assert "Another tenant" not in str(llm.chat_calls)
    assert (thread_summary["tenant_id"], user_summary["tenant_id"]) == ("acme", "acme")
    for store in (turns_store, memories, summaries_store):
        assert store.queries
        for sql, params, partition_key in store.queries:
            assert "c.tenant_id = @tenant_id AND c.user_id = @user_id" in sql
            assert (params["@tenant_id"], params["@user_id"]) == ("acme", "u1")
            assert partition_key == ["acme", "u1"]
    assert summaries_store.reads == [("summary_u1_t1", ["acme", "u1"]), ("user_summary_u1", ["acme", "u1"])]
    assert [doc["id"] for doc in summaries_store.upserts] == ["summary_u1_t1", "user_summary_u1"]
    assert {doc["tenant_id"] for doc in summaries_store.upserts} == {"acme"}


def test_the_threads_of_one_owner_have_summaries_with_different_ids() -> None:
    turns_store = FakeStore([_turn("In thread one."), {**_turn("In thread two."), "id": "turn2", "thread_id": "t2"}])
    summaries_store = FakeStore()
    llm = FakeLLMService([{"overview": "One."}, {"overview": "Two."}])
    service = _pipeline(FakeStore(), llm, turns_store=turns_store, summaries_store=summaries_store)

    first = service.generate_thread_summary("u1", "t1")
    second = service.generate_thread_summary("u1", "t2")

    assert (first["id"], second["id"]) == ("summary_u1_t1", "summary_u1_t2")
    assert {doc["id"] for doc in summaries_store.docs} == {"summary_u1_t1", "summary_u1_t2"}


def test_build_procedural_context_requires_user_id() -> None:
    with pytest.raises(Exception, match="user_id is required"):
        fake = FakeLLMService([])
        _pipeline(FakeStore(), fake).build_procedural_context("")


def test_a_pipeline_without_a_tenant_works_in_the_default_one() -> None:
    llm = FakeLLMService([])
    memories = FakeStore()
    service = PipelineService(
        memories,
        llm.chat_client,
        llm.embeddings_client,
        containers=_containers_for_store(memories),
    )

    service.build_procedural_context("u1")

    (_, params, partition_key), *_ = memories.queries
    assert (params["@tenant_id"], partition_key) == (DEFAULT_TENANT_ID, [DEFAULT_TENANT_ID, "u1"])


@pytest.mark.parametrize("tenant_id", ["", "  ", "a/b", "a?b", "a#b", "a\\b", "a" * 257, 7])
def test_a_pipeline_refuses_a_tenant_id_that_is_not_valid(tenant_id) -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        _pipeline(FakeStore(), FakeLLMService([]), tenant_id=tenant_id)
