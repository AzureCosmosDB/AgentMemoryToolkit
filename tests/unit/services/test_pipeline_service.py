from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.services.pipeline import PipelineService, _StoreContainerAdapter


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
    def __init__(self, docs: list[dict[str, Any]] | None = None):
        self.docs = [dict(doc) for doc in (docs or [])]
        self.upserts: list[dict[str, Any]] = []
        self.supersede_calls: list[tuple[str, str, str]] = []

    def query(
        self,
        sql: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: Any = None,
        cross_partition: bool = False,
    ) -> list[dict[str, Any]]:
        del partition_key, cross_partition
        params = {p["name"]: p["value"] for p in (parameters or [])}
        docs = [dict(doc) for doc in self.docs]
        uid = params.get("@user_id", params.get("@uid"))
        if uid is not None:
            docs = [doc for doc in docs if doc.get("user_id") == uid]
        if "@thread_id" in params:
            docs = [doc for doc in docs if doc.get("thread_id") == params["@thread_id"]]
        if "@type" in params:
            docs = [doc for doc in docs if doc.get("type") == params["@type"]]
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
        if "superseded_by" in sql:
            docs = [doc for doc in docs if not doc.get("superseded_by")]
        if "IS_DEFINED(c.metadata.lesson)" in sql:
            docs = [doc for doc in docs if doc.get("metadata", {}).get("lesson")]
        if "source_memory_ids" not in sql and "ORDER BY c.created_at DESC" in sql:
            docs.sort(key=lambda doc: doc.get("created_at", ""), reverse=True)
        elif "ORDER BY c.version DESC" in sql:
            docs.sort(key=lambda doc: int(doc.get("version") or 0), reverse=True)
        elif "ORDER BY c._ts DESC" in sql:
            docs.sort(key=lambda doc: int(doc.get("_ts") or 0), reverse=True)
        return docs

    def read_item(self, item_id: str, partition_key: Any) -> dict[str, Any]:
        del partition_key
        for doc in self.docs:
            if doc.get("id") == item_id:
                return dict(doc)
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        raise CosmosResourceNotFoundError(message=f"not found: {item_id}")

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        body = dict(record)
        self.upserts.append(body)
        self.docs = [doc for doc in self.docs if doc.get("id") != body.get("id")]
        self.docs.append(body)
        return body

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
) -> PipelineService:
    return PipelineService(
        memories_store,
        llm.chat_client,
        llm.embeddings_client,
        containers=_containers_for_store(memories_store, turns_store=turns_store, summaries_store=summaries_store),
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


def test_extract_memories_happy_path_writes_fact_and_episodic() -> None:
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
                "episodic": [
                    {
                        "scope_type": "project",
                        "scope_value": "CI",
                        "situation": "CI tests flaked intermittently",
                        "action_taken": "Added retries",
                        "outcome": "Tests stabilized",
                        "lesson": "Use retries for flaky CI tests.",
                        "confidence": 0.8,
                    }
                ],
            }
        ]
    )

    result = _pipeline(store, llm, turns_store=turns_store).extract_memories("u1", "t1")

    assert result["fact_count"] == 1
    assert result["episodic_count"] == 1
    assert result["updated_count"] == 0
    assert [doc["type"] for doc in store.upserts] == ["fact", "episodic"]
    assert set(store.upserts[0]["tags"]) == {"sys:fact", "sys:auto-extracted", "topic:ui"}
    assert llm.chat_calls
    assert llm.embed_calls == [
        [
            "The user prefers dark mode.",
            "CI tests flaked intermittently → Added retries → Tests stabilized",
        ]
    ]


def test_extract_memories_contradict_supersedes_existing_fact() -> None:
    old = _fact("old_fact", "The user prefers light mode.")
    store = FakeStore([old])
    turns_store = FakeStore([_turn("Actually, I prefer dark mode now.")])
    llm = FakeLLMService(
        [
            {
                "facts": [
                    {
                        "text": "The user prefers dark mode.",
                        "action": "CONTRADICT",
                        "supersedes_id": "old_fact",
                        "category": "preference",
                    }
                ]
            }
        ]
    )

    result = _pipeline(store, llm, turns_store=turns_store).extract_memories("u1", "t1")

    assert result["fact_count"] == 1
    assert result["contradicted_count"] == 1
    assert store.supersede_calls[0][2] == "contradict"
    old_doc = next(doc for doc in store.docs if doc["id"] == "old_fact")
    assert old_doc["superseded_by"] == store.upserts[0]["id"]
    assert old_doc["supersede_reason"] == "contradict"
    assert old_doc["superseded_at"]


def test_synthesize_procedural_produces_procedural_memory() -> None:
    store = FakeStore(
        [
            _fact("f1", "Always use bullet points.", salience=0.9),
            {
                "id": "e1",
                "user_id": "u1",
                "thread_id": "t1",
                "role": "system",
                "type": "episodic",
                "content": "Past project",
                "metadata": {"lesson": "Keep examples small."},
                "salience": 0.8,
                "created_at": "2025-01-02T00:00:00+00:00",
            },
        ]
    )
    llm = FakeLLMService([{"system_prompt": "Use concise bullet points."}])

    result = _pipeline(store, llm).synthesize_procedural("u1")

    assert result["status"] == "synthesized"
    proc = result["procedural"]
    assert proc["type"] == "procedural"
    assert proc["content"] == "Use concise bullet points."
    assert proc["source_fact_ids"] == ["f1"]
    assert proc["source_episodic_ids"] == ["e1"]
    assert store.upserts == [proc]


def test_reconcile_memories_returns_three_bucket_counts() -> None:
    store = FakeStore(
        [
            _fact("f1", "User likes coffee", ts=1),
            _fact("f2", "User prefers coffee", ts=2),
            _fact("f3", "User is vegetarian", ts=3),
            _fact("f4", "User eats steak", ts=4),
        ]
    )
    llm = FakeLLMService(
        [
            {
                "duplicate_groups": [
                    {"source_ids": ["f1", "f2"], "merged_content": "The user prefers coffee.", "confidence": 0.9}
                ],
                "contradicted_pairs": [{"winner_id": "f3", "loser_id": "f4", "reason": "newer evidence"}],
                "kept_ids": [],
            }
        ]
    )

    result = _pipeline(store, llm).reconcile_memories("u1", n=4)

    assert result == {"kept": 1, "merged": 2, "contradicted": 1}


def test_reconcile_memories_tombstones_losers_with_reasons() -> None:
    store = FakeStore(
        [
            _fact("f1", "User likes coffee", ts=1),
            _fact("f2", "User prefers coffee", ts=2),
            _fact("f3", "User is vegetarian", ts=3),
            _fact("f4", "User eats steak", ts=4),
        ]
    )
    llm = FakeLLMService(
        [
            {
                "duplicate_groups": [{"source_ids": ["f1", "f2"], "merged_content": "The user prefers coffee."}],
                "contradicted_pairs": [{"winner_id": "f3", "loser_id": "f4", "reason": "explicit correction"}],
                "kept_ids": [],
            }
        ]
    )

    _pipeline(store, llm).reconcile_memories("u1", n=4)

    by_id = {doc["id"]: doc for doc in store.docs}
    assert by_id["f1"]["supersede_reason"] == "duplicate"
    assert by_id["f2"]["supersede_reason"] == "duplicate"
    assert by_id["f4"]["supersede_reason"] == "contradict"
    assert by_id["f4"]["superseded_by"] == "f3"


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
    service = PipelineService(FakeStore(), object(), embeddings, containers=containers)
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
    assert summaries_container.upsert_item.call_args.kwargs["body"]["id"] == "summary_u1_t1"
    memories_container.method_calls == []
    turns_container.method_calls == []


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


def test_build_procedural_context_returns_active_procedural() -> None:
    store = FakeStore(
        [
            {
                "id": "proc_u1_1",
                "user_id": "u1",
                "thread_id": "__procedural__",
                "type": "procedural",
                "version": 1,
                "content": "Old prompt",
                "superseded_by": "proc_u1_2",
            },
            {
                "id": "proc_u1_2",
                "user_id": "u1",
                "thread_id": "__procedural__",
                "type": "procedural",
                "version": 2,
                "content": "Active prompt",
            },
        ]
    )

    fake = FakeLLMService([])
    result = _pipeline(store, fake).build_procedural_context("u1")
    assert result == "Active prompt"


def test_build_procedural_context_requires_user_id() -> None:
    with pytest.raises(Exception, match="user_id is required"):
        fake = FakeLLMService([])
        _pipeline(FakeStore(), fake).build_procedural_context("")
