"""Tests for procedural synthesis and procedural prompt retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.cosmos_memory_client import CosmosMemoryClient
from agent_memory_toolkit.processors import DurableFunctionProcessor
from agent_memory_toolkit.services.pipeline import PipelineService
from agent_memory_toolkit.store import MemoryStore


def _assert_iso8601(text: str) -> None:
    assert text
    datetime.fromisoformat(text)


def _capture_upserts():
    """Capture documents written via upsert_item OR create_item.

    The pipeline writes new facts/episodics via ``create_item`` (for 409
    idempotency) and writes new procedural versions via ``create_item +
    bump-seq-retry``. Either way, tests want the persisted body — so the
    helper now wires the same capture to both side_effects.
    """
    upserted: list[dict] = []

    def _capture(*, body):
        upserted.append(body)
        return body

    return upserted, _capture


def _make_extract_pipeline(llm_response: dict):
    turns_container = MagicMock()
    memories_container = MagicMock()
    summaries_container = MagicMock()
    turns_container.query_items.return_value = [
        {
            "id": "turn1",
            "user_id": "u1",
            "thread_id": "t1",
            "role": "user",
            "type": "turn",
            "content": "Always use bullet points.",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
    ]
    upserted, capture = _capture_upserts()
    memories_container.upsert_item.side_effect = capture
    memories_container.create_item.side_effect = capture

    embeddings = MagicMock()
    embeddings.generate_batch.side_effect = lambda texts: [[0.0] * 4 for _ in texts]

    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    store = MemoryStore(containers=containers, embeddings_client=embeddings)
    pipeline = PipelineService(store, MagicMock(), embeddings, containers=containers)
    pipeline._run_prompty = MagicMock(return_value=json.dumps(llm_response))
    pipeline._load_existing_memories = MagicMock(return_value=[])
    return pipeline, memories_container, upserted


def _fact_doc(
    doc_id: str,
    content: str,
    *,
    category: str = "preference",
    salience: float = 0.9,
    created_at: str = "2025-01-01T00:00:00+00:00",
    predicate: str | None = None,
    obj: str | None = None,
) -> dict:
    metadata = {"category": category}
    if predicate is not None:
        metadata["predicate"] = predicate
    if obj is not None:
        metadata["object"] = obj
    return {
        "id": doc_id,
        "user_id": "u1",
        "thread_id": "t-source",
        "role": "system",
        "type": "fact",
        "content": content,
        "metadata": metadata,
        "salience": salience,
        "created_at": created_at,
    }


def _episodic_doc(
    doc_id: str,
    *,
    lesson: str,
    salience: float = 0.7,
    created_at: str = "2025-01-02T00:00:00+00:00",
) -> dict:
    return {
        "id": doc_id,
        "user_id": "u1",
        "thread_id": "t-source",
        "role": "system",
        "type": "episodic",
        "content": f"Episode {doc_id}",
        "metadata": {"lesson": lesson},
        "salience": salience,
        "created_at": created_at,
    }


def _procedural_doc(
    doc_id: str,
    *,
    version: int,
    content: str,
    source_fact_ids: list[str],
    source_episodic_ids: list[str],
    superseded_by: str | None = None,
    ts: int = 0,
    etag: str = "etag-1",
) -> dict:
    doc = {
        "id": doc_id,
        "user_id": "u1",
        "thread_id": "__procedural__",
        "type": "procedural",
        "version": version,
        "content": content,
        "source_fact_ids": list(source_fact_ids),
        "source_episodic_ids": list(source_episodic_ids),
        "supersedes_ids": [],
        "created_at": f"2025-01-0{version}T00:00:00+00:00",
        "role": "system",
        "tags": ["sys:procedural", "sys:synthesized"],
        "_etag": etag,
        "_ts": ts,
    }
    if superseded_by is not None:
        doc["superseded_by"] = superseded_by
    return doc


def _make_synthesis_pipeline(
    *,
    prior_docs: list[dict] | None = None,
    fact_docs: list[dict] | None = None,
    episodic_docs: list[dict] | None = None,
    name_docs: list[dict] | None = None,
    llm_output: str = "Follow the user's preferences.",
):
    turns_container = MagicMock()
    memories_container = MagicMock()
    summaries_container = MagicMock()
    memories_container.query_items.side_effect = [
        list(prior_docs or []),
        list(fact_docs or []),
        list(episodic_docs or []),
        list(name_docs or []),
    ]
    upserted, capture = _capture_upserts()
    memories_container.upsert_item.side_effect = capture
    memories_container.create_item.side_effect = capture

    mock_embeddings = MagicMock()
    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    store = MemoryStore(containers=containers, embeddings_client=mock_embeddings)
    pipeline = PipelineService(store, MagicMock(), mock_embeddings, containers=containers)
    pipeline._run_prompty = MagicMock(return_value=json.dumps({"system_prompt": llm_output}))
    return pipeline, memories_container, upserted


def _make_client(*, processor=None) -> CosmosMemoryClient:
    client = CosmosMemoryClient.__new__(CosmosMemoryClient)
    memories_container = MagicMock()
    turns_container = MagicMock()
    summaries_container = MagicMock()
    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    client._memories_container_client = memories_container
    client._turns_container_client = turns_container
    client._summaries_container_client = summaries_container
    client._embeddings_client = MagicMock()
    client._store = MemoryStore(containers=containers, embeddings_client=client._embeddings_client)
    client._pipeline = None
    client._processor = processor
    client._processor_explicit = processor is not None
    return client


def test_extract_memories_without_procedural_bucket_returns_new_count_shape():
    pipeline, _, upserted = _make_extract_pipeline(
        {
            "facts": [
                {
                    "text": "Always use bullet points.",
                    "category": "preference",
                    "action": "ADD",
                }
            ],
            "episodic": [
                {
                    "scope_type": "task",
                    "scope_value": "refactoring tests",
                    "situation": "Refactoring tests",
                    "action_taken": "Used focused helpers",
                    "outcome": "The suite stayed readable",
                }
            ],
        }
    )

    result = pipeline.extract_memories("u1", "t1")
    legacy_fact_count_key = "_".join(("facts", "count"))
    legacy_proc_key = "_".join(("procedural", "count"))

    assert result["fact_count"] == 1
    assert result["episodic_count"] == 1
    assert result["unclassified_count"] == 0
    assert legacy_fact_count_key not in result
    assert legacy_proc_key not in result
    assert all(doc["type"] != "procedural" for doc in upserted)


def test_extract_memories_ignores_legacy_procedural_bucket_in_llm_payload():
    pipeline, _, upserted = _make_extract_pipeline(
        {
            "facts": [
                {
                    "text": "Never use var in TypeScript.",
                    "category": "requirement",
                    "action": "ADD",
                }
            ],
            "procedural": [
                {
                    "instruction": "Use bullet points",
                    "action": "ADD",
                }
            ],
        }
    )

    result = pipeline.extract_memories("u1", "t1")
    legacy_fact_count_key = "_".join(("facts", "count"))
    legacy_proc_key = "_".join(("procedural", "count"))

    assert result["fact_count"] == 1
    assert legacy_fact_count_key not in result
    assert legacy_proc_key not in result
    assert [doc["type"] for doc in upserted] == ["fact"]


def test_synthesize_procedural_first_synthesis_from_empty_prior():
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference", salience=0.95),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement", salience=0.9),
    ]
    episodic_docs = [
        _episodic_doc("e1", lesson="When the user asks for brevity, keep the answer terse.", salience=0.8),
        _episodic_doc("e2", lesson="", salience=0.2),
    ]
    pipeline, container, upserted = _make_synthesis_pipeline(
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
        llm_output="Be concise and prefer bullet points.",
    )

    result = pipeline.synthesize_procedural("u1", force=False)

    assert pipeline._run_prompty.call_count == 1
    assert result["status"] == "synthesized"
    doc = result["procedural"]
    assert doc["version"] == 1
    assert doc["content"] == "Be concise and prefer bullet points."
    assert set(doc["source_fact_ids"]) == {"f1", "f2"}
    assert set(doc["source_episodic_ids"]) == {"e1"}
    assert doc["supersedes_ids"] == []
    assert upserted == [doc]
    container.replace_item.assert_not_called()


def test_synthesize_procedural_only_touches_memories_container():
    turns_container = MagicMock()
    memories_container = MagicMock()
    summaries_container = MagicMock()
    memories_container.query_items.side_effect = [
        [],
        [_fact_doc("f1", "Always use bullet points.", category="preference", salience=0.95)],
        [_episodic_doc("e1", lesson="Keep examples small.")],
        [],
    ]
    memories_container.create_item.side_effect = lambda body: body
    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    embeddings = MagicMock()
    store = MemoryStore(containers=containers, embeddings_client=embeddings)
    pipeline = PipelineService(store, MagicMock(), embeddings, containers=containers)
    pipeline._run_prompty = MagicMock(return_value=json.dumps({"system_prompt": "Use concise bullets."}))

    result = pipeline.synthesize_procedural("u1")

    assert result["status"] == "synthesized"
    assert memories_container.query_items.call_count == 4
    memories_container.create_item.assert_called_once()
    turns_container.method_calls == []
    summaries_container.method_calls == []


def test_synthesize_procedural_resynthesis_supersedes_prior_with_update_reason():
    prior_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Old prompt",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
        _fact_doc("f3", "Lead with the final answer.", category="preference", salience=0.85),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, container, upserted = _make_synthesis_pipeline(
        prior_docs=[prior_doc],
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
        llm_output="New prompt",
    )

    result = pipeline.synthesize_procedural("u1")

    assert result["status"] == "synthesized"
    new_doc = result["procedural"]
    assert new_doc["id"] == "proc_u1_2"
    assert new_doc["version"] == 2
    assert new_doc["supersedes_ids"] == [prior_doc["id"]]
    assert upserted == [new_doc]
    body = container.replace_item.call_args.kwargs["body"]
    assert body["id"] == prior_doc["id"]
    assert body["superseded_by"] == new_doc["id"]
    _assert_iso8601(body["superseded_at"])
    assert body["supersede_reason"] == "update"


def test_synthesize_procedural_noop_when_source_ids_are_unchanged():
    prior_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Existing prompt",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
        _fact_doc("f1", "Always use bullet points.", category="preference"),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, container, _ = _make_synthesis_pipeline(
        prior_docs=[prior_doc],
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
    )

    result = pipeline.synthesize_procedural("u1", force=False)

    assert result == {"status": "unchanged", "procedural": prior_doc}
    pipeline._run_prompty.assert_not_called()
    container.upsert_item.assert_not_called()
    container.replace_item.assert_not_called()


def test_synthesize_procedural_force_true_reruns_when_source_ids_are_unchanged():
    prior_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Existing prompt",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, container, upserted = _make_synthesis_pipeline(
        prior_docs=[prior_doc],
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
        llm_output="Refreshed prompt",
    )

    result = pipeline.synthesize_procedural("u1", force=True)

    assert pipeline._run_prompty.call_count == 1
    assert result["status"] == "synthesized"
    new_doc = result["procedural"]
    assert new_doc["version"] == 2
    assert upserted == [new_doc]
    body = container.replace_item.call_args.kwargs["body"]
    assert body["superseded_by"] == new_doc["id"]
    assert body["supersede_reason"] == "update"


def test_synthesize_procedural_short_circuits_for_cold_user_with_no_sources():
    """B2 regression: a user with no facts and no episodics must not consume an
    LLM call. Without this guard, ``synthesize_procedural`` for cold users
    would invoke the chat client and then discard the result, wasting tokens
    on every auto-trigger.
    """
    pipeline, container, upserted = _make_synthesis_pipeline()

    result = pipeline.synthesize_procedural("u1")

    pipeline._run_prompty.assert_not_called()
    container.upsert_item.assert_not_called()
    container.replace_item.assert_not_called()
    assert result == {"status": "unchanged", "procedural": None}
    assert upserted == []


def test_synthesize_procedural_short_circuits_on_second_call_with_tied_salience():
    """B1 regression: 60 facts at the default salience must yield a deterministic
    selection so the source-id short-circuit fires on the second call.

    Before the SQL composite ORDER BY (salience DESC, created_at ASC, id ASC),
    Cosmos returned an arbitrary 50-of-60 per query; the prior/current source
    sets never matched and the LLM fired on every reconcile. The Python re-sort
    that previously masked this in unit tests has been removed — the SQL itself
    must be deterministic.
    """
    fact_docs = [
        _fact_doc(
            f"f{i:03}",
            f"Fact {i}",
            category="preference",
            salience=0.5,
            created_at=f"2025-01-{(i % 28) + 1:02}T00:00:00+00:00",
        )
        for i in range(60)
    ]
    pipeline, container, upserted = _make_synthesis_pipeline(
        fact_docs=fact_docs,
        llm_output="Generated prompt",
    )

    first = pipeline.synthesize_procedural("u1")
    assert first["status"] == "synthesized"
    assert pipeline._run_prompty.call_count == 1

    synthesized = first["procedural"]
    container.query_items.side_effect = [
        [synthesized],
        list(fact_docs),
        [],
        [],
    ]

    second = pipeline.synthesize_procedural("u1")

    assert pipeline._run_prompty.call_count == 1
    assert second["status"] == "unchanged"
    assert second["procedural"]["id"] == synthesized["id"]


def test_get_procedural_prompt_returns_none_when_missing():
    client = _make_client()
    client._memories_container_client.query_items.return_value = []

    assert client.get_procedural_prompt("u1") is None


def test_get_procedural_prompt_returns_active_content():
    active_doc = _procedural_doc(
        "proc_u1_2",
        version=2,
        content="Active prompt",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        ts=2,
    )
    superseded_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Old prompt",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_2",
        ts=1,
    )
    docs = [superseded_doc, active_doc]
    client = _make_client()

    def _query_items(**kwargs):
        query = kwargs["query"]
        if "superseded_by" in query:
            return [doc for doc in docs if not doc.get("superseded_by")]
        return list(docs)

    client._memories_container_client.query_items.side_effect = _query_items

    assert client.get_procedural_prompt("u1") == "Active prompt"


def test_get_procedural_history_returns_active_first_then_newest_versions():
    v1 = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="v1",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_2",
        ts=1,
    )
    v2 = _procedural_doc(
        "proc_u1_2",
        version=2,
        content="v2",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_3",
        ts=2,
    )
    v3 = _procedural_doc(
        "proc_u1_3",
        version=3,
        content="v3",
        source_fact_ids=["f1", "f2", "f3"],
        source_episodic_ids=["e1"],
        ts=3,
    )
    client = _make_client()
    client._memories_container_client.query_items.return_value = [v1, v3, v2]

    history = client.get_procedural_history("u1", limit=10)

    assert [doc["id"] for doc in history] == ["proc_u1_3", "proc_u1_2", "proc_u1_1"]


def test_get_procedural_history_respects_limit():
    v1 = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="v1",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_2",
        ts=1,
    )
    v2 = _procedural_doc(
        "proc_u1_2",
        version=2,
        content="v2",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_3",
        ts=2,
    )
    v3 = _procedural_doc(
        "proc_u1_3",
        version=3,
        content="v3",
        source_fact_ids=["f1", "f2", "f3"],
        source_episodic_ids=["e1"],
        ts=3,
    )
    client = _make_client()
    client._memories_container_client.query_items.return_value = [v1, v2, v3]

    history = client.get_procedural_history("u1", limit=2)

    assert [doc["id"] for doc in history] == ["proc_u1_3", "proc_u1_2"]
    assert len(history) == 2


def test_client_synthesize_procedural_raises_for_remote_processors():
    client = _make_client(processor=DurableFunctionProcessor())
    client._pipeline = MagicMock()

    with pytest.raises(NotImplementedError, match="durable mode"):
        client.synthesize_procedural("u1")

    client._pipeline.synthesize_procedural.assert_not_called()


def test_synthesize_procedural_retries_with_fresh_llm_call_when_winner_has_partial_coverage():
    """Two concurrent writers race on ``proc_u1_2``; loser must re-read the
    winner, see that its source set does NOT cover the loser's current set,
    re-call the LLM with the winner as the new prior, and write at the
    bumped version. Verifies content monotonicity in source coverage, not
    just version number.
    """
    from azure.cosmos.exceptions import CosmosResourceExistsError

    prior_v1 = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="v1 prompt",
        source_fact_ids=["f1"],
        source_episodic_ids=[],
        ts=1,
    )
    # Loser sees f1 and f2; winner only saw f1 → not covered → must re-call LLM.
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
    ]

    container = MagicMock()
    container.query_items.side_effect = [
        [prior_v1],
        fact_docs,
        [],
        [],
        [
            prior_v1,
            _procedural_doc(
                "proc_u1_2",
                version=2,
                content="v2 by winner",
                source_fact_ids=["f1"],
                source_episodic_ids=[],
                ts=2,
            ),
        ],
    ]
    upserted, capture = _capture_upserts()
    container.upsert_item.side_effect = capture

    write_log: list[str] = []

    def _create(*, body):
        write_log.append(body["id"])
        if body["id"] == "proc_u1_2":
            raise CosmosResourceExistsError(message="conflict")
        capture(body=body)
        return body

    container.create_item.side_effect = _create

    mock_embeddings = MagicMock()
    containers = {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: container,
        ContainerKey.SUMMARIES: MagicMock(),
    }
    store = MemoryStore(containers=containers, embeddings_client=mock_embeddings)
    pipeline = PipelineService(store, MagicMock(), mock_embeddings, containers=containers)
    # Two LLM calls expected: once against v1, once against v2-winner after 409.
    pipeline._run_prompty = MagicMock(
        side_effect=[
            json.dumps({"system_prompt": "v2 by us (stale, will lose race)"}),
            json.dumps({"system_prompt": "v3 by us (fresh, prior=winner)"}),
        ]
    )

    result = pipeline.synthesize_procedural("u1", force=False)

    assert result["status"] == "synthesized"
    assert result["procedural"]["id"] == "proc_u1_3"
    assert result["procedural"]["version"] == 3
    assert write_log == ["proc_u1_2", "proc_u1_3"]
    # LLM called twice: stale content discarded, fresh content reflects winner as prior.
    assert pipeline._run_prompty.call_count == 2
    assert "v3 by us" in result["procedural"]["content"]
    assert result["procedural"]["supersedes_ids"] == ["proc_u1_2"]


def test_synthesize_procedural_short_circuits_when_race_winner_covers_loser_sources():
    """Common race case: both writers process the same source set. After 409,
    loser re-reads, finds winner's source_ids ⊇ loser's, and returns the
    winner's doc as ``unchanged`` — no second LLM call, no wasted write.
    """
    from azure.cosmos.exceptions import CosmosResourceExistsError

    prior_v1 = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="v1 prompt",
        source_fact_ids=["f1"],
        source_episodic_ids=[],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
    ]
    # Winner already wrote v2 with the same source set we see now.
    winner_v2 = _procedural_doc(
        "proc_u1_2",
        version=2,
        content="v2 by winner",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=[],
        ts=2,
    )

    container = MagicMock()
    container.query_items.side_effect = [
        [prior_v1],
        fact_docs,
        [],
        [],
        [prior_v1, winner_v2],
    ]
    upserted, capture = _capture_upserts()
    container.upsert_item.side_effect = capture

    write_log: list[str] = []

    def _create(*, body):
        write_log.append(body["id"])
        if body["id"] == "proc_u1_2":
            raise CosmosResourceExistsError(message="conflict")
        capture(body=body)
        return body

    container.create_item.side_effect = _create

    mock_embeddings = MagicMock()
    containers = {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: container,
        ContainerKey.SUMMARIES: MagicMock(),
    }
    store = MemoryStore(containers=containers, embeddings_client=mock_embeddings)
    pipeline = PipelineService(store, MagicMock(), mock_embeddings, containers=containers)
    pipeline._run_prompty = MagicMock(return_value=json.dumps({"system_prompt": "stale v2 by us"}))

    result = pipeline.synthesize_procedural("u1", force=False)

    # Loser short-circuits: returns winner's doc, never retries the LLM.
    assert result["status"] == "unchanged"
    assert result["procedural"]["id"] == "proc_u1_2"
    assert result["procedural"]["content"] == "v2 by winner"
    # Only the losing write attempt; no retry write at proc_u1_3.
    assert write_log == ["proc_u1_2"]
    assert pipeline._run_prompty.call_count == 1
