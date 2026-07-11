from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService


def _service() -> AsyncPipelineService:
    p = AsyncPipelineService.__new__(AsyncPipelineService)
    p._memories_container = MagicMock()
    p._embed_batch = AsyncMock()
    p._embed_one = AsyncMock(return_value=[0.1, 0.2])
    p._upsert_memory = AsyncMock(side_effect=lambda doc: doc)
    p._mark_superseded = AsyncMock(return_value=True)
    return p


def _fact(fid: str, content: str, embedding=None, tags=None, metadata=None) -> dict:
    return {
        "id": fid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "role": "system",
        "content": content,
        "content_hash": "0" * 32,
        "confidence": 0.8,
        "salience": 0.7,
        "tags": list(tags or ["sys:fact"]),
        "metadata": dict(metadata or {"category": "preference"}),
        "created_at": "2025-01-01T00:00:00+00:00",
        "embedding": embedding or [1.0, 0.0],
    }


def _episode(eid: str, content: str) -> dict:
    return {
        "id": eid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "episodic",
        "role": "system",
        "content": content,
        "content_hash": "1" * 32,
        "confidence": 0.8,
        "salience": 0.7,
        "tags": ["sys:episodic", "sys:dup-candidate"],
        "metadata": {
            "scope_type": "project",
            "scope_value": "CI",
            "lesson": content,
            "outcome_valence": "positive",
        },
        "created_at": "2025-01-01T00:00:00+00:00",
        "embedding": [1.0, 0.0],
    }


@pytest.mark.asyncio
async def test_vector_distance_function_reads_container_policy():
    # The distance function comes from the container's vector embedding policy
    # (read once, cached), NOT an env var.
    p = _service()
    p._memories_container.read = AsyncMock(
        return_value={
            "vectorEmbeddingPolicy": {"vectorEmbeddings": [{"path": "/embedding", "distanceFunction": "dotproduct"}]}
        }
    )
    assert await p._vector_distance_function() == "dotproduct"
    assert await p._vector_distance_function() == "dotproduct"
    assert p._memories_container.read.await_count == 1


@pytest.mark.asyncio
async def test_vector_candidates_orders_nearest_first_by_distance_function():
    # Regression: async _vector_candidates must order most-similar-first per the
    # container's distanceFunction. For cosine/dotproduct higher score = more
    # similar (DESC); for euclidean lower distance = more similar (ASC). A missing
    # DESC silently fetched the LEAST-similar rows when the pool exceeded top_k.
    p = _service()
    captured: dict[str, str] = {}

    async def fake_query_items(_container, *, query, parameters):
        captured["query"] = query
        return [
            {"id": "near", "content": "a", "type": "fact", "score": 0.95},
            {"id": "far", "content": "b", "type": "fact", "score": 0.10},
        ]

    p._query_items = AsyncMock(side_effect=fake_query_items)

    p._distance_function_cache = "cosine"
    out = await p._vector_candidates(user_id="u1", embedding=[1.0, 0.0], memory_type="fact", top_k=2, exclude_ids=set())
    # Cosmos rejects an explicit ASC/DESC on ORDER BY VectorDistance(); it orders
    # most-similar-first server-side. Direction-awareness lives in the Python sort.
    assert "ORDER BY VectorDistance(c.embedding, @vec)" in captured["query"]
    assert "VectorDistance(c.embedding, @vec) DESC" not in captured["query"]
    assert "VectorDistance(c.embedding, @vec) ASC" not in captured["query"]
    assert [c["id"] for c in out] == ["near", "far"]

    p._distance_function_cache = "euclidean"
    out = await p._vector_candidates(user_id="u1", embedding=[1.0, 0.0], memory_type="fact", top_k=2, exclude_ids=set())
    assert "VectorDistance(c.embedding, @vec) ASC" not in captured["query"]
    # euclidean: lower distance = more similar, so 0.10 ("far" label) sorts first.
    # euclidean: lower distance = more similar, so 0.10 ("far" label) sorts first.
    assert [c["id"] for c in out] == ["far", "near"]


@pytest.mark.asyncio
async def test_dedup_extracted_folds_near_dup_in_place_and_keeps_novel():
    p = _service()
    p._vector_distance_function = AsyncMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]
    p._nearest_active_full = AsyncMock(
        side_effect=[
            ({"id": "existing-1", "content": "same", "type": "fact"}, 0.99),
            (None, 0.0),
        ]
    )
    p._apply_inplace_update = AsyncMock(return_value=True)
    extracted = {
        "facts": [_fact("f-dup", "restatement"), _fact("f-novel", "brand new")],
        "episodic": [],
        "updates": [],
    }

    out = await p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-novel"]
    p._apply_inplace_update.assert_awaited_once()
    target, new_doc = p._apply_inplace_update.call_args.args
    assert target["id"] == "existing-1"
    assert new_doc["id"] == "f-dup"
    assert out["updates"][-1]["inplace_updated"] == 1


@pytest.mark.asyncio
async def test_dedup_extracted_failed_inplace_update_keeps_new_doc():
    p = _service()
    p._vector_distance_function = AsyncMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._nearest_active_full = AsyncMock(return_value=({"id": "existing-1", "content": "same", "type": "fact"}, 0.99))
    p._apply_inplace_update = AsyncMock(return_value=False)
    extracted = {"facts": [_fact("f-dup", "restatement")], "episodic": [], "updates": []}

    out = await p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-dup"]
    assert all(op.get("op") != "stats" or "inplace_updated" not in op for op in out["updates"])


@pytest.mark.asyncio
async def test_dedup_extracted_below_threshold_is_novel():
    p = _service()
    p._vector_distance_function = AsyncMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._nearest_active_full = AsyncMock(return_value=({"id": "existing-1", "content": "near", "type": "fact"}, 0.85))
    p._apply_inplace_update = AsyncMock(return_value=True)
    extracted = {"facts": [_fact("f-new", "somewhat similar")], "episodic": [], "updates": []}

    out = await p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-new"]
    p._apply_inplace_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_euclidean_disables_inplace_folding():
    p = _service()
    p._vector_distance_function = AsyncMock(return_value="euclidean")
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._nearest_active_full = AsyncMock()
    p._apply_inplace_update = AsyncMock()
    extracted = {"facts": [_fact("f-new", "near identical")], "episodic": [], "updates": []}

    out = await p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-new"]
    p._nearest_active_full.assert_not_awaited()
    p._apply_inplace_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_inplace_update_recency_wins_and_unions():
    p = _service()
    p._upsert_item = AsyncMock()
    neighbor = _fact("existing-1", "old content", tags=["sys:fact", "topic:a"])
    neighbor["confidence"] = 0.6
    neighbor["salience"] = 0.5
    neighbor["updated_at"] = "2025-01-01T00:00:00+00:00"
    neighbor["_etag"] = "etag-xyz"
    new_doc = _fact(
        "f-new", "new richer content", embedding=[0.5, 0.5], tags=["sys:fact", "topic:b", "sys:dup-candidate"]
    )
    new_doc["confidence"] = 0.9
    new_doc["salience"] = 0.8

    ok = await p._apply_inplace_update(neighbor, new_doc)

    assert ok is True
    written = p._upsert_item.call_args.kwargs["body"]
    assert written["id"] == "existing-1"
    assert written["content"] == "new richer content"
    assert written["embedding"] == [0.5, 0.5]
    assert written["salience"] == 0.8
    assert written["confidence"] == 0.9
    assert "topic:a" in written["tags"] and "topic:b" in written["tags"]
    assert "sys:dup-candidate" not in written["tags"]
    assert "_etag" not in written


@pytest.mark.asyncio
async def test_nearest_active_full_returns_full_doc_and_skips_excluded():
    p = _service()
    doc_a = _fact("a", "first")
    doc_b = _fact("b", "second")

    async def query_items(_container, *, query, parameters):
        del query, parameters
        return [{"doc": doc_a, "score": 0.99}, {"doc": doc_b, "score": 0.80}]

    p._query_items = AsyncMock(side_effect=query_items)
    neighbor, score = await p._nearest_active_full(
        user_id="u1", embedding=[1.0, 0.0], memory_type="fact", exclude_ids={"a"}
    )
    assert neighbor["id"] == "b"
    assert score == 0.80


@pytest.mark.asyncio
async def test_dedup_extracted_memories_flag_off_is_noop(monkeypatch):
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_vector_enabled", lambda: False)
    p = _service()
    extracted = {"facts": [_fact("f1", "content")], "episodic": [], "updates": []}

    out = await p.dedup_extracted_memories("u1", extracted)

    assert out is extracted
    p._embed_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_dedup_extracted_memories_passes_user_id_per_concurrent_call():
    # Two concurrent dedup calls for different users must each query with their own
    # user_id (no shared mutable state leaking one user's id into another's query).
    p = _service()
    p._vector_distance_function = AsyncMock(return_value="cosine")
    seen_users: list[str] = []

    async def nearest(*, user_id, embedding, memory_type, exclude_ids):
        del embedding, memory_type, exclude_ids
        seen_users.append(user_id)
        return None, 0.0

    p._nearest_active_full = AsyncMock(side_effect=nearest)

    async def run(uid):
        p2 = _service()
        p2._vector_distance_function = AsyncMock(return_value="cosine")
        p2._nearest_active_full = AsyncMock(side_effect=nearest)
        p2._embed_batch.return_value = [[1.0, 0.0]]
        await p2.dedup_extracted_memories(uid, {"facts": [_fact("f", "c")], "episodic": [], "updates": []})

    await asyncio.gather(run("userA"), run("userB"))
    assert set(seen_users) == {"userA", "userB"}


@pytest.mark.asyncio
async def test_reconcile_memory_type_routes_episodic_and_procedural_noop():
    p = _service()
    p._run_prompty = AsyncMock()

    episodic_result = await p.reconcile_memories("u1", memory_type="episodic")
    assert episodic_result == {"kept": 0, "merged": 0, "contradicted": 0}

    procedural_result = await p.reconcile_memories("u1", memory_type="procedural")
    assert procedural_result == {"kept": 0, "merged": 0, "contradicted": 0}

    p._run_prompty.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_fact_contradiction_only():
    p = _service()
    facts = [
        _fact("f1", "User's deadline is March 1"),
        _fact("f2", "User's deadline is March 15"),
    ]
    facts[0]["created_at"] = "2024-01-01T00:00:00+00:00"
    facts[1]["created_at"] = "2024-02-01T00:00:00+00:00"
    p._active_memories_for_reconcile = AsyncMock(return_value=facts)
    p._run_prompty = AsyncMock(
        return_value=json.dumps(
            {
                "duplicate_groups": [{"merged_content": "ignored", "source_ids": ["f1", "f2"]}],
                "contradicted_pairs": [{"winner_id": "f2", "loser_id": "f1", "reason": "more recent"}],
                "kept_ids": ["f2"],
            }
        )
    )

    result = await p.reconcile_memories("u1", memory_type="fact")

    assert result == {"kept": 1, "merged": 0, "contradicted": 1}
    p._upsert_memory.assert_not_awaited()
    assert p._mark_superseded.await_count == 1
    assert p._mark_superseded.call_args.args[0]["id"] == "f1"
    assert p._mark_superseded.call_args.args[1] == "f2"
    assert p._mark_superseded.call_args.kwargs["reason"] == "contradict"
    assert p._run_prompty.call_args.args[0] == "dedup.prompty"
