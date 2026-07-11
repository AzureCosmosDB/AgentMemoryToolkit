from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from azure.cosmos.agent_memory.services.pipeline import PipelineService


def _make_pipeline() -> PipelineService:
    p = PipelineService.__new__(PipelineService)
    p._memories_container = MagicMock()
    p._container = p._memories_container
    p._embeddings = MagicMock()
    p._embed_batch = MagicMock()
    p._embed_one = MagicMock(return_value=[1.0])
    p._run_prompty = MagicMock(
        return_value=json.dumps({"duplicate_groups": [], "contradicted_pairs": [], "kept_ids": []})
    )
    p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
    p._mark_superseded = MagicMock(return_value=True)
    return p


def _doc(mid: str, content: str, memory_type: str = "fact", **extra: Any) -> dict[str, Any]:
    tags = extra.pop("tags", [f"sys:{memory_type}"])
    metadata = extra.pop(
        "metadata",
        {"category": "preference"}
        if memory_type == "fact"
        else {
            "scope_type": "project",
            "scope_value": "demo",
            "lesson": content,
            "outcome_valence": "neutral",
        },
    )
    return {
        "id": mid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": memory_type,
        "role": "system",
        "content": content,
        "content_hash": mid,
        "confidence": 0.8,
        "salience": 0.7,
        "tags": tags,
        "metadata": metadata,
        "prompt_id": "extract_memories.prompty",
        "prompt_version": "v1",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        **extra,
    }


def test_vector_distance_function_reads_container_policy() -> None:
    # The distance function comes from the container's vector embedding policy
    # (read once, cached), NOT an env var.
    p = _make_pipeline()
    p._memories_container.read.return_value = {
        "vectorEmbeddingPolicy": {"vectorEmbeddings": [{"path": "/embedding", "distanceFunction": "euclidean"}]}
    }
    assert p._vector_distance_function() == "euclidean"
    # Cached: a later policy change is not re-read within the instance's lifetime.
    p._memories_container.read.return_value = {
        "vectorEmbeddingPolicy": {"vectorEmbeddings": [{"path": "/embedding", "distanceFunction": "cosine"}]}
    }
    assert p._vector_distance_function() == "euclidean"
    assert p._memories_container.read.call_count == 1


def test_distance_function_not_cached_on_read_failure() -> None:
    # A transient container.read() failure must NOT poison the cache: it returns an
    # uncached cosine default so the next call self-heals to the real (euclidean)
    # policy. Caching cosine here would silently mis-handle a euclidean container.
    p = _make_pipeline()
    euclid = {"vectorEmbeddingPolicy": {"vectorEmbeddings": [{"path": "/embedding", "distanceFunction": "euclidean"}]}}
    p._memories_container.read = MagicMock(side_effect=[RuntimeError("429 throttled"), euclid])

    # First call: transient failure -> cosine, but NOT cached.
    assert p._vector_distance_function() == "cosine"
    assert getattr(p, "_distance_function_cache", None) is None

    # Second call: read succeeds -> real euclidean policy, now cached.
    assert p._vector_distance_function() == "euclidean"
    assert p._distance_function_cache == "euclidean"


def test_vector_candidates_orders_nearest_first_by_distance_function() -> None:
    # Parity with async: ORDER BY direction follows the container distanceFunction.
    p = _make_pipeline()
    captured: dict[str, str] = {}

    def query_items(*, query: str, parameters, **kwargs):
        del parameters, kwargs
        captured["query"] = query
        return iter(
            [
                {"id": "near", "content": "a", "type": "fact", "score": 0.95},
                {"id": "far", "content": "b", "type": "fact", "score": 0.10},
            ]
        )

    p._memories_container.query_items.side_effect = query_items

    p._distance_function_cache = "cosine"
    out = p._vector_candidates(user_id="u1", embedding=[1.0, 0.0], memory_type="fact", top_k=2, exclude_ids=set())
    # Cosmos rejects an explicit ASC/DESC on ORDER BY VectorDistance(); it orders
    # most-similar-first server-side. Direction-awareness lives in the Python sort.
    assert "ORDER BY VectorDistance(c.embedding, @vec)" in captured["query"]
    assert "VectorDistance(c.embedding, @vec) DESC" not in captured["query"]
    assert "VectorDistance(c.embedding, @vec) ASC" not in captured["query"]
    assert [c["id"] for c in out] == ["near", "far"]

    p._distance_function_cache = "euclidean"
    out = p._vector_candidates(user_id="u1", embedding=[1.0, 0.0], memory_type="fact", top_k=2, exclude_ids=set())
    assert "VectorDistance(c.embedding, @vec) ASC" not in captured["query"]
    # euclidean: lower distance = more similar, so 0.10 ("far" label) sorts first.
    assert [c["id"] for c in out] == ["far", "near"]


def test_dedup_extracted_folds_near_dup_in_place_and_keeps_novel() -> None:
    # Write-time in-place dedup: a new fact whose nearest active neighbor is at/above
    # DEDUP_SIM_HIGH is folded into that neighbor in place (dropped from the ADD set);
    # a fact with no close neighbor is novel and passes through to persist.
    p = _make_pipeline()
    p._vector_distance_function = MagicMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0], [0.0, 1.0]]
    p._nearest_active_full = MagicMock(
        side_effect=[
            ({"id": "existing-1", "content": "same", "type": "fact"}, 0.99),
            (None, 0.0),
        ]
    )
    p._apply_inplace_update = MagicMock(return_value=True)
    extracted = {
        "facts": [
            _doc("f-dup", "restatement of existing"),
            _doc("f-novel", "brand new fact"),
        ],
        "episodic": [],
        "updates": [],
    }

    out = p.dedup_extracted_memories("u1", extracted)

    # f-dup folded in place (removed from ADD set); f-novel kept.
    assert [doc["id"] for doc in out["facts"]] == ["f-novel"]
    p._apply_inplace_update.assert_called_once()
    target, new_doc = p._apply_inplace_update.call_args.args
    assert target["id"] == "existing-1"
    assert new_doc["id"] == "f-dup"
    assert out["updates"][-1]["inplace_updated"] == 1


def test_dedup_extracted_failed_inplace_update_keeps_new_doc() -> None:
    # If the in-place upsert fails, the new doc must NOT be lost - it stays in the
    # result so persist ADDs it as a novel record.
    p = _make_pipeline()
    p._vector_distance_function = MagicMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._nearest_active_full = MagicMock(return_value=({"id": "existing-1", "content": "same", "type": "fact"}, 0.99))
    p._apply_inplace_update = MagicMock(return_value=False)
    extracted = {"facts": [_doc("f-dup", "restatement")], "episodic": [], "updates": []}

    out = p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-dup"]
    assert all(op.get("op") != "stats" or "inplace_updated" not in op for op in out["updates"])


def test_dedup_extracted_below_threshold_is_novel() -> None:
    # A neighbor below DEDUP_SIM_HIGH is not a near-duplicate: the new fact is novel
    # and no in-place update happens.
    p = _make_pipeline()
    p._vector_distance_function = MagicMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._nearest_active_full = MagicMock(return_value=({"id": "existing-1", "content": "near", "type": "fact"}, 0.85))
    p._apply_inplace_update = MagicMock(return_value=True)
    extracted = {"facts": [_doc("f-new", "somewhat similar")], "episodic": [], "updates": []}

    out = p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-new"]
    p._apply_inplace_update.assert_not_called()


def test_dedup_second_batch_dup_of_same_target_is_dropped_once() -> None:
    # Two new facts that both fold into the SAME existing neighbor: only the first
    # refreshes it; the second is dropped without re-writing the target.
    p = _make_pipeline()
    p._vector_distance_function = MagicMock(return_value="cosine")
    p._embed_batch.return_value = [[1.0, 0.0], [1.0, 0.0]]
    p._nearest_active_full = MagicMock(
        side_effect=[
            ({"id": "existing-1", "content": "same", "type": "fact"}, 0.99),
            ({"id": "existing-1", "content": "same", "type": "fact"}, 0.98),
        ]
    )
    p._apply_inplace_update = MagicMock(return_value=True)
    extracted = {
        "facts": [_doc("f-a", "restate one"), _doc("f-b", "restate two")],
        "episodic": [],
        "updates": [],
    }

    out = p.dedup_extracted_memories("u1", extracted)

    assert out["facts"] == []
    assert p._apply_inplace_update.call_count == 1
    assert out["updates"][-1]["inplace_updated"] == 1


def test_euclidean_disables_inplace_folding() -> None:
    # On euclidean distance the cosine-calibrated DEDUP_SIM_HIGH is not comparable,
    # so in-place folding is disabled and every extracted doc passes through as-is.
    p = _make_pipeline()
    p._vector_distance_function = MagicMock(return_value="euclidean")
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._nearest_active_full = MagicMock()
    p._apply_inplace_update = MagicMock()
    extracted = {"facts": [_doc("f-new", "near identical")], "episodic": [], "updates": []}

    out = p.dedup_extracted_memories("u1", extracted)

    assert [doc["id"] for doc in out["facts"]] == ["f-new"]
    p._nearest_active_full.assert_not_called()
    p._apply_inplace_update.assert_not_called()


def test_apply_inplace_update_recency_wins_and_unions() -> None:
    # The refreshed doc keeps the neighbor's id but takes the new content/embedding,
    # max salience/confidence, unioned tags (minus sys:dup-candidate), bumped updated_at.
    p = _make_pipeline()
    neighbor = _doc("existing-1", "old content", confidence=0.6, salience=0.5, tags=["sys:fact", "topic:a"])
    neighbor["_etag"] = "etag-xyz"
    new_doc = _doc(
        "f-new",
        "new richer content",
        confidence=0.9,
        salience=0.8,
        tags=["sys:fact", "topic:b", "sys:dup-candidate"],
        embedding=[0.5, 0.5],
    )

    ok = p._apply_inplace_update(neighbor, new_doc)

    assert ok is True
    # ETag optimistic concurrency: goes through replace_item with IfNotModified.
    call = p._memories_container.replace_item.call_args
    assert call.kwargs["etag"] == "etag-xyz"
    written = call.kwargs["body"]
    assert written["id"] == "existing-1"
    assert written["content"] == "new richer content"  # recency wins
    assert written["embedding"] == [0.5, 0.5]
    assert written["salience"] == 0.8
    assert written["confidence"] == 0.9
    assert "topic:a" in written["tags"] and "topic:b" in written["tags"]
    assert "sys:dup-candidate" not in written["tags"]
    assert "_etag" not in written
    assert written["updated_at"] != neighbor["updated_at"]


def test_apply_inplace_update_shorter_restatement_keeps_richer_content() -> None:
    p = _make_pipeline()
    neighbor = _doc(
        "existing-1",
        "March 1, room 204, deluxe suite",
        confidence=0.6,
        salience=0.5,
        tags=["sys:fact", "topic:a"],
        embedding=[0.1, 0.2],
    )
    neighbor["_etag"] = "etag-xyz"
    new_doc = _doc(
        "f-new",
        "March 1",
        confidence=0.9,
        salience=0.8,
        tags=["sys:fact", "topic:b"],
        embedding=[0.5, 0.5],
    )

    ok = p._apply_inplace_update(neighbor, new_doc)

    assert ok is True
    written = p._memories_container.replace_item.call_args.kwargs["body"]
    assert written["content"] == "March 1, room 204, deluxe suite"  # richer content kept
    assert written["embedding"] == [0.1, 0.2]  # matching embedding kept
    assert written["salience"] == 0.8  # metadata still recency-wins
    assert written["confidence"] == 0.9
    assert "topic:a" in written["tags"] and "topic:b" in written["tags"]
    # A concurrent writer (ETag mismatch) must NOT clobber; caller ADDs novel.
    from azure.cosmos.exceptions import CosmosAccessConditionFailedError

    p = _make_pipeline()
    p._memories_container.replace_item.side_effect = CosmosAccessConditionFailedError(message="etag")
    neighbor = _doc("existing-1", "old", tags=["sys:fact"])
    neighbor["_etag"] = "stale"
    new_doc = _doc("f-new", "old restated", embedding=[0.5, 0.5], tags=["sys:fact"])

    assert p._apply_inplace_update(neighbor, new_doc) is False


def test_nearest_active_full_returns_full_doc_and_skips_excluded() -> None:
    p = _make_pipeline()
    doc_a = _doc("a", "first")
    doc_b = _doc("b", "second")

    def query_items(*, query: str, parameters, **kwargs):
        del query, parameters, kwargs
        return iter(
            [
                {"doc": doc_a, "score": 0.99},
                {"doc": doc_b, "score": 0.80},
            ]
        )

    p._memories_container.query_items.side_effect = query_items
    # Exclude the closest -> falls through to the next candidate.
    neighbor, score = p._nearest_active_full(user_id="u1", embedding=[1.0, 0.0], memory_type="fact", exclude_ids={"a"})
    assert neighbor["id"] == "b"
    assert score == 0.80


def test_dedup_extracted_flag_off_is_noop(monkeypatch) -> None:
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_vector_enabled", lambda: False)
    p = _make_pipeline()
    extracted = {"facts": [_doc("f1", "content")], "episodic": [], "updates": []}

    out = p.dedup_extracted_memories("u1", extracted)

    assert out is extracted
    p._embed_batch.assert_not_called()


def test_reconcile_memory_type_routing_episodic_and_procedural() -> None:
    # Episodic and procedural reconcile are no-ops (their near-dups fold at write
    # time / have no contradiction semantics): no LLM call, zeroed counts.
    p = _make_pipeline()

    episodic_result = p.reconcile_memories("u1", memory_type="episodic")
    assert episodic_result == {"kept": 0, "merged": 0, "contradicted": 0}

    procedural_result = p.reconcile_memories("u1", memory_type="procedural")
    assert procedural_result == {"kept": 0, "merged": 0, "contradicted": 0}

    p._run_prompty.assert_not_called()


def test_reconcile_fact_contradiction_only() -> None:
    # The fact reconcile path applies only contradicted_pairs; duplicate_groups in
    # the LLM response are ignored (write-time in-place dedup owns paraphrases).
    p = _make_pipeline()
    facts = [
        _doc("f1", "User's deadline is March 1", created_at="2024-01-01T00:00:00+00:00"),
        _doc("f2", "User's deadline is March 15", created_at="2024-02-01T00:00:00+00:00"),
    ]
    p._memories_container.query_items.return_value = iter(facts)
    p._run_prompty = MagicMock(
        return_value=json.dumps(
            {
                "duplicate_groups": [{"merged_content": "ignored", "source_ids": ["f1", "f2"]}],
                "contradicted_pairs": [{"winner_id": "f2", "loser_id": "f1", "reason": "more recent"}],
                "kept_ids": ["f2"],
            }
        )
    )

    result = p.reconcile_memories("u1", memory_type="fact")

    assert result == {"kept": 1, "merged": 0, "contradicted": 1}
    # No merged doc upserted; only the loser superseded.
    p._upsert_memory.assert_not_called()
    assert p._mark_superseded.call_count == 1
    assert p._mark_superseded.call_args.args[0]["id"] == "f1"
    assert p._mark_superseded.call_args.args[1] == "f2"
    assert p._mark_superseded.call_args.kwargs["reason"] == "contradict"
    assert p._run_prompty.call_args.args[0] == "dedup.prompty"


def test_reconcile_skips_chained_contradiction() -> None:
    # (A>B) then (B>C) must not tombstone C in favor of an already-dead B.
    p = _make_pipeline()
    facts = [_doc("A", "a"), _doc("B", "b"), _doc("C", "c")]
    p._memories_container.query_items.return_value = iter(facts)
    p._run_prompty = MagicMock(
        return_value=json.dumps(
            {
                "contradicted_pairs": [
                    {"winner_id": "A", "loser_id": "B", "reason": "x"},
                    {"winner_id": "B", "loser_id": "C", "reason": "y"},
                ],
                "kept_ids": [],
            }
        )
    )

    result = p.reconcile_memories("u1", memory_type="fact")

    # Only the first pair applies; the chained (B>C) is skipped since B is dead.
    assert result["contradicted"] == 1
    assert p._mark_superseded.call_count == 1
    assert p._mark_superseded.call_args.args[0]["id"] == "B"
