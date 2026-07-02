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
async def test_candidate_mode_clears_tags_only_for_survivors():
    # Latent-bug regression (async mirror): consumed sources must not be
    # re-upserted by tag-clearing, which would resurrect them without superseded_by.
    p = _service()
    f1 = _fact("f1", "a", tags=["sys:fact", "sys:dup-candidate"])
    f2 = _fact("f2", "b", tags=["sys:fact", "sys:dup-candidate"])
    f3 = _fact("f3", "c", tags=["sys:fact", "sys:dup-candidate"])
    p._build_candidate_clusters = AsyncMock(return_value=([[f1, f2, f3]], 3, [f1, f2, f3]))
    p._reconcile_pool = AsyncMock(return_value=({"kept": 1, "merged": 2, "contradicted": 0}, {"f1", "f2"}))
    cleared: list[str] = []

    async def clear(docs):
        cleared.extend(d["id"] for d in docs)

    p._clear_dup_candidate_tags = AsyncMock(side_effect=clear)
    p._emit_reconcile_outcome = MagicMock()

    result = await p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)

    assert cleared == ["f3"]
    assert result == {
        "kept": 1,
        "merged": 2,
        "contradicted": 0,
        "reconcile_clusters_sent": 1,
        "reconcile_llm_calls_saved": 2,
    }


@pytest.mark.asyncio
async def test_candidate_mode_clears_tags_on_orphan_seeds():
    # Async mirror: orphan dup-candidate seeds (no cluster) get their stale tag cleared.
    p = _service()
    orphan = _fact("orphan", "lonely", tags=["sys:fact", "sys:dup-candidate"])
    c1 = _fact("c1", "a", tags=["sys:fact", "sys:dup-candidate"])
    c2 = _fact("c2", "b", tags=["sys:fact", "sys:dup-candidate"])
    p._build_candidate_clusters = AsyncMock(return_value=([[c1, c2]], 3, [orphan, c1, c2]))
    p._reconcile_pool = AsyncMock(return_value=({"kept": 2, "merged": 0, "contradicted": 0}, set()))
    cleared: list[str] = []

    async def clear(docs):
        cleared.extend(d["id"] for d in docs)

    p._clear_dup_candidate_tags = AsyncMock(side_effect=clear)
    p._emit_reconcile_outcome = MagicMock()

    await p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)

    assert set(cleared) == {"c1", "c2", "orphan"}


@pytest.mark.asyncio
async def test_sweep_survives_one_cluster_failure():
    # Async mirror: one failing cluster must not abort the sweep; failed cluster's
    # tags are retained, remaining cluster + orphan are still cleared.
    p = _service()
    c1 = [
        _fact("a1", "x", tags=["sys:fact", "sys:dup-candidate"]),
        _fact("a2", "y", tags=["sys:fact", "sys:dup-candidate"]),
    ]
    c2 = [
        _fact("b1", "p", tags=["sys:fact", "sys:dup-candidate"]),
        _fact("b2", "q", tags=["sys:fact", "sys:dup-candidate"]),
    ]
    orphan = _fact("o1", "lonely", tags=["sys:fact", "sys:dup-candidate"])
    p._build_candidate_clusters = AsyncMock(return_value=([c1, c2], 5, [*c1, *c2, orphan]))
    p._reconcile_pool = AsyncMock(
        side_effect=[RuntimeError("truncated LLM response"), ({"kept": 2, "merged": 0, "contradicted": 0}, set())]
    )
    cleared: list[str] = []

    async def clear(docs):
        cleared.extend(d["id"] for d in docs)

    p._clear_dup_candidate_tags = AsyncMock(side_effect=clear)
    p._emit_reconcile_outcome = MagicMock()

    result = await p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)

    assert "a1" not in cleared and "a2" not in cleared
    assert {"b1", "b2", "o1"} <= set(cleared)
    assert result["reconcile_clusters_sent"] == 2


@pytest.mark.asyncio
async def test_full_rebuild_clears_survivor_tags(monkeypatch):
    # Async mirror: full_rebuild full-pool path clears survivor dup-candidate tags.
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_reconcile_mode", lambda: "candidate")
    p = _service()
    pool = [
        _fact("f1", "a", tags=["sys:fact", "sys:dup-candidate"]),
        _fact("f2", "b", tags=["sys:fact", "sys:dup-candidate"]),
    ]
    p._active_memories_for_reconcile = AsyncMock(return_value=pool)
    p._reconcile_pool = AsyncMock(return_value=({"kept": 1, "merged": 1, "contradicted": 0}, {"f1"}))
    cleared: list[str] = []

    async def clear(docs):
        cleared.extend(d["id"] for d in docs)

    p._clear_dup_candidate_tags = AsyncMock(side_effect=clear)
    p._emit_reconcile_outcome = MagicMock()

    await p.reconcile_memories("u1", n=50, memory_type="fact", full_rebuild=True)

    assert cleared == ["f2"]


@pytest.mark.asyncio
async def test_dedup_extracted_memories_flag_off_is_noop(monkeypatch):
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_vector_enabled", lambda: False)
    p = _service()
    extracted = {"facts": [_fact("f1", "User likes tea")], "episodic": [], "updates": []}

    out = await p.dedup_extracted_memories("u1", extracted)

    assert out is extracted
    p._embed_batch.assert_not_called()


@pytest.mark.asyncio
async def test_euclidean_disables_near_exact_autodrop():
    # Async mirror: euclidean disables the cosine-calibrated near-exact auto-drop;
    # the near-identical new memory is kept + tagged for LLM reconcile.
    p = _service()
    p._vector_distance_function = AsyncMock(return_value="euclidean")
    fact = _fact("f-new", "near identical")
    fact.pop("embedding", None)
    p._embed_batch.return_value = [[1.0, 0.0]]
    p._vector_candidates = AsyncMock(
        return_value=[{"id": "existing", "content": "same", "type": "fact", "score": 0.05}]
    )

    out = await p.dedup_extracted_memories("u1", {"facts": [fact], "episodic": [], "updates": []})

    assert [doc["id"] for doc in out["facts"]] == ["f-new"]
    assert out["facts"][0]["tags"][-1] == "sys:dup-candidate"
    assert out["updates"][-1]["vector_dedup_skipped"] == 0
    assert out["updates"][-1]["dup_candidates_tagged"] == 1


@pytest.mark.asyncio
async def test_dedup_extracted_memories_passes_user_id_per_concurrent_call():
    p = _service()
    seen: list[tuple[str, str]] = []

    async def vector_candidates(**kwargs):
        await asyncio.sleep(0)
        exclude_ids = kwargs["exclude_ids"]
        doc_id = next(iter(exclude_ids))
        seen.append((doc_id, kwargs["user_id"]))
        return []

    p._vector_candidates = AsyncMock(side_effect=vector_candidates)
    user_a_doc = _fact("user-a-doc", "User A likes tea")
    user_b_doc = _fact("user-b-doc", "User B likes coffee")

    await asyncio.gather(
        p.dedup_extracted_memories("user-a", {"facts": [user_a_doc], "episodic": [], "updates": []}),
        p.dedup_extracted_memories("user-b", {"facts": [user_b_doc], "episodic": [], "updates": []}),
    )

    assert dict(seen) == {"user-a-doc": "user-a", "user-b-doc": "user-b"}


@pytest.mark.asyncio
async def test_dedup_extracted_memories_vector_ladder_and_intra_batch():
    p = _service()
    docs = [
        _fact("drop-existing", "User likes tea"),
        _fact("tag-existing", "User likes coffee"),
        _fact("clean", "User likes water"),
        _fact("batch-keeper", "User likes green tea"),
        _fact("drop-batch", "User likes green tea too"),
    ]
    for doc in docs:
        doc.pop("embedding", None)
    p._embed_batch.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [0.7, 0.7],
        [0.7, 0.7],
    ]
    p._vector_candidates = AsyncMock(
        side_effect=[
            [{"id": "old-high", "content": "User likes tea.", "type": "fact", "score": 0.99}],
            [{"id": "old-mid", "content": "User enjoys coffee.", "type": "fact", "score": 0.90}],
            [{"id": "old-low", "content": "Unrelated", "type": "fact", "score": 0.20}],
            [],
            [],
        ]
    )

    out = await p.dedup_extracted_memories("u1", {"facts": docs, "episodic": [], "updates": []})

    # Intra-batch new-vs-new dedup was removed: drop-batch (no existing match)
    # is now kept; same-batch near-dups are deferred to reconcile.
    ids = [doc["id"] for doc in out["facts"]]
    assert ids == ["tag-existing", "clean", "batch-keeper", "drop-batch"]
    assert all("embedding" in doc for doc in out["facts"])
    tagged = out["facts"][0]
    assert "sys:dup-candidate" in tagged["tags"]
    assert tagged["metadata"]["dup_of"] == "old-mid"
    assert tagged["metadata"]["dup_score"] == 0.90
    assert "sys:dup-candidate" not in out["facts"][1]["tags"]
    stats = out["updates"][-1]
    assert stats["vector_dedup_skipped"] == 1
    assert stats["dup_candidates_tagged"] == 1


@pytest.mark.asyncio
async def test_dedup_skips_underspecified_doc_verbatim():
    # Parity with sync: a doc with no/unknown type is passed through untouched
    # and never runs vector dedup (async previously defaulted type to the bucket).
    p = _service()
    p._vector_candidates = AsyncMock(return_value=[{"id": "x", "content": "c", "type": "fact", "score": 0.99}])
    doc = _fact("f1", "content")
    doc.pop("type")
    doc.pop("embedding", None)
    p._embed_batch.return_value = [[1.0, 0.0]]

    out = await p.dedup_extracted_memories("u1", {"facts": [doc], "episodic": [], "updates": []})

    assert [d["id"] for d in out["facts"]] == ["f1"]
    assert "sys:dup-candidate" not in out["facts"][0].get("tags", [])
    p._vector_candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_memory_type_routes_episodic_merge_only(monkeypatch):
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_reconcile_mode", lambda: "full_pool")
    p = _service()
    episodes = [_episode("ep_1", "CI failed then retries fixed it"), _episode("ep_2", "CI failed; retries fixed it")]
    p._active_memories_for_reconcile = AsyncMock(return_value=episodes)
    p._run_prompty = AsyncMock(
        return_value=json.dumps(
            {
                "duplicate_groups": [
                    {
                        "merged_content": "CI failed, then retries fixed it",
                        "source_ids": ["ep_1", "ep_2"],
                        "confidence": 0.9,
                        "salience": 0.8,
                    }
                ],
                "kept_ids": [],
                "contradicted_pairs": [{"winner_id": "ep_1", "loser_id": "ep_2"}],
            }
        )
    )

    result = await p.reconcile_memories("u1", memory_type="episodic")

    assert result == {"kept": 0, "merged": 2, "contradicted": 0}
    assert p._run_prompty.await_args.kwargs["inputs"].keys() == {"episodics_text"}
    assert p._run_prompty.await_args.args[0] == "dedup_episodic.prompty"


@pytest.mark.asyncio
async def test_candidate_reconcile_builds_connected_component():
    p = _service()
    docs = {
        "f1": _fact("f1", "User likes aisle seats", tags=["sys:fact", "sys:dup-candidate"]),
        "f2": _fact("f2", "User prefers aisle seats"),
        "f3": _fact("f3", "User enjoys aisle seats"),
    }

    async def query_items(_container, **kwargs):
        params = {p["name"]: p["value"] for p in kwargs.get("parameters", [])}
        if params.get("@tag") == "sys:dup-candidate":
            return [docs["f1"]]
        ids = [value for name, value in params.items() if name.startswith("@id")]
        return [docs[mid] for mid in ids if mid in docs]

    p._query_items = AsyncMock(side_effect=query_items)
    p._vector_candidates = AsyncMock(
        side_effect=[
            [
                {"id": "f2", "content": docs["f2"]["content"], "type": "fact", "score": 0.90},
                {"id": "f3", "content": docs["f3"]["content"], "type": "fact", "score": 0.88},
            ],
            [
                {"id": "f2", "content": docs["f2"]["content"], "type": "fact", "score": 0.90},
                {"id": "f3", "content": docs["f3"]["content"], "type": "fact", "score": 0.88},
            ],
            [
                {"id": "f1", "content": docs["f1"]["content"], "type": "fact", "score": 0.90},
                {"id": "f3", "content": docs["f3"]["content"], "type": "fact", "score": 0.89},
            ],
            [
                {"id": "f1", "content": docs["f1"]["content"], "type": "fact", "score": 0.88},
                {"id": "f2", "content": docs["f2"]["content"], "type": "fact", "score": 0.89},
            ],
        ]
    )
    p._run_prompty = AsyncMock(
        return_value=json.dumps(
            {
                "duplicate_groups": [
                    {
                        "merged_content": "User prefers aisle seats",
                        "source_ids": ["f1", "f2", "f3"],
                        "confidence": 0.9,
                        "salience": 0.8,
                    }
                ],
                "contradicted_pairs": [],
                "kept_ids": [],
            }
        )
    )

    result = await p.reconcile_memories("u1")

    assert result == {
        "kept": 0,
        "merged": 3,
        "contradicted": 0,
        "reconcile_clusters_sent": 1,
        "reconcile_llm_calls_saved": 2,
    }
    p._run_prompty.assert_awaited_once()
    facts_text = p._run_prompty.await_args.kwargs["inputs"]["facts_text"]
    assert all(fid in facts_text for fid in ["f1", "f2", "f3"])
