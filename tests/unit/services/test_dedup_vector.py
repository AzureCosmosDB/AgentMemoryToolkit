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


def test_dedup_extracted_vector_ladder_and_intra_batch() -> None:
    p = _make_pipeline()
    p._embed_batch.return_value = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.99, 0.01]]
    p._vector_candidates = MagicMock(
        side_effect=[
            [{"id": "existing-high", "content": "same", "type": "fact", "score": 0.99}],
            [{"id": "existing-mid", "content": "near", "type": "fact", "score": 0.85}],
            [{"id": "existing-low", "content": "far", "type": "fact", "score": 0.20}],
            [{"id": "existing-low-2", "content": "far", "type": "fact", "score": 0.10}],
        ]
    )
    extracted = {
        "facts": [
            _doc("f-high", "drop against existing"),
            _doc("f-mid", "tag against existing"),
            _doc("f-clean", "keep clean"),
            _doc("f-intra", "near-dup in batch is now kept (deferred to reconcile)"),
        ],
        "episodic": [],
        "updates": [],
    }

    out = p.dedup_extracted_memories("u1", extracted)

    # Intra-batch new-vs-new dedup was removed: f-intra is compared only against
    # persisted memories (Cosmos) -> novel here -> kept; reconcile catches any
    # same-batch near-dups later.
    assert [doc["id"] for doc in out["facts"]] == ["f-mid", "f-clean", "f-intra"]
    assert out["facts"][0]["tags"][-1] == "sys:dup-candidate"
    assert out["facts"][0]["metadata"]["dup_of"] == "existing-mid"
    assert out["facts"][0]["metadata"]["dup_score"] == 0.85
    assert "sys:dup-candidate" not in out["facts"][1]["tags"]
    assert all("embedding" in doc for doc in out["facts"])
    assert out["updates"][-1]["vector_dedup_skipped"] == 1
    assert out["updates"][-1]["dup_candidates_tagged"] == 1


def test_candidate_mode_clears_tags_only_for_survivors() -> None:
    # Latent-bug regression: a source consumed (superseded) by a merge must NOT be
    # re-upserted by tag-clearing, which would resurrect it without superseded_by.
    p = _make_pipeline()
    f1 = _doc("f1", "a", tags=["sys:fact", "sys:dup-candidate"])
    f2 = _doc("f2", "b", tags=["sys:fact", "sys:dup-candidate"])
    f3 = _doc("f3", "c", tags=["sys:fact", "sys:dup-candidate"])
    p._build_candidate_clusters = MagicMock(return_value=([[f1, f2, f3]], 3, [f1, f2, f3]))
    p._reconcile_pool = MagicMock(return_value=({"kept": 1, "merged": 2, "contradicted": 0}, {"f1", "f2"}))
    cleared: list[str] = []
    p._clear_dup_candidate_tags = MagicMock(side_effect=lambda docs: cleared.extend(d["id"] for d in docs))
    p._emit_reconcile_outcome = MagicMock()

    result = p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)

    assert cleared == ["f3"]
    assert result == {
        "kept": 1,
        "merged": 2,
        "contradicted": 0,
        "reconcile_clusters_sent": 1,
        "reconcile_llm_calls_saved": 2,
    }


def test_candidate_mode_clears_tags_on_orphan_seeds() -> None:
    # Seeds tagged sys:dup-candidate that never join a cluster (no near-duplicate)
    # must have their stale tag cleared so future sweeps don't re-scan them forever.
    p = _make_pipeline()
    orphan = _doc("orphan", "lonely", tags=["sys:fact", "sys:dup-candidate"])
    c1 = _doc("c1", "a", tags=["sys:fact", "sys:dup-candidate"])
    c2 = _doc("c2", "b", tags=["sys:fact", "sys:dup-candidate"])
    p._build_candidate_clusters = MagicMock(return_value=([[c1, c2]], 3, [orphan, c1, c2]))
    p._reconcile_pool = MagicMock(return_value=({"kept": 2, "merged": 0, "contradicted": 0}, set()))
    cleared: list[str] = []
    p._clear_dup_candidate_tags = MagicMock(side_effect=lambda docs: cleared.extend(d["id"] for d in docs))
    p._emit_reconcile_outcome = MagicMock()

    p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)

    # Cluster survivors (c1, c2) plus the orphan all get cleared.
    assert set(cleared) == {"c1", "c2", "orphan"}


def test_euclidean_disables_near_exact_autodrop() -> None:
    # On euclidean distance the cosine-calibrated DEDUP_SIM_HIGH auto-drop is
    # disabled: a near-identical existing memory must NOT silently drop the new
    # one. It falls through to borderline tagging (LLM reconcile adjudicates).
    p = _make_pipeline()
    p._vector_distance_function = MagicMock(return_value="euclidean")
    p._embed_batch.return_value = [[1.0, 0.0]]
    # euclidean score = distance; 0.05 is "near-exact" (would drop under cosine rules).
    p._vector_candidates = MagicMock(
        return_value=[{"id": "existing", "content": "same", "type": "fact", "score": 0.05}]
    )
    extracted = {"facts": [_doc("f-new", "near identical")], "episodic": [], "updates": []}

    out = p.dedup_extracted_memories("u1", extracted)

    # Not dropped — kept and tagged for LLM reconcile instead.
    assert [doc["id"] for doc in out["facts"]] == ["f-new"]
    assert out["facts"][0]["tags"][-1] == "sys:dup-candidate"
    assert out["updates"][-1]["vector_dedup_skipped"] == 0
    assert out["updates"][-1]["dup_candidates_tagged"] == 1


def test_candidate_mode_has_no_inmemory_backstop() -> None:
    # The periodic full-pool backstop is no longer driven by an in-memory sweep
    # counter (unreliable on the FA per-worker singleton). Candidate mode does
    # ONLY clustering now; the full-pool pass is requested by the caller via
    # full_rebuild on a persisted-counter cadence.
    p = _make_pipeline()
    assert not hasattr(p, "_next_reconcile_sweep")
    p._build_candidate_clusters = MagicMock(return_value=([], 0, []))
    p._active_memories_for_reconcile = MagicMock()
    p._emit_reconcile_outcome = MagicMock()

    # Many sweeps in a row never escalate to a full-pool pass on their own.
    for _ in range(30):
        p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)
    p._build_candidate_clusters.assert_called()
    p._active_memories_for_reconcile.assert_not_called()


def test_full_rebuild_bypasses_candidate_mode(monkeypatch) -> None:
    # Public reconcile(full_rebuild=True) must take the full-pool single-LLM-pass
    # path even under candidate mode, so it catches dissimilar contradictions.
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_reconcile_mode", lambda: "candidate")
    p = _make_pipeline()
    pool = [_doc("f1", "User is vegetarian"), _doc("f2", "User loves steak")]
    p._active_memories_for_reconcile = MagicMock(return_value=pool)
    p._reconcile_pool = MagicMock(return_value=({"kept": 1, "merged": 0, "contradicted": 1}, set()))
    p._reconcile_candidate_mode = MagicMock()
    p._emit_reconcile_outcome = MagicMock()

    result = p.reconcile_memories("u1", n=50, memory_type="fact", full_rebuild=True)

    p._reconcile_candidate_mode.assert_not_called()
    p._reconcile_pool.assert_called_once_with("u1", "fact", pool)
    assert result["contradicted"] == 1


def test_full_rebuild_clears_survivor_tags(monkeypatch) -> None:
    # full_rebuild full-pool path must clear sys:dup-candidate on survivors so it
    # doesn't leave stale tags/metadata on user-visible memories.
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_reconcile_mode", lambda: "candidate")
    p = _make_pipeline()
    pool = [
        _doc("f1", "a", tags=["sys:fact", "sys:dup-candidate"]),
        _doc("f2", "b", tags=["sys:fact", "sys:dup-candidate"]),
    ]
    p._active_memories_for_reconcile = MagicMock(return_value=pool)
    # f1 consumed (superseded), f2 survives.
    p._reconcile_pool = MagicMock(return_value=({"kept": 1, "merged": 1, "contradicted": 0}, {"f1"}))
    cleared: list[str] = []
    p._clear_dup_candidate_tags = MagicMock(side_effect=lambda docs: cleared.extend(d["id"] for d in docs))
    p._emit_reconcile_outcome = MagicMock()

    p.reconcile_memories("u1", n=50, memory_type="fact", full_rebuild=True)

    # Survivor f2 cleared; consumed f1 not re-upserted (would resurrect it).
    assert cleared == ["f2"]


def test_sweep_survives_one_cluster_failure() -> None:
    # A truncated/malformed LLM response on one cluster must not abort the sweep:
    # remaining clusters still reconcile, orphan clearing still runs, and the failed
    # cluster's tags are RETAINED (not cleared) so it retries next sweep.
    p = _make_pipeline()
    c1 = [
        _doc("a1", "x", tags=["sys:fact", "sys:dup-candidate"]),
        _doc("a2", "y", tags=["sys:fact", "sys:dup-candidate"]),
    ]
    c2 = [
        _doc("b1", "p", tags=["sys:fact", "sys:dup-candidate"]),
        _doc("b2", "q", tags=["sys:fact", "sys:dup-candidate"]),
    ]
    orphan = _doc("o1", "lonely", tags=["sys:fact", "sys:dup-candidate"])
    seeds = [c1[0], c1[1], c2[0], c2[1], orphan]
    p._build_candidate_clusters = MagicMock(return_value=([c1, c2], 5, seeds))
    p._reconcile_pool = MagicMock(
        side_effect=[RuntimeError("truncated LLM response"), ({"kept": 2, "merged": 0, "contradicted": 0}, set())]
    )
    cleared: list[str] = []
    p._clear_dup_candidate_tags = MagicMock(side_effect=lambda docs: cleared.extend(d["id"] for d in docs))
    p._emit_reconcile_outcome = MagicMock()

    result = p._reconcile_candidate_mode("u1", n=50, memory_type="fact", started_at=0.0)

    # c1 failed -> its tags retained (not cleared) for retry; c2 survivors + orphan cleared.
    assert "a1" not in cleared and "a2" not in cleared
    assert {"b1", "b2", "o1"} <= set(cleared)
    assert result["reconcile_clusters_sent"] == 2
    assert result["kept"] == 2


def test_dedup_extracted_flag_off_is_noop(monkeypatch) -> None:
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_vector_enabled", lambda: False)
    p = _make_pipeline()
    extracted = {"facts": [_doc("f1", "content")], "episodic": [], "updates": []}

    out = p.dedup_extracted_memories("u1", extracted)

    assert out is extracted
    p._embed_batch.assert_not_called()


def test_reconcile_memory_type_routing_episodic_and_procedural(monkeypatch) -> None:
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_reconcile_mode", lambda: "full_pool")
    p = _make_pipeline()
    episodes = [_doc("e1", "episode one", "episodic"), _doc("e2", "episode two", "episodic")]
    p._memories_container.query_items.return_value = iter(episodes)
    p._run_prompty.return_value = json.dumps({"duplicate_groups": [], "kept_ids": ["e1", "e2"]})

    result = p.reconcile_memories("u1", memory_type="episodic")

    assert result["contradicted"] == 0
    assert p._run_prompty.call_args.args[0] == "dedup_episodic.prompty"
    assert "episodics_text" in p._run_prompty.call_args.kwargs["inputs"]

    p._run_prompty.reset_mock()
    assert p.reconcile_memories("u1", memory_type="procedural")["reconcile_clusters_sent"] == 0
    p._run_prompty.assert_not_called()


def test_candidate_mode_connected_components() -> None:
    p = _make_pipeline()
    seed = _doc(
        "f1",
        "User likes coffee",
        embedding=[1.0, 0.0],
        tags=["sys:fact", "sys:dup-candidate"],
        metadata={"category": "preference", "dup_of": "f2", "dup_score": 0.9},
    )
    neighbor = _doc("f2", "User loves coffee", embedding=[0.95, 0.05])

    def query_items(*, query: str, parameters: list[dict[str, Any]], **kwargs: Any):
        del kwargs
        params = {p["name"]: p["value"] for p in parameters}
        if "ARRAY_CONTAINS" in query:
            return iter([seed])
        ids = {value for name, value in params.items() if name.startswith("@id")}
        if ids:
            return iter([doc for doc in (seed, neighbor) if doc["id"] in ids])
        return iter([seed, neighbor])

    p._memories_container.query_items.side_effect = query_items
    p._vector_candidates = MagicMock(
        return_value=[{"id": "f2", "content": neighbor["content"], "type": "fact", "score": 0.9}]
    )
    p._run_prompty.return_value = json.dumps(
        {
            "duplicate_groups": [{"merged_content": "User likes coffee.", "source_ids": ["f1", "f2"]}],
            "contradicted_pairs": [],
            "kept_ids": [],
        }
    )

    result = p.reconcile_memories("u1", memory_type="fact")

    assert result["reconcile_clusters_sent"] == 1
    assert result["merged"] == 2
    p._run_prompty.assert_called_once()
    assert p._run_prompty.call_args.args[0] == "dedup.prompty"
