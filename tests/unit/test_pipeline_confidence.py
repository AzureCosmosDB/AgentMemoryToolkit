"""Tests for ProcessingPipeline.extract_memories confidence + unclassified handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from azure.cosmos.agent_memory.store import MemoryStore

TENANT = "acme"


def _make_pipeline(llm_response: dict):
    turns_container = MagicMock()
    memories_container = MagicMock()
    summaries_container = MagicMock()
    # Single turn so the pipeline doesn't bail on "no memories found".
    turns_container.query_items.return_value = iter(
        [
            {
                "id": "turn1",
                "user_id": "u1",
                "thread_id": "t1",
                "role": "user",
                "type": "turn",
                "content": "I prefer dark mode.",
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        ]
    )
    upserted: list[dict] = []
    memories_container.upsert_item.side_effect = lambda body: upserted.append(body) or body
    memories_container.create_item.side_effect = lambda body: upserted.append(body) or body

    chat = MagicMock()
    embeddings = MagicMock()
    embeddings.generate_batch.side_effect = lambda texts: [[0.0] * 4 for _ in texts]

    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    store = MemoryStore(containers=containers, embeddings_client=embeddings, tenant_id=TENANT)
    pipeline = PipelineService(store, chat, embeddings, containers=containers, tenant_id=TENANT)
    # Avoid real LLM/prompty calls.
    pipeline._run_prompty = MagicMock(return_value=json.dumps(llm_response))

    return pipeline, upserted


def test_extract_stamps_top_level_confidence_on_facts():
    pipeline, upserted = _make_pipeline(
        {
            "facts": [
                {
                    "text": "User prefers dark mode",
                    "category": "preference",
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "dark mode",
                    "confidence": 0.92,
                    "salience": 0.6,
                    "action": "ADD",
                }
            ]
        }
    )

    result = pipeline.extract_memories("u1", "t1")

    facts = [d for d in upserted if d["type"] == "fact"]
    assert len(facts) == 1
    assert facts[0]["confidence"] == pytest.approx(0.92)
    # confidence must NOT live under metadata anymore.
    assert "confidence" not in facts[0]["metadata"]
    assert result["fact_count"] == 1


def test_extract_defaults_confidence_to_half_when_missing():
    pipeline, upserted = _make_pipeline(
        {
            "facts": [{"text": "User likes coffee", "action": "ADD"}],
            "episodic": [],
        }
    )

    pipeline.extract_memories("u1", "t1")

    for doc in upserted:
        assert doc["confidence"] == 0.5, f"missing default for {doc['type']} {doc['id']}"


def test_extract_reads_and_writes_inside_the_pipelines_tenant():
    pipeline, upserted = _make_pipeline({"facts": [{"text": "User likes coffee", "action": "ADD"}], "episodic": []})

    pipeline.extract_memories("u1", "t1")

    assert [d["type"] for d in upserted] == ["fact"]
    assert (upserted[0]["tenant_id"], upserted[0]["user_id"], upserted[0]["thread_id"]) == (TENANT, "u1", "t1")
    turns_queries = pipeline._turns_container.query_items.call_args_list
    assert len(turns_queries) == 2, "the turns to extract, and the turns before them as context"
    for call in turns_queries:
        params = {p["name"]: p["value"] for p in call.kwargs["parameters"]}
        assert "c.tenant_id = @tenant_id AND c.user_id = @user_id AND c.thread_id = @thread_id" in call.kwargs["query"]
        assert (params["@tenant_id"], params["@user_id"], params["@thread_id"]) == (TENANT, "u1", "t1")
        assert call.kwargs["partition_key"] == [TENANT, "u1"]
    stamped = pipeline._turns_container.patch_item.call_args
    assert stamped.kwargs["partition_key"] == [TENANT, "u1"]


class TestMarkSupersededDoesNotMutate:
    """``_mark_superseded`` must not mutate its input dict before the write.

    If the write fails (412/transient), callers retrying would otherwise see
    a dict already carrying ``superseded_by`` and lose the ability to detect
    "no, this fact has not yet been marked superseded" downstream.
    """

    def test_input_dict_unchanged_on_success(self):
        from azure.core import MatchConditions

        from azure.cosmos.agent_memory.services.pipeline import PipelineService

        pipeline = PipelineService.__new__(PipelineService)
        pipeline._store = None
        pipeline._memories_container = MagicMock()

        old_doc = {"id": "fact-1", "_etag": "etag-1", "content": "x"}
        snapshot = dict(old_doc)

        result = pipeline._mark_superseded(old_doc, "fact-2", reason="duplicate")

        assert result is True
        assert old_doc == snapshot
        body = pipeline._memories_container.replace_item.call_args.kwargs["body"]
        assert body["superseded_by"] == "fact-2"
        assert body["supersede_reason"] == "duplicate"
        assert "superseded_at" in body
        assert (
            pipeline._memories_container.replace_item.call_args.kwargs["match_condition"]
            == MatchConditions.IfNotModified
        )

    def test_input_dict_unchanged_on_failure(self):
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        from azure.cosmos.agent_memory.services.pipeline import PipelineService

        pipeline = PipelineService.__new__(PipelineService)
        pipeline._store = None
        pipeline._memories_container = MagicMock()
        pipeline._memories_container.replace_item.side_effect = CosmosAccessConditionFailedError(
            message="412",
            response=None,
        )

        old_doc = {"id": "fact-1", "_etag": "etag-1", "content": "x"}
        snapshot = dict(old_doc)

        result = pipeline._mark_superseded(old_doc, "fact-2", reason="duplicate")

        assert result is False
        assert old_doc == snapshot


class TestGenerateUserSummaryThreadIdsObservabilityOnly:
    """``thread_ids`` must NOT filter the SQL query.

    A user-summary roll-up may run after several change-feed batches have
    accumulated against the user counter; ``thread_ids`` from the last
    crossing batch is a strict subset of the threads that contributed
    memories in the cross-counter window. Filtering the query by it would
    permanently exclude pre-watermark memories from threads in earlier
    batches (the ``c.created_at > @since`` bound moves past them on the
    next persist).
    """

    def _build_pipeline(self):
        from azure.cosmos.agent_memory.services.pipeline import PipelineService

        pipeline = PipelineService.__new__(PipelineService)
        pipeline._tenant_id = TENANT
        pipeline._embeddings = MagicMock()
        pipeline._embeddings.generate.return_value = [0.1] * 8
        pipeline._upsert_summary = MagicMock(side_effect=lambda doc: doc)
        pipeline._memories_container = MagicMock()
        pipeline._summaries_container = MagicMock()
        pipeline._chat = MagicMock()
        return pipeline

    def test_the_user_summary_reads_the_owners_partition_of_its_tenant(self):
        pipeline = self._build_pipeline()
        pipeline._summaries_container.read_item.side_effect = CosmosResourceNotFoundError(message="not found")
        pipeline._memories_container.query_items.return_value = iter([])
        pipeline._summaries_container.query_items.return_value = iter(
            [{"id": "summary_u1_t1", "type": "thread_summary", "thread_id": "t1", "content": "User likes coffee."}]
        )

        with patch.object(pipeline, "_run_prompty", return_value='{"key_facts":["likes coffee"]}'):
            pipeline.generate_user_summary(user_id="u1")

        pipeline._summaries_container.read_item.assert_called_once_with(
            item="user_summary_u1", partition_key=[TENANT, "u1"]
        )
        for container_mock in (pipeline._memories_container, pipeline._summaries_container):
            call = container_mock.query_items.call_args
            params = {p["name"]: p["value"] for p in call.kwargs["parameters"]}
            assert "c.tenant_id = @tenant_id AND c.user_id = @user_id" in call.kwargs["query"]
            assert (params["@tenant_id"], params["@user_id"]) == (TENANT, "u1")
            assert call.kwargs["partition_key"] == [TENANT, "u1"]
            assert "enable_cross_partition_query" not in call.kwargs
        stored = pipeline._upsert_summary.call_args.args[0]
        assert (stored["tenant_id"], stored["user_id"], stored["thread_id"]) == (TENANT, "u1", "__user_summary__")

    def test_thread_ids_does_not_appear_in_query_or_parameters(self):
        pipeline = self._build_pipeline()
        # No prior user-summary; first-pass full generation.
        pipeline._summaries_container.read_item.side_effect = CosmosResourceNotFoundError(message="not found")
        # No facts/episodics in MEMORIES for this user.
        pipeline._memories_container.query_items.return_value = iter([])
        # Two thread summaries on different threads; the IN filter would drop t3.
        pipeline._summaries_container.query_items.return_value = iter(
            [
                {
                    "id": "summary_u1_t1",
                    "user_id": "u1",
                    "thread_id": "t1",
                    "type": "thread_summary",
                    "content": "User likes coffee.",
                    "created_at": "2025-01-01T00:00:00+00:00",
                },
                {
                    "id": "summary_u1_t3",
                    "user_id": "u1",
                    "thread_id": "t3",
                    "type": "thread_summary",
                    "content": "User lives in Seattle.",
                    "created_at": "2025-01-01T00:00:01+00:00",
                },
            ]
        )

        with patch.object(
            pipeline,
            "_run_prompty",
            return_value='{"key_facts":["likes coffee","lives in Seattle"]}',
        ):
            pipeline.generate_user_summary(user_id="u1", thread_ids=["t1"])

        for container_mock in (pipeline._memories_container, pipeline._summaries_container):
            call = container_mock.query_items.call_args
            query = call.kwargs["query"]
            params = call.kwargs["parameters"]
            assert "IN (@tid" not in query
            assert "@tid" not in query
            assert not any(p["name"].startswith("@tid") for p in params)

        upserted = pipeline._upsert_summary.call_args.args[0]
        # Both threads must contribute to the resulting summary metadata.
        assert sorted(upserted["metadata"]["thread_ids"]) == ["t1", "t3"]


# ---------------------------------------------------------------------------
# Legacy episodic fields in extract_memories.prompty responses
# ---------------------------------------------------------------------------


def test_extract_memories_ignores_legacy_episodic_payloads(caplog):
    pipeline, upserted = _make_pipeline(
        {
            "episodic": [
                {
                    "scope_type": "trip",
                    "scope_value": "Paris",
                    "confidence": 0.95,
                    "salience": 0.8,
                }
            ]
        }
    )

    with caplog.at_level("WARNING", logger="azure.cosmos.agent_memory.pipeline"):
        result = pipeline.extract_memories("u1", "t1")

    assert result["episodic_count"] == 0
    assert not any(d["type"] == "episodic" for d in upserted)
    assert not any("dropping malformed episodic" in rec.getMessage() for rec in caplog.records)


def test_extract_compound_statement_yields_facts_across_categories():
    """A single user turn that combines preference + requirement must produce two
    facts, not one merged fact. Regression guard on the pipeline plumbing."""
    pipeline, upserted = _make_pipeline(
        {
            "facts": [
                {
                    "text": "The user does not eat meat.",
                    "category": "preference",
                    "subject": "user",
                    "predicate": "dietary_restriction",
                    "object": "no meat",
                    "confidence": 1.0,
                    "salience": 0.9,
                    "tags": ["topic:diet"],
                },
                {
                    "text": "The user requires wheelchair-accessible restaurants.",
                    "category": "requirement",
                    "subject": "user",
                    "predicate": "accessibility_requirement",
                    "object": "wheelchair-accessible restaurants",
                    "confidence": 1.0,
                    "salience": 0.95,
                    "tags": ["topic:accessibility"],
                },
            ]
        }
    )
    pipeline.extract_memories("u1", "t1")

    facts = [d for d in upserted if d["type"] == "fact"]
    assert len(facts) == 2
    by_category = {f["metadata"]["category"]: f for f in facts}
    assert set(by_category) == {"preference", "requirement"}
    assert by_category["preference"]["content"] == "The user does not eat meat."
    assert by_category["requirement"]["content"] == "The user requires wheelchair-accessible restaurants."
