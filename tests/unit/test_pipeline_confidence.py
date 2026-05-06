"""Tests for ProcessingPipeline.extract_memories confidence + unclassified handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.pipeline import ProcessingPipeline


def _make_pipeline(llm_response: dict):
    container = MagicMock()
    # Single turn so the pipeline doesn't bail on "no memories found".
    container.query_items.return_value = iter(
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
    # Capture upserts for inspection.
    upserted: list[dict] = []
    container.upsert_item.side_effect = lambda body: upserted.append(body) or body

    llm = MagicMock()
    embeddings = MagicMock()
    embeddings.generate_batch.side_effect = lambda texts: [[0.0] * 4 for _ in texts]

    pipeline = ProcessingPipeline(
        cosmos_container=container,
        chat_client=llm,
        embeddings_client=embeddings,
    )
    # Avoid real LLM/prompty calls.
    pipeline._run_prompty = MagicMock(return_value=json.dumps(llm_response))
    pipeline._load_existing_memories = MagicMock(return_value=[])

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
    assert result["facts_count"] == 1


def test_extract_defaults_confidence_to_half_when_missing():
    pipeline, upserted = _make_pipeline(
        {
            "facts": [{"text": "User likes coffee", "action": "ADD"}],
            "procedural": [{"instruction": "Greet warmly", "action": "ADD"}],
            "episodic": [
                {
                    "situation": "Trying X",
                    "action_taken": "Did Y",
                    "outcome": "Worked",
                }
            ],
        }
    )

    pipeline.extract_memories("u1", "t1")

    for doc in upserted:
        assert doc["confidence"] == 0.5, f"missing default for {doc['type']} {doc['id']}"


def test_extract_routes_unclassified_to_fact_with_tag():
    pipeline, upserted = _make_pipeline(
        {
            "unclassified": [
                {
                    "text": "Weird ambiguous thing about the user",
                    "confidence": 0.45,
                    "salience": 0.4,
                    "tags": ["ambig"],
                    "reason": "could be fact or episodic",
                }
            ]
        }
    )

    result = pipeline.extract_memories("u1", "t1")

    assert len(upserted) == 1
    doc = upserted[0]
    assert doc["type"] == "fact"
    assert "sys:unclassified" in doc["tags"]
    assert "sys:fact" in doc["tags"]
    assert "topic:ambig" in doc["tags"]
    assert doc["confidence"] == pytest.approx(0.45)
    assert doc["metadata"]["unclassified_reason"] == "could be fact or episodic"
    assert result["unclassified_count"] == 1
    assert result["facts_count"] == 0


def test_extract_episodic_carries_confidence():
    pipeline, upserted = _make_pipeline(
        {
            "episodic": [
                {
                    "situation": "Setup CI",
                    "action_taken": "Added Ruff",
                    "outcome": "Faster lint",
                    "confidence": 0.8,
                    "salience": 0.7,
                }
            ]
        }
    )
    pipeline.extract_memories("u1", "t1")
    [ep] = [d for d in upserted if d["type"] == "episodic"]
    assert ep["confidence"] == pytest.approx(0.8)


def test_extract_procedural_carries_confidence():
    pipeline, upserted = _make_pipeline(
        {
            "procedural": [
                {
                    "instruction": "Use ruff format",
                    "category": "workflow",
                    "source": "explicit_instruction",
                    "confidence": 0.95,
                    "salience": 0.9,
                    "action": "ADD",
                }
            ]
        }
    )
    pipeline.extract_memories("u1", "t1")
    [pr] = [d for d in upserted if d["type"] == "procedural"]
    assert pr["confidence"] == pytest.approx(0.95)


class TestMarkSupersededDoesNotMutate:
    """``_mark_superseded`` must not mutate its input dict before the write.

    If the write fails (412/transient), callers retrying would otherwise see
    a dict already carrying ``superseded_by`` and lose the ability to detect
    "no, this fact has not yet been marked superseded" downstream.
    """

    def test_input_dict_unchanged_on_success(self):
        from azure.core import MatchConditions

        from agent_memory_toolkit.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline.__new__(ProcessingPipeline)
        pipeline._container = MagicMock()

        old_doc = {"id": "fact-1", "_etag": "etag-1", "content": "x"}
        snapshot = dict(old_doc)

        result = pipeline._mark_superseded(old_doc, "fact-2", reason="duplicate")

        assert result is True
        assert old_doc == snapshot
        body = pipeline._container.replace_item.call_args.kwargs["body"]
        assert body["superseded_by"] == "fact-2"
        assert body["supersede_reason"] == "duplicate"
        assert "superseded_at" in body
        assert pipeline._container.replace_item.call_args.kwargs["match_condition"] == MatchConditions.IfNotModified

    def test_input_dict_unchanged_on_failure(self):
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        from agent_memory_toolkit.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline.__new__(ProcessingPipeline)
        pipeline._container = MagicMock()
        pipeline._container.replace_item.side_effect = CosmosAccessConditionFailedError(message="412", response=None)

        old_doc = {"id": "fact-1", "_etag": "etag-1", "content": "x"}
        snapshot = dict(old_doc)

        result = pipeline._mark_superseded(old_doc, "fact-2", reason="contradiction")

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
        from agent_memory_toolkit.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline.__new__(ProcessingPipeline)
        pipeline._embeddings = MagicMock()
        pipeline._embeddings.generate.return_value = [0.1] * 8
        pipeline._upsert_memory = MagicMock()
        pipeline._container = MagicMock()
        pipeline._chat = MagicMock()
        return pipeline

    def test_thread_ids_does_not_appear_in_query_or_parameters(self):
        pipeline = self._build_pipeline()
        # No prior user-summary; first-pass full generation.
        pipeline._container.read_item.side_effect = Exception("not found")
        # Two memories on different threads; the IN filter would drop t3.
        pipeline._container.query_items.return_value = iter(
            [
                {
                    "id": "f1",
                    "user_id": "u1",
                    "thread_id": "t1",
                    "type": "fact",
                    "content": "User likes coffee.",
                    "created_at": "2025-01-01T00:00:00+00:00",
                },
                {
                    "id": "f3",
                    "user_id": "u1",
                    "thread_id": "t3",
                    "type": "fact",
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

        call = pipeline._container.query_items.call_args
        query = call.kwargs["query"]
        params = call.kwargs["parameters"]

        assert "IN (" not in query
        assert "@tid" not in query
        assert not any(p["name"].startswith("@tid") for p in params)

        upserted = pipeline._upsert_memory.call_args.args[0]
        # Both threads must contribute to the resulting summary metadata.
        assert sorted(upserted["metadata"]["thread_ids"]) == ["t1", "t3"]


class TestDrainPipelineResources:
    """``_init_pipeline`` must drain prior sync resources on re-entry.

    Calling ``connect_cosmos()`` more than once on the same async client
    (container switch, credential rotation, reconnect) must not leak the
    predecessor sync EmbeddingsClient or sync CosmosClient.
    """

    def test_reentry_closes_prior_sync_embeddings_and_cosmos(self):
        from agent_memory_toolkit.aio.cosmos_memory_client import (
            AsyncCosmosMemoryClient,
        )

        client = AsyncCosmosMemoryClient.__new__(AsyncCosmosMemoryClient)
        prior_embed = MagicMock()
        prior_cosmos = MagicMock()
        client._sync_embeddings_client = prior_embed
        client._sync_cosmos_client = prior_cosmos

        client._drain_pipeline_resources()

        prior_embed.close.assert_called_once()
        prior_cosmos.close.assert_called_once()
        assert client._sync_embeddings_client is None
        assert client._sync_cosmos_client is None

    def test_drain_is_safe_when_attributes_unset(self):
        from agent_memory_toolkit.aio.cosmos_memory_client import (
            AsyncCosmosMemoryClient,
        )

        client = AsyncCosmosMemoryClient.__new__(AsyncCosmosMemoryClient)
        # No attribute set yet — must not AttributeError.
        client._drain_pipeline_resources()
