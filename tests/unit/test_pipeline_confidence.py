"""Tests for ProcessingPipeline.extract_memories confidence + unclassified handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

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
    assert result["facts_count"] == 1


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
