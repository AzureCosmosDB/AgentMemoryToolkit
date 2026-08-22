from __future__ import annotations

import json
from unittest.mock import MagicMock

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory.services.pipeline import PipelineService


def _service() -> PipelineService:
    svc = PipelineService.__new__(PipelineService)
    svc._transcript_metadata_keys = None
    svc._prompt_lineage = lambda _filename: {"prompt_id": "p", "prompt_version": "v1"}  # type: ignore[method-assign]
    return svc


def _turn(**overrides):
    doc = {
        "id": "turn1",
        "user_id": "alice",
        "thread_id": "thread1",
        "role": "user",
        "type": "turn",
        "content": "remember the launch checklist",
        "metadata": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "tenant_id": "acme",
        "scope_type": "team",
        "scope_id": "eng",
        "scope_key": "team:eng",
        "acl": {"read": ["team:eng"], "write": ["team:eng"], "annotate": [], "forget": ["team:eng"]},
        "provenance": {"created_by": "user:alice"},
    }
    doc.update(overrides)
    return doc


def test_extract_memories_durable_inherits_source_turn_scope():
    svc = _service()
    svc._run_prompty = lambda *_args, **_kwargs: json.dumps(  # type: ignore[method-assign]
        {"facts": [{"text": "The launch checklist matters.", "category": "preference"}]}
    )

    result = svc.extract_memories_durable("alice", "thread1", turns=[_turn()])

    fact = result["facts"][0]
    assert fact["tenant_id"] == "acme"
    assert fact["scope_type"] == "team"
    assert fact["scope_id"] == "eng"
    assert fact["scope_key"] == "team:eng"
    assert fact["acl"]["read"] == ["team:eng"]


def test_build_episode_docs_inherits_source_turn_scope():
    svc = _service()
    svc._run_prompty = lambda *_args, **_kwargs: json.dumps(  # type: ignore[method-assign]
        {
            "episodes": [
                {
                    "title": "Launch prep",
                    "summary": "Discussed launch prep.",
                    "events": [{"sequence": 1, "description": "Mentioned checklist", "source_turn_ids": ["turn1"]}],
                }
            ]
        }
    )

    docs = svc._build_episode_docs("alice", "thread1", [_turn()], segment_key="seg")

    assert docs[0]["tenant_id"] == "acme"
    assert docs[0]["scope_key"] == "team:eng"
    assert docs[0]["acl"]["read"] == ["team:eng"]


def test_generate_thread_summary_durable_inherits_source_turn_scope():
    svc = _service()
    summaries = MagicMock()
    summaries.read_item.side_effect = CosmosResourceNotFoundError(message="missing")
    turns = MagicMock()
    turns.query_items.return_value = [_turn()]
    svc._summaries_container = summaries
    svc._turns_container = turns
    svc._run_prompty = lambda *_args, **_kwargs: json.dumps(  # type: ignore[method-assign]
        {"overview": "Launch prep summary", "topics": ["launch"]}
    )

    summary = svc.generate_thread_summary_durable("alice", "thread1")

    assert summary["tenant_id"] == "acme"
    assert summary["scope_type"] == "team"
    assert summary["scope_id"] == "eng"
    assert summary["scope_key"] == "team:eng"
    assert summary["acl"]["read"] == ["team:eng"]
