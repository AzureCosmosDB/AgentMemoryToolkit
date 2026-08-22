"""Async tests for atomic procedural synthesis and compiled procedural context."""

from __future__ import annotations

from typing import Any

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from tests.unit.services.test_extract_dry import (
    _async_containers_for_store,
    _AsyncChat,
    _AsyncEmbeddings,
    _AsyncStore,
)


class _AsyncProceduralStore(_AsyncStore):
    async def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        del partition_key, cross_partition
        params = {p["name"]: p["value"] for p in (parameters or [])}
        scope_key = params.get("@scope_key")
        memory_type = params.get("@type", params.get("@memory_type"))
        docs = [dict(doc) for doc in self.docs]
        if scope_key is not None:
            docs = [
                doc
                for doc in docs
                if (doc.get("scope_key") or (f"user:{doc.get('user_id')}" if doc.get("user_id") else None)) == scope_key
            ]
        if memory_type is not None:
            docs = [doc for doc in docs if doc.get("type") == memory_type]
        if "c.status='active'" in sql:
            docs = [doc for doc in docs if doc.get("status") == "active"]
        if "superseded_by" in sql:
            docs = [doc for doc in docs if not doc.get("superseded_by")]
        return docs

    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        if any(doc.get("id") == body.get("id") for doc in self.docs):
            raise CosmosResourceExistsError(message="conflict")
        self.docs.append(dict(body))
        return dict(body)


def _service(
    store: _AsyncProceduralStore,
    responses: list[dict[str, Any]] | None = None,
) -> AsyncPipelineService:
    return AsyncPipelineService(
        store,
        _AsyncChat(responses or []),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store),
    )


def _fact() -> dict[str, Any]:
    return {
        "id": "fact-raw-1",
        "user_id": "u1",
        "type": "fact",
        "content": "The user explicitly said to run targeted tests before reporting success.",
        "metadata": {"category": "preference"},
        "salience": 0.9,
        "created_at": "2025-01-01T00:00:00+00:00",
    }


def _episode() -> dict[str, Any]:
    return {
        "id": "episode-raw-1",
        "user_id": "u1",
        "type": "episodic",
        "content": "A retry investigation succeeded.",
        "lessons": ["Retry transient CI failures once before escalating."],
        "salience": 0.8,
        "created_at": "2025-01-01T00:01:00+00:00",
    }


def _procedure(
    name: str,
    *,
    grounded_in: list[str],
    source_kind: str,
    summary: str = "Run targeted tests before reporting success.",
) -> dict[str, Any]:
    return {
        "name": name,
        "summary": summary,
        "retrieval_text": summary,
        "procedure_kind": "behavioral_policy",
        "scope_type": "user",
        "scope_value": None,
        "activation_conditions": [],
        "preconditions": [],
        "steps": [],
        "success_conditions": [],
        "failure_conditions": [],
        "safety_constraints": [],
        "source_kind": source_kind,
        "grounded_in": grounded_in,
        "confidence": 0.8,
    }


@pytest.mark.asyncio
async def test_synthesize_procedural_extracts_atomic_procedures_and_gates_provenance() -> None:
    store = _AsyncProceduralStore([_fact(), _episode()])
    service = _service(
        store,
        [
            {
                "procedures": [
                    _procedure(
                        "Targeted testing",
                        grounded_in=["fact-1"],
                        source_kind="explicit_user_instruction",
                    ),
                    _procedure(
                        "Retry CI failures",
                        grounded_in=["ep-1"],
                        source_kind="episode_distillation",
                        summary="Retry transient CI failures once before escalating.",
                    ),
                ]
            }
        ],
    )

    result = await service.synthesize_procedural("u1")

    assert result == {"status": "synthesized", "procedures_created": 2, "procedures_skipped": 0}
    procedures = [doc for doc in store.docs if doc.get("type") == "procedural"]
    assert len(procedures) == 2
    assert {doc["name"]: doc["status"] for doc in procedures} == {
        "Targeted testing": "active",
        "Retry CI failures": "candidate",
    }
    assert {doc["name"]: doc["source_kind"] for doc in procedures} == {
        "Targeted testing": "explicit_user_instruction",
        "Retry CI failures": "episode_distillation",
    }
    for doc in procedures:
        assert doc["id"].startswith("proc_")
        assert doc["type"] == "procedural"
        assert doc["name"]
        assert doc["retrieval_text"]
        assert doc["thread_id"] == "__procedural__"
        assert doc["embedding"] == [1.0]
        # utility_score is seeded from the LLM confidence (0.8), not hard-wired to 0.5.
        assert doc["utility_score"] == 0.8


@pytest.mark.asyncio
async def test_synthesize_procedural_is_idempotent_by_scope_and_name() -> None:
    store = _AsyncProceduralStore([_fact()])
    response = {
        "procedures": [
            _procedure(
                "Targeted testing",
                grounded_in=["fact-1"],
                source_kind="explicit_user_instruction",
            )
        ]
    }
    service = _service(store, [response, response])

    first = await service.synthesize_procedural("u1")
    second = await service.synthesize_procedural("u1")

    assert first == {"status": "synthesized", "procedures_created": 1, "procedures_skipped": 0}
    assert second == {"status": "synthesized", "procedures_created": 0, "procedures_skipped": 1}
    procedures = [doc for doc in store.docs if doc.get("type") == "procedural"]
    assert len(procedures) == 1


@pytest.mark.asyncio
async def test_build_procedural_context_uses_active_procedures_only() -> None:
    active = {
        "id": "proc-active",
        "user_id": "u1",
        "type": "procedural",
        "status": "active",
        "name": "Targeted testing",
        "summary": "Run targeted tests before reporting success.",
        "retrieval_text": "tests success",
        "procedure_kind": "behavioral_policy",
        "scope_type": "user",
        "scope_value": None,
        "priority": 10,
        "source_authority": "high",
        "version": 1,
    }
    candidate = {
        **active,
        "id": "proc-candidate",
        "status": "candidate",
        "name": "Candidate policy",
        "summary": "Do not include this candidate procedure.",
    }
    service = _service(_AsyncProceduralStore([active, candidate]))

    context = await service.build_procedural_context("u1")

    assert "Run targeted tests before reporting success." in context
    assert "Do not include this candidate procedure." not in context
    assert await service.build_procedural_context("missing-user") == ""
