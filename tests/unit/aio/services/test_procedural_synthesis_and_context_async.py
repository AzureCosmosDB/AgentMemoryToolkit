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
async def test_synthesize_procedural_applies_provenance_gate() -> None:
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
                        source_kind="explicit_user_instruction",
                        summary="Retry transient CI failures once before escalating.",
                    ),
                ]
            }
        ],
    )

    result = await service.synthesize_procedural("u1")

    assert result == {"status": "synthesized", "procedures_created": 2, "procedures_skipped": 0}
    procedures = [doc for doc in store.docs if doc.get("type") == "procedural"]
    assert {doc["name"]: doc["status"] for doc in procedures} == {
        "Targeted testing": "active",
        "Retry CI failures": "candidate",
    }
    assert {doc["name"]: doc["source_kind"] for doc in procedures}["Retry CI failures"] == "episode_distillation"
    assert procedures[0]["thread_id"] == "__procedural__"
    assert procedures[0]["embedding"] == [1.0]


@pytest.mark.asyncio
async def test_build_procedural_context_uses_active_procedures_only() -> None:
    active = {
        "id": "proc-active",
        "scope_key": "user:u1",
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


def _active_procedure(
    name: str,
    *,
    summary: str,
    procedure_kind: str = "behavioral_policy",
    scope_type: str = "user",
    retrieval_text: str | None = None,
    activation_conditions: list[str] | None = None,
    priority: int = 0,
    source_authority: str = "low",
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"proc-{name.lower().replace(' ', '-')}",
        "scope_key": "user:u1",
        "type": "procedural",
        "status": "active",
        "name": name,
        "summary": summary,
        "retrieval_text": retrieval_text if retrieval_text is not None else summary,
        "procedure_kind": procedure_kind,
        "scope_type": scope_type,
        "scope_value": None,
        "activation_conditions": activation_conditions or [],
        "steps": steps or [],
        "priority": priority,
        "source_authority": source_authority,
        "version": 1,
    }


@pytest.mark.asyncio
async def test_build_procedural_context_includes_task_matching_workflow_only() -> None:
    workflow = _active_procedure(
        "Partition workflow",
        summary="Use partition key diagnostics.",
        procedure_kind="workflow",
        scope_type="domain",
        retrieval_text="cosmos partition routing diagnostics",
        activation_conditions=["when debugging partition fanout"],
        steps=[{"sequence": 1, "instruction": "Inspect partition routing."}],
    )
    service = _service(_AsyncProceduralStore([workflow]))

    matching_context = await service.build_procedural_context("u1", task="debug partition latency")

    assert "Partition workflow" in matching_context
    assert "Inspect partition routing." in matching_context
    assert await service.build_procedural_context("u1") == ""
    assert await service.build_procedural_context("u1", task="summarize billing invoices") == ""


@pytest.mark.asyncio
async def test_build_procedural_context_includes_policy_kinds_regardless_of_scope() -> None:
    global_policy = _active_procedure(
        "Global reporting",
        summary="Always report validation status.",
        scope_type="global",
    )
    team_rule = _active_procedure(
        "Team test rule",
        summary="Run targeted tests before reporting success.",
        procedure_kind="decision_rule",
        scope_type="team",
    )
    unmatched_workflow = _active_procedure(
        "Unmatched workflow",
        summary="A task workflow that should not appear without a matching task.",
        procedure_kind="workflow",
        scope_type="project",
    )
    service = _service(_AsyncProceduralStore([global_policy, team_rule, unmatched_workflow]))

    context = await service.build_procedural_context("u1")

    # behavioral_policy + decision_rule are standing policies regardless of tenancy scope
    assert "Global reporting" in context
    assert "Team test rule" in context
    # a task procedure (workflow) is only injected when the task matches
    assert "Unmatched workflow" not in context


@pytest.mark.asyncio
async def test_build_procedural_context_orders_policies_by_priority_then_authority() -> None:
    lower_priority = _active_procedure(
        "Lower priority",
        summary="Lower priority policy.",
        priority=1,
        source_authority="mandatory",
    )
    higher_priority = _active_procedure(
        "Higher priority",
        summary="Higher priority policy.",
        priority=10,
        source_authority="low",
    )
    mandatory_authority = _active_procedure(
        "Mandatory authority",
        summary="Mandatory authority policy.",
        priority=10,
        source_authority="mandatory",
    )
    service = _service(_AsyncProceduralStore([lower_priority, higher_priority, mandatory_authority]))

    context = await service.build_procedural_context("u1")

    assert context.index("Mandatory authority") < context.index("Higher priority")
    assert context.index("Higher priority") < context.index("Lower priority")


@pytest.mark.asyncio
async def test_synthesize_procedural_downgrades_episode_only_explicit_claim() -> None:
    store = _AsyncProceduralStore([_episode()])
    service = _service(
        store,
        [
            {
                "procedures": [
                    _procedure(
                        "Episode claim",
                        grounded_in=["ep-1"],
                        source_kind="explicit_user_instruction",
                        summary="Retry transient CI failures once before escalating.",
                    )
                ]
            }
        ],
    )

    result = await service.synthesize_procedural("u1")

    assert result == {"status": "synthesized", "procedures_created": 1, "procedures_skipped": 0}
    procedure = next(doc for doc in store.docs if doc.get("type") == "procedural")
    assert procedure["status"] == "candidate"
    assert procedure["source_kind"] == "episode_distillation"


@pytest.mark.asyncio
async def test_synthesize_procedural_downgrades_episode_only_organization_policy() -> None:
    store = _AsyncProceduralStore([_episode()])
    service = _service(
        store,
        [
            {
                "procedures": [
                    _procedure(
                        "Org policy from episode",
                        grounded_in=["ep-1"],
                        source_kind="organization_policy",
                        summary="Retry transient CI failures once before escalating.",
                    )
                ]
            }
        ],
    )

    result = await service.synthesize_procedural("u1")

    assert result == {"status": "synthesized", "procedures_created": 1, "procedures_skipped": 0}
    procedure = next(doc for doc in store.docs if doc.get("type") == "procedural")
    assert procedure["status"] == "candidate"
    assert procedure["source_kind"] == "episode_distillation"


@pytest.mark.asyncio
async def test_synthesize_procedural_ungrounded_claim_is_candidate() -> None:
    store = _AsyncProceduralStore([_fact()])
    service = _service(
        store,
        [
            {
                "procedures": [
                    _procedure(
                        "Ungrounded org policy",
                        grounded_in=["nonexistent"],
                        source_kind="organization_policy",
                    )
                ]
            }
        ],
    )

    result = await service.synthesize_procedural("u1")

    assert result == {"status": "synthesized", "procedures_created": 1, "procedures_skipped": 0}
    procedure = next(doc for doc in store.docs if doc.get("type") == "procedural")
    assert procedure["status"] == "candidate"


@pytest.mark.asyncio
async def test_synthesize_procedural_skips_malformed_workflow_and_creates_valid_sibling() -> None:
    malformed = _procedure(
        "Empty workflow",
        grounded_in=["fact-1"],
        source_kind="explicit_user_instruction",
        summary="Malformed workflow with no steps.",
    )
    malformed["procedure_kind"] = "workflow"
    valid = _procedure(
        "Valid policy",
        grounded_in=["fact-1"],
        source_kind="explicit_user_instruction",
    )
    store = _AsyncProceduralStore([_fact()])
    service = _service(store, [{"procedures": [malformed, valid]}])

    result = await service.synthesize_procedural("u1")

    assert result == {"status": "synthesized", "procedures_created": 1, "procedures_skipped": 1}
    procedures = [doc for doc in store.docs if doc.get("type") == "procedural"]
    assert [doc["name"] for doc in procedures] == ["Valid policy"]


@pytest.mark.asyncio
async def test_synthesize_procedural_quarantines_retryable_and_non_retryable_llm_errors() -> None:
    retryable_service = _service(_AsyncProceduralStore([_fact()]))

    async def raise_retryable(filename: str, inputs: dict[str, Any]) -> str:
        del filename, inputs
        raise RuntimeError("rate limit")

    retryable_service._run_prompty = raise_retryable  # type: ignore[method-assign]

    retryable_result = await retryable_service.synthesize_procedural("u1")

    assert retryable_result == {"status": "deferred", "procedures_created": 0}

    non_retryable_service = _service(_AsyncProceduralStore([_fact()]))

    async def raise_non_retryable(filename: str, inputs: dict[str, Any]) -> str:
        del filename, inputs
        raise RuntimeError("content_filter")

    non_retryable_service._run_prompty = raise_non_retryable  # type: ignore[method-assign]

    non_retryable_result = await non_retryable_service.synthesize_procedural("u1")

    assert non_retryable_result == {"status": "skipped", "procedures_created": 0}


@pytest.mark.asyncio
async def test_synthesize_procedural_maps_multi_procedure_lineage() -> None:
    store = _AsyncProceduralStore([_fact(), _episode()])
    service = _service(
        store,
        [
            {
                "procedures": [
                    _procedure(
                        "Fact lineage",
                        grounded_in=["fact-1"],
                        source_kind="explicit_user_instruction",
                    ),
                    _procedure(
                        "Episode lineage",
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
    procedures = {doc["name"]: doc for doc in store.docs if doc.get("type") == "procedural"}
    assert procedures["Fact lineage"]["source_fact_ids"] == ["fact-raw-1"]
    assert procedures["Fact lineage"]["source_episodic_ids"] == []
    assert procedures["Episode lineage"]["source_fact_ids"] == []
    assert procedures["Episode lineage"]["source_episodic_ids"] == ["episode-raw-1"]
