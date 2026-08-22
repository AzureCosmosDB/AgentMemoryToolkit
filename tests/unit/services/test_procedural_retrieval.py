from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.store import MemoryStore


def _containers(*, memories: Any = None, turns: Any = None, summaries: Any = None) -> dict[ContainerKey, Any]:
    return {
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.TURNS: turns if turns is not None else MagicMock(),
        ContainerKey.SUMMARIES: summaries if summaries is not None else MagicMock(),
    }


def _params_by_name(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {param["name"]: param["value"] for param in call_kwargs["parameters"]}


def test_retrieve_procedures_filters_active_procedures_by_default() -> None:
    ranked_docs = [
        {"id": "proc-1", "type": "procedural", "status": "active", "similarity_score": 0.1},
        {"id": "proc-2", "type": "procedural", "status": "active", "similarity_score": 0.2},
    ]
    memories = MagicMock()
    memories.query_items.return_value = ranked_docs
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    result = store.retrieve_procedures("u1", "cosmos db retry", top_k=2)

    assert result == ranked_docs
    call_kwargs = memories.query_items.call_args.kwargs
    assert "TOP 2" in call_kwargs["query"]
    assert "c.type = @type" in call_kwargs["query"]
    assert "c.scope_key = @scope_key" in call_kwargs["query"]
    assert "c.status = @status" in call_kwargs["query"]
    assert "VectorDistance(c.embedding, @embedding)" in call_kwargs["query"]
    assert "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@type"] == "procedural"
    assert params["@scope_key"] == "user:u1"
    assert params["@status"] == "active"
    assert params["@embedding"] == [0.1, 0.2]
    assert params["@kw0"] == "cosmos"


def test_retrieve_procedures_status_none_drops_status_filter() -> None:
    ranked_docs = [{"id": "proc-domain", "type": "procedural", "similarity_score": 0.1}]
    memories = MagicMock()
    memories.query_items.return_value = ranked_docs
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.3, 0.4]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    result = store.retrieve_procedures(
        "u1",
        "partition key",
        procedure_kind="recovery_strategy",
        status=None,
    )

    assert result == ranked_docs
    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.procedure_kind = @procedure_kind" in call_kwargs["query"]
    assert "c.status = @status" not in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@type"] == "procedural"
    assert params["@scope_key"] == "user:u1"
    assert params["@procedure_kind"] == "recovery_strategy"
    assert "@status" not in params


def test_retrieve_procedures_filters_by_kind_and_can_include_superseded() -> None:
    ranked_docs = [{"id": "proc-workflow", "type": "procedural", "similarity_score": 0.1}]
    memories = MagicMock()
    memories.query_items.return_value = ranked_docs
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.5, 0.6]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    result = store.retrieve_procedures(
        "u1",
        "workflow retry",
        procedure_kind="workflow",
        include_superseded=True,
    )

    assert result == ranked_docs
    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.procedure_kind = @procedure_kind" in call_kwargs["query"]
    assert "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))" not in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@procedure_kind"] == "workflow"
    assert params["@status"] == "active"
