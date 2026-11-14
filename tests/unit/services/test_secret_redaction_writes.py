"""Literal credentials are redacted wherever the pipeline turns model output into a stored memory.

Extraction redacts a fact before its hash and id are derived (the activity result lands in Durable
history); persistence, episodes, summaries and procedures redact before embedding, so the vector
matches the stored text; and every direct container write passes the guard.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import compute_content_hash
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.services._pipeline_helpers import ID_SEED_SEP
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from azure.cosmos.agent_memory.store import MemoryStore
from tests.unit.aio.services.test_procedural_synthesis_and_context_async import AsyncChat as AsyncProcedureChat
from tests.unit.aio.services.test_procedural_synthesis_and_context_async import (
    AsyncEmbeddings as AsyncProcedureEmbeddings,
)
from tests.unit.aio.services.test_procedural_synthesis_and_context_async import AsyncMemories as AsyncProcedureMemories
from tests.unit.services.test_extract_dry import (
    _async_containers_for_store,
    _AsyncChat,
    _AsyncEmbeddings,
    _AsyncStore,
    _containers_for_store,
    _response,
    _Store,
    _SyncChat,
    _SyncEmbeddings,
    _turn,
)
from tests.unit.services.test_procedural_synthesis_and_context import Chat as ProcedureChat
from tests.unit.services.test_procedural_synthesis_and_context import Embeddings as ProcedureEmbeddings
from tests.unit.services.test_procedural_synthesis_and_context import Memories as ProcedureMemories
from tests.unit.services.test_procedural_synthesis_and_context import answer as procedure_answer
from tests.unit.services.test_procedural_synthesis_and_context import containers_for as procedure_containers
from tests.unit.services.test_procedural_synthesis_and_context import fact as procedure_source_fact

# GitHub tokens, a format the write guard redacts; built here so the source holds no token literal.
TOKEN = "ghp_" + "A1b2" * 9
OTHER_TOKEN = "ghp_" + "Z9y8" * 9
ACCOUNT_KEY = "k" * 86 + "=="
CONNECTION_STRING = f"AccountEndpoint=https://acct.documents.azure.com:443/;AccountKey={ACCOUNT_KEY};"


def _embedded(embeddings: _SyncEmbeddings) -> list[str]:
    return [text for call in embeddings.calls for text in call]


def _fact_doc(content: str) -> dict[str, Any]:
    content_hash = compute_content_hash(content)
    seed = ID_SEED_SEP.join(("u1", "t1", content_hash))
    return {
        "id": f"fact_{hashlib.sha256(seed.encode()).hexdigest()[:32]}",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": content,
        "content_hash": content_hash,
        "confidence": 0.9,
        "salience": 0.8,
        "tags": ["sys:fact", "sys:auto-extracted"],
        "prompt_id": "extract_memories.prompty",
        "prompt_version": "v1",
        "metadata": {"category": "preference"},
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }


def _episode_answer(summary: str) -> dict[str, Any]:
    return {
        "title": "Deployed staging",
        "summary": summary,
        "started_at": "2025-01-01T00:01:00+00:00",
        "ended_at": "2025-01-01T00:02:00+00:00",
        "participants": [],
        "events": [
            {
                "sequence": 1,
                "description": "The deploy ran.",
                "occurred_at": "2025-01-01T00:01:00+00:00",
                "source_turn_ids": ["turn-1"],
            }
        ],
        "outcome": None,
        "lessons": [],
        "salience": 0.8,
        "confidence": 0.9,
    }


# --- persistence -------------------------------------------------------------------------------------------


def test_persist_redacts_secrets_before_embedding() -> None:
    """Redacting only at write time would drop the vector built from the raw text, leaving the fact unsearchable."""
    store, embeddings = _Store([]), _SyncEmbeddings()
    service = PipelineService(store, _SyncChat([]), embeddings, containers=_containers_for_store(store))
    secret = _fact_doc(f"The user's Cosmos DB credentials are `{CONNECTION_STRING}`.")
    clean = _fact_doc("The user prefers Ruff for linting.")

    service.persist_extracted_memories("u1", {"facts": [secret, clean], "episodic": [], "updates": []})

    assert not any(ACCOUNT_KEY in text for text in _embedded(embeddings))
    stored = {doc["id"]: doc for doc in store.docs}
    assert ACCOUNT_KEY not in stored[secret["id"]]["content"]
    assert stored[secret["id"]]["content_hash"] == compute_content_hash(stored[secret["id"]]["content"])
    assert stored[secret["id"]]["metadata"]["classification"] == "secret"
    assert "sys:secret" in stored[secret["id"]]["tags"]
    assert stored[secret["id"]]["embedding"] == [1.0]
    assert stored[clean["id"]]["content"] == "The user prefers Ruff for linting."
    assert "sys:secret" not in stored[clean["id"]]["tags"]
    assert stored[clean["id"]]["embedding"] == [1.0]


async def test_async_persist_redacts_secrets_before_embedding() -> None:
    store, embeddings = _AsyncStore([]), _AsyncEmbeddings()
    service = AsyncPipelineService(store, _AsyncChat([]), embeddings, containers=_async_containers_for_store(store))
    secret = _fact_doc(f"The user's Cosmos DB credentials are `{CONNECTION_STRING}`.")
    clean = _fact_doc("The user prefers Ruff for linting.")

    await service.persist_extracted_memories("u1", {"facts": [secret, clean], "episodic": [], "updates": []})

    assert not any(ACCOUNT_KEY in text for text in _embedded(embeddings))
    stored = {doc["id"]: doc for doc in store.docs}
    assert ACCOUNT_KEY not in stored[secret["id"]]["content"]
    assert stored[secret["id"]]["metadata"]["classification"] == "secret"
    assert "sys:secret" in stored[secret["id"]]["tags"]
    assert stored[secret["id"]]["embedding"] == [1.0]
    assert "sys:secret" not in stored[clean["id"]]["tags"]


# --- extraction --------------------------------------------------------------------------------------------


def _extract(text: str, **fact: Any) -> dict[str, Any]:
    response = _response()
    response["facts"][0].update(text=text, **fact)
    memories, turns = _Store([]), _Store([_turn(0)])
    service = PipelineService(
        memories,
        _SyncChat([response]),
        _SyncEmbeddings(),
        containers=_containers_for_store(memories, turns_store=turns),
    )
    return service.extract_memories_durable("u1", "t1")["facts"][0]


async def _async_extract(text: str, **fact: Any) -> dict[str, Any]:
    response = _response()
    response["facts"][0].update(text=text, **fact)
    memories, turns = _AsyncStore([]), _AsyncStore([_turn(0)])
    service = AsyncPipelineService(
        memories,
        _AsyncChat([response]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories, turns_store=turns),
    )
    return (await service.extract_memories_durable("u1", "t1"))["facts"][0]


def test_extracted_fact_is_redacted_before_its_hash_and_id_are_derived() -> None:
    """The id comes from the redacted text, so it never depends on which credential the turn quoted."""
    fact = _extract(f"The staging token {TOKEN} is shared.")
    same_fact_other_token = _extract(f"The staging token {OTHER_TOKEN} is shared.")

    assert fact["id"] == same_fact_other_token["id"]
    assert TOKEN not in fact["content"]
    assert fact["content_hash"] == compute_content_hash(fact["content"])
    assert fact["metadata"]["classification"] == "secret"
    assert "sys:secret" in fact["tags"]


async def test_async_extracted_fact_is_redacted_before_its_hash_and_id_are_derived() -> None:
    fact = await _async_extract(f"The staging token {TOKEN} is shared.")
    same_fact_other_token = await _async_extract(f"The staging token {OTHER_TOKEN} is shared.")

    assert fact["id"] == same_fact_other_token["id"]
    assert TOKEN not in fact["content"]
    assert fact["content_hash"] == compute_content_hash(fact["content"])
    assert fact["metadata"]["classification"] == "secret"


def test_extracted_fact_metadata_is_redacted_before_it_leaves_extraction() -> None:
    """The activity result lands in Durable history, so model-written metadata is cleaned here too."""
    fact = _extract("The staging token was rotated.", temporal_context=f"old token {TOKEN} expired Friday")

    assert TOKEN not in repr(fact)
    assert fact["metadata"]["classification"] == "secret"


async def test_async_extracted_fact_metadata_is_redacted_before_it_leaves_extraction() -> None:
    fact = await _async_extract("The staging token was rotated.", temporal_context=f"old token {TOKEN} expired Friday")

    assert TOKEN not in repr(fact)
    assert fact["metadata"]["classification"] == "secret"


def test_a_clean_extracted_fact_is_not_flagged() -> None:
    fact = _extract("The user prefers Ruff for linting.")

    assert "sys:secret" not in fact["tags"]
    assert "classification" not in fact["metadata"]


# --- episodes ----------------------------------------------------------------------------------------------


def test_an_episode_is_redacted_before_it_is_embedded(monkeypatch) -> None:
    """The episode text reached the embedding provider raw; the write-time guard then dropped its vector."""
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    memories, turns, embeddings = _Store([]), _Store([_turn(1), _turn(2)]), _SyncEmbeddings()
    service = PipelineService(
        memories,
        _SyncChat([{"episodes": [_episode_answer(f"The user deployed with token {TOKEN} and the tests passed.")]}]),
        embeddings,
        containers=_containers_for_store(memories, turns_store=turns),
    )

    service.extract_episodes("u1", "t1", flush=True)

    assert not any(TOKEN in text for text in _embedded(embeddings))
    episode = next(doc for doc in memories.docs if doc.get("type") == "episodic")
    assert TOKEN not in repr(episode)
    assert episode["embedding"] == [1.0]


async def test_async_an_episode_is_redacted_before_it_is_embedded(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    memories, turns, embeddings = _AsyncStore([]), _AsyncStore([_turn(1), _turn(2)]), _AsyncEmbeddings()
    service = AsyncPipelineService(
        memories,
        _AsyncChat([{"episodes": [_episode_answer(f"The user deployed with token {TOKEN} and the tests passed.")]}]),
        embeddings,
        containers=_async_containers_for_store(memories, turns_store=turns),
    )

    await service.extract_episodes("u1", "t1", flush=True)

    assert not any(TOKEN in text for text in _embedded(embeddings))
    episode = next(doc for doc in memories.docs if doc.get("type") == "episodic")
    assert TOKEN not in repr(episode)
    assert episode["embedding"] == [1.0]


# --- procedures --------------------------------------------------------------------------------------------


def _procedure_summary() -> str:
    return f"Deploy staging with {TOKEN}."


def test_a_synthesized_procedure_is_redacted_before_it_is_embedded() -> None:
    memories, embeddings = ProcedureMemories([procedure_source_fact()]), ProcedureEmbeddings()
    chat = ProcedureChat([{"procedures": [procedure_answer(summary=_procedure_summary())]}])
    containers = procedure_containers(memories)
    service = PipelineService(
        MemoryStore(containers=containers, embeddings_client=embeddings), chat, embeddings, containers=containers
    )

    assert service.synthesize_procedural("u1")["procedures_created"] == 1

    assert _embedded(embeddings) and not any(TOKEN in text for text in _embedded(embeddings))
    procedure = next(doc for doc in memories.docs if doc.get("type") == "procedural")
    assert TOKEN not in repr(procedure)
    assert procedure["metadata"]["classification"] == "secret"
    assert procedure["embedding"] == [1.0], "the vector was built from the redacted text and is kept"


async def test_async_a_synthesized_procedure_is_redacted_before_it_is_embedded() -> None:
    memories, embeddings = ProcedureMemories([procedure_source_fact()]), AsyncProcedureEmbeddings()
    chat = AsyncProcedureChat([{"procedures": [procedure_answer(summary=_procedure_summary())]}])
    containers = procedure_containers(AsyncProcedureMemories(memories))
    service = AsyncPipelineService(
        AsyncMemoryStore(containers=containers, embeddings_client=embeddings), chat, embeddings, containers=containers
    )

    assert (await service.synthesize_procedural("u1"))["procedures_created"] == 1

    assert _embedded(embeddings) and not any(TOKEN in text for text in _embedded(embeddings))
    procedure = next(doc for doc in memories.docs if doc.get("type") == "procedural")
    assert TOKEN not in repr(procedure)
    assert procedure["embedding"] == [1.0]


# --- direct container writes -------------------------------------------------------------------------------


def _procedural_doc_quoting_a_token() -> dict[str, Any]:
    secret = f"token {TOKEN}"
    return {
        "id": "proc_1",
        "user_id": "u1",
        "type": "procedural",
        "content": f"Deploy {secret}",
        "summary": f"Use {secret}",
        "steps": [{"sequence": 1, "instruction": f"login {secret}"}],
    }


def _written_body(containers: dict[ContainerKey, Any], helper: str) -> dict[str, Any]:
    target = ContainerKey.SUMMARIES if helper == "_upsert_summary" else ContainerKey.MEMORIES
    method = "create_item" if helper == "_create_memory" else "upsert_item"
    return getattr(containers[target], method).call_args.kwargs["body"]


@pytest.mark.parametrize("helper", ["_upsert_memory", "_create_memory", "_upsert_summary"])
def test_direct_pipeline_writes_are_sanitized(helper) -> None:
    """These helpers write straight to the container and never pass through the store."""
    containers = {key: MagicMock() for key in (ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES)}
    service = PipelineService(MagicMock(), _SyncChat([]), _SyncEmbeddings(), containers=containers)
    original = _procedural_doc_quoting_a_token()

    getattr(service, helper)(original)

    body = _written_body(containers, helper)
    assert TOKEN not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert TOKEN in repr(original), "the caller's document is not rewritten"


@pytest.mark.parametrize("helper", ["_upsert_memory", "_create_memory", "_upsert_summary"])
async def test_async_direct_pipeline_writes_are_sanitized(helper) -> None:
    containers = {key: AsyncMock() for key in (ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES)}
    service = AsyncPipelineService(MagicMock(), _AsyncChat([]), _AsyncEmbeddings(), containers=containers)

    await getattr(service, helper)(_procedural_doc_quoting_a_token())

    body = _written_body(containers, helper)
    assert TOKEN not in repr(body)
    assert body["metadata"]["classification"] == "secret"
