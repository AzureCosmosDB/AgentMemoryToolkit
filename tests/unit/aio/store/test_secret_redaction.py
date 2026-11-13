"""The async store redacts literal credentials before it derives, embeds or writes anything."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import compute_content_hash
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore

# A GitHub token, a format the write guard redacts; built here so the source holds no token literal.
TOKEN = "ghp_" + "A1b2" * 9
SECRET = f"token {TOKEN}"


def _container():
    container = MagicMock()
    container.upsert_item = AsyncMock()
    container.replace_item = AsyncMock()
    return container


def _containers(*, turns=None, memories=None):
    return {
        ContainerKey.TURNS: turns if turns is not None else _container(),
        ContainerKey.MEMORIES: memories if memories is not None else _container(),
        ContainerKey.SUMMARIES: _container(),
    }


def _stored_doc(**overrides):
    doc = {
        "id": "m1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": "hello",
        "metadata": {},
        "created_at": "2025-01-01T00:00:00+00:00",
        "tags": [],
    }
    doc.update(overrides)
    return doc


def _embeddings(*, vector=None, batch=None):
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=vector or [0.1, 0.2])
    embeddings.generate_batch = AsyncMock(return_value=batch or [[0.1, 0.2]])
    return embeddings


async def test_async_add_redacts_before_deriving_hash_and_embedding():
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.add(user_id="u1", role="user", content=f"Deploy with {SECRET}", memory_type="fact", thread_id="t1")

    body = memories.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in embeddings.generate.call_args.args[0], "the vector would otherwise encode the secret"
    assert TOKEN not in body["content"]
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert body["embedding"] == [0.1, 0.2]
    assert body["metadata"]["classification"] == "secret"
    assert "sys:secret" in body["tags"]


async def test_async_add_redacts_a_turn_and_flags_it():
    turns = _container()
    store = AsyncMemoryStore(containers=_containers(turns=turns))

    await store.add(
        user_id="u1", role="user", content=f"Deploy with {SECRET}", thread_id="t1", metadata={"note": SECRET}
    )

    body = turns.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert "sys:secret" in body["tags"]


async def test_async_add_leaves_a_clean_record_unflagged():
    memories = _container()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.add(user_id="u1", role="user", content="Prefers Ruff.", memory_type="fact", thread_id="t1")

    body = memories.upsert_item.call_args.kwargs["body"]
    assert "sys:secret" not in body["tags"]
    assert "classification" not in body["metadata"]


async def test_async_add_discards_a_supplied_embedding_built_from_secret_bearing_text():
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.add(user_id="u1", role="user", content=f"Deploy with {SECRET}", memory_type="fact", embedding=[9.9])

    body = memories.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]
    assert TOKEN not in embeddings.generate.call_args.args[0]


async def test_async_add_does_not_regenerate_a_discarded_embedding_when_embedding_is_off():
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.add(
        user_id="u1", role="user", content=f"Deploy with {SECRET}", memory_type="fact", embedding=[9.9], embed=False
    )

    assert "embedding" not in memories.upsert_item.call_args.kwargs["body"]
    embeddings.generate.assert_not_called()


async def test_async_add_keeps_a_supplied_embedding_for_clean_text():
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.add(user_id="u1", role="user", content="Prefers Ruff.", memory_type="fact", embedding=[9.9])

    assert memories.upsert_item.call_args.kwargs["body"]["embedding"] == [9.9]
    embeddings.generate.assert_not_called()


async def test_async_upsert_memory_redacts_a_prebuilt_record_and_drops_its_stale_vector():
    memories = _container()
    store = AsyncMemoryStore(containers=_containers(memories=memories))
    raw = f"Deploy with {SECRET}"
    record = {
        "id": "fact_1",
        "type": "fact",
        "user_id": "u1",
        "thread_id": "t1",
        "content": raw,
        "content_hash": compute_content_hash(raw),
        "embedding": [9.9],
    }

    await store.upsert_memory(record)

    body = memories.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in repr(body)
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert "embedding" not in body


@pytest.mark.parametrize("supplied", [None, [9.9]], ids=["no-vector", "stale-vector"])
async def test_async_push_embeds_only_redacted_text(supplied):
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)
    raw = f"Deploy with {SECRET}"
    record = {
        "id": "fact_1",
        "type": "fact",
        "user_id": "u1",
        "thread_id": "t1",
        "content": raw,
        "content_hash": compute_content_hash(raw),
    }
    if supplied is not None:
        record["embedding"] = supplied
    local = [record]

    await store.push(local)

    assert TOKEN not in repr(embeddings.generate_batch.call_args.args[0])
    body = memories.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in body["content"]
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert body["embedding"] == [0.1, 0.2]
    assert local[0]["embedding"] == [0.1, 0.2]


async def test_async_update_redacts_new_content_and_metadata():
    memories = _container()
    memories.read_item = AsyncMock(
        return_value=_stored_doc(embedding=[9.9], content_hash=compute_content_hash("hello"))
    )
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.update(
        "m1", user_id="u1", thread_id="t1", memory_type="fact", content=f"Use {SECRET}", metadata={"note": SECRET}
    )

    body = memories.replace_item.call_args.kwargs["body"]
    assert TOKEN not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert "embedding" not in body
