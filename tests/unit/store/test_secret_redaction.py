"""The store redacts literal credentials before it derives, embeds or writes anything."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import compute_content_hash
from azure.cosmos.agent_memory.store import MemoryStore

# A GitHub token, a format the write guard redacts; built here so the source holds no token literal.
TOKEN = "ghp_" + "A1b2" * 9
SECRET = f"token {TOKEN}"


def _containers(*, turns=None, memories=None):
    return {
        ContainerKey.TURNS: turns if turns is not None else MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
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


def test_add_redacts_before_deriving_hash_and_embedding():
    """Redaction must precede every derived value, not just the stored content."""
    memories = MagicMock()
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.add(user_id="u1", role="user", content=f"Deploy with {SECRET}", memory_type="fact", thread_id="t1")

    body = memories.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in embeddings.generate.call_args.args[0], "the vector would otherwise encode the secret"
    assert TOKEN not in body["content"]
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert body["embedding"] == [0.1, 0.2]
    assert body["metadata"]["classification"] == "secret"
    assert "sys:secret" in body["tags"]


def test_add_redacts_a_turn_and_flags_it():
    turns = MagicMock()
    store = MemoryStore(containers=_containers(turns=turns))

    store.add(user_id="u1", role="user", content=f"Deploy with {SECRET}", thread_id="t1", metadata={"note": SECRET})

    body = turns.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert body["metadata"]["secret_patterns"] == ["github_token"]
    assert "sys:secret" in body["tags"]


def test_add_leaves_a_clean_record_unflagged():
    memories = MagicMock()
    store = MemoryStore(containers=_containers(memories=memories))

    store.add(user_id="u1", role="user", content="Prefers Ruff.", memory_type="fact", thread_id="t1")

    body = memories.upsert_item.call_args.kwargs["body"]
    assert "sys:secret" not in body["tags"]
    assert "classification" not in body["metadata"]


def test_add_discards_a_supplied_embedding_built_from_secret_bearing_text():
    """Only the content was redacted; the caller's vector still encoded the credential."""
    memories = MagicMock()
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.add(user_id="u1", role="user", content=f"Deploy with {SECRET}", memory_type="fact", embedding=[9.9])

    body = memories.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]
    assert TOKEN not in embeddings.generate.call_args.args[0]


def test_add_does_not_regenerate_a_discarded_embedding_when_embedding_is_off():
    memories = MagicMock()
    embeddings = MagicMock()
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.add(
        user_id="u1", role="user", content=f"Deploy with {SECRET}", memory_type="fact", embedding=[9.9], embed=False
    )

    assert "embedding" not in memories.upsert_item.call_args.kwargs["body"]
    embeddings.generate.assert_not_called()


def test_add_keeps_a_supplied_embedding_for_clean_text():
    memories = MagicMock()
    embeddings = MagicMock()
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.add(user_id="u1", role="user", content="Prefers Ruff.", memory_type="fact", embedding=[9.9])

    assert memories.upsert_item.call_args.kwargs["body"]["embedding"] == [9.9]
    embeddings.generate.assert_not_called()


def test_upsert_memory_redacts_a_prebuilt_record_and_drops_its_stale_vector():
    memories = MagicMock()
    store = MemoryStore(containers=_containers(memories=memories))
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

    store.upsert_memory(record)

    body = memories.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in repr(body)
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert "embedding" not in body
    assert record["content"] == raw, "the caller's record is not rewritten"


@pytest.mark.parametrize("supplied", [None, [9.9]], ids=["no-vector", "stale-vector"])
def test_push_embeds_only_redacted_text(supplied):
    """Records built outside add() arrive with their own derived fields; push() used to embed them unsanitized."""
    memories = MagicMock()
    embeddings = MagicMock()
    embeddings.generate_batch.return_value = [[0.1, 0.2]]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)
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

    store.push(local)

    assert TOKEN not in repr(embeddings.generate_batch.call_args.args[0])
    body = memories.upsert_item.call_args.kwargs["body"]
    assert TOKEN not in body["content"]
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert body["embedding"] == [0.1, 0.2]
    assert local[0]["embedding"] == [0.1, 0.2], "the caller gets the vector of the stored text"


def test_update_redacts_new_content_and_metadata():
    memories = MagicMock()
    memories.read_item.return_value = _stored_doc(embedding=[9.9], content_hash=compute_content_hash("hello"))
    store = MemoryStore(containers=_containers(memories=memories))

    store.update(
        "m1", user_id="u1", thread_id="t1", memory_type="fact", content=f"Use {SECRET}", metadata={"note": SECRET}
    )

    body = memories.replace_item.call_args.kwargs["body"]
    assert TOKEN not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert body["content_hash"] == compute_content_hash(body["content"])
    assert "embedding" not in body, "the old vector no longer matches the redacted content"
