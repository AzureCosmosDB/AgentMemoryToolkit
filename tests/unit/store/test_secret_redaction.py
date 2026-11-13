"""The store redacts literal credentials before it derives, embeds or writes anything."""

from __future__ import annotations

import copy
import logging
from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosAccessConditionFailedError

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


def test_upsert_memory_without_an_embedder_redacts_a_prebuilt_record_and_drops_its_stale_vector():
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
    assert "embedding" not in body, "the store has no embeddings client to rebuild it with"
    assert record["content"] == raw, "the caller's record is not rewritten"


def _prebuilt(content, **overrides):
    return {
        "id": "fact_1",
        "type": "fact",
        "user_id": "u1",
        "thread_id": "t1",
        "content": content,
        "content_hash": compute_content_hash(content),
        **overrides,
    }


def _embedder(vector=None):
    embeddings = MagicMock()
    embeddings.generate.return_value = vector or [0.1, 0.2]
    return embeddings


def test_upsert_memory_rebuilds_a_dropped_vector_from_the_redacted_text():
    """Dropping it removed the record from vector search; the stale vector was built from the raw text."""
    memories = MagicMock()
    embeddings = _embedder()
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.upsert_memory(_prebuilt(f"Deploy with {SECRET}", embedding=[9.9]))

    body = memories.upsert_item.call_args.kwargs["body"]
    embeddings.generate.assert_called_once_with(body["content"])
    assert TOKEN not in body["content"]
    assert body["embedding"] == [0.1, 0.2]


def test_upsert_memory_rebuilds_a_procedures_vector_from_its_retrieval_text():
    memories = MagicMock()
    embeddings = _embedder()
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)
    record = _prebuilt("Deploy the service", type="procedural", retrieval_text=f"deploy with {SECRET}", embedding=[9.9])

    store.upsert_memory(record)

    body = memories.upsert_item.call_args.kwargs["body"]
    embeddings.generate.assert_called_once_with(body["retrieval_text"])
    assert TOKEN not in body["retrieval_text"]
    assert body["embedding"] == [0.1, 0.2]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_prebuilt(f"Deploy with {SECRET}"), None),
        (_prebuilt("Prefers Ruff.", embedding=[9.9]), [9.9]),
    ],
    ids=["secret-without-a-vector", "clean-with-a-vector"],
)
def test_upsert_memory_embeds_only_to_replace_a_vector_the_guard_dropped(record, expected):
    memories = MagicMock()
    embeddings = _embedder()
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.upsert_memory(record)

    embeddings.generate.assert_not_called()
    assert memories.upsert_item.call_args.kwargs["body"].get("embedding") == expected


def test_upsert_memory_stores_without_a_vector_when_embedding_fails(caplog):
    caplog.set_level(logging.DEBUG)
    memories = MagicMock()
    embeddings = MagicMock()
    embeddings.generate.side_effect = RuntimeError(f"quota exceeded for {SECRET}")
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.upsert_memory(_prebuilt(f"Deploy with {SECRET}", embedding=[9.9]))

    assert "embedding" not in memories.upsert_item.call_args.kwargs["body"]
    assert "embedding generation failed" in caplog.text
    assert TOKEN not in caplog.text and "quota exceeded" not in caplog.text, "a log never quotes the error's text"


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


def test_update_without_an_embedder_redacts_new_content_and_metadata_and_drops_the_vector():
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
    assert "embedding" not in body, "the store has no embeddings client to rebuild it with"


def _update_store(stored, embeddings):
    memories = MagicMock()
    memories.read_item.side_effect = lambda **_kwargs: copy.deepcopy(stored)
    return MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings), memories


def test_update_rebuilds_the_vector_from_the_redacted_content():
    """The old vector was built from the old, clean text; dropping it only removed the record from vector search."""
    embeddings = _embedder()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content=f"Use {SECRET}")

    body = memories.replace_item.call_args.kwargs["body"]
    embeddings.generate.assert_called_once_with(body["content"])
    assert TOKEN not in body["content"]
    assert body["embedding"] == [0.1, 0.2]


def test_a_metadata_only_update_rebuilds_the_vector_of_a_record_that_already_held_a_secret():
    embeddings = _embedder()
    store, memories = _update_store(_stored_doc(content=f"Use {SECRET}", embedding=[9.9]), embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", metadata={"note": "rotated"})

    body = memories.replace_item.call_args.kwargs["body"]
    embeddings.generate.assert_called_once_with(body["content"])
    assert TOKEN not in repr(body)
    assert body["embedding"] == [0.1, 0.2]


def test_update_rebuilds_the_vector_when_the_content_changes():
    embeddings = _embedder()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    embeddings.generate.assert_called_once_with("Prefers Ruff.")
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [0.1, 0.2]


@pytest.mark.parametrize(
    "changes",
    [{"metadata": {"note": "rotated"}}, {"role": "user"}, {"content": "hello"}],
    ids=["metadata", "role", "same-content"],
)
def test_update_keeps_the_vector_when_the_text_it_encodes_is_unchanged(changes):
    embeddings = _embedder()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", **changes)

    embeddings.generate.assert_not_called()
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [9.9]


def test_update_does_not_embed_a_record_that_has_no_vector():
    embeddings = _embedder()
    store, memories = _update_store(_stored_doc(), embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content=f"Use {SECRET}")

    embeddings.generate.assert_not_called()
    assert "embedding" not in memories.replace_item.call_args.kwargs["body"]


def test_update_keeps_a_procedures_vector_when_only_its_content_changes():
    """A procedure is embedded from its retrieval_text, which a content update leaves as it was."""
    embeddings = _embedder()
    stored = _stored_doc(type="procedural", retrieval_text="deploy the service", embedding=[9.9])
    store, memories = _update_store(stored, embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="procedural", content="Run the deploy script.")

    embeddings.generate.assert_not_called()
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [9.9]


def test_update_rebuilds_a_procedures_vector_from_its_retrieval_text():
    """Redacting the content drops the vector although the text it encodes did not change."""
    embeddings = _embedder()
    stored = _stored_doc(type="procedural", retrieval_text="deploy the service", embedding=[9.9])
    store, memories = _update_store(stored, embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="procedural", content=f"Use {SECRET}")

    embeddings.generate.assert_called_once_with("deploy the service")
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [0.1, 0.2]


def test_update_drops_the_stale_vector_when_embedding_fails(caplog):
    caplog.set_level(logging.DEBUG)
    embeddings = MagicMock()
    embeddings.generate.side_effect = RuntimeError(f"quota exceeded for {SECRET}")
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    assert "embedding" not in memories.replace_item.call_args.kwargs["body"]
    assert "embedding generation failed" in caplog.text
    assert TOKEN not in caplog.text and "quota exceeded" not in caplog.text, "a log never quotes the error's text"


def test_an_etag_conflict_retry_does_not_embed_the_same_text_again(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)
    embeddings = _embedder()
    store, memories = _update_store(_stored_doc(embedding=[9.9], _etag="v1"), embeddings)
    memories.replace_item.side_effect = [CosmosAccessConditionFailedError(message="412", response=None), None]

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    assert memories.replace_item.call_count == 2
    embeddings.generate.assert_called_once_with("Prefers Ruff.")
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [0.1, 0.2]


def _tagged_store(tags=()):
    memories = MagicMock()
    memories.read_item.side_effect = lambda **_kwargs: _stored_doc(tags=list(tags))
    return MemoryStore(containers=_containers(memories=memories)), memories


def test_add_tags_drops_a_credential_tag_and_flags_the_record():
    store, memories = _tagged_store(["topic:deploy"])

    store.add_tags("m1", "u1", "t1", "fact", [TOKEN, "Release"])

    body = memories.replace_item.call_args.kwargs["body"]
    assert body["tags"] == ["release", "sys:secret", "topic:deploy"]
    assert TOKEN not in repr(body) and TOKEN.lower() not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert body["metadata"]["secret_patterns"] == ["github_token"]


def test_add_tags_checks_a_tag_before_it_is_lowercased():
    """An AWS access key id is only recognized in upper case, and tags are stored lower case."""
    store, memories = _tagged_store()

    store.add_tags("m1", "u1", "t1", "fact", ["AKIA" + "IOSFODNN7EXAMPLE"])

    body = memories.replace_item.call_args.kwargs["body"]
    assert "akiaiosfodnn7example" not in repr(body).lower()
    assert body["tags"] == ["sys:secret"]
    assert body["metadata"]["secret_patterns"] == ["aws_access_key_id"]


def test_add_tags_leaves_a_clean_record_unflagged_and_remove_tags_never_flags():
    store, memories = _tagged_store(["a", "b"])

    store.add_tags("m1", "u1", "t1", "fact", ["c"])
    assert memories.replace_item.call_args.kwargs["body"]["tags"] == ["a", "b", "c"]
    assert "classification" not in memories.replace_item.call_args.kwargs["body"]["metadata"]

    store.remove_tags("m1", "u1", "t1", "fact", [TOKEN, "a"])
    assert memories.replace_item.call_args.kwargs["body"]["tags"] == ["b"]
    assert "classification" not in memories.replace_item.call_args.kwargs["body"]["metadata"]


def test_add_tags_logs_the_formats_found_never_the_tag(caplog):
    caplog.set_level(logging.DEBUG)
    store, _memories = _tagged_store()

    store.add_tags("m1", "u1", "t1", "fact", [TOKEN])

    assert "patterns=['github_token']" in caplog.text
    assert TOKEN not in caplog.text and TOKEN.lower() not in caplog.text
