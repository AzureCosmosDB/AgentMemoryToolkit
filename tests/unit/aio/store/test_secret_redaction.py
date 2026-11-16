"""The async store redacts literal credentials before it derives, embeds or writes anything."""

from __future__ import annotations

import copy
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.cosmos.exceptions import CosmosAccessConditionFailedError

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


async def test_async_upsert_memory_without_an_embedder_redacts_a_prebuilt_record_and_drops_its_stale_vector():
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
    assert "embedding" not in body, "the store has no embeddings client to rebuild it with"


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


async def test_async_upsert_memory_rebuilds_a_dropped_vector_from_the_redacted_text():
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.upsert_memory(_prebuilt(f"Deploy with {SECRET}", embedding=[9.9]))

    body = memories.upsert_item.call_args.kwargs["body"]
    embeddings.generate.assert_awaited_once_with(body["content"])
    assert TOKEN not in body["content"]
    assert body["embedding"] == [0.1, 0.2]


async def test_async_upsert_memory_rebuilds_a_procedures_vector_from_its_retrieval_text():
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)
    record = _prebuilt("Deploy the service", type="procedural", retrieval_text=f"deploy with {SECRET}", embedding=[9.9])

    await store.upsert_memory(record)

    body = memories.upsert_item.call_args.kwargs["body"]
    embeddings.generate.assert_awaited_once_with(body["retrieval_text"])
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
async def test_async_upsert_memory_embeds_only_to_replace_a_vector_the_guard_dropped(record, expected):
    memories = _container()
    embeddings = _embeddings()
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.upsert_memory(record)

    embeddings.generate.assert_not_called()
    assert memories.upsert_item.call_args.kwargs["body"].get("embedding") == expected


async def test_async_upsert_memory_stores_without_a_vector_when_embedding_fails(caplog):
    caplog.set_level(logging.DEBUG)
    memories = _container()
    embeddings = _embeddings()
    embeddings.generate = AsyncMock(side_effect=RuntimeError(f"quota exceeded for {SECRET}"))
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.upsert_memory(_prebuilt(f"Deploy with {SECRET}", embedding=[9.9]))

    assert "embedding" not in memories.upsert_item.call_args.kwargs["body"]
    assert "embedding generation failed" in caplog.text
    assert TOKEN not in caplog.text and "quota exceeded" not in caplog.text, "a log never quotes the error's text"


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


async def test_async_update_without_an_embedder_redacts_new_content_and_metadata_and_keeps_the_previous_vector():
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
    assert body["embedding"] == [9.9], "the store has no embeddings client to rebuild it with"


def _update_store(stored, embeddings):
    memories = _container()
    memories.read_item = AsyncMock(side_effect=lambda **_kwargs: copy.deepcopy(stored))
    return AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings), memories


def _failing_embedder():
    embeddings = _embeddings()
    embeddings.generate = AsyncMock(side_effect=RuntimeError(f"quota exceeded for {SECRET}"))
    return embeddings


async def test_async_update_without_an_embedder_keeps_the_previous_vector_when_the_content_changes():
    store, memories = _update_store(_stored_doc(embedding=[9.9]), None)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    body = memories.replace_item.call_args.kwargs["body"]
    assert body["content"] == "Prefers Ruff."
    assert body["embedding"] == [9.9]


async def test_async_update_rebuilds_the_vector_from_the_redacted_content():
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content=f"Use {SECRET}")

    body = memories.replace_item.call_args.kwargs["body"]
    embeddings.generate.assert_awaited_once_with(body["content"])
    assert TOKEN not in body["content"]
    assert body["embedding"] == [0.1, 0.2]


async def test_an_async_metadata_only_update_rebuilds_the_vector_of_a_record_that_already_held_a_secret():
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(content=f"Use {SECRET}", embedding=[9.9]), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", metadata={"note": "rotated"})

    body = memories.replace_item.call_args.kwargs["body"]
    embeddings.generate.assert_awaited_once_with(body["content"])
    assert TOKEN not in repr(body)
    assert body["embedding"] == [0.1, 0.2]


async def test_async_update_rebuilds_the_vector_when_the_content_changes():
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    embeddings.generate.assert_awaited_once_with("Prefers Ruff.")
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [0.1, 0.2]


@pytest.mark.parametrize(
    "changes",
    [{"metadata": {"note": "rotated"}}, {"role": "user"}, {"content": "hello"}],
    ids=["metadata", "role", "same-content"],
)
async def test_async_update_keeps_the_vector_when_the_text_it_encodes_is_unchanged(changes):
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", **changes)

    embeddings.generate.assert_not_called()
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [9.9]


async def test_async_update_does_not_embed_a_record_that_has_no_vector():
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content=f"Use {SECRET}")

    embeddings.generate.assert_not_called()
    assert "embedding" not in memories.replace_item.call_args.kwargs["body"]


async def test_async_update_drops_the_vector_of_a_record_whose_text_is_emptied_without_embedding_it():
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(embedding=[9.9]), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="")

    embeddings.generate.assert_not_called()
    assert "embedding" not in memories.replace_item.call_args.kwargs["body"]


async def test_async_update_keeps_a_procedures_vector_when_only_its_content_changes():
    embeddings = _embeddings()
    stored = _stored_doc(type="procedural", retrieval_text="deploy the service", embedding=[9.9])
    store, memories = _update_store(stored, embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="procedural", content="Run the deploy script.")

    embeddings.generate.assert_not_called()
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [9.9]


async def test_async_update_rebuilds_a_procedures_vector_from_its_retrieval_text():
    embeddings = _embeddings()
    stored = _stored_doc(type="procedural", retrieval_text="deploy the service", embedding=[9.9])
    store, memories = _update_store(stored, embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="procedural", content=f"Use {SECRET}")

    embeddings.generate.assert_awaited_once_with("deploy the service")
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [0.1, 0.2]


@pytest.mark.parametrize(
    "content",
    ["Prefers Ruff.", f"Use {SECRET}"],
    ids=["clean-content", "content-with-a-secret"],
)
async def test_async_update_keeps_the_previous_vector_when_embedding_fails(content, caplog):
    caplog.set_level(logging.DEBUG)
    store, memories = _update_store(_stored_doc(embedding=[9.9]), _failing_embedder())

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content=content)

    body = memories.replace_item.call_args.kwargs["body"]
    assert body["embedding"] == [9.9]
    assert TOKEN not in repr(body)
    assert "embedding generation failed for m1 (RuntimeError" in caplog.text
    assert "keeping its previous vector" in caplog.text
    assert TOKEN not in caplog.text and "quota exceeded" not in caplog.text, "a log never quotes the error's text"
    assert "Prefers Ruff." not in caplog.text, "a log never quotes the memory"


@pytest.mark.parametrize("embedder", ["fails", "missing"])
@pytest.mark.parametrize(
    "changes",
    [{"metadata": {"note": "rotated"}}, {"content": "Prefers Ruff."}],
    ids=["metadata-only", "clean-content"],
)
async def test_async_update_drops_a_vector_built_from_a_secret_when_it_cannot_be_rebuilt(changes, embedder, caplog):
    caplog.set_level(logging.DEBUG)
    embeddings = _failing_embedder() if embedder == "fails" else None
    store, memories = _update_store(_stored_doc(content=f"Use {SECRET}", embedding=[9.9]), embeddings)

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", **changes)

    body = memories.replace_item.call_args.kwargs["body"]
    assert "embedding" not in body
    assert TOKEN not in repr(body)
    assert "dropping its previous vector" in caplog.text
    assert TOKEN not in caplog.text and "quota exceeded" not in caplog.text


@pytest.mark.parametrize(
    ("retrieval_text", "expected"),
    [("deploy the service", [9.9]), (f"deploy with {SECRET}", None)],
    ids=["retrieval-text-clean", "retrieval-text-held-a-secret"],
)
async def test_async_update_judges_a_procedures_vector_by_the_text_it_was_built_from(retrieval_text, expected):
    stored = _stored_doc(type="procedural", retrieval_text=retrieval_text, content=f"Use {SECRET}", embedding=[9.9])
    store, memories = _update_store(stored, _failing_embedder())

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="procedural", metadata={"note": "rotated"})

    body = memories.replace_item.call_args.kwargs["body"]
    assert body.get("embedding") == expected
    assert TOKEN not in repr(body)


async def test_an_async_etag_conflict_retry_keeps_the_vector_it_reads_again_when_embedding_fails(monkeypatch):
    async def _noop_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    memories = _container()
    memories.read_item = AsyncMock(
        side_effect=[_stored_doc(embedding=[9.9], _etag="v1"), _stored_doc(embedding=[7.7], _etag="v2")]
    )
    memories.replace_item = AsyncMock(
        side_effect=[CosmosAccessConditionFailedError(message="412", response=None), None]
    )
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=_failing_embedder())

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    assert memories.replace_item.await_count == 2
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [7.7]


async def test_an_async_etag_conflict_retry_does_not_embed_the_same_text_again(monkeypatch):
    async def _noop_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    embeddings = _embeddings()
    store, memories = _update_store(_stored_doc(embedding=[9.9], _etag="v1"), embeddings)
    memories.replace_item = AsyncMock(
        side_effect=[CosmosAccessConditionFailedError(message="412", response=None), None]
    )

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="Prefers Ruff.")

    assert memories.replace_item.await_count == 2
    embeddings.generate.assert_awaited_once_with("Prefers Ruff.")
    assert memories.replace_item.call_args.kwargs["body"]["embedding"] == [0.1, 0.2]


def _tagged_store(tags=()):
    memories = _container()
    memories.read_item = AsyncMock(side_effect=lambda **_kwargs: _stored_doc(tags=list(tags)))
    return AsyncMemoryStore(containers=_containers(memories=memories)), memories


async def test_async_add_tags_drops_a_credential_tag_and_flags_the_record():
    store, memories = _tagged_store(["topic:deploy"])

    await store.add_tags("m1", "u1", "t1", "fact", [TOKEN, "Release"])

    body = memories.replace_item.call_args.kwargs["body"]
    assert body["tags"] == ["release", "sys:secret", "topic:deploy"]
    assert TOKEN not in repr(body) and TOKEN.lower() not in repr(body)
    assert body["metadata"]["classification"] == "secret"
    assert body["metadata"]["secret_patterns"] == ["github_token"]


async def test_async_add_tags_checks_a_tag_before_it_is_lowercased():
    store, memories = _tagged_store()

    await store.add_tags("m1", "u1", "t1", "fact", ["AKIA" + "IOSFODNN7EXAMPLE"])

    body = memories.replace_item.call_args.kwargs["body"]
    assert "akiaiosfodnn7example" not in repr(body).lower()
    assert body["tags"] == ["sys:secret"]
    assert body["metadata"]["secret_patterns"] == ["aws_access_key_id"]


async def test_async_add_tags_leaves_a_clean_record_unflagged_and_remove_tags_never_flags():
    store, memories = _tagged_store(["a", "b"])

    await store.add_tags("m1", "u1", "t1", "fact", ["c"])
    assert memories.replace_item.call_args.kwargs["body"]["tags"] == ["a", "b", "c"]
    assert "classification" not in memories.replace_item.call_args.kwargs["body"]["metadata"]

    await store.remove_tags("m1", "u1", "t1", "fact", [TOKEN, "a"])
    assert memories.replace_item.call_args.kwargs["body"]["tags"] == ["b"]
    assert "classification" not in memories.replace_item.call_args.kwargs["body"]["metadata"]


async def test_async_add_tags_logs_the_formats_found_never_the_tag(caplog):
    caplog.set_level(logging.DEBUG)
    store, _memories = _tagged_store()

    await store.add_tags("m1", "u1", "t1", "fact", [TOKEN])

    assert "patterns=['github_token']" in caplog.text
    assert TOKEN not in caplog.text and TOKEN.lower() not in caplog.text
