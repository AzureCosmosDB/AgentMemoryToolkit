"""Unit tests for AsyncCosmosMemoryStore."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryStore
from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
)
from agent_memory_toolkit.models import MemoryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AsyncIterator:
    """Simple async iterator over a list of items."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _make_record(**overrides) -> MemoryRecord:
    defaults = {
        "id": str(uuid.uuid4()),
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "content": "hello",
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def _make_doc(**overrides) -> dict:
    defaults = {
        "id": str(uuid.uuid4()),
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "hello",
        "metadata": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def store():
    return AsyncCosmosMemoryStore(
        endpoint="https://fake.documents.azure.com:443/",
        credential=MagicMock(),
        database="testdb",
        container="testcont",
    )


@pytest.fixture
def connected_store(store):
    """A store with a mocked container client already attached."""
    store._cosmos_client = MagicMock()
    store._container_client = MagicMock()
    # Make upsert_item, replace_item, delete_item async
    store._container_client.upsert_item = AsyncMock()
    store._container_client.replace_item = AsyncMock()
    store._container_client.delete_item = AsyncMock()
    return store


# ===================================================================
# connect()
# ===================================================================


async def test_connect_success(store):
    mock_cosmos_cls = MagicMock()
    mock_db = MagicMock()
    mock_container = MagicMock()
    mock_cosmos_cls.return_value = mock_cosmos_cls
    mock_cosmos_cls.get_database_client.return_value = mock_db
    mock_db.get_container_client.return_value = mock_container

    with patch(
        "agent_memory_toolkit.aio.cosmos_memory_client.CosmosClient",
        mock_cosmos_cls,
        create=True,
    ), patch.dict(
        "sys.modules",
        {"azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls)},
    ):
        await store.connect()

    assert store._cosmos_client is not None
    assert store._container_client is not None


async def test_connect_missing_endpoint():
    store = AsyncCosmosMemoryStore(endpoint=None, credential=MagicMock())
    with pytest.raises(ConfigurationError):
        await store.connect()


async def test_connect_missing_credential():
    store = AsyncCosmosMemoryStore(
        endpoint="https://x.documents.azure.com:443/", credential=None
    )
    with pytest.raises(ConfigurationError):
        await store.connect()


# ===================================================================
# _require_connected()
# ===================================================================


async def test_require_connected_before_connect(store):
    with pytest.raises(CosmosNotConnectedError):
        store._require_connected()


async def test_require_connected_after_connect(connected_store):
    # Should not raise
    connected_store._require_connected()


# ===================================================================
# upsert()
# ===================================================================


async def test_upsert_single(connected_store):
    record = _make_record()
    await connected_store.upsert(record)
    connected_store._container_client.upsert_item.assert_awaited_once()
    body = connected_store._container_client.upsert_item.call_args.kwargs["body"]
    assert body["id"] == record.id


async def test_upsert_not_connected(store):
    with pytest.raises(CosmosNotConnectedError):
        await store.upsert(_make_record())


async def test_upsert_cosmos_failure(connected_store):
    connected_store._container_client.upsert_item.side_effect = Exception("boom")
    with pytest.raises(CosmosOperationError):
        await connected_store.upsert(_make_record())


# ===================================================================
# upsert_batch()
# ===================================================================


async def test_upsert_batch(connected_store):
    records = [_make_record() for _ in range(5)]
    await connected_store.upsert_batch(records, batch_size=2)
    assert connected_store._container_client.upsert_item.await_count == 5


async def test_upsert_batch_single_batch(connected_store):
    records = [_make_record() for _ in range(3)]
    await connected_store.upsert_batch(records, batch_size=10)
    assert connected_store._container_client.upsert_item.await_count == 3


# ===================================================================
# get_memories()
# ===================================================================


async def test_get_memories_no_filters(connected_store):
    docs = [_make_doc(), _make_doc()]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.get_memories()
    assert len(results) == 2
    connected_store._container_client.query_items.assert_called_once()


async def test_get_memories_with_filters(connected_store):
    docs = [_make_doc(user_id="u1")]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.get_memories(user_id="u1", role="user")
    assert len(results) == 1
    call_kwargs = connected_store._container_client.query_items.call_args.kwargs
    assert "@user_id" in str(call_kwargs["parameters"])


async def test_get_memories_recent_k(connected_store):
    docs = [_make_doc(content="older"), _make_doc(content="newer")]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.get_memories(recent_k=2)
    # recent_k reverses the result so oldest-first
    assert results[0]["content"] == "newer"
    assert results[1]["content"] == "older"
    query = connected_store._container_client.query_items.call_args.kwargs["query"]
    assert "TOP @recent_k" in query


async def test_get_memories_not_connected(store):
    with pytest.raises(CosmosNotConnectedError):
        await store.get_memories()


# ===================================================================
# get_thread()
# ===================================================================


async def test_get_thread(connected_store):
    docs = [_make_doc(content="second"), _make_doc(content="first")]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.get_thread(thread_id="t1")
    # get_thread reverses to chronological
    assert results[0]["content"] == "first"
    assert results[1]["content"] == "second"


async def test_get_thread_with_recent_k(connected_store):
    docs = [_make_doc(content="c"), _make_doc(content="b"), _make_doc(content="a")]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.get_thread(thread_id="t1", recent_k=2)
    # Slices first 2 then reverses
    assert len(results) == 2


# ===================================================================
# update()
# ===================================================================


async def test_update_success(connected_store):
    doc = _make_doc(id="m1")
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator([doc])
    )
    await connected_store.update(memory_id="m1", content="updated")
    connected_store._container_client.replace_item.assert_awaited_once()
    call_kwargs = connected_store._container_client.replace_item.call_args.kwargs
    assert call_kwargs["body"]["content"] == "updated"


async def test_update_not_found(connected_store):
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator([])
    )
    with pytest.raises(MemoryNotFoundError):
        await connected_store.update(memory_id="missing")


async def test_update_partial_fields(connected_store):
    doc = _make_doc(id="m1", role="user", content="old")
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator([doc])
    )
    await connected_store.update(memory_id="m1", role="agent", metadata={"key": "val"})
    body = connected_store._container_client.replace_item.call_args.kwargs["body"]
    assert body["role"] == "agent"
    assert body["metadata"] == {"key": "val"}
    assert body["content"] == "old"  # unchanged
    assert "updated_at" in body


# ===================================================================
# delete()
# ===================================================================


async def test_delete_success(connected_store):
    doc = _make_doc(id="m1", user_id="u1", thread_id="t1")
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator([doc])
    )
    await connected_store.delete(memory_id="m1", user_id="u1", thread_id="t1")
    connected_store._container_client.delete_item.assert_awaited_once_with(
        item="m1", partition_key=["u1", "t1"]
    )


async def test_delete_not_found(connected_store):
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator([])
    )
    with pytest.raises(MemoryNotFoundError):
        await connected_store.delete(memory_id="x", user_id="u1", thread_id="t1")


# ===================================================================
# close()
# ===================================================================


async def test_close(connected_store):
    mock_client = AsyncMock()
    connected_store._cosmos_client = mock_client
    await connected_store.close()
    mock_client.close.assert_awaited_once()
    assert connected_store._cosmos_client is None
    assert connected_store._container_client is None


async def test_close_calls_cosmos_client():
    store = AsyncCosmosMemoryStore(
        endpoint="https://fake.documents.azure.com:443/",
        credential=MagicMock(),
    )
    mock_client = AsyncMock()
    store._cosmos_client = mock_client
    store._container_client = MagicMock()
    await store.close()
    mock_client.close.assert_awaited_once()
    assert store._cosmos_client is None
    assert store._container_client is None


async def test_close_noop_when_not_connected(store):
    await store.close()  # should not raise


# ===================================================================
# async context manager
# ===================================================================


async def test_context_manager():
    store = AsyncCosmosMemoryStore(
        endpoint="https://fake.documents.azure.com:443/",
        credential=MagicMock(),
    )
    mock_client = AsyncMock()
    store._cosmos_client = mock_client
    store._container_client = MagicMock()

    async with store as s:
        assert s is store
    mock_client.close.assert_awaited_once()


# ===================================================================
# vector_search()
# ===================================================================


async def test_vector_search(connected_store):
    docs = [_make_doc(content="result")]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.vector_search(
        query_vector=[0.1, 0.2], user_id="u1", top_k=3
    )
    assert len(results) == 1
    query = connected_store._container_client.query_items.call_args.kwargs["query"]
    assert "VectorDistance" in query


async def test_vector_search_hybrid(connected_store):
    docs = [_make_doc()]
    connected_store._container_client.query_items = MagicMock(
        return_value=AsyncIterator(docs)
    )
    results = await connected_store.vector_search(
        query_vector=[0.1],
        hybrid_search=True,
        search_terms="weather",
        top_k=5,
    )
    assert len(results) == 1
    query = connected_store._container_client.query_items.call_args.kwargs["query"]
    assert "RRF" in query
    assert "FullTextScore" in query
