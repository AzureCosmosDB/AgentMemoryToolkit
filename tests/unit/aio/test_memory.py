"""Unit tests for AsyncAgentMemory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_toolkit.aio.memory import AsyncAgentMemory
from agent_memory_toolkit.exceptions import CosmosNotConnectedError
from agent_memory_toolkit.models import MemoryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


@pytest.fixture
def memory():
    """AsyncAgentMemory with default credential disabled (no azure.identity import)."""
    return AsyncAgentMemory(use_default_credential=False)


@pytest.fixture
def memory_with_cosmos(memory):
    """AsyncAgentMemory with a mocked cosmos store already attached."""
    mock_store = AsyncMock()
    # _require_connected is a sync method, so override with a plain MagicMock
    mock_store._require_connected = MagicMock()
    memory._cosmos_store = mock_store
    return memory


# ===================================================================
# Constructor
# ===================================================================


async def test_constructor_default_credential_disabled():
    mem = AsyncAgentMemory(use_default_credential=False)
    assert mem.local_memory == []
    assert mem._cosmos_store is None


async def test_constructor_default_credential_enabled():
    mock_cred = MagicMock()
    with patch(
        "agent_memory_toolkit.aio.memory.DefaultAzureCredential",
        return_value=mock_cred,
        create=True,
    ):
        mem = AsyncAgentMemory(use_default_credential=True)
    assert mem._cosmos_credential is not None


# ===================================================================
# add_local()
# ===================================================================


async def test_add_local_valid(memory):
    memory.add_local(user_id="u1", role="user", content="hi")
    assert len(memory.local_memory) == 1
    m = memory.local_memory[0]
    assert m["user_id"] == "u1"
    assert m["role"] == "user"
    assert m["content"] == "hi"
    assert m["type"] == "turn"


async def test_add_local_all_fields(memory):
    memory.add_local(
        user_id="u1",
        role="agent",
        content="response",
        memory_type="summary",
        agent_id="bot-1",
        metadata={"k": "v"},
        thread_id="t-custom",
    )
    m = memory.local_memory[0]
    assert m["role"] == "agent"
    assert m["type"] == "summary"
    assert m["agent_id"] == "bot-1"
    assert m["metadata"] == {"k": "v"}
    assert m["thread_id"] == "t-custom"


async def test_add_local_invalid_role(memory):
    with pytest.raises(ValueError, match="role must be one of"):
        memory.add_local(user_id="u1", role="invalid", content="hi")


async def test_add_local_invalid_type(memory):
    with pytest.raises(ValueError, match="type must be one of"):
        memory.add_local(user_id="u1", role="user", content="hi", memory_type="bad")


# ===================================================================
# get_local()
# ===================================================================


async def test_get_local_no_filter(memory):
    memory.add_local(user_id="u1", role="user", content="a")
    memory.add_local(user_id="u2", role="agent", content="b")
    results = memory.get_local()
    assert len(results) == 2


async def test_get_local_with_filters(memory):
    memory.add_local(user_id="u1", role="user", content="a")
    memory.add_local(user_id="u1", role="agent", content="b")
    memory.add_local(user_id="u2", role="user", content="c")

    results = memory.get_local(user_id="u1", role="user")
    assert len(results) == 1
    assert results[0]["content"] == "a"


async def test_get_local_by_id(memory):
    memory.add_local(user_id="u1", role="user", content="x")
    mid = memory.local_memory[0]["id"]
    results = memory.get_local(memory_id=mid)
    assert len(results) == 1
    assert results[0]["id"] == mid


async def test_get_local_by_type(memory):
    memory.add_local(user_id="u1", role="user", content="a", memory_type="turn")
    memory.add_local(user_id="u1", role="agent", content="b", memory_type="summary")
    results = memory.get_local(memory_type="summary")
    assert len(results) == 1


# ===================================================================
# update_local()
# ===================================================================


async def test_update_local_success(memory):
    memory.add_local(user_id="u1", role="user", content="old")
    mid = memory.local_memory[0]["id"]
    memory.update_local(memory_id=mid, content="new")
    assert memory.local_memory[0]["content"] == "new"
    assert "updated_at" in memory.local_memory[0]


async def test_update_local_not_found(memory):
    with pytest.raises(KeyError, match="No memory found"):
        memory.update_local(memory_id="nonexistent", content="x")


async def test_update_local_partial(memory):
    memory.add_local(user_id="u1", role="user", content="old", memory_type="turn")
    mid = memory.local_memory[0]["id"]
    memory.update_local(memory_id=mid, metadata={"k": "v"})
    m = memory.local_memory[0]
    assert m["content"] == "old"  # unchanged
    assert m["metadata"] == {"k": "v"}


# ===================================================================
# delete_local()
# ===================================================================


async def test_delete_local_success(memory):
    memory.add_local(user_id="u1", role="user", content="a")
    memory.add_local(user_id="u1", role="agent", content="b")
    mid = memory.local_memory[0]["id"]
    memory.delete_local(mid)
    assert len(memory.local_memory) == 1
    assert memory.local_memory[0]["content"] == "b"


async def test_delete_local_not_found(memory):
    with pytest.raises(KeyError, match="No memory found"):
        memory.delete_local("nonexistent")


# ===================================================================
# connect_cosmos()
# ===================================================================


async def test_connect_cosmos():
    mem = AsyncAgentMemory(
        cosmos_endpoint="https://fake.documents.azure.com:443/",
        cosmos_credential=MagicMock(),
        use_default_credential=False,
    )
    mock_store_cls = MagicMock()
    mock_store_instance = AsyncMock()
    mock_store_cls.return_value = mock_store_instance

    with patch(
        "agent_memory_toolkit.aio.memory.AsyncCosmosMemoryStore", mock_store_cls
    ):
        await mem.connect_cosmos()

    mock_store_instance.connect.assert_awaited_once()
    assert mem._cosmos_store is mock_store_instance


# ===================================================================
# add_cosmos()
# ===================================================================


async def test_add_cosmos(memory_with_cosmos):
    await memory_with_cosmos.add_cosmos(
        user_id="u1", role="user", content="hello"
    )
    memory_with_cosmos._cosmos_store.upsert.assert_awaited_once()
    record = memory_with_cosmos._cosmos_store.upsert.call_args.kwargs.get("record")
    if record is None:
        record = memory_with_cosmos._cosmos_store.upsert.call_args.args[0]
    assert isinstance(record, MemoryRecord)
    assert record.content == "hello"


async def test_add_cosmos_not_connected(memory):
    with pytest.raises(CosmosNotConnectedError):
        await memory.add_cosmos(user_id="u1", role="user", content="hi")


# ===================================================================
# push_to_cosmos()
# ===================================================================


async def test_push_to_cosmos(memory_with_cosmos):
    memory_with_cosmos.add_local(user_id="u1", role="user", content="a")
    memory_with_cosmos.add_local(user_id="u1", role="agent", content="b")

    await memory_with_cosmos.push_to_cosmos(batch_size=5)
    memory_with_cosmos._cosmos_store.upsert_batch.assert_awaited_once()
    call_args = memory_with_cosmos._cosmos_store.upsert_batch.call_args
    records = call_args.args[0] if call_args.args else call_args.kwargs["records"]
    assert len(records) == 2


async def test_push_to_cosmos_not_connected(memory):
    memory.add_local(user_id="u1", role="user", content="a")
    with pytest.raises(CosmosNotConnectedError):
        await memory.push_to_cosmos()


async def test_push_to_cosmos_invalid_batch_size(memory_with_cosmos):
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        await memory_with_cosmos.push_to_cosmos(batch_size=0)


# ===================================================================
# search_cosmos()
# ===================================================================


async def test_search_cosmos(memory_with_cosmos):
    memory_with_cosmos._embeddings_client = AsyncMock()
    memory_with_cosmos._embeddings_client.generate = AsyncMock(
        return_value=[0.1, 0.2, 0.3]
    )
    memory_with_cosmos._cosmos_store.vector_search = AsyncMock(
        return_value=[{"id": "m1", "content": "result"}]
    )

    results = await memory_with_cosmos.search_cosmos(
        search_terms="weather", user_id="u1", top_k=3
    )
    assert len(results) == 1
    memory_with_cosmos._embeddings_client.generate.assert_awaited_once_with("weather")
    memory_with_cosmos._cosmos_store.vector_search.assert_awaited_once()


async def test_search_cosmos_not_connected(memory):
    with pytest.raises(CosmosNotConnectedError):
        await memory.search_cosmos(search_terms="test")


# ===================================================================
# get_memories() / get_thread()
# ===================================================================


async def test_get_memories_delegates(memory_with_cosmos):
    memory_with_cosmos._cosmos_store.get_memories = AsyncMock(return_value=[{"id": "x"}])
    result = await memory_with_cosmos.get_memories(user_id="u1")
    assert len(result) == 1
    memory_with_cosmos._cosmos_store.get_memories.assert_awaited_once()


async def test_get_thread_delegates(memory_with_cosmos):
    memory_with_cosmos._cosmos_store.get_thread = AsyncMock(return_value=[{"id": "x"}])
    result = await memory_with_cosmos.get_thread(thread_id="t1")
    assert len(result) == 1


# ===================================================================
# Cosmos ops without connect
# ===================================================================


async def test_cosmos_ops_without_connect(memory):
    with pytest.raises(CosmosNotConnectedError):
        await memory.get_memories()
    with pytest.raises(CosmosNotConnectedError):
        await memory.get_thread(thread_id="t1")
    with pytest.raises(CosmosNotConnectedError):
        await memory.update_cosmos(memory_id="m1")
    with pytest.raises(CosmosNotConnectedError):
        await memory.delete_cosmos(memory_id="m1", thread_id="t1", user_id="u1")


# ===================================================================
# close()
# ===================================================================


async def test_close(memory_with_cosmos):
    memory_with_cosmos._embeddings_client = AsyncMock()
    memory_with_cosmos._processing_client = AsyncMock()

    await memory_with_cosmos.close()
    memory_with_cosmos._cosmos_store.close.assert_awaited_once()
    memory_with_cosmos._embeddings_client.close.assert_awaited_once()
    memory_with_cosmos._processing_client.close.assert_awaited_once()


async def test_close_without_cosmos(memory):
    memory._embeddings_client = AsyncMock()
    memory._processing_client = AsyncMock()
    await memory.close()  # should not raise


# ===================================================================
# async context manager
# ===================================================================


async def test_context_manager(memory_with_cosmos):
    memory_with_cosmos._embeddings_client = AsyncMock()
    memory_with_cosmos._processing_client = AsyncMock()

    async with memory_with_cosmos as m:
        assert m is memory_with_cosmos
    memory_with_cosmos._cosmos_store.close.assert_awaited_once()
