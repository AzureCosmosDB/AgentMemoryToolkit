"""Unit tests for AsyncCosmosMemoryClient (unified async client)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    CosmosNotConnectedError,
    MemoryNotFoundError,
    ValidationError,
)

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


def _make_client(**overrides) -> AsyncCosmosMemoryClient:
    """Build an AsyncCosmosMemoryClient with credential auto-resolution disabled."""
    defaults: dict = {"use_default_credential": False}
    defaults.update(overrides)
    return AsyncCosmosMemoryClient(**defaults)


def _connected_client() -> tuple[AsyncCosmosMemoryClient, MagicMock]:
    """Return a client with a mocked container client already wired up."""
    client = _make_client()
    container = MagicMock()
    container.upsert_item = AsyncMock()
    container.replace_item = AsyncMock()
    container.delete_item = AsyncMock()
    client._container_client = container
    return client, container


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


# ===================================================================
# Constructor
# ===================================================================


class TestConstructor:
    def test_default_credential_disabled(self):
        mem = _make_client()
        assert mem.local_memory == []
        assert mem._container_client is None
        assert mem._cosmos_credential is None
        assert mem._ai_foundry_credential is None

    def test_default_credential_enabled(self):
        mock_cred = MagicMock()
        mock_module = MagicMock()
        mock_module.DefaultAzureCredential.return_value = mock_cred

        with patch.dict("sys.modules", {"azure.identity.aio": mock_module}):
            mem = AsyncCosmosMemoryClient(use_default_credential=True)
        assert mem._cosmos_credential is not None


# ===================================================================
# Local CRUD (synchronous)
# ===================================================================


class TestAddLocal:
    def test_add_local_valid(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="hi", thread_id="t1")

        assert len(mem.local_memory) == 1
        m = mem.local_memory[0]
        assert m["user_id"] == "u1"
        assert m["role"] == "user"
        assert m["content"] == "hi"
        assert m["type"] == "turn"

    def test_add_local_all_fields(self):
        mem = _make_client()
        mem.add_local(
            user_id="u1",
            role="agent",
            content="response",
            memory_type="summary",
            agent_id="bot-1",
            metadata={"k": "v"},
            thread_id="t-custom",
        )
        m = mem.local_memory[0]
        assert m["role"] == "agent"
        assert m["type"] == "summary"
        assert m["agent_id"] == "bot-1"
        assert m["metadata"] == {"k": "v"}
        assert m["thread_id"] == "t-custom"

    def test_add_local_invalid_role(self):
        mem = _make_client()
        with pytest.raises(ValidationError, match="role must be one of"):
            mem.add_local(user_id="u1", role="invalid", content="hi", thread_id="t1")

    def test_add_local_invalid_type(self):
        mem = _make_client()
        with pytest.raises(ValidationError, match="type must be one of"):
            mem.add_local(user_id="u1", role="user", content="hi", memory_type="bad")

    def test_add_local_turn_requires_thread_id(self):
        mem = _make_client()
        with pytest.raises(ValidationError, match="thread_id is required"):
            mem.add_local(user_id="u1", role="user", content="hi")
        # Validation must run BEFORE append — otherwise an orphan turn
        # with thread_id=None would persist and pollute pk on push.
        assert mem.local_memory == []
        assert mem._unflushed_turn_counts == {}


class TestGetLocal:
    def test_get_local_no_filter(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u2", role="agent", content="b", thread_id="t1")
        results = mem.get_local()
        assert len(results) == 2

    def test_get_local_with_filters(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u1", role="agent", content="b", thread_id="t1")
        mem.add_local(user_id="u2", role="user", content="c", thread_id="t1")

        results = mem.get_local(user_id="u1", role="user")
        assert len(results) == 1
        assert results[0]["content"] == "a"

    def test_get_local_by_id(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="x", thread_id="t1")
        mid = mem.local_memory[0]["id"]
        results = mem.get_local(memory_id=mid)
        assert len(results) == 1
        assert results[0]["id"] == mid


class TestUpdateLocal:
    def test_update_local_success(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="old", thread_id="t1")
        mid = mem.local_memory[0]["id"]
        mem.update_local(memory_id=mid, content="new")
        assert mem.local_memory[0]["content"] == "new"
        assert "updated_at" in mem.local_memory[0]

    def test_update_local_not_found(self):
        mem = _make_client()
        with pytest.raises(MemoryNotFoundError):
            mem.update_local(memory_id="nonexistent", content="x")

    def test_update_local_partial(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="old", memory_type="turn", thread_id="t1")
        mid = mem.local_memory[0]["id"]
        mem.update_local(memory_id=mid, metadata={"k": "v"})
        m = mem.local_memory[0]
        assert m["content"] == "old"
        assert m["metadata"] == {"k": "v"}


class TestDeleteLocal:
    def test_delete_local_success(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u1", role="agent", content="b", thread_id="t1")
        mid = mem.local_memory[0]["id"]
        mem.delete_local(mid)
        assert len(mem.local_memory) == 1
        assert mem.local_memory[0]["content"] == "b"

    def test_delete_local_not_found(self):
        mem = _make_client()
        with pytest.raises(MemoryNotFoundError):
            mem.delete_local("nonexistent")


# ===================================================================
# Cosmos connection (async)
# ===================================================================


class TestConnectCosmos:
    async def test_connect_cosmos_success(self):
        mem = _make_client(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_credential=MagicMock(),
        )
        mock_cosmos_cls = MagicMock()
        mock_db = MagicMock()
        mock_container = MagicMock()
        mock_cosmos_cls.return_value = mock_cosmos_cls
        mock_cosmos_cls.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        with patch.dict(
            "sys.modules",
            {"azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls)},
        ):
            await mem.connect_cosmos()

        assert mem._container_client is mock_container


class TestCreateMemoryStore:
    async def test_create_memory_store_with_counter_container(self):
        mem = _make_client()
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = AsyncMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()
        mock_throughput_cls = MagicMock(side_effect=lambda **kwargs: type("Throughput", (), kwargs)())

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_db.create_container_if_not_exists = AsyncMock(
            side_effect=[mock_memories_container, mock_counter_container, mock_lease_container]
        )

        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls),
                "azure.cosmos": MagicMock(
                    PartitionKey=MagicMock(),
                    ThroughputProperties=mock_throughput_cls,
                ),
            },
        ):
            await mem.create_memory_store(
                endpoint="https://fake.documents.azure.com:443/",
                credential="fake-key",
                embedding_dimensions=256,
                throughput_mode="autoscale",
                autoscale_max_ru=1000,
            )

        mock_client.create_database_if_not_exists.assert_awaited_once()
        memories_call = mock_db.create_container_if_not_exists.await_args_list[0]
        counter_call = mock_db.create_container_if_not_exists.await_args_list[1]
        lease_call = mock_db.create_container_if_not_exists.await_args_list[2]
        vec_policy = memories_call.kwargs["vector_embedding_policy"]
        assert vec_policy["vectorEmbeddings"][0]["dimensions"] == 256
        ft_policy = memories_call.kwargs["full_text_policy"]
        assert ft_policy["defaultLanguage"] == "en-US"
        assert counter_call.kwargs["id"] == "counter"
        assert counter_call.kwargs["offer_throughput"].auto_scale_max_throughput == 1000
        assert lease_call.kwargs["id"] == "leases"
        assert lease_call.kwargs["offer_throughput"].auto_scale_max_throughput == 1000
        assert "vector_embedding_policy" not in counter_call.kwargs
        assert mem._container_client is mock_memories_container

    async def test_create_memory_store_defaults_to_serverless(self):
        mem = _make_client(cosmos_throughput_mode="serverless")
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = AsyncMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_db.create_container_if_not_exists = AsyncMock(
            side_effect=[mock_memories_container, mock_counter_container, mock_lease_container]
        )

        with patch.dict("os.environ", {"COSMOS_DB_AUTOSCALE_MAX_RU": "not-an-int"}, clear=False):
            with patch.dict(
                "sys.modules",
                {
                    "azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls),
                    "azure.cosmos": MagicMock(
                        PartitionKey=MagicMock(),
                        ThroughputProperties=MagicMock(),
                    ),
                },
            ):
                await mem.create_memory_store(
                    endpoint="https://fake.documents.azure.com:443/",
                    credential="fake-key",
                    throughput_mode="serverless",
                )

        for call in mock_db.create_container_if_not_exists.await_args_list:
            assert "offer_throughput" not in call.kwargs

    def test_constructor_ignores_invalid_autoscale_env_in_serverless_mode(self):
        with patch.dict("os.environ", {"COSMOS_DB_AUTOSCALE_MAX_RU": "not-an-int"}, clear=False):
            mem = _make_client(cosmos_throughput_mode="serverless")

        assert mem._cosmos_autoscale_max_ru is None

    def test_constructor_rejects_invalid_throughput_mode(self):
        with pytest.raises(ConfigurationError, match="expected 'serverless' or 'autoscale'"):
            _make_client(cosmos_throughput_mode="invalid")


class TestRequireCosmos:
    async def test_require_cosmos_before_connect(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem._require_cosmos()

    async def test_require_cosmos_after_connect(self):
        mem, _ = _connected_client()
        await mem._require_cosmos()  # should not raise


# ===================================================================
# Cosmos CRUD (async, mock _container_client)
# ===================================================================


class TestAddCosmos:
    async def test_add_cosmos(self):
        mem, container = _connected_client()
        await mem.add_cosmos(user_id="u1", role="user", content="hello")

        container.upsert_item.assert_awaited_once()
        body = container.upsert_item.call_args.kwargs["body"]
        assert body["content"] == "hello"
        assert body["user_id"] == "u1"

    async def test_add_cosmos_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem.add_cosmos(user_id="u1", role="user", content="hi")


class TestPushToCosmos:
    async def test_push_to_cosmos(self):
        mem, container = _connected_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u1", role="agent", content="b", thread_id="t1")

        await mem.push_to_cosmos(batch_size=5)

        assert container.upsert_item.await_count == 2

    async def test_push_to_cosmos_not_connected(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        with pytest.raises(CosmosNotConnectedError):
            await mem.push_to_cosmos()

    async def test_push_to_cosmos_invalid_batch_size(self):
        mem, _ = _connected_client()
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            await mem.push_to_cosmos(batch_size=0)


class TestGetMemories:
    async def test_no_filters(self):
        mem, container = _connected_client()
        docs = [_make_doc(), _make_doc()]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        results = await mem.get_memories()
        assert len(results) == 2

    async def test_with_filters(self):
        mem, container = _connected_client()
        docs = [_make_doc(user_id="u1")]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        results = await mem.get_memories(user_id="u1", role="user")
        assert len(results) == 1
        call_kwargs = container.query_items.call_args.kwargs
        assert "@user_id" in str(call_kwargs["parameters"])

    async def test_recent_k(self):
        mem, container = _connected_client()
        docs = [_make_doc(id="newer"), _make_doc(id="older")]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        results = await mem.get_memories(recent_k=2)
        query = container.query_items.call_args.kwargs["query"]
        assert "TOP @recent_k" in query
        assert "ORDER BY c._ts DESC" in query
        # Reversed to chronological
        assert results[0]["id"] == "older"
        assert results[1]["id"] == "newer"

    async def test_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem.get_memories()


class TestGetThread:
    async def test_basic(self):
        mem, container = _connected_client()
        docs = [_make_doc(content="second"), _make_doc(content="first")]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        result = await mem.get_thread(thread_id="t1")
        # Reversed to chronological
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"

    async def test_with_recent_k(self):
        mem, container = _connected_client()
        docs = [_make_doc(content="c"), _make_doc(content="b"), _make_doc(content="a")]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        result = await mem.get_thread(thread_id="t1", recent_k=2)
        assert len(result) == 2


class TestUpdateCosmos:
    async def test_success(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1")
        container.query_items = MagicMock(return_value=AsyncIterator([doc]))

        await mem.update_cosmos(memory_id="m1", content="updated")

        container.replace_item.assert_awaited_once()
        body = container.replace_item.call_args.kwargs["body"]
        assert body["content"] == "updated"
        assert "updated_at" in body

    async def test_not_found(self):
        mem, container = _connected_client()
        container.query_items = MagicMock(return_value=AsyncIterator([]))

        with pytest.raises(MemoryNotFoundError):
            await mem.update_cosmos(memory_id="missing")

    async def test_partial_fields(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1", role="user", content="old")
        container.query_items = MagicMock(return_value=AsyncIterator([doc]))

        await mem.update_cosmos(memory_id="m1", role="agent", metadata={"key": "val"})
        body = container.replace_item.call_args.kwargs["body"]
        assert body["role"] == "agent"
        assert body["metadata"] == {"key": "val"}
        assert body["content"] == "old"
        assert "updated_at" in body


class TestDeleteCosmos:
    async def test_success(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1", user_id="u1", thread_id="t1")
        container.query_items = MagicMock(return_value=AsyncIterator([doc]))

        await mem.delete_cosmos(memory_id="m1", user_id="u1", thread_id="t1")

        container.delete_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])

    async def test_not_found(self):
        mem, container = _connected_client()
        container.query_items = MagicMock(return_value=AsyncIterator([]))

        with pytest.raises(MemoryNotFoundError):
            await mem.delete_cosmos(memory_id="x", user_id="u1", thread_id="t1")


class TestGetUserSummary:
    async def test_filters_by_type(self):
        mem, container = _connected_client()
        doc = _make_doc(type="user_summary")
        container.query_items = MagicMock(return_value=AsyncIterator([doc]))

        result = await mem.get_user_summary(user_id="u1")

        call_kwargs = container.query_items.call_args.kwargs
        assert "user_summary" in call_kwargs["query"]
        assert result == [doc]


# ===================================================================
# Cosmos guard
# ===================================================================


class TestCosmosGuard:
    async def test_cosmos_ops_without_connect(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem.get_memories()
        with pytest.raises(CosmosNotConnectedError):
            await mem.get_thread(thread_id="t1")
        with pytest.raises(CosmosNotConnectedError):
            await mem.update_cosmos(memory_id="m1")
        with pytest.raises(CosmosNotConnectedError):
            await mem.delete_cosmos(memory_id="m1", thread_id="t1", user_id="u1")


# ===================================================================
# Search (async)
# ===================================================================


class TestSearchCosmos:
    async def test_search_cosmos(self):
        mem, container = _connected_client()
        docs = [_make_doc()]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1, 0.2, 0.3])

        results = await mem.search_cosmos(search_terms="weather", user_id="u1", top_k=3)

        assert len(results) == 1
        mem._embeddings_client.generate.assert_awaited_once_with("weather")
        query = container.query_items.call_args.kwargs["query"]
        assert "VectorDistance" in query

    async def test_search_hybrid(self):
        mem, container = _connected_client()
        docs = [_make_doc()]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1])

        results = await mem.search_cosmos(
            search_terms="weather Seattle",
            hybrid_search=True,
            top_k=5,
        )

        assert len(results) == 1
        query = container.query_items.call_args.kwargs["query"]
        assert "RRF" in query
        assert "FullTextScore" in query

    async def test_search_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem.search_cosmos(search_terms="test")

    async def test_search_empty_terms(self):
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="search_terms must be a non-empty string"):
            await mem.search_cosmos(search_terms="")

    async def test_search_whitespace_only_terms(self):
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="search_terms must be a non-empty string"):
            await mem.search_cosmos(search_terms="   ")


# ===================================================================
# Processing delegation (async)
# ===================================================================


class TestGenerateThreadSummary:
    async def test_generate_thread_summary(self):
        mem, container = _connected_client()
        mock_pipeline = MagicMock()
        mock_pipeline.generate_thread_summary = MagicMock(return_value={"status": "ok"})
        mem._pipeline = mock_pipeline

        result = await mem.generate_thread_summary(user_id="u1", thread_id="t1")

        mock_pipeline.generate_thread_summary.assert_called_once_with(
            "u1",
            "t1",
            None,
        )
        assert result == {"status": "ok"}


# ===================================================================
# close() and context manager
# ===================================================================


class TestClose:
    async def test_close_with_cosmos(self):
        mem, _ = _connected_client()
        mock_cosmos = AsyncMock()
        mem._cosmos_client = mock_cosmos
        mem._embeddings_client = AsyncMock()

        await mem.close()

        mock_cosmos.close.assert_awaited_once()
        assert mem._cosmos_client is None
        assert mem._container_client is None

    async def test_close_without_cosmos(self):
        mem = _make_client()
        mem._embeddings_client = AsyncMock()
        await mem.close()  # should not raise

    async def test_context_manager(self):
        mem, _ = _connected_client()
        mock_cosmos = AsyncMock()
        mem._cosmos_client = mock_cosmos
        mem._embeddings_client = AsyncMock()

        async with mem as m:
            assert m is mem

        mock_cosmos.close.assert_awaited_once()
