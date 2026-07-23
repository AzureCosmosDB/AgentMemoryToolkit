"""Unit tests for AsyncCosmosMemoryClient (unified async client)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import (
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
    """Return a client with mocked split containers already wired up."""
    client = _make_client()
    container = MagicMock()
    turns = MagicMock()
    summaries = MagicMock()
    container.id = "memories"
    turns.id = "memories_turns"
    summaries.id = "memories_summaries"
    for c in (container, turns, summaries):
        c.upsert_item = AsyncMock()
        c.replace_item = AsyncMock()
        c.delete_item = AsyncMock()
        c.query_items = MagicMock(return_value=AsyncIterator([]))
    client._memories_container_client = container
    client._turns_container_client = turns
    client._summaries_container_client = summaries
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
        assert mem._memories_container_client is None
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
# Injected embeddings / chat clients
# ===================================================================


class _FakeEmbeddings:
    """Minimal stand-in for AsyncEmbeddingsClient used to verify injection."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeChat:
    """Minimal stand-in for AsyncChatClient used to verify injection."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class TestInjectedModelClients:
    def test_injected_clients_are_used_and_not_owned(self):
        emb = _FakeEmbeddings()
        chat = _FakeChat()
        mem = _make_client(embeddings_client=emb, chat_client=chat)

        assert mem._embeddings_client is emb
        assert mem._chat_client is chat
        assert mem._owns_embeddings_client is False
        assert mem._owns_chat_client is False

    def test_default_clients_are_built_and_owned(self):
        mem = _make_client()

        assert mem._embeddings_client is not None
        assert mem._chat_client is not None
        assert mem._owns_embeddings_client is True
        assert mem._owns_chat_client is True

    def test_clients_can_be_injected_independently(self):
        emb = _FakeEmbeddings()
        mem = _make_client(embeddings_client=emb)

        assert mem._embeddings_client is emb
        assert mem._owns_embeddings_client is False
        # Chat client was not injected, so the toolkit builds and owns it.
        assert mem._owns_chat_client is True

    async def test_close_does_not_close_injected_clients(self):
        emb = _FakeEmbeddings()
        chat = _FakeChat()
        mem = _make_client(embeddings_client=emb, chat_client=chat)

        await mem.close()

        assert emb.closed is False
        assert chat.closed is False


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
            memory_type="thread_summary",
            agent_id="bot-1",
            metadata={"k": "v"},
            thread_id="t-custom",
        )
        m = mem.local_memory[0]
        assert m["role"] == "agent"
        assert m["type"] == "thread_summary"
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
        # Validation must run BEFORE append - otherwise an orphan turn
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

        assert mem._memories_container_client is mock_container


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
        mock_turns_container = MagicMock()
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists = AsyncMock(
            side_effect=[
                mock_memories_container,
                mock_turns_container,
                mock_summaries_container,
                mock_counter_container,
                mock_lease_container,
            ]
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
        summaries_call = mock_db.create_container_if_not_exists.await_args_list[2]
        counter_call = mock_db.create_container_if_not_exists.await_args_list[3]
        lease_call = mock_db.create_container_if_not_exists.await_args_list[4]
        vec_policy = memories_call.kwargs["vector_embedding_policy"]
        assert vec_policy["vectorEmbeddings"][0]["dimensions"] == 256
        ft_policy = memories_call.kwargs["full_text_policy"]
        assert ft_policy["defaultLanguage"] == "en-US"
        assert counter_call.kwargs["id"] == "counter"
        assert counter_call.kwargs["offer_throughput"].auto_scale_max_throughput == 1000
        assert lease_call.kwargs["id"] == "leases"
        assert lease_call.kwargs["offer_throughput"].auto_scale_max_throughput == 1000
        assert summaries_call.kwargs["id"] == "memories_summaries"
        assert "vector_embedding_policy" in summaries_call.kwargs
        assert "full_text_policy" in summaries_call.kwargs
        assert summaries_call.kwargs["indexing_policy"]["vectorIndexes"][0]["path"] == "/embedding"
        assert summaries_call.kwargs["indexing_policy"]["compositeIndexes"][0][-1] == {
            "path": "/version",
            "order": "descending",
        }
        assert "vector_embedding_policy" not in counter_call.kwargs
        assert mem._memories_container_client is mock_memories_container

    async def test_create_memory_store_turns_container_uses_30_day_ttl(self):
        mem = _make_client()
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = AsyncMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()
        mock_turns_container = MagicMock()

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists = AsyncMock(
            side_effect=[
                mock_memories_container,
                mock_turns_container,
                mock_summaries_container,
                mock_counter_container,
                mock_lease_container,
            ]
        )

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
                turns_container="memories_turns",
            )

        turns_call = mock_db.create_container_if_not_exists.await_args_list[1]
        assert turns_call.kwargs["id"] == "memories_turns"
        assert turns_call.kwargs["default_ttl"] == 2_592_000
        # The turns container is always provisioned with a vector index + full-text
        # policy so it is primed for search_turns() even when turn
        # embeddings are disabled. Vector indexes use quantizedFlat.
        assert "vector_embedding_policy" in turns_call.kwargs
        assert "full_text_policy" in turns_call.kwargs
        assert turns_call.kwargs["indexing_policy"]["vectorIndexes"][0]["type"] == "quantizedFlat"
        assert mem._turns_container_client is mock_turns_container

    async def test_create_memory_store_defaults_to_serverless(self):
        # serverless mode ignores autoscale config entirely, even an invalid value.
        mem = _make_client(cosmos_throughput_mode="serverless", cosmos_autoscale_max_ru="not-an-int")
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = AsyncMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_turns_container = MagicMock()
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists = AsyncMock(
            side_effect=[
                mock_memories_container,
                mock_turns_container,
                mock_summaries_container,
                mock_counter_container,
                mock_lease_container,
            ]
        )

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

    def test_constructor_ignores_autoscale_in_serverless_mode(self):
        # Even an invalid autoscale value is ignored in serverless mode.
        mem = _make_client(cosmos_throughput_mode="serverless", cosmos_autoscale_max_ru="not-an-int")

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


class TestValidateTopology:
    async def test_validate_topology_succeeds_on_healthy_deploy(self):
        mem = _make_client()
        memories = MagicMock(id="memories")
        turns = MagicMock(id="memories_turns")
        summaries = MagicMock(id="memories_summaries")
        memories.read = AsyncMock()
        turns.read = AsyncMock()
        summaries.read = AsyncMock()
        mem._memories_container_client = memories
        mem._turns_container_client = turns
        mem._summaries_container_client = summaries

        await mem.validate_topology()

        memories.read.assert_awaited_once()
        turns.read.assert_awaited_once()
        summaries.read.assert_awaited_once()

    async def test_validate_topology_raises_on_missing_container(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem = _make_client()
        mem._memories_container_client = MagicMock(id="memories")
        mem._turns_container_client = MagicMock(id="memories_turns")
        mem._summaries_container_client = MagicMock(id="memories_summaries")
        mem._memories_container_client.read = AsyncMock()
        mem._turns_container_client.read = AsyncMock()
        mem._summaries_container_client.read = AsyncMock(side_effect=CosmosResourceNotFoundError(message="missing"))

        with pytest.raises(RuntimeError, match="memories_summaries"):
            await mem.validate_topology()

    async def test_validate_topology_raises_when_not_connected(self):
        mem = _make_client()

        with pytest.raises(RuntimeError, match="call connect_cosmos"):
            await mem.validate_topology()


# ===================================================================
# Cosmos CRUD (async, mock _memories_container_client)
# ===================================================================


class TestAddCosmos:
    async def test_add_cosmos(self):
        mem, container = _connected_client()
        # Suppress the background cadence task to keep the test focused on the CRUD write.
        mem._maybe_auto_trigger = AsyncMock()
        await mem.add_cosmos(user_id="u1", role="user", content="hello", thread_id="t1")
        # Drain any pending background tasks (none expected since we stubbed the trigger).
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        turns = mem._turns_container_client
        turns.upsert_item.assert_awaited_once()
        body = turns.upsert_item.call_args.kwargs["body"]
        assert body["content"] == "hello"
        assert body["user_id"] == "u1"

    async def test_add_cosmos_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem.add_cosmos(user_id="u1", role="user", content="hi", thread_id="t1")

    async def test_add_cosmos_turn_requires_thread_id(self):
        """Turn writes must declare a thread_id so the auto-trigger counter can group them."""
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="thread_id is required"):
            await mem.add_cosmos(user_id="u1", role="user", content="hi")  # memory_type='turn' default

    async def test_add_cosmos_non_turn_does_not_require_thread_id(self):
        """Non-turn writes (facts, episodics, etc.) work without thread_id and skip cadence."""
        mem, container = _connected_client()
        trigger = AsyncMock()
        mem._maybe_auto_trigger = trigger

        await mem.add_cosmos(user_id="u1", role="user", content="prefers dark mode", memory_type="fact")
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        container.upsert_item.assert_awaited_once()
        trigger.assert_not_awaited()

    async def test_add_cosmos_turn_schedules_cadence(self):
        """A turn write must schedule the auto-trigger as a background task so cadence
        env vars apply whether the caller uses the local buffer or writes through directly."""
        mem, _ = _connected_client()
        trigger = AsyncMock()
        mem._maybe_auto_trigger = trigger

        await mem.add_cosmos(user_id="u1", role="user", content="hello", thread_id="t1")
        # Drain the background task so the AsyncMock records the call.
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        trigger.assert_awaited_once_with({("u1", "t1"): 1})


class TestPushToCosmos:
    async def test_push_to_cosmos(self):
        mem, container = _connected_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u1", role="agent", content="b", thread_id="t1")

        await mem.push_to_cosmos(batch_size=5)

        assert mem._turns_container_client.upsert_item.await_count == 2

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
        mem, _ = _connected_client()
        docs = [_make_doc(content="second"), _make_doc(content="first")]
        mem._turns_container_client.query_items = MagicMock(return_value=AsyncIterator(docs))

        result = await mem.get_thread(thread_id="t1")
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"

    async def test_with_recent_k(self):
        mem, _ = _connected_client()
        docs = [_make_doc(content="c"), _make_doc(content="b"), _make_doc(content="a")]
        mem._turns_container_client.query_items = MagicMock(return_value=AsyncIterator(docs))

        result = await mem.get_thread(thread_id="t1", recent_k=2)
        assert len(result) == 2


class TestUpdateCosmos:
    async def test_success(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1", type="fact")
        container.read_item = AsyncMock(return_value=doc)
        container.replace_item = AsyncMock()

        await mem.update_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact", content="updated")

        container.read_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])
        container.replace_item.assert_awaited_once()
        body = container.replace_item.call_args.kwargs["body"]
        assert body["content"] == "updated"
        assert body["type"] == "fact"
        assert "updated_at" in body

    async def test_not_found(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, container = _connected_client()
        container.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))

        with pytest.raises(MemoryNotFoundError):
            await mem.update_cosmos(memory_id="missing", user_id="u1", thread_id="t1", memory_type="fact")

    async def test_partial_fields(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1", role="user", content="old", type="fact")
        container.read_item = AsyncMock(return_value=doc)
        container.replace_item = AsyncMock()

        await mem.update_cosmos(
            memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact", role="agent", metadata={"key": "val"}
        )
        body = container.replace_item.call_args.kwargs["body"]
        assert body["role"] == "agent"
        assert body["metadata"] == {"key": "val"}
        assert body["content"] == "old"
        assert "updated_at" in body


class TestDeleteCosmos:
    async def test_success(self):
        mem, container = _connected_client()
        container.read_item = AsyncMock(return_value=_make_doc(id="m1", type="fact"))
        container.delete_item = AsyncMock()

        await mem.delete_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact")

        container.delete_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])

    async def test_not_found(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, container = _connected_client()
        container.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))
        container.delete_item = AsyncMock()

        with pytest.raises(MemoryNotFoundError):
            await mem.delete_cosmos(memory_id="x", user_id="u1", thread_id="t1", memory_type="fact")

        container.delete_item.assert_not_awaited()


class TestGetUserSummary:
    async def test_returns_doc_when_present(self):
        mem, _ = _connected_client()
        summaries = mem._summaries_container_client
        doc = _make_doc(type="user_summary", id="user_summary_u1")
        summaries.read_item = AsyncMock(return_value=doc)

        result = await mem.get_user_summary(user_id="u1")

        call_kwargs = summaries.read_item.call_args.kwargs
        assert call_kwargs["item"] == "user_summary_u1"
        assert call_kwargs["partition_key"] == ["u1", "__user_summary__"]
        assert result == doc

    async def test_returns_none_when_absent(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, _ = _connected_client()
        summaries = mem._summaries_container_client
        summaries.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))

        result = await mem.get_user_summary(user_id="u1")

        assert result is None


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
            await mem.update_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact")
        with pytest.raises(CosmosNotConnectedError):
            await mem.delete_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact")


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

    async def test_search_uses_keyword_params_for_hybrid_sql(self):
        mem, container = _connected_client()
        container.query_items = MagicMock(return_value=AsyncIterator([_make_doc()]))

        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1])

        await mem.search_cosmos(search_terms="weather Seattle", top_k=3)

        query = container.query_items.call_args.kwargs["query"]
        parameters = container.query_items.call_args.kwargs["parameters"]
        assert "ORDER BY RANK RRF" in query
        assert "FullTextScore(c.content, @kw0, @kw1)" in query
        assert {"name": "@kw0", "value": "weather"} in parameters
        assert {"name": "@kw1", "value": "seattle"} in parameters

    async def test_search_all_stopwords_uses_vector_sql(self):
        mem, container = _connected_client()
        container.query_items = MagicMock(return_value=AsyncIterator([_make_doc()]))

        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1])

        await mem.search_cosmos(search_terms="what is the", top_k=3)

        query = container.query_items.call_args.kwargs["query"]
        parameters = container.query_items.call_args.kwargs["parameters"]
        assert "ORDER BY VectorDistance" in query
        assert "RRF" not in query
        assert not any(parameter["name"].startswith("@kw") for parameter in parameters)

    async def test_search_hybrid(self):
        mem, container = _connected_client()
        docs = [_make_doc()]
        container.query_items = MagicMock(return_value=AsyncIterator(docs))

        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1])

        results = await mem.search_cosmos(
            search_terms="weather Seattle",
            top_k=5,
        )

        assert len(results) == 1
        query = container.query_items.call_args.kwargs["query"]
        assert "RRF" in query
        assert "FullTextScore" in query

    async def test_search_turns(self):
        mem, container = _connected_client()
        turns = mem._turns_container_client
        turns.query_items = MagicMock(return_value=AsyncIterator([_make_doc()]))

        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1, 0.2, 0.3])

        results = await mem.search_turns(search_terms="weather", user_id="u1", thread_id="t1", top_k=3)

        assert len(results) == 1
        mem._embeddings_client.generate.assert_awaited_once_with("weather")
        turns.query_items.assert_called_once()
        container.query_items.assert_not_called()
        assert "VectorDistance" in turns.query_items.call_args.kwargs["query"]

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

    async def test_search_episodic_forwards_search_options(self):
        containers = {key: MagicMock() for key in ContainerKey}
        store = AsyncMemoryStore(containers=containers)
        store.search = AsyncMock(return_value=[])

        await store.search_episodic(
            user_id="u1",
            search_terms="weather",
            top_k=2,
            min_salience=0.4,
            include_superseded=True,
        )

        store.search.assert_awaited_once_with(
            search_terms="weather",
            user_id="u1",
            memory_types=["episodic"],
            top_k=2,
            min_salience=0.4,
            include_superseded=True,
        )

    async def test_build_episodic_context_forwards_search_options(self):
        containers = {key: MagicMock() for key in ContainerKey}
        store = AsyncMemoryStore(containers=containers)
        store.search_episodic = AsyncMock(return_value=[])

        context = await store.build_episodic_context("u1", "weather", top_k=2)

        assert context == ""
        store.search_episodic.assert_awaited_once_with("u1", "weather", top_k=2)


# ===================================================================
# Processing delegation (async)
# ===================================================================


class TestGenerateThreadSummary:
    async def test_generate_thread_summary(self):
        mem, container = _connected_client()
        mock_pipeline = AsyncMock()
        mock_pipeline.generate_thread_summary.return_value = {"status": "ok"}
        mock_pipeline._store = mem._get_store()
        mock_pipeline._containers = dict(mem._containers)
        mem._pipeline = mock_pipeline

        result = await mem.generate_thread_summary(user_id="u1", thread_id="t1")

        mock_pipeline.generate_thread_summary.assert_awaited_once_with(
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
        assert mem._memories_container_client is None

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


async def test_list_tags_delegates_to_store():
    mem, container = _connected_client()
    container.query_items = MagicMock(return_value=AsyncIterator([["topic:python", "sys:fact"]]))

    assert await mem.list_tags("u1") == ["topic:python"]

    kwargs = container.query_items.call_args.kwargs
    assert "SELECT VALUE c.tags" in kwargs["query"]
    assert kwargs["parameters"] == [{"name": "@user_id", "value": "u1"}]


class TestAsyncSearchCosmosUnifiedRetrieval:
    def _stub_store(self, mem, *, memories, turns=None, turns_raises=False):
        store = MagicMock()
        store.search = AsyncMock(return_value=list(memories))
        if turns_raises:
            store.search_turns = AsyncMock(side_effect=RuntimeError("turns not embedded"))
        else:
            store.search_turns = AsyncMock(return_value=list(turns or []))
        mem._get_store = MagicMock(return_value=store)
        return store

    async def test_default_does_not_search_turns(self):
        mem, _ = _connected_client()
        store = self._stub_store(mem, memories=[{"content": "fact A", "type": "fact"}])

        out = await mem.search_cosmos("q", user_id="u1")

        assert out == [{"content": "fact A", "type": "fact"}]
        store.search_turns.assert_not_called()

    async def test_include_turns_appends_and_dedups(self):
        mem, _ = _connected_client()
        self._stub_store(
            mem,
            memories=[{"content": "fact A", "type": "fact"}],
            turns=[
                {"content": "fact A", "type": "turn"},  # dup -> skipped
                {"content": "raw dialogue B", "type": "turn"},
            ],
        )

        out = await mem.search_cosmos("q", user_id="u1", include_turns=True)

        assert out == [
            {"content": "fact A", "type": "fact"},
            {"content": "raw dialogue B", "type": "turn"},
        ]

    async def test_include_turns_failure_returns_memories_only(self):
        mem, _ = _connected_client()
        self._stub_store(mem, memories=[{"content": "fact A", "type": "fact"}], turns_raises=True)

        out = await mem.search_cosmos("q", user_id="u1", include_turns=True)

        assert out == [{"content": "fact A", "type": "fact"}]

    async def test_include_summaries_appends_after_facts_and_dedups(self):
        mem, _ = _connected_client()
        store = self._stub_store(mem, memories=[{"content": "fact A", "type": "fact"}])
        store.search_summaries = AsyncMock(
            return_value=[
                {"content": "session overview", "type": "thread_summary"},
                {"content": "fact A", "type": "user_summary"},  # dup -> skipped
            ]
        )

        out = await mem.search_cosmos("q", user_id="u1", include_summaries=True)

        assert out == [
            {"content": "fact A", "type": "fact"},
            {"content": "session overview", "type": "thread_summary"},
        ]

    async def test_facts_summaries_turns_ordering(self):
        mem, _ = _connected_client()
        store = self._stub_store(
            mem,
            memories=[{"content": "fact A", "type": "fact"}],
            turns=[{"content": "raw turn C", "type": "turn"}],
        )
        store.search_summaries = AsyncMock(return_value=[{"content": "summary B", "type": "thread_summary"}])

        out = await mem.search_cosmos("q", user_id="u1", include_summaries=True, include_turns=True)

        assert [r["content"] for r in out] == ["fact A", "summary B", "raw turn C"]

    async def test_include_summaries_failure_returns_memories_only(self):
        mem, _ = _connected_client()
        store = self._stub_store(mem, memories=[{"content": "fact A", "type": "fact"}])
        store.search_summaries = AsyncMock(side_effect=RuntimeError("no summaries index"))

        out = await mem.search_cosmos("q", user_id="u1", include_summaries=True)

        assert out == [{"content": "fact A", "type": "fact"}]

    async def test_search_summaries_delegates_to_store(self):
        mem, _ = _connected_client()
        store = self._stub_store(mem, memories=[])
        store.search_summaries = AsyncMock(return_value=[{"content": "s", "type": "user_summary"}])

        out = await mem.search_summaries("q", user_id="u1")

        assert out == [{"content": "s", "type": "user_summary"}]
        store.search_summaries.assert_awaited_once()
