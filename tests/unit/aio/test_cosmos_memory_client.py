"""Unit tests for AsyncCosmosMemoryClient (unified async client)."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._partitioning import DEFAULT_TENANT_ID, PARTITION_KEY_PATHS
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.processors import AsyncInProcessProcessor
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import (
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EARLIER_KEY = ["/user_id", "/thread_id"]


def _keyed_container(paths=PARTITION_KEY_PATHS, container_id="container") -> MagicMock:
    """A container whose metadata names its partition key, as ``read()`` returns it."""
    properties = {"id": container_id, "partitionKey": {"paths": list(paths), "kind": "MultiHash", "version": 2}}
    return MagicMock(id=container_id, read=AsyncMock(return_value=properties))


async def _assert_discarded(mem, cosmos_client) -> None:
    """A failed connect or store creation closes the Cosmos client it opened and leaves ``mem`` unconnected."""
    cosmos_client.close.assert_awaited_once()
    for attribute in (
        "_cosmos_client",
        "_memories_container_client",
        "_turns_container_client",
        "_summaries_container_client",
        "_counter_container_client",
        "_store",
        "_pipeline",
    ):
        assert getattr(mem, attribute) is None, attribute
    with pytest.raises(CosmosNotConnectedError):
        await mem._require_cosmos()


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


def _connected_client(**client_kwargs) -> tuple[AsyncCosmosMemoryClient, MagicMock]:
    """Return a client with mocked split containers already wired up."""
    client = _make_client(**client_kwargs)
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
# Tenant binding
# ===================================================================


class TestTenant:
    def test_a_client_is_bound_to_the_default_tenant_unless_given_one(self):
        assert _make_client().tenant_id == DEFAULT_TENANT_ID

    def test_the_tenant_is_the_constructor_argument(self):
        assert _make_client(tenant_id="acme").tenant_id == "acme"

    @pytest.mark.parametrize("tenant_id", ["", "a/b", 7])
    def test_a_tenant_that_cannot_be_stored_is_refused(self, tenant_id):
        with pytest.raises(ValidationError):
            _make_client(tenant_id=tenant_id)

    def test_the_tenant_cannot_be_changed_after_construction(self):
        mem = _make_client(tenant_id="acme")

        with pytest.raises(AttributeError):
            mem.tenant_id = "globex"

    def test_the_store_and_the_pipeline_are_bound_to_the_clients_tenant(self):
        mem, _ = _connected_client(tenant_id="acme")

        assert mem._get_store().tenant_id == "acme"
        assert mem._get_pipeline()._tenant_id == "acme"

    @pytest.mark.parametrize(
        "method", ["update_cosmos", "delete_memory", "delete_turn", "add_tags", "remove_tags", "get_memory_history"]
    )
    def test_a_memory_is_addressed_by_its_owner_not_its_thread(self, method):
        assert "thread_id" not in inspect.signature(getattr(AsyncCosmosMemoryClient, method)).parameters


class _CustomProcessor:
    """A processor of its own: it reports no tenant."""


class TestProcessorTenant:
    @staticmethod
    def _processor(tenant_id: str) -> AsyncInProcessProcessor:
        """A processor over a real pipeline of ``tenant_id``."""
        return AsyncInProcessProcessor(
            cosmos_container=MagicMock(),
            turns_container=MagicMock(),
            summaries_container=MagicMock(),
            chat_client=MagicMock(),
            embeddings_client=MagicMock(),
            tenant_id=tenant_id,
        )

    def test_a_processor_of_another_tenant_is_refused(self):
        with pytest.raises(ConfigurationError, match=rf"tenant '{DEFAULT_TENANT_ID}'.*tenant 'acme'"):
            _make_client(tenant_id="acme", processor=self._processor(DEFAULT_TENANT_ID))

    def test_a_processor_of_the_clients_tenant_is_used(self):
        processor = self._processor("acme")

        assert _make_client(tenant_id="acme", processor=processor)._get_processor() is processor

    @pytest.mark.parametrize(
        "processor",
        [
            pytest.param(_CustomProcessor(), id="custom"),
            pytest.param(AsyncInProcessProcessor(pipeline=MagicMock()), id="pipeline-double"),
        ],
    )
    def test_a_processor_that_reports_no_tenant_is_not_checked(self, processor):
        assert _make_client(tenant_id="acme", processor=processor)._get_processor() is processor


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
        # with no thread_id would be buffered and pushed.
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
    CONTAINERS = ("memories", "memories_turns", "memories_summaries", "counter")

    @staticmethod
    @contextmanager
    def _serving(containers):
        """Patch ``azure.cosmos.aio`` so ``connect_cosmos`` finds ``containers`` by id; yields the Cosmos client."""
        cosmos_client = MagicMock(close=AsyncMock())
        cosmos_client.get_database_client.return_value.get_container_client.side_effect = containers.__getitem__
        aio = MagicMock(CosmosClient=MagicMock(return_value=cosmos_client))
        with patch.dict("sys.modules", {"azure.cosmos.aio": aio}):
            yield cosmos_client

    @staticmethod
    async def _connect(mem):
        await mem.connect_cosmos(endpoint="https://fake.documents.azure.com:443/", credential="fake-key")

    async def test_connect_cosmos_success(self):
        mem = _make_client(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_credential=MagicMock(),
        )
        mock_cosmos_cls = MagicMock()
        mock_db = MagicMock()
        mock_container = _keyed_container()
        mock_cosmos_cls.return_value = mock_cosmos_cls
        mock_cosmos_cls.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        with patch.dict(
            "sys.modules",
            {"azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls)},
        ):
            await mem.connect_cosmos()

        assert mem._memories_container_client is mock_container

    @pytest.mark.parametrize("earlier", CONTAINERS)
    async def test_a_container_with_the_earlier_key_is_refused(self, earlier):
        containers = {name: _keyed_container(container_id=name) for name in self.CONTAINERS}
        containers[earlier] = _keyed_container(EARLIER_KEY, container_id=earlier)
        mem = _make_client()

        with self._serving(containers) as cosmos_client:
            with pytest.raises(
                CosmosOperationError,
                match=(
                    rf"Failed to connect to Cosmos DB \(async\): container '{earlier}' has partition key "
                    r"\['/user_id', '/thread_id'\].*\['/tenant_id', '/user_id'\]"
                ),
            ):
                await self._connect(mem)

        await _assert_discarded(mem, cosmos_client)

    async def test_the_client_is_reset_even_when_closing_it_fails(self):
        containers = {name: _keyed_container(container_id=name) for name in self.CONTAINERS}
        containers["memories_turns"] = _keyed_container(EARLIER_KEY, container_id="memories_turns")
        mem = _make_client()

        with self._serving(containers) as cosmos_client:
            cosmos_client.close.side_effect = RuntimeError("close failed")
            with pytest.raises(
                CosmosOperationError,
                match=r"Failed to connect to Cosmos DB \(async\): container 'memories_turns' has partition key",
            ):
                await self._connect(mem)

        await _assert_discarded(mem, cosmos_client)

    async def test_a_container_that_does_not_exist_is_refused(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        containers = {name: _keyed_container(container_id=name) for name in self.CONTAINERS}
        containers["counter"].read.side_effect = CosmosResourceNotFoundError(message="missing")
        mem = _make_client()

        with self._serving(containers) as cosmos_client:
            with pytest.raises(
                CosmosOperationError,
                match=r"Failed to connect to Cosmos DB \(async\): container 'counter' does not exist",
            ):
                await self._connect(mem)

        await _assert_discarded(mem, cosmos_client)

    async def test_containers_keyed_by_tenant_and_user_connect_after_all_four_are_read(self):
        containers = {name: _keyed_container(container_id=name) for name in self.CONTAINERS}
        mem = _make_client()

        with self._serving(containers) as cosmos_client:
            await self._connect(mem)

        cosmos_client.close.assert_not_awaited()
        assert mem._memories_container_client is containers["memories"]
        assert mem._turns_container_client is containers["memories_turns"]
        assert mem._summaries_container_client is containers["memories_summaries"]
        assert mem._counter_container_client is containers["counter"]
        for container in containers.values():
            container.read.assert_awaited_once()


class TestPreferredLocations:
    @staticmethod
    async def _cosmos_client_kwargs(entry_point: str = "connect_cosmos", **client_kwargs) -> dict:
        mock_cosmos_cls = MagicMock()
        database = MagicMock(
            create_container_if_not_exists=AsyncMock(),
            get_container_client=MagicMock(return_value=_keyed_container()),
        )
        mock_cosmos_cls.return_value.create_database_if_not_exists = AsyncMock(return_value=database)
        mock_cosmos_cls.return_value.get_database_client.return_value = database
        mem = _make_client(**client_kwargs)
        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls),
                "azure.cosmos": MagicMock(PartitionKey=MagicMock(), ThroughputProperties=MagicMock()),
            },
        ):
            await getattr(mem, entry_point)(endpoint="https://fake.documents.azure.com:443/", credential="fake-key")
        return mock_cosmos_cls.call_args.kwargs

    @pytest.mark.parametrize("entry_point", ["connect_cosmos", "create_memory_store"])
    async def test_constructor_argument_reaches_the_async_cosmos_client(self, monkeypatch, entry_point):
        monkeypatch.delenv("COSMOS_DB_PREFERRED_REGIONS", raising=False)

        kwargs = await self._cosmos_client_kwargs(entry_point, cosmos_preferred_locations="West US 3, West US 2")

        assert kwargs["preferred_locations"] == ["West US 3", "West US 2"]

    async def test_preferred_locations_default_to_account_order(self, monkeypatch):
        monkeypatch.delenv("COSMOS_DB_PREFERRED_REGIONS", raising=False)
        assert (await self._cosmos_client_kwargs())["preferred_locations"] == []

    @pytest.mark.parametrize("entry_point", ["connect_cosmos", "create_memory_store"])
    async def test_preferred_locations_read_from_environment(self, monkeypatch, entry_point):
        monkeypatch.setenv("COSMOS_DB_PREFERRED_REGIONS", "West US 3")
        assert (await self._cosmos_client_kwargs(entry_point))["preferred_locations"] == ["West US 3"]


class TestCreateMemoryStore:
    @staticmethod
    async def _created_containers() -> list[dict]:
        """The arguments of the five ``create_container_if_not_exists`` calls ``create_memory_store`` makes."""
        mock_cosmos_cls = MagicMock()
        mock_db = AsyncMock()
        mock_cosmos_cls.return_value.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        partition_key = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls),
                "azure.cosmos": MagicMock(PartitionKey=partition_key, ThroughputProperties=MagicMock()),
            },
        ):
            await _make_client().create_memory_store(
                endpoint="https://fake.documents.azure.com:443/", credential="fake-key"
            )
        return [call.kwargs for call in mock_db.create_container_if_not_exists.await_args_list]

    async def test_the_containers_are_partitioned_by_tenant_and_user(self):
        memories, turns, summaries, counter, leases = await self._created_containers()

        for container in (memories, turns, summaries, counter):
            assert container["partition_key"].path == ["/tenant_id", "/user_id"]
            assert container["partition_key"].kind == "MultiHash"
        assert leases["partition_key"].path == "/id"

    async def test_the_summaries_composite_index_starts_with_the_partition_key(self):
        summaries = (await self._created_containers())[2]

        assert summaries["indexing_policy"]["compositeIndexes"] == [
            [
                {"path": "/tenant_id", "order": "ascending"},
                {"path": "/user_id", "order": "ascending"},
                {"path": "/thread_id", "order": "ascending"},
                {"path": "/version", "order": "descending"},
            ]
        ]

    async def test_create_memory_store_with_counter_container(self):
        mem = _make_client()
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = AsyncMock()
        mock_memories_container = _keyed_container()
        mock_counter_container = _keyed_container()
        mock_lease_container = MagicMock()
        mock_throughput_cls = MagicMock(side_effect=lambda **kwargs: type("Throughput", (), kwargs)())

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_turns_container = _keyed_container()
        mock_summaries_container = _keyed_container()
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
        assert counter_call.kwargs["default_ttl"] == 7_776_000
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
        mock_memories_container = _keyed_container()
        mock_counter_container = _keyed_container()
        mock_lease_container = MagicMock()
        mock_turns_container = _keyed_container()

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_summaries_container = _keyed_container()
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
        mock_memories_container = _keyed_container()
        mock_counter_container = _keyed_container()
        mock_lease_container = MagicMock()

        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mock_turns_container = _keyed_container()
        mock_summaries_container = _keyed_container()
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

    @pytest.mark.parametrize("close_fails", [False, True], ids=["closes", "close-fails"])
    @pytest.mark.parametrize("earlier", ["memories", "memories_turns", "memories_summaries", "counter"])
    async def test_an_existing_container_with_the_earlier_key_is_refused(self, earlier, close_fails):
        containers = {
            name: _keyed_container(container_id=name)
            for name in ("memories", "memories_turns", "memories_summaries", "counter")
        }
        containers[earlier] = _keyed_container(EARLIER_KEY, container_id=earlier)
        mock_cosmos_cls = MagicMock()
        mock_cosmos_cls.return_value.close = AsyncMock(
            side_effect=RuntimeError("close failed") if close_fails else None
        )
        mock_db = AsyncMock()
        mock_db.create_container_if_not_exists = AsyncMock(side_effect=[*containers.values(), MagicMock()])
        mock_cosmos_cls.return_value.create_database_if_not_exists = AsyncMock(return_value=mock_db)
        mem = _make_client()

        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos.aio": MagicMock(CosmosClient=mock_cosmos_cls),
                "azure.cosmos": MagicMock(PartitionKey=MagicMock(), ThroughputProperties=MagicMock()),
            },
        ):
            with pytest.raises(
                CosmosOperationError,
                match=rf"container '{earlier}' has partition key \['/user_id', '/thread_id'\].*recreate the container",
            ):
                await mem.create_memory_store(endpoint="https://fake.documents.azure.com:443/", credential="fake-key")

        await _assert_discarded(mem, mock_cosmos_cls.return_value)

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
    CONTAINER_ATTRIBUTES = {
        "memories": "_memories_container_client",
        "memories_turns": "_turns_container_client",
        "memories_summaries": "_summaries_container_client",
        "counter": "_counter_container_client",
    }

    def _connected(self, earlier=None):
        mem = _make_client()
        containers = {
            name: _keyed_container(EARLIER_KEY if name == earlier else PARTITION_KEY_PATHS, container_id=name)
            for name in self.CONTAINER_ATTRIBUTES
        }
        for name, attribute in self.CONTAINER_ATTRIBUTES.items():
            setattr(mem, attribute, containers[name])
        return mem, containers

    async def test_validate_topology_succeeds_on_healthy_deploy(self):
        mem, containers = self._connected()

        await mem.validate_topology()

        for container in containers.values():
            container.read.assert_awaited_once()

    @pytest.mark.parametrize("earlier", ["memories", "memories_turns", "memories_summaries", "counter"])
    async def test_validate_topology_raises_on_a_container_with_the_earlier_key(self, earlier):
        mem, _ = self._connected(earlier)

        with pytest.raises(RuntimeError, match=rf"validate_topology: container '{earlier}' has partition key"):
            await mem.validate_topology()

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
    async def test_upsert_memory(self):
        mem, container = _connected_client()
        # Suppress the background cadence task to keep the test focused on the CRUD write.
        mem._maybe_auto_trigger = AsyncMock()
        await mem.upsert_memory(user_id="u1", role="user", content="hello", thread_id="t1")
        # Drain any pending background tasks (none expected since we stubbed the trigger).
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        turns = mem._turns_container_client
        turns.upsert_item.assert_awaited_once()
        body = turns.upsert_item.call_args.kwargs["body"]
        assert body["content"] == "hello"
        assert body["user_id"] == "u1"

    async def test_upsert_memory_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            await mem.upsert_memory(user_id="u1", role="user", content="hi", thread_id="t1")

    async def test_upsert_memory_turn_requires_thread_id(self):
        """Turn writes must declare a thread_id so the auto-trigger counter can group them."""
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="thread_id is required"):
            await mem.upsert_memory(user_id="u1", role="user", content="hi")  # memory_type='turn' default

    async def test_upsert_memory_non_turn_does_not_require_thread_id(self):
        """Non-turn writes (facts, episodics, etc.) work without thread_id and skip cadence."""
        mem, container = _connected_client()
        trigger = AsyncMock()
        mem._maybe_auto_trigger = trigger

        await mem.upsert_memory(user_id="u1", role="user", content="prefers dark mode", memory_type="fact")
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        container.upsert_item.assert_awaited_once()
        trigger.assert_not_awaited()

    async def test_upsert_memory_turn_schedules_cadence(self):
        """A turn write must schedule the auto-trigger as a background task so cadence
        env vars apply whether the caller uses the local buffer or writes through directly."""
        mem, _ = _connected_client()
        trigger = AsyncMock()
        mem._maybe_auto_trigger = trigger

        await mem.upsert_memory(user_id="u1", role="user", content="hello", thread_id="t1")
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
        mem, container = _connected_client(tenant_id="acme")
        doc = _make_doc(id="m1", type="fact", tenant_id="acme")
        container.read_item = AsyncMock(return_value=doc)
        container.replace_item = AsyncMock()

        await mem.update_cosmos(memory_id="m1", user_id="u1", memory_type="fact", content="updated")

        container.read_item.assert_awaited_once_with(item="m1", partition_key=["acme", "u1"])
        container.replace_item.assert_awaited_once()
        body = container.replace_item.call_args.kwargs["body"]
        assert body["content"] == "updated"
        assert body["type"] == "fact"
        assert body["tenant_id"] == "acme"
        assert "updated_at" in body

    async def test_not_found(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, container = _connected_client()
        container.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))

        with pytest.raises(MemoryNotFoundError):
            await mem.update_cosmos(memory_id="missing", user_id="u1", memory_type="fact")

    async def test_partial_fields(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1", role="user", content="old", type="fact")
        container.read_item = AsyncMock(return_value=doc)
        container.replace_item = AsyncMock()

        await mem.update_cosmos(memory_id="m1", user_id="u1", memory_type="fact", role="agent", metadata={"key": "val"})
        body = container.replace_item.call_args.kwargs["body"]
        assert body["role"] == "agent"
        assert body["metadata"] == {"key": "val"}
        assert body["content"] == "old"
        assert "updated_at" in body


class TestDeleteCosmos:
    async def test_success(self):
        mem, container = _connected_client(tenant_id="acme")
        container.read_item = AsyncMock(return_value=_make_doc(id="m1", type="fact", tenant_id="acme"))
        container.delete_item = AsyncMock()

        await mem.delete_memory(memory_id="m1", user_id="u1", memory_type="fact")

        container.delete_item.assert_awaited_once_with(item="m1", partition_key=["acme", "u1"])

    async def test_not_found(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, container = _connected_client()
        container.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))
        container.delete_item = AsyncMock()

        with pytest.raises(MemoryNotFoundError):
            await mem.delete_memory(memory_id="x", user_id="u1", memory_type="fact")

        container.delete_item.assert_not_awaited()


class TestGetUserSummary:
    async def test_returns_doc_when_present(self):
        mem, _ = _connected_client(tenant_id="acme")
        summaries = mem._summaries_container_client
        doc = _make_doc(type="user_summary", id="user_summary_u1", tenant_id="acme")
        summaries.read_item = AsyncMock(return_value=doc)

        result = await mem.get_user_summary(user_id="u1")

        call_kwargs = summaries.read_item.call_args.kwargs
        assert call_kwargs["item"] == "user_summary_u1"
        assert call_kwargs["partition_key"] == ["acme", "u1"]
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
            await mem.update_cosmos(memory_id="m1", user_id="u1", memory_type="fact")
        with pytest.raises(CosmosNotConnectedError):
            await mem.delete_memory(memory_id="m1", user_id="u1", memory_type="fact")


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
        store.query = AsyncMock(return_value=[])
        store._embed = AsyncMock(return_value=[0.1])

        await store.search_episodic(
            user_id="u1",
            search_terms="weather",
            top_k=2,
            min_salience=0.4,
            include_superseded=True,
        )

        query, parameters = store.query.await_args.args[:2]
        assert "c.type = @type" in query
        assert "VectorDistance(c.embedding, @embedding)" in query
        assert {"name": "@type", "value": "episodic"} in parameters

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
    mem, container = _connected_client(tenant_id="acme")
    container.query_items = MagicMock(return_value=AsyncIterator([["topic:python", "sys:fact"]]))

    assert await mem.list_tags("u1") == ["topic:python"]

    kwargs = container.query_items.call_args.kwargs
    assert "SELECT VALUE c.tags" in kwargs["query"]
    assert "c.tenant_id = @tenant_id AND c.user_id = @user_id" in kwargs["query"]
    assert kwargs["parameters"] == [{"name": "@tenant_id", "value": "acme"}, {"name": "@user_id", "value": "u1"}]
    assert kwargs["partition_key"] == ["acme", "u1"]


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


class TestAsyncDeleteHelpers:
    @pytest.mark.asyncio
    async def test_delete_turn_delegates_with_turn_type(self):
        mem = _make_client()
        mem.delete_memory = AsyncMock()
        await mem.delete_turn("turn-1", user_id="u1")
        mem.delete_memory.assert_awaited_once_with("turn-1", user_id="u1", memory_type="turn")

    @pytest.mark.asyncio
    async def test_delete_thread_summary_returns_true_then_false(self):
        mem = _make_client()
        mem.delete_memory = AsyncMock()
        assert await mem.delete_thread_summary("u1", "t1") is True
        mem.delete_memory.assert_awaited_once_with("summary_u1_t1", user_id="u1", memory_type="thread_summary")
        mem.delete_memory = AsyncMock(side_effect=MemoryNotFoundError(memory_id="x", user_id="u1"))
        assert await mem.delete_thread_summary("u1", "t1") is False

    @pytest.mark.asyncio
    async def test_delete_user_summary_uses_deterministic_id(self):
        mem = _make_client()
        mem.delete_memory = AsyncMock()
        assert await mem.delete_user_summary("u1") is True
        mem.delete_memory.assert_awaited_once_with("user_summary_u1", user_id="u1", memory_type="user_summary")

    @pytest.mark.asyncio
    async def test_delete_thread_deletes_turns_and_summary_and_counts(self):
        mem = _make_client()
        mem.get_thread = AsyncMock(return_value=[{"id": "turn-1"}, {"id": "turn-2"}])
        mem.delete_memory = AsyncMock()
        deleted = await mem.delete_thread("u1", "t1")
        assert deleted == 3  # 2 turns + 1 summary
        assert mem.get_thread.call_args.kwargs["include_superseded"] is True

    @pytest.mark.asyncio
    async def test_delete_thread_skips_missing_turn(self):
        mem = _make_client()
        mem.get_thread = AsyncMock(return_value=[{"id": "turn-1"}, {"id": "turn-2"}])

        async def _delete(memory_id, **kwargs):
            if memory_id == "turn-1":
                raise MemoryNotFoundError(memory_id="turn-1", user_id="u1")

        mem.delete_memory = AsyncMock(side_effect=_delete)
        deleted = await mem.delete_thread("u1", "t1", include_summary=False)
        assert deleted == 1

    @pytest.mark.asyncio
    async def test_delete_thread_requires_ids(self):
        mem = _make_client()
        with pytest.raises(ValidationError):
            await mem.delete_thread("", "t1")


# ===================================================================
# Tenant isolation: a client bound to a tenant reads, writes and deletes only inside it
# ===================================================================

_READS = {
    "get_memories": lambda mem: mem.get_memories(user_id="u1", thread_id="t1"),
    "get_memories_of_every_user": lambda mem: mem.get_memories(),
    "get_thread": lambda mem: mem.get_thread(thread_id="t1", user_id="u1"),
    "get_thread_of_every_user": lambda mem: mem.get_thread(thread_id="t1"),
    "get_thread_summary": lambda mem: mem.get_thread_summary("u1", "t1"),
    "get_episodes": lambda mem: mem.get_episodes("u1"),
    "list_tags": lambda mem: mem.list_tags("u1"),
    "get_procedural_memories": lambda mem: mem.get_procedural_memories("u1"),
    "get_procedural_history": lambda mem: mem.get_procedural_history("u1"),
    "get_memory_history": lambda mem: mem.get_memory_history("m1", "u1"),
    "search_cosmos": lambda mem: mem.search_cosmos("weather", user_id="u1"),
    "search_cosmos_by_keywords_of_every_user": lambda mem: mem.search_cosmos("weather Seattle"),
    "search_turns": lambda mem: mem.search_turns("weather", user_id="u1", thread_id="t1"),
    "search_summaries": lambda mem: mem.search_summaries("weather", user_id="u1"),
    "search_episodic_memories": lambda mem: mem.search_episodic_memories("u1", "weather"),
    "retrieve_procedures": lambda mem: mem.retrieve_procedures("u1", "weather"),
}


class TestTenantIsolation:
    @pytest.mark.parametrize("read", _READS.values(), ids=_READS.keys())
    async def test_every_query_a_read_sends_is_confined_to_the_tenant(self, read):
        mem, _ = _connected_client(tenant_id="acme")
        mem._embeddings_client = AsyncMock()
        mem._embeddings_client.generate = AsyncMock(return_value=[0.1, 0.2, 0.3])

        await read(mem)

        sent = [
            call.kwargs
            for container in (
                mem._memories_container_client,
                mem._turns_container_client,
                mem._summaries_container_client,
            )
            for call in container.query_items.call_args_list
        ]
        assert sent
        for query in sent:
            assert "c.tenant_id = @tenant_id" in query["query"]
            assert {"name": "@tenant_id", "value": "acme"} in query["parameters"]
            assert query.get("partition_key") in (None, ["acme", "u1"])

    async def test_a_written_turn_carries_the_tenant(self):
        mem, _ = _connected_client(tenant_id="acme")
        mem._maybe_auto_trigger = AsyncMock()

        await mem.upsert_memory(user_id="u1", role="user", content="hello", thread_id="t1")
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        body = mem._turns_container_client.upsert_item.call_args.kwargs["body"]
        assert (body["tenant_id"], body["user_id"], body["thread_id"]) == ("acme", "u1", "t1")

    async def test_pushed_memories_carry_the_tenant(self):
        mem, memories = _connected_client(tenant_id="acme")
        mem._maybe_auto_trigger = AsyncMock()
        mem.add_local(user_id="u1", role="user", content="prefers dark mode", memory_type="fact")
        mem.add_local(user_id="u1", role="user", content="hello", thread_id="t1")

        await mem.push_to_cosmos()
        await asyncio.gather(*list(mem._background_tasks), return_exceptions=True)

        written = [*memories.upsert_item.call_args_list, *mem._turns_container_client.upsert_item.call_args_list]
        assert [call.kwargs["body"]["tenant_id"] for call in written] == ["acme", "acme"]

    async def test_a_record_that_names_another_tenant_is_refused(self):
        mem, memories = _connected_client(tenant_id="acme")
        mem.add_local(user_id="u1", role="user", content="prefers dark mode", memory_type="fact")
        mem.local_memory[0]["tenant_id"] = "globex"

        with pytest.raises(ValidationError):
            await mem.push_to_cosmos()

        memories.upsert_item.assert_not_awaited()

    async def test_a_thread_is_deleted_from_the_owners_partition_of_the_tenant(self):
        mem, _ = _connected_client(tenant_id="acme")
        turns, summaries = mem._turns_container_client, mem._summaries_container_client
        turn = _make_doc(id="turn-1", tenant_id="acme")
        turns.query_items = MagicMock(return_value=AsyncIterator([turn]))
        turns.read_item = AsyncMock(return_value=turn)
        summaries.read_item = AsyncMock(
            return_value=_make_doc(id="summary_u1_t1", type="thread_summary", tenant_id="acme")
        )

        assert await mem.delete_thread("u1", "t1") == 2

        for container in (turns, summaries):
            container.delete_item.assert_awaited_once()
            assert container.delete_item.call_args.kwargs["partition_key"] == ["acme", "u1"]
