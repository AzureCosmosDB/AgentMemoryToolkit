"""Unit tests for CosmosMemoryClient (unified sync client)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.exceptions import (
    ConfigurationError,
    CosmosNotConnectedError,
    MemoryNotFoundError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**overrides) -> CosmosMemoryClient:
    """Build a CosmosMemoryClient with credential auto-resolution disabled."""
    defaults: dict = {"use_default_credential": False}
    defaults.update(overrides)
    return CosmosMemoryClient(**defaults)


def _connected_client() -> tuple[CosmosMemoryClient, MagicMock]:
    """Return a client with mocked split containers already wired up."""
    client = _make_client()
    container = MagicMock()
    turns = MagicMock()
    summaries = MagicMock()
    container.id = "memories"
    turns.id = "memories_turns"
    summaries.id = "memories_summaries"
    container.query_items.return_value = []
    turns.query_items.return_value = []
    summaries.query_items.return_value = []
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
    def test_default_credential_created_when_flag_true(self):
        """use_default_credential=True creates independent DefaultAzureCredential instances."""
        sentinel = MagicMock(name="default-cred")
        mock_module = MagicMock()
        mock_module.DefaultAzureCredential.return_value = sentinel

        with patch.dict("sys.modules", {"azure.identity": mock_module}):
            mem = CosmosMemoryClient(use_default_credential=True)
            # Two independent instances — one per consumer (cosmos + AI Foundry)
            # — so close() can tear each down without affecting the other.
            assert mock_module.DefaultAzureCredential.call_count == 2
            assert mem._cosmos_credential is sentinel
            assert mem._ai_foundry_credential is sentinel
            assert mem._owns_cosmos_credential is True
            assert mem._owns_ai_foundry_credential is True

    def test_no_credential_when_flag_false(self):
        """use_default_credential=False leaves credentials as None."""
        mem = _make_client()
        assert mem._cosmos_credential is None
        assert mem._ai_foundry_credential is None


# ===================================================================
# Injected embeddings / chat clients
# ===================================================================


class _FakeEmbeddings:
    """Minimal stand-in for EmbeddingsClient used to verify injection."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeChat:
    """Minimal stand-in for ChatClient used to verify injection."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
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

    def test_close_does_not_close_injected_clients(self):
        emb = _FakeEmbeddings()
        chat = _FakeChat()
        mem = _make_client(embeddings_client=emb, chat_client=chat)

        mem.close()

        assert emb.closed is False
        assert chat.closed is False


# ===================================================================
# Local CRUD
# ===================================================================


class TestAddLocal:
    def test_add_local_valid(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="hello", thread_id="t1")

        assert len(mem.local_memory) == 1
        m = mem.local_memory[0]
        assert m["user_id"] == "u1"
        assert m["role"] == "user"
        assert m["content"] == "hello"
        assert m["type"] == "turn"
        assert "id" in m
        assert "created_at" in m

    def test_add_local_all_fields(self):
        mem = _make_client()
        mem.add_local(
            user_id="u1",
            role="agent",
            content="hi",
            memory_type="thread_summary",
            agent_id="a1",
            metadata={"k": "v"},
            thread_id="t1",
        )

        m = mem.local_memory[0]
        assert m["role"] == "agent"
        assert m["type"] == "thread_summary"
        assert m["metadata"] == {"k": "v"}
        assert m["thread_id"] == "t1"

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

    def test_add_local_non_turn_thread_id_optional(self):
        mem = _make_client()
        # Non-turn types (thread_summary, fact, etc.) accept an omitted thread_id;
        # _make_memory auto-generates a UUID for hierarchical-PK validity.
        mem.add_local(user_id="u1", role="user", content="profile", memory_type="user_summary")
        assert len(mem.local_memory) == 1


class TestGetLocal:
    def test_get_local_no_filters(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u2", role="agent", content="b", thread_id="t1")

        results = mem.get_local()
        assert len(results) == 2

    def test_get_local_with_filters(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", memory_type="turn", thread_id="t1")
        mem.add_local(user_id="u1", role="agent", content="b", memory_type="turn", thread_id="t1")
        mem.add_local(user_id="u2", role="user", content="c", memory_type="thread_summary")

        results = mem.get_local(user_id="u1", role="user", memory_types=["turn"])
        assert len(results) == 1
        assert results[0]["content"] == "a"

    def test_get_local_by_id(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="target", thread_id="t1")
        mem.add_local(user_id="u1", role="user", content="other", thread_id="t1")
        mid = mem.local_memory[0]["id"]

        results = mem.get_local(memory_id=mid)
        assert len(results) == 1
        assert results[0]["content"] == "target"


class TestUpdateLocal:
    def test_update_local_success(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="old", thread_id="t1")
        mid = mem.local_memory[0]["id"]

        mem.update_local(mid, content="new", metadata={"k": "v"})

        m = mem.local_memory[0]
        assert m["content"] == "new"
        assert m["metadata"] == {"k": "v"}
        assert "updated_at" in m

    def test_update_local_not_found(self):
        mem = _make_client()
        with pytest.raises(MemoryNotFoundError):
            mem.update_local("nonexistent-id", content="x")

    def test_update_local_partial(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="old", thread_id="t1")
        mid = mem.local_memory[0]["id"]

        mem.update_local(mid, role="agent", metadata={"k": "v"})

        m = mem.local_memory[0]
        assert m["role"] == "agent"
        assert m["metadata"] == {"k": "v"}
        assert m["content"] == "old"  # unchanged


class TestDeleteLocal:
    def test_delete_local_success(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="x", thread_id="t1")
        mid = mem.local_memory[0]["id"]

        mem.delete_local(mid)
        assert len(mem.local_memory) == 0

    def test_delete_local_not_found(self):
        mem = _make_client()
        with pytest.raises(MemoryNotFoundError):
            mem.delete_local("nonexistent-id")


# ===================================================================
# Cosmos connection
# ===================================================================


class TestAutoCreateOnInit:
    def test_auto_creates_store_when_endpoint_provided(self):
        """Constructor calls create_memory_store() when cosmos_endpoint is set."""
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()
        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_turns_container = MagicMock()
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists.side_effect = [
            mock_memories_container,
            mock_turns_container,
            mock_summaries_container,
            mock_counter_container,
            mock_lease_container,
        ]

        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos": MagicMock(
                    CosmosClient=mock_cosmos_cls,
                    PartitionKey=MagicMock(),
                    ThroughputProperties=MagicMock(),
                ),
            },
        ):
            mem = _make_client(
                cosmos_endpoint="https://fake.documents.azure.com:443/",
                cosmos_credential="fake-key",
            )

        assert mem._memories_container_client is mock_memories_container
        assert mem._turns_container_client is mock_turns_container
        assert mem._summaries_container_client is mock_summaries_container
        mock_client.create_database_if_not_exists.assert_called_once()
        assert mock_db.create_container_if_not_exists.call_count == 5


class TestRequireCosmos:
    def test_require_cosmos_before_connect(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            mem._require_cosmos()


class TestValidateTopology:
    def test_validate_topology_succeeds_on_healthy_deploy(self):
        mem = _make_client()
        memories = MagicMock(id="memories")
        turns = MagicMock(id="memories_turns")
        summaries = MagicMock(id="memories_summaries")
        mem._memories_container_client = memories
        mem._turns_container_client = turns
        mem._summaries_container_client = summaries

        mem.validate_topology()

        memories.read.assert_called_once()
        turns.read.assert_called_once()
        summaries.read.assert_called_once()

    def test_validate_topology_raises_on_missing_container(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem = _make_client()
        mem._memories_container_client = MagicMock(id="memories")
        mem._turns_container_client = MagicMock(id="memories_turns")
        mem._summaries_container_client = MagicMock(id="memories_summaries")
        mem._summaries_container_client.read.side_effect = CosmosResourceNotFoundError(message="missing")

        with pytest.raises(RuntimeError, match="memories_summaries"):
            mem.validate_topology()

    def test_validate_topology_raises_when_not_connected(self):
        mem = _make_client()

        with pytest.raises(RuntimeError, match="call connect_cosmos"):
            mem.validate_topology()


class TestCreateMemoryStore:
    def test_create_memory_store_with_custom_dimensions(self):
        """Explicit create_memory_store() call with custom embedding dimensions."""
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()
        mock_throughput_cls = MagicMock(side_effect=lambda **kwargs: type("Throughput", (), kwargs)())
        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_turns_container = MagicMock()
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists.side_effect = [
            mock_memories_container,
            mock_turns_container,
            mock_summaries_container,
            mock_counter_container,
            mock_lease_container,
        ]

        # Start local-only, then create store explicitly
        mem = _make_client()

        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos": MagicMock(
                    CosmosClient=mock_cosmos_cls,
                    PartitionKey=MagicMock(),
                    ThroughputProperties=mock_throughput_cls,
                ),
            },
        ):
            mem.create_memory_store(
                endpoint="https://fake.documents.azure.com:443/",
                credential="fake-key",
                embedding_dimensions=256,
                throughput_mode="autoscale",
                autoscale_max_ru=1000,
            )

        mock_client.create_database_if_not_exists.assert_called_once()
        memories_call = mock_db.create_container_if_not_exists.call_args_list[0]
        summaries_call = mock_db.create_container_if_not_exists.call_args_list[2]
        counter_call = mock_db.create_container_if_not_exists.call_args_list[3]
        lease_call = mock_db.create_container_if_not_exists.call_args_list[4]
        vec_policy = memories_call.kwargs["vector_embedding_policy"]
        assert vec_policy["vectorEmbeddings"][0]["dimensions"] == 256
        ft_policy = memories_call.kwargs["full_text_policy"]
        assert ft_policy["defaultLanguage"] == "en-US"
        assert counter_call.kwargs["id"] == "counter"
        assert counter_call.kwargs["offer_throughput"].auto_scale_max_throughput == 1000
        assert lease_call.kwargs["id"] == "leases"
        assert lease_call.kwargs["offer_throughput"].auto_scale_max_throughput == 1000
        assert summaries_call.kwargs["id"] == "memories_summaries"
        assert "vector_embedding_policy" not in summaries_call.kwargs
        assert "full_text_policy" not in summaries_call.kwargs
        assert summaries_call.kwargs["indexing_policy"]["compositeIndexes"][0][-1] == {
            "path": "/version",
            "order": "descending",
        }
        assert "vector_embedding_policy" not in counter_call.kwargs
        assert mem._memories_container_client is mock_memories_container

    def test_create_memory_store_turns_container_uses_30_day_ttl(self):
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()
        mock_turns_container = MagicMock()
        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists.side_effect = [
            mock_memories_container,
            mock_turns_container,
            mock_summaries_container,
            mock_counter_container,
            mock_lease_container,
        ]

        mem = _make_client()

        with patch.dict(
            "sys.modules",
            {
                "azure.cosmos": MagicMock(
                    CosmosClient=mock_cosmos_cls,
                    PartitionKey=MagicMock(),
                    ThroughputProperties=MagicMock(),
                ),
            },
        ):
            mem.create_memory_store(
                endpoint="https://fake.documents.azure.com:443/",
                credential="fake-key",
                turns_container="memories_turns",
            )

        turns_call = mock_db.create_container_if_not_exists.call_args_list[1]
        assert turns_call.kwargs["id"] == "memories_turns"
        assert turns_call.kwargs["default_ttl"] == 2_592_000
        # The turns container is always provisioned with a vector index + full-text
        # policy so it is primed for search_turns() even when turn
        # embeddings are disabled. Vector indexes use quantizedFlat.
        assert "vector_embedding_policy" in turns_call.kwargs
        assert "full_text_policy" in turns_call.kwargs
        assert turns_call.kwargs["indexing_policy"]["vectorIndexes"][0]["type"] == "quantizedFlat"
        assert mem._turns_container_client is mock_turns_container

    def test_create_memory_store_defaults_to_serverless(self):
        mock_cosmos_cls = MagicMock()
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_memories_container = MagicMock()
        mock_counter_container = MagicMock()
        mock_lease_container = MagicMock()
        mock_cosmos_cls.return_value = mock_client
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_turns_container = MagicMock()
        mock_summaries_container = MagicMock()
        mock_db.create_container_if_not_exists.side_effect = [
            mock_memories_container,
            mock_turns_container,
            mock_summaries_container,
            mock_counter_container,
            mock_lease_container,
        ]

        mem = _make_client(cosmos_throughput_mode="serverless")

        with patch.dict("os.environ", {"COSMOS_DB_AUTOSCALE_MAX_RU": "not-an-int"}, clear=False):
            with patch.dict(
                "sys.modules",
                {
                    "azure.cosmos": MagicMock(
                        CosmosClient=mock_cosmos_cls,
                        PartitionKey=MagicMock(),
                        ThroughputProperties=MagicMock(),
                    ),
                },
            ):
                mem.create_memory_store(
                    endpoint="https://fake.documents.azure.com:443/",
                    credential="fake-key",
                    throughput_mode="serverless",
                )

        for call in mock_db.create_container_if_not_exists.call_args_list:
            assert "offer_throughput" not in call.kwargs

    def test_constructor_ignores_invalid_autoscale_env_in_serverless_mode(self):
        with patch.dict("os.environ", {"COSMOS_DB_AUTOSCALE_MAX_RU": "not-an-int"}, clear=False):
            mem = _make_client(cosmos_throughput_mode="serverless")

        assert mem._cosmos_autoscale_max_ru is None

    def test_constructor_rejects_invalid_throughput_mode(self):
        with pytest.raises(ConfigurationError, match="expected 'serverless' or 'autoscale'"):
            _make_client(cosmos_throughput_mode="invalid")


# ===================================================================
# Cosmos CRUD (mock _memories_container_client)
# ===================================================================


class TestAddCosmos:
    def test_add_cosmos(self):
        mem, container = _connected_client()
        # Suppress cadence work — the trigger path is exercised in
        # tests/unit/test_auto_trigger.py; this test just asserts the CRUD write.
        mem._maybe_auto_trigger = MagicMock()
        mem.add_cosmos(user_id="u1", role="user", content="hello", thread_id="t1")

        turns = mem._turns_container_client
        turns.upsert_item.assert_called_once()
        body = turns.upsert_item.call_args.kwargs["body"]
        assert body["content"] == "hello"
        assert body["user_id"] == "u1"
        assert body["role"] == "user"

    def test_add_cosmos_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            mem.add_cosmos(user_id="u1", role="user", content="hi", thread_id="t1")

    def test_add_cosmos_turn_requires_thread_id(self):
        """Turn writes must declare a thread_id so the auto-trigger counter can group them."""
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="thread_id is required"):
            mem.add_cosmos(user_id="u1", role="user", content="hi")  # memory_type='turn' default

    def test_add_cosmos_non_turn_does_not_require_thread_id(self):
        """Non-turn writes (facts, episodics, etc.) work without thread_id and skip cadence."""
        mem, container = _connected_client()
        trigger = MagicMock()
        mem._maybe_auto_trigger = trigger

        mem.add_cosmos(user_id="u1", role="user", content="prefers dark mode", memory_type="fact")

        container.upsert_item.assert_called_once()
        trigger.assert_not_called()

    def test_add_cosmos_turn_triggers_cadence(self):
        """A turn write must bump the auto-trigger counter so cadence env vars apply
        whether the caller uses the local buffer or writes through directly."""
        mem, _ = _connected_client()
        trigger = MagicMock()
        mem._maybe_auto_trigger = trigger

        mem.add_cosmos(user_id="u1", role="user", content="hello", thread_id="t1")

        trigger.assert_called_once_with({("u1", "t1"): 1})

    def test_add_cosmos_swallows_cadence_failure(self):
        """If the cadence trigger raises, the add_cosmos call must still succeed —
        the user's turn was written; cadence is best-effort telemetry."""
        mem, _ = _connected_client()
        mem._maybe_auto_trigger = MagicMock(side_effect=RuntimeError("boom"))

        # Should NOT raise — the write succeeded.
        result_id = mem.add_cosmos(user_id="u1", role="user", content="hi", thread_id="t1")

        assert isinstance(result_id, str)
        mem._turns_container_client.upsert_item.assert_called_once()


class TestPushToCosmos:
    def test_push_to_cosmos(self):
        mem, container = _connected_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        mem.add_local(user_id="u1", role="agent", content="b", thread_id="t1")

        mem.push_to_cosmos()

        assert mem._turns_container_client.upsert_item.call_count == 2

    def test_push_to_cosmos_not_connected(self):
        mem = _make_client()
        mem.add_local(user_id="u1", role="user", content="a", thread_id="t1")
        with pytest.raises(CosmosNotConnectedError):
            mem.push_to_cosmos()

    def test_push_to_cosmos_invalid_batch_size(self):
        mem, _ = _connected_client()
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            mem.push_to_cosmos(batch_size=0)

    def test_push_to_cosmos_embeds_non_turn_memories(self):
        """Non-turn memories must be embedded on push so vector search works."""
        mem, container = _connected_client()
        # Wire a fake embeddings client that returns deterministic vectors.
        embed_calls: list[list[str]] = []

        def _generate_batch(texts: list[str]) -> list[list[float]]:
            embed_calls.append(list(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

        mem._embeddings_client = MagicMock()
        mem._embeddings_client.generate_batch.side_effect = _generate_batch

        mem.add_local(
            user_id="u1",
            role="user",
            content="user prefers dark mode",
            memory_type="fact",
        )
        mem.add_local(user_id="u1", role="user", content="hello", thread_id="t1")  # turn

        mem.push_to_cosmos()

        # Only the fact (non-turn) should have been included in the batch embed call.
        assert embed_calls == [["user prefers dark mode"]]
        fact_bodies = [c.kwargs["body"] for c in container.upsert_item.call_args_list]
        turn_bodies = [c.kwargs["body"] for c in mem._turns_container_client.upsert_item.call_args_list]
        fact_body = next(b for b in fact_bodies if b["type"] == "fact")
        turn_body = next(b for b in turn_bodies if b["type"] == "turn")
        assert fact_body["embedding"] == [0.1, 0.2, 0.3]
        assert "embedding" not in turn_body

    def test_push_to_cosmos_caches_embeddings_in_local_memory(self):
        """Repeat push_to_cosmos() must not re-embed the same non-turn records."""
        mem, container = _connected_client()
        embed_calls: list[list[str]] = []

        def _generate_batch(texts: list[str]) -> list[list[float]]:
            embed_calls.append(list(texts))
            return [[0.5, 0.6, 0.7] for _ in texts]

        mem._embeddings_client = MagicMock()
        mem._embeddings_client.generate_batch.side_effect = _generate_batch

        mem.add_local(user_id="u1", role="user", content="fact one", memory_type="fact")
        mem.push_to_cosmos()
        mem.push_to_cosmos()

        # Second push should not re-embed — embedding is cached on local_memory.
        assert embed_calls == [["fact one"]]
        assert mem.local_memory[0]["embedding"] == [0.5, 0.6, 0.7]


class TestGetMemories:
    def test_no_filters(self):
        mem, container = _connected_client()
        doc = _make_doc()
        container.query_items.return_value = [doc]

        result = mem.get_memories()

        call_kwargs = container.query_items.call_args.kwargs
        assert "WHERE" in call_kwargs["query"]
        assert "superseded_by" in call_kwargs["query"]
        assert result == [doc]

    def test_with_filters(self):
        mem, container = _connected_client()
        doc = _make_doc(type="fact")
        container.query_items.return_value = [doc]

        mem.get_memories(
            memory_id="m1",
            user_id="u1",
            thread_id="t1",
            role="user",
            memory_types=["fact"],
        )

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "WHERE" in query
        params = call_kwargs["parameters"]
        param_names = {p["name"] for p in params}
        assert "@memory_id" in param_names
        assert "@user_id" in param_names
        assert "@thread_id" in param_names
        assert "@role" in param_names
        assert "@memory_type_0" in param_names

    def test_recent_k(self):
        mem, container = _connected_client()
        doc1 = _make_doc(id="old")
        doc2 = _make_doc(id="new")
        container.query_items.return_value = [doc2, doc1]

        result = mem.get_memories(recent_k=2)

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "TOP @recent_k" in query
        assert "ORDER BY c._ts DESC" in query
        # Reversed to chronological
        assert result[0]["id"] == "old"
        assert result[1]["id"] == "new"


class TestGetThread:
    def test_basic(self, sample_memory_dicts):
        mem, _ = _connected_client()
        turns = mem._turns_container_client
        turns.query_items.return_value = list(reversed(sample_memory_dicts))

        result = mem.get_thread(thread_id="t1")

        call_kwargs = turns.query_items.call_args.kwargs
        params = call_kwargs["parameters"]
        assert any(p["name"] == "@thread_id" for p in params)
        assert len(result) == 3

    def test_with_recent_k(self, sample_memory_dicts):
        mem, _ = _connected_client()
        turns = mem._turns_container_client
        turns.query_items.return_value = list(reversed(sample_memory_dicts))

        result = mem.get_thread(thread_id="t1", recent_k=2)
        assert len(result) == 2


class TestUpdateCosmos:
    def test_success(self):
        mem, container = _connected_client()
        doc = _make_doc(id="m1", type="fact")
        container.read_item = MagicMock(return_value=doc.copy())
        container.replace_item = MagicMock()

        mem.update_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact", content="updated")

        container.read_item.assert_called_once_with(item="m1", partition_key=["u1", "t1"])
        container.replace_item.assert_called_once()
        body = container.replace_item.call_args.kwargs["body"]
        assert body["content"] == "updated"
        assert body["type"] == "fact"
        assert "updated_at" in body

    def test_not_found(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, container = _connected_client()
        container.read_item = MagicMock(side_effect=CosmosResourceNotFoundError(message="404"))

        with pytest.raises(MemoryNotFoundError):
            mem.update_cosmos(memory_id="no-such-id", user_id="u1", thread_id="t1", memory_type="fact", content="x")


class TestDeleteCosmos:
    def test_success(self):
        mem, container = _connected_client()
        container.read_item = MagicMock(return_value=_make_doc(id="m1", type="fact"))
        container.delete_item = MagicMock()

        mem.delete_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact")

        container.delete_item.assert_called_once_with(item="m1", partition_key=["u1", "t1"])

    def test_not_found(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, container = _connected_client()
        container.read_item = MagicMock(side_effect=CosmosResourceNotFoundError(message="404"))
        container.delete_item = MagicMock()

        with pytest.raises(MemoryNotFoundError):
            mem.delete_cosmos(memory_id="nope", user_id="u1", thread_id="t1", memory_type="fact")

        container.delete_item.assert_not_called()


class TestGetUserSummary:
    def test_returns_doc_when_present(self):
        mem, _ = _connected_client()
        summaries = mem._summaries_container_client
        doc = _make_doc(type="user_summary", id="user_summary_u1")
        summaries.read_item.return_value = doc

        result = mem.get_user_summary(user_id="u1")

        call_kwargs = summaries.read_item.call_args.kwargs
        assert call_kwargs["item"] == "user_summary_u1"
        assert call_kwargs["partition_key"] == ["u1", "__user_summary__"]
        assert result == doc

    def test_returns_none_when_absent(self):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        mem, _ = _connected_client()
        summaries = mem._summaries_container_client
        summaries.read_item.side_effect = CosmosResourceNotFoundError(message="404")

        result = mem.get_user_summary(user_id="u1")

        assert result is None


# ===================================================================
# Search
# ===================================================================


class TestSearchCosmos:
    def test_search_cosmos(self):
        mem, container = _connected_client()
        container.query_items.return_value = [_make_doc()]

        mem._embeddings_client = MagicMock()
        mem._embeddings_client.generate.return_value = [0.1, 0.2, 0.3]

        result = mem.search_cosmos(search_terms="weather", user_id="u1", top_k=3)

        mem._embeddings_client.generate.assert_called_once_with("weather")
        call_kwargs = container.query_items.call_args.kwargs
        assert "VectorDistance" in call_kwargs["query"]
        assert len(result) == 1

    def test_search_hybrid(self):
        mem, container = _connected_client()
        container.query_items.return_value = [_make_doc()]

        mem._embeddings_client = MagicMock()
        mem._embeddings_client.generate.return_value = [0.1, 0.2]

        mem.search_cosmos(
            search_terms="weather Seattle",
            hybrid_search=True,
            top_k=5,
        )

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "RANK RRF" in query
        assert "FullTextScore" in query

    def test_search_turns(self):
        mem, container = _connected_client()
        turns = mem._turns_container_client
        turns.query_items.return_value = [_make_doc()]

        mem._embeddings_client = MagicMock()
        mem._embeddings_client.generate.return_value = [0.1, 0.2, 0.3]

        result = mem.search_turns(search_terms="weather", user_id="u1", thread_id="t1", top_k=3)

        mem._embeddings_client.generate.assert_called_once_with("weather")
        turns.query_items.assert_called_once()
        container.query_items.assert_not_called()
        assert "VectorDistance" in turns.query_items.call_args.kwargs["query"]
        assert len(result) == 1

    def test_search_not_connected(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            mem.search_cosmos(search_terms="test")

    def test_search_empty_terms(self):
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="search_terms must be a non-empty string"):
            mem.search_cosmos(search_terms="")

    def test_search_whitespace_only_terms(self):
        mem, _ = _connected_client()
        with pytest.raises(ValidationError, match="search_terms must be a non-empty string"):
            mem.search_cosmos(search_terms="   ")


# ===================================================================
# Processing delegation
# ===================================================================


class TestGenerateThreadSummary:
    def test_generate_thread_summary(self):
        mem, container = _connected_client()
        mock_pipeline = MagicMock()
        mock_pipeline.generate_thread_summary.return_value = {"status": "ok"}
        mock_pipeline._store = mem._get_store()
        mock_pipeline._containers = dict(mem._containers)
        mem._pipeline = mock_pipeline

        result = mem.generate_thread_summary(user_id="u1", thread_id="t1")

        mock_pipeline.generate_thread_summary.assert_called_once_with(
            "u1",
            "t1",
            None,
        )
        assert result == {"status": "ok"}


# ===================================================================
# Guard clause
# ===================================================================


class TestCosmosGuard:
    def test_get_memories_without_connect(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            mem.get_memories()

    def test_cosmos_ops_without_connect(self):
        mem = _make_client()
        with pytest.raises(CosmosNotConnectedError):
            mem.get_thread(thread_id="t1")
        with pytest.raises(CosmosNotConnectedError):
            mem.update_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact")
        with pytest.raises(CosmosNotConnectedError):
            mem.delete_cosmos(memory_id="m1", user_id="u1", thread_id="t1", memory_type="fact")


# ===================================================================
# close() and context manager
# ===================================================================


class TestClose:
    def test_close_with_cosmos(self):
        mem, _ = _connected_client()
        mock_cosmos = MagicMock()
        mem._cosmos_client = mock_cosmos

        mem.close()

        mock_cosmos.close.assert_called_once()
        assert mem._cosmos_client is None
        assert mem._memories_container_client is None

    def test_close_without_cosmos(self):
        mem = _make_client()
        mem.close()  # should not raise

    def test_context_manager(self):
        mem, _ = _connected_client()
        mock_cosmos = MagicMock()
        mem._cosmos_client = mock_cosmos

        with mem as m:
            assert m is mem

        mock_cosmos.close.assert_called_once()


def test_list_tags_delegates_to_store():
    mem, container = _connected_client()
    container.query_items.return_value = [["topic:python", "sys:fact"]]

    assert mem.list_tags("u1") == ["topic:python"]

    kwargs = container.query_items.call_args.kwargs
    assert "SELECT VALUE c.tags" in kwargs["query"]
    assert kwargs["parameters"] == [{"name": "@user_id", "value": "u1"}]


class TestSyncCadenceThresholdsForwarding:
    """A ``cadence_thresholds`` mapping on the sync client is forwarded to the auto-trigger.

    This lets callers set per-turn cadence in-process instead of mutating ``os.environ``.
    """

    def test_cadence_thresholds_forwarded(self):
        thresholds = {"FACT_EXTRACTION_EVERY_N": 3, "DEDUP_EVERY_N": 2}
        mem = CosmosMemoryClient(use_default_credential=False, cadence_thresholds=thresholds)
        mem._get_processor = MagicMock(return_value=MagicMock())
        mem._get_counter_container = MagicMock(return_value=MagicMock())

        with patch("azure.cosmos.agent_memory.cosmos_memory_client.maybe_trigger_steps") as mock_trigger:
            mem._maybe_auto_trigger({("u1", "t1"): 1})

        mock_trigger.assert_called_once()
        assert mock_trigger.call_args.kwargs["thresholds"] == thresholds

    def test_defaults_to_none_when_unset(self):
        mem = CosmosMemoryClient(use_default_credential=False)
        mem._get_processor = MagicMock(return_value=MagicMock())
        mem._get_counter_container = MagicMock(return_value=MagicMock())

        with patch("azure.cosmos.agent_memory.cosmos_memory_client.maybe_trigger_steps") as mock_trigger:
            mem._maybe_auto_trigger({("u1", "t1"): 1})

        # None preserves the env-only behavior (the auto-trigger treats None as defaults).
        assert mock_trigger.call_args.kwargs["thresholds"] is None


class TestSyncCadenceThresholdsNormalization:
    """The sync client normalizes ``cadence_thresholds`` at construction time."""

    def test_defensive_copy_isolates_later_mutation(self):
        thresholds = {"FACT_EXTRACTION_EVERY_N": 3}
        mem = CosmosMemoryClient(use_default_credential=False, cadence_thresholds=thresholds)
        thresholds["FACT_EXTRACTION_EVERY_N"] = 99
        assert mem._cadence_thresholds == {"FACT_EXTRACTION_EVERY_N": 3}

    def test_string_values_are_coerced_to_int(self):
        mem = CosmosMemoryClient(use_default_credential=False, cadence_thresholds={"DEDUP_EVERY_N": "5"})
        assert mem._cadence_thresholds == {"DEDUP_EVERY_N": 5}

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError):
            CosmosMemoryClient(use_default_credential=False, cadence_thresholds={"DEDUP_EVERY_N": -1})

    def test_non_int_value_rejected(self):
        with pytest.raises(ValueError):
            CosmosMemoryClient(use_default_credential=False, cadence_thresholds={"DEDUP_EVERY_N": "x"})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError):
            CosmosMemoryClient(use_default_credential=False, cadence_thresholds=[("DEDUP_EVERY_N", 5)])
