"""Unit tests for CosmosMemoryStore (sync Cosmos DB client)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.cosmos_memory_client import CosmosMemoryStore
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

def _make_store(
    endpoint: str | None = "https://fake.documents.azure.com",
    credential: str | None = "fake-key",
    database: str = "ai_memory",
    container: str = "memories",
) -> CosmosMemoryStore:
    return CosmosMemoryStore(
        endpoint=endpoint,
        credential=credential,
        database=database,
        container=container,
    )


def _connected_store() -> tuple[CosmosMemoryStore, MagicMock]:
    """Return a store with a mocked container client already wired up."""
    store = _make_store()
    container = MagicMock()
    store._container_client = container
    return store, container


def _make_record(**overrides) -> MemoryRecord:
    defaults = dict(
        id=str(uuid.uuid4()),
        user_id="u1",
        thread_id="t1",
        role="user",
        content="hello",
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    @patch("azure.cosmos.CosmosClient")
    def test_connect_success(self, MockCosmosClient):
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_container = MagicMock()
        MockCosmosClient.return_value = mock_client
        mock_client.get_database_client.return_value = mock_db
        mock_db.get_container_client.return_value = mock_container

        store = _make_store()
        store.connect()

        MockCosmosClient.assert_called_once_with(
            "https://fake.documents.azure.com", credential="fake-key"
        )
        mock_client.get_database_client.assert_called_once_with("ai_memory")
        mock_db.get_container_client.assert_called_once_with("memories")
        assert store._container_client is mock_container

    def test_connect_missing_endpoint(self):
        store = _make_store(endpoint=None)
        with pytest.raises(ConfigurationError) as exc_info:
            store.connect()
        assert exc_info.value.parameter == "endpoint"

    def test_connect_missing_credential(self):
        store = _make_store(credential=None)
        with pytest.raises(ConfigurationError) as exc_info:
            store.connect()
        assert exc_info.value.parameter == "credential"


# ---------------------------------------------------------------------------
# _require_connected()
# ---------------------------------------------------------------------------


class TestRequireConnected:
    def test_raises_when_not_connected(self):
        store = _make_store()
        with pytest.raises(CosmosNotConnectedError):
            store._require_connected()


# ---------------------------------------------------------------------------
# create_store()
# ---------------------------------------------------------------------------


class TestCreateStore:
    @patch("azure.cosmos.CosmosClient")
    def test_create_store_success(self, MockCosmosClient):
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_container = MagicMock()
        MockCosmosClient.return_value = mock_client
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = mock_container

        store = _make_store()
        store.create_store(embedding_dimensions=256)

        mock_client.create_database_if_not_exists.assert_called_once_with(
            id="ai_memory"
        )
        call_kwargs = mock_db.create_container_if_not_exists.call_args
        assert call_kwargs.kwargs["id"] == "memories"

        # Verify vector embedding policy includes correct dimensions
        vec_policy = call_kwargs.kwargs["vector_embedding_policy"]
        assert vec_policy["vectorEmbeddings"][0]["dimensions"] == 256

        # Verify full-text policy
        ft_policy = call_kwargs.kwargs["full_text_policy"]
        assert ft_policy["defaultLanguage"] == "en-US"

        assert store._container_client is mock_container


# ---------------------------------------------------------------------------
# upsert / upsert_batch
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_success(self):
        store, container = _connected_store()
        rec = _make_record()
        store.upsert(rec)
        container.upsert_item.assert_called_once()
        body = container.upsert_item.call_args.kwargs["body"]
        assert body["id"] == rec.id
        assert body["content"] == "hello"

    def test_upsert_not_connected(self):
        store = _make_store()
        with pytest.raises(CosmosNotConnectedError):
            store.upsert(_make_record())

    def test_upsert_cosmos_error(self):
        store, container = _connected_store()
        container.upsert_item.side_effect = RuntimeError("boom")
        with pytest.raises(CosmosOperationError):
            store.upsert(_make_record())

    def test_upsert_batch(self):
        store, container = _connected_store()
        records = [_make_record(id=f"r{i}") for i in range(3)]
        store.upsert_batch(records)
        assert container.upsert_item.call_count == 3


# ---------------------------------------------------------------------------
# get_memories()
# ---------------------------------------------------------------------------


class TestGetMemories:
    def test_no_filters(self, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict]

        result = store.get_memories()

        call_kwargs = container.query_items.call_args.kwargs
        assert "WHERE" not in call_kwargs["query"]
        assert result == [sample_memory_dict]

    def test_all_filters(self, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict]

        store.get_memories(
            memory_id="m1",
            user_id="u1",
            thread_id="t1",
            role="user",
            memory_type="turn",
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
        assert "@memory_type" in param_names

    def test_recent_k(self, sample_memory_dict):
        store, container = _connected_store()
        # Simulate docs returned in DESC order
        doc1 = {**sample_memory_dict, "id": "old"}
        doc2 = {**sample_memory_dict, "id": "new"}
        container.query_items.return_value = [doc2, doc1]

        result = store.get_memories(recent_k=2)

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "TOP @recent_k" in query
        assert "ORDER BY c._ts DESC" in query
        # Result should be reversed to chronological
        assert result[0]["id"] == "old"
        assert result[1]["id"] == "new"


# ---------------------------------------------------------------------------
# get_thread()
# ---------------------------------------------------------------------------


class TestGetThread:
    def test_basic(self, sample_memory_dicts):
        store, container = _connected_store()
        container.query_items.return_value = list(reversed(sample_memory_dicts))

        result = store.get_thread(thread_id="t1")

        call_kwargs = container.query_items.call_args.kwargs
        assert "@thread_id" in call_kwargs["query"] or any(
            p["name"] == "@thread_id" for p in call_kwargs["parameters"]
        )
        # Should be reversed to chronological
        assert len(result) == 3

    def test_with_recent_k(self, sample_memory_dicts):
        store, container = _connected_store()
        # Return 3 docs (DESC order), get_thread truncates + reverses
        container.query_items.return_value = list(reversed(sample_memory_dicts))

        result = store.get_thread(thread_id="t1", recent_k=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_success(self, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict.copy()]

        store.update(memory_id=sample_memory_dict["id"], content="updated")

        container.replace_item.assert_called_once()
        body = container.replace_item.call_args.kwargs["body"]
        assert body["content"] == "updated"
        assert "updated_at" in body

    def test_not_found(self):
        store, container = _connected_store()
        container.query_items.return_value = []

        with pytest.raises(MemoryNotFoundError):
            store.update(memory_id="no-such-id", content="x")


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    def test_success(self, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict]

        store.delete(
            memory_id=sample_memory_dict["id"],
            user_id=sample_memory_dict["user_id"],
            thread_id=sample_memory_dict["thread_id"],
        )

        container.delete_item.assert_called_once_with(
            item=sample_memory_dict["id"],
            partition_key=[
                sample_memory_dict["user_id"],
                sample_memory_dict["thread_id"],
            ],
        )

    def test_not_found(self):
        store, container = _connected_store()
        container.query_items.return_value = []

        with pytest.raises(MemoryNotFoundError):
            store.delete(memory_id="nope", user_id="u1", thread_id="t1")


# ---------------------------------------------------------------------------
# vector_search()
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_vector_only(self, sample_embedding, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict]

        result = store.vector_search(query_vector=sample_embedding, top_k=3)

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "VectorDistance" in query
        assert "RRF" not in query
        assert result == [sample_memory_dict]

    def test_hybrid(self, sample_embedding, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict]

        store.vector_search(
            query_vector=sample_embedding,
            hybrid_search=True,
            search_terms="weather Seattle",
            top_k=5,
        )

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "RANK RRF" in query
        assert "FullTextScore" in query
        params = call_kwargs["parameters"]
        key_terms_param = [p for p in params if p["name"] == "@key_terms"]
        assert key_terms_param[0]["value"] == "weather Seattle"

    def test_with_filters(self, sample_embedding, sample_memory_dict):
        store, container = _connected_store()
        container.query_items.return_value = [sample_memory_dict]

        store.vector_search(
            query_vector=sample_embedding,
            user_id="u1",
            role="user",
        )

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "WHERE" in query


# ---------------------------------------------------------------------------
# get_user_summary()
# ---------------------------------------------------------------------------


class TestGetUserSummary:
    def test_filters_by_type(self, sample_memory_dict):
        store, container = _connected_store()
        summary_doc = {**sample_memory_dict, "type": "user_summary"}
        container.query_items.return_value = [summary_doc]

        result = store.get_user_summary(user_id="u1")

        call_kwargs = container.query_items.call_args.kwargs
        query = call_kwargs["query"]
        assert "user_summary" in query
        assert result == [summary_doc]
