"""Unit tests for the synchronous AgentMemory orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.exceptions import CosmosNotConnectedError
from agent_memory_toolkit.memory import AgentMemory
from agent_memory_toolkit.models import MemoryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**overrides) -> AgentMemory:
    """Build an AgentMemory with credential auto-resolution disabled."""
    defaults = dict(use_default_credential=False)
    defaults.update(overrides)
    return AgentMemory(**defaults)


# ===================================================================
# Constructor
# ===================================================================


class TestConstructor:
    """Tests 1-2: credential resolution."""

    def test_default_credential_created_when_flag_true(self):
        """use_default_credential=True creates a DefaultAzureCredential."""
        sentinel = MagicMock(name="default-cred")
        mock_module = MagicMock()
        mock_module.DefaultAzureCredential.return_value = sentinel

        with patch.dict("sys.modules", {"azure.identity": mock_module}):
            mem = AgentMemory(use_default_credential=True)
            mock_module.DefaultAzureCredential.assert_called_once()
            assert mem._cosmos_credential is sentinel
            assert mem._ai_foundry_credential is sentinel

    def test_no_credential_when_flag_false(self):
        """use_default_credential=False leaves credentials as None."""
        mem = _make_agent()
        assert mem._cosmos_credential is None
        assert mem._ai_foundry_credential is None


# ===================================================================
# Local CRUD
# ===================================================================


class TestAddLocal:
    """Tests 3-4: add_local."""

    def test_add_local_valid(self):
        mem = _make_agent()
        mem.add_local(user_id="u1", role="user", content="hello")

        assert len(mem.local_memory) == 1
        m = mem.local_memory[0]
        assert m["user_id"] == "u1"
        assert m["role"] == "user"
        assert m["content"] == "hello"
        assert m["type"] == "turn"
        assert "id" in m
        assert "created_at" in m

    def test_add_local_invalid_role(self):
        mem = _make_agent()
        with pytest.raises(ValueError, match="role must be one of"):
            mem.add_local(user_id="u1", role="invalid", content="hi")


class TestGetLocal:
    """Tests 5-6: get_local."""

    def test_get_local_no_filters(self):
        mem = _make_agent()
        mem.add_local(user_id="u1", role="user", content="a")
        mem.add_local(user_id="u2", role="agent", content="b")

        results = mem.get_local()
        assert len(results) == 2

    def test_get_local_with_filters(self):
        mem = _make_agent()
        mem.add_local(user_id="u1", role="user", content="a", memory_type="turn")
        mem.add_local(user_id="u1", role="agent", content="b", memory_type="turn")
        mem.add_local(user_id="u2", role="user", content="c", memory_type="summary")

        # AND logic: user_id=u1 AND role=user AND type=turn
        results = mem.get_local(user_id="u1", role="user", memory_type="turn")
        assert len(results) == 1
        assert results[0]["content"] == "a"


class TestUpdateLocal:
    """Tests 7-8: update_local."""

    def test_update_local_success(self):
        mem = _make_agent()
        mem.add_local(user_id="u1", role="user", content="old")
        mid = mem.local_memory[0]["id"]

        mem.update_local(mid, content="new", metadata={"k": "v"})

        m = mem.local_memory[0]
        assert m["content"] == "new"
        assert m["metadata"] == {"k": "v"}
        assert "updated_at" in m

    def test_update_local_not_found(self):
        mem = _make_agent()
        with pytest.raises(KeyError, match="No memory found"):
            mem.update_local("nonexistent-id", content="x")


class TestDeleteLocal:
    """Tests 9-10: delete_local."""

    def test_delete_local_success(self):
        mem = _make_agent()
        mem.add_local(user_id="u1", role="user", content="x")
        mid = mem.local_memory[0]["id"]

        mem.delete_local(mid)
        assert len(mem.local_memory) == 0

    def test_delete_local_not_found(self):
        mem = _make_agent()
        with pytest.raises(KeyError, match="No memory found"):
            mem.delete_local("nonexistent-id")


# ===================================================================
# Cosmos delegation
# ===================================================================


class TestConnectCosmos:
    """Test 11: connect_cosmos."""

    @patch("agent_memory_toolkit.memory.CosmosMemoryStore", autospec=True)
    def test_connect_cosmos(self, mock_store_cls):
        mem = _make_agent(cosmos_endpoint="https://test.documents.azure.com:443/")
        mock_instance = mock_store_cls.return_value

        mem.connect_cosmos()

        mock_store_cls.assert_called_once_with(
            endpoint="https://test.documents.azure.com:443/",
            credential=None,
            database="ai_memory",
            container="memories",
        )
        mock_instance.connect.assert_called_once()
        assert mem._cosmos_store is mock_instance


class TestAddCosmos:
    """Test 12: add_cosmos."""

    def test_add_cosmos(self):
        mem = _make_agent()
        mock_store = MagicMock()
        mem._cosmos_store = mock_store

        mem.add_cosmos(user_id="u1", role="user", content="hello")

        mock_store.upsert.assert_called_once()
        record = mock_store.upsert.call_args[0][0]
        assert isinstance(record, MemoryRecord)
        assert record.user_id == "u1"
        assert record.role == "user"
        assert record.content == "hello"


class TestPushToCosmos:
    """Test 13: push_to_cosmos."""

    def test_push_to_cosmos(self):
        mem = _make_agent()
        mock_store = MagicMock()
        mem._cosmos_store = mock_store

        mem.add_local(user_id="u1", role="user", content="a")
        mem.add_local(user_id="u1", role="agent", content="b")

        mem.push_to_cosmos()

        mock_store.upsert_batch.assert_called_once()
        records = mock_store.upsert_batch.call_args[0][0]
        assert len(records) == 2
        assert all(isinstance(r, MemoryRecord) for r in records)


class TestGetMemories:
    """Test 14: get_memories."""

    def test_get_memories(self):
        mem = _make_agent()
        mock_store = MagicMock()
        mock_store.get_memories.return_value = [{"id": "1", "content": "hi"}]
        mem._cosmos_store = mock_store

        result = mem.get_memories(user_id="u1", role="user")

        mock_store.get_memories.assert_called_once_with(
            memory_id=None,
            user_id="u1",
            thread_id=None,
            role="user",
            memory_type=None,
            recent_k=None,
        )
        assert result == [{"id": "1", "content": "hi"}]


class TestSearchCosmos:
    """Test 15: search_cosmos."""

    def test_search_cosmos(self):
        mem = _make_agent()
        mock_store = MagicMock()
        mock_store.vector_search.return_value = [{"id": "1", "score": 0.95}]
        mem._cosmos_store = mock_store

        mock_embed = MagicMock()
        mock_embed.generate.return_value = [0.1, 0.2, 0.3]
        mem._embeddings_client = mock_embed

        result = mem.search_cosmos(search_terms="weather", user_id="u1", top_k=3)

        mock_embed.generate.assert_called_once_with("weather")
        mock_store.vector_search.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3],
            user_id="u1",
            role=None,
            memory_type=None,
            thread_id=None,
            hybrid_search=False,
            search_terms="weather",
            top_k=3,
        )
        assert result == [{"id": "1", "score": 0.95}]


# ===================================================================
# Processing delegation
# ===================================================================


class TestGenerateThreadSummary:
    """Test 16: generate_thread_summary."""

    def test_generate_thread_summary(self):
        mem = _make_agent()
        mock_proc = MagicMock()
        mock_proc.generate_thread_summary.return_value = {"status": "ok"}
        mem._processing_client = mock_proc

        result = mem.generate_thread_summary(user_id="u1", thread_id="t1")

        mock_proc.generate_thread_summary.assert_called_once_with(
            user_id="u1",
            thread_id="t1",
            recent_k=None,
            poll_interval=2.0,
            timeout=120.0,
        )
        assert result == {"status": "ok"}


# ===================================================================
# Guard clause
# ===================================================================


class TestCosmosGuard:
    """Test 17: Cosmos op without connect raises CosmosNotConnectedError."""

    def test_get_memories_without_connect(self):
        mem = _make_agent()
        with pytest.raises(CosmosNotConnectedError):
            mem.get_memories()
