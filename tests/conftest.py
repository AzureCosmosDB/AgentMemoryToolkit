"""Shared test fixtures for Agent Memory Toolkit tests."""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_user_id():
    return "test-user-001"


@pytest.fixture
def sample_thread_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_memory_dict(sample_user_id, sample_thread_id):
    """A raw memory dict as returned by _make_memory or Cosmos queries."""
    return {
        "id": str(uuid.uuid4()),
        "user_id": sample_user_id,
        "thread_id": sample_thread_id,
        "role": "user",
        "type": "turn",
        "content": "What is the weather in Seattle?",
        "metadata": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_memory_dicts(sample_user_id, sample_thread_id):
    """A list of memory dicts simulating a conversation thread."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": str(uuid.uuid4()),
            "user_id": sample_user_id,
            "thread_id": sample_thread_id,
            "role": "user",
            "type": "turn",
            "content": "What is the weather in Seattle?",
            "metadata": {},
            "created_at": now.isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": sample_user_id,
            "thread_id": sample_thread_id,
            "role": "agent",
            "type": "turn",
            "content": "The current weather in Seattle is 55°F and cloudy.",
            "metadata": {},
            "created_at": now.isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": sample_user_id,
            "thread_id": sample_thread_id,
            "role": "user",
            "type": "turn",
            "content": "Can you book me a hotel near Pike Place Market?",
            "metadata": {},
            "created_at": now.isoformat(),
        },
    ]


@pytest.fixture
def sample_embedding():
    """A fake embedding vector (10 dimensions for speed)."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.fixture
def mock_credential():
    """A mock Azure TokenCredential."""
    cred = MagicMock()
    cred.get_token.return_value = MagicMock(token="fake-token", expires_on=9999999999)
    return cred


# ---------------------------------------------------------------------------
# Integration-test fixtures
# ---------------------------------------------------------------------------

INTEGRATION_ENABLED = os.environ.get("AGENT_MEMORY_RUN_INTEGRATION", "").lower() == "true"


@pytest.fixture(scope="session")
def cosmos_endpoint():
    """Cosmos DB endpoint from env vars (integration tests)."""
    return os.environ.get("COSMOS_DB_ENDPOINT", "")


@pytest.fixture(scope="session")
def cosmos_database():
    """Cosmos DB database name for integration tests."""
    return os.environ.get("COSMOS_DB_DATABASE", "ai_memory_integration_test")


@pytest.fixture(scope="session")
def cosmos_container():
    """Cosmos DB container name for integration tests."""
    return os.environ.get("COSMOS_DB_MEMORIES_CONTAINER", "memories_integration_test")


@pytest.fixture(scope="session")
def cosmos_key():
    """Cosmos DB account key (used as a fallback when control-plane RBAC is not available)."""
    return os.environ.get("COSMOS_DB_KEY", "")


@pytest.fixture(scope="session")
def ai_foundry_endpoint():
    """Azure OpenAI endpoint from env vars."""
    return os.environ.get("AI_FOUNDRY_ENDPOINT", "")


@pytest.fixture(scope="session")
def ai_foundry_api_key():
    """Azure OpenAI API key from env vars (optional — Entra ID is preferred)."""
    return os.environ.get("AI_FOUNDRY_API_KEY", "")


@pytest.fixture(scope="session")
def embedding_deployment_name():
    """Embedding model deployment name."""
    return os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")


@pytest.fixture(scope="session")
def embedding_dimensions():
    """Embedding dimensions from env vars."""
    raw = os.environ.get("AI_FOUNDRY_EMBEDDING_DIMENSIONS", "1536")
    return int(raw) if raw else 1536


@pytest.fixture(scope="session")
def chat_deployment_name():
    """LLM deployment name used by the processing pipeline."""
    return os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")


@pytest.fixture
def unique_user_id():
    """Unique user ID to isolate integration test data."""
    return f"integ-test-user-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_thread_id():
    """Unique thread ID to isolate integration test data."""
    return str(uuid.uuid4())
