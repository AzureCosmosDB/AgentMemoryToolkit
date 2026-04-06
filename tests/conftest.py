"""Shared test fixtures for Agent Memory Toolkit tests."""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

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
