"""Unit tests for agent_memory_toolkit.exceptions."""

import pytest

from agent_memory_toolkit.exceptions import (
    AgentMemoryError,
    AuthenticationError,
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    EmbeddingError,
    MemoryNotFoundError,
    OrchestrationTimeoutError,
    ProcessingError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

ALL_SUBTYPES = [
    ConfigurationError,
    ValidationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
    EmbeddingError,
    ProcessingError,
    OrchestrationTimeoutError,
    AuthenticationError,
]


@pytest.mark.parametrize("exc_cls", ALL_SUBTYPES, ids=lambda c: c.__name__)
def test_all_exceptions_inherit_from_agent_memory_error(exc_cls):
    assert issubclass(exc_cls, AgentMemoryError)


def test_catch_all_subtypes():
    """try/except AgentMemoryError catches every subtype."""
    for cls in ALL_SUBTYPES:
        with pytest.raises(AgentMemoryError):
            raise cls("test")


# ---------------------------------------------------------------------------
# ConfigurationError
# ---------------------------------------------------------------------------


def test_configuration_error_parameter_kwarg():
    err = ConfigurationError(parameter="cosmos_endpoint")
    assert err.parameter == "cosmos_endpoint"
    assert "Missing or invalid configuration: cosmos_endpoint" in str(err)


def test_configuration_error_custom_message():
    err = ConfigurationError("custom msg", parameter="p")
    assert str(err) == "custom msg"
    assert err.parameter == "p"


# ---------------------------------------------------------------------------
# MemoryNotFoundError
# ---------------------------------------------------------------------------


def test_memory_not_found_full_context():
    err = MemoryNotFoundError(memory_id="m1", user_id="u1", thread_id="t1")
    assert err.memory_id == "m1"
    assert err.user_id == "u1"
    assert err.thread_id == "t1"
    msg = str(err)
    assert "m1" in msg
    assert "u1" in msg
    assert "t1" in msg


def test_memory_not_found_partial_context():
    err = MemoryNotFoundError(memory_id="m2")
    assert err.memory_id == "m2"
    assert err.user_id is None
    assert err.thread_id is None
    msg = str(err)
    assert "m2" in msg
    assert "user_id" not in msg


# ---------------------------------------------------------------------------
# CosmosNotConnectedError
# ---------------------------------------------------------------------------


def test_cosmos_not_connected_default_message():
    err = CosmosNotConnectedError()
    assert "connect_cosmos()" in str(err)


# ---------------------------------------------------------------------------
# OrchestrationTimeoutError
# ---------------------------------------------------------------------------


def test_orchestration_timeout_with_attrs():
    err = OrchestrationTimeoutError(timeout=30.0, status_url="https://example.com/status")
    assert err.timeout == 30.0
    assert err.status_url == "https://example.com/status"
    msg = str(err)
    assert "30.0" in msg
    assert "https://example.com/status" in msg
