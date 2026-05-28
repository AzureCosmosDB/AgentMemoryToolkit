"""Unit tests for agent_memory_toolkit.exceptions."""

import pytest

from agent_memory_toolkit.exceptions import (
    AgentMemoryError,
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    LLMError,
    MemoryNotFoundError,
    ValidationError,
)

ALL_SUBTYPES = [
    ConfigurationError,
    ValidationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
    LLMError,
]


@pytest.mark.parametrize("exc_cls", ALL_SUBTYPES, ids=lambda c: c.__name__)
def test_all_exceptions_inherit_from_agent_memory_error(exc_cls):
    assert issubclass(exc_cls, AgentMemoryError)


def test_catch_all_subtypes():
    """try/except AgentMemoryError catches every subtype."""
    for cls in ALL_SUBTYPES:
        with pytest.raises(AgentMemoryError):
            raise cls("test")


def test_configuration_error_parameter_kwarg():
    err = ConfigurationError(parameter="cosmos_endpoint")
    assert err.parameter == "cosmos_endpoint"
    assert "Missing or invalid configuration: cosmos_endpoint" in str(err)


def test_configuration_error_custom_message():
    err = ConfigurationError("custom msg", parameter="p")
    assert str(err) == "custom msg"
    assert err.parameter == "p"


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


def test_cosmos_not_connected_default_message():
    err = CosmosNotConnectedError()
    assert "connect_cosmos()" in str(err)


def test_llm_error():
    err = LLMError("test error")
    assert str(err) == "test error"


def test_llm_error_inherits():
    err = LLMError("boom")
    assert isinstance(err, AgentMemoryError)


def test_llm_error_caught_by_base():
    with pytest.raises(AgentMemoryError):
        raise LLMError("oops")
