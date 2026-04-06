"""Unit tests for AsyncProcessingClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_toolkit.aio.processing import AsyncProcessingClient
from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    OrchestrationTimeoutError,
    ProcessingError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockAsyncContextManager:
    """Simulates an aiohttp response context manager."""

    def __init__(self, return_value):
        self._return_value = return_value

    async def __aenter__(self):
        return self._return_value

    async def __aexit__(self, *args):
        pass


def _mock_response(json_data: dict):
    resp = MagicMock()
    resp.json = AsyncMock(return_value=json_data)
    return resp


@pytest.fixture
def client():
    return AsyncProcessingClient(
        endpoint="https://my-func.azurewebsites.net",
        key="test-key",
        poll_interval=0.01,
        timeout=1.0,
    )


@pytest.fixture
def client_with_session(client):
    """Client with a pre-attached mock aiohttp session."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    client._session = session
    return client, session


# ===================================================================
# invoke_orchestrator() — immediate completion
# ===================================================================


async def test_invoke_immediate_completion(client_with_session):
    client, session = client_with_session

    start_resp = _mock_response({"runtimeStatus": "Completed", "output": "done"})
    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))

    result = await client.invoke_orchestrator({"user_id": "u1"})
    assert result["runtimeStatus"] == "Completed"


# ===================================================================
# invoke_orchestrator() — polls multiple times
# ===================================================================


async def test_invoke_polls_until_completed(client_with_session):
    client, session = client_with_session

    start_resp = _mock_response({
        "id": "inst-1",
        "statusQueryGetUri": "https://status/inst-1",
    })
    poll_running = _mock_response({"runtimeStatus": "Running"})
    poll_completed = _mock_response({"runtimeStatus": "Completed", "output": "ok"})

    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))
    session.get = MagicMock(
        side_effect=[
            MockAsyncContextManager(poll_running),
            MockAsyncContextManager(poll_completed),
        ]
    )

    result = await client.invoke_orchestrator({"data": 1})
    assert result["runtimeStatus"] == "Completed"
    assert session.get.call_count == 2


# ===================================================================
# invoke_orchestrator() — Failed
# ===================================================================


async def test_invoke_failed(client_with_session):
    client, session = client_with_session

    start_resp = _mock_response({
        "id": "inst-2",
        "statusQueryGetUri": "https://status/inst-2",
    })
    poll_failed = _mock_response({
        "runtimeStatus": "Failed",
        "output": "something went wrong",
    })

    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))
    session.get = MagicMock(
        return_value=MockAsyncContextManager(poll_failed),
    )

    with pytest.raises(ProcessingError, match="something went wrong"):
        await client.invoke_orchestrator({"data": 1})


# ===================================================================
# invoke_orchestrator() — timeout
# ===================================================================


async def test_invoke_timeout():
    client = AsyncProcessingClient(
        endpoint="https://my-func.azurewebsites.net",
        key="k",
        poll_interval=0.01,
        timeout=0.05,
    )
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    client._session = session

    start_resp = _mock_response({
        "id": "inst-3",
        "statusQueryGetUri": "https://status/inst-3",
    })
    poll_running = _mock_response({"runtimeStatus": "Running"})

    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))
    session.get = MagicMock(return_value=MockAsyncContextManager(poll_running))

    with pytest.raises(OrchestrationTimeoutError):
        await client.invoke_orchestrator({"data": 1})


# ===================================================================
# invoke_orchestrator() — missing endpoint
# ===================================================================


async def test_invoke_missing_endpoint():
    client = AsyncProcessingClient(endpoint=None, key="k")
    with pytest.raises(ConfigurationError):
        await client.invoke_orchestrator({"data": 1})


# ===================================================================
# generate_thread_summary() — payload correct
# ===================================================================


async def test_generate_thread_summary_payload(client_with_session):
    client, session = client_with_session

    start_resp = _mock_response({"runtimeStatus": "Completed", "output": "summary"})
    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))

    result = await client.generate_thread_summary(
        user_id="u1", thread_id="t1", recent_k=5
    )
    assert result["runtimeStatus"] == "Completed"

    # Check the JSON payload sent to session.post
    call_kwargs = session.post.call_args.kwargs
    payload = call_kwargs["json"]
    assert payload["user_id"] == "u1"
    assert payload["thread_id"] == "t1"
    assert payload["thread_summary_only"] is True
    assert payload["recent_k"] == 5


async def test_generate_thread_summary_no_recent_k(client_with_session):
    client, session = client_with_session
    start_resp = _mock_response({"runtimeStatus": "Completed"})
    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))

    await client.generate_thread_summary(user_id="u1", thread_id="t1")
    payload = session.post.call_args.kwargs["json"]
    assert "recent_k" not in payload


# ===================================================================
# extract_facts()
# ===================================================================


async def test_extract_facts_payload(client_with_session):
    client, session = client_with_session
    start_resp = _mock_response({"runtimeStatus": "Completed"})
    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))

    await client.extract_facts(user_id="u1", thread_id="t1")
    payload = session.post.call_args.kwargs["json"]
    assert payload["extract_facts_only"] is True


# ===================================================================
# close()
# ===================================================================


async def test_close(client_with_session):
    client, session = client_with_session
    await client.close()
    session.close.assert_awaited_once()
    assert client._session is None


async def test_close_noop_when_no_session():
    client = AsyncProcessingClient(endpoint="https://x.azurewebsites.net")
    await client.close()  # should not raise


# ===================================================================
# async context manager
# ===================================================================


async def test_context_manager():
    client = AsyncProcessingClient(
        endpoint="https://func.azurewebsites.net", key="k"
    )
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    client._session = session

    async with client as c:
        assert c is client
    session.close.assert_awaited_once()


# ===================================================================
# URL construction
# ===================================================================


async def test_url_includes_key(client_with_session):
    client, session = client_with_session
    start_resp = _mock_response({"runtimeStatus": "Completed"})
    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))

    await client.invoke_orchestrator({"data": 1})
    url = session.post.call_args.args[0]
    assert "?code=test-key" in url
    assert "/orchestrators/memory_orchestrator" in url


async def test_url_no_key():
    client = AsyncProcessingClient(
        endpoint="https://func.azurewebsites.net",
        key=None,
        poll_interval=0.01,
        timeout=1.0,
    )
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    client._session = session

    start_resp = _mock_response({"runtimeStatus": "Completed"})
    session.post = MagicMock(return_value=MockAsyncContextManager(start_resp))

    await client.invoke_orchestrator({"data": 1})
    url = session.post.call_args.args[0]
    assert "?code=" not in url
