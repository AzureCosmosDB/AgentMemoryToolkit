"""Unit tests for ProcessingClient (sync Durable Functions client)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    OrchestrationTimeoutError,
    ProcessingError,
)
from agent_memory_toolkit.processing import ProcessingClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_ENDPOINT = "https://myfunc.azurewebsites.net"


def _make_client(**overrides) -> ProcessingClient:
    defaults = dict(endpoint=FAKE_ENDPOINT, key=None, poll_interval=0.05, timeout=5.0)
    defaults.update(overrides)
    return ProcessingClient(**defaults)


def _urlopen_response(body: dict) -> MagicMock:
    """Create a context-manager mock that returns *body* as JSON bytes."""
    data = json.dumps(body).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = data
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# invoke_orchestrator()
# ---------------------------------------------------------------------------


class TestInvokeOrchestrator:
    @patch("urllib.request.urlopen")
    def test_immediate_completion(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "inst-1", "statusQueryGetUri": "https://status/1"}
        )
        poll_resp = _urlopen_response(
            {"runtimeStatus": "Completed", "output": {"ok": True}}
        )
        mock_urlopen.side_effect = [start_resp, poll_resp]

        client = _make_client()
        result = client.invoke_orchestrator({"key": "value"})

        assert result["runtimeStatus"] == "Completed"
        assert result["output"] == {"ok": True}
        assert mock_urlopen.call_count == 2

    @patch("urllib.request.urlopen")
    def test_polls_multiple_times(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "inst-2", "statusQueryGetUri": "https://status/2"}
        )
        running_resp_1 = _urlopen_response({"runtimeStatus": "Running"})
        running_resp_2 = _urlopen_response({"runtimeStatus": "Running"})
        done_resp = _urlopen_response(
            {"runtimeStatus": "Completed", "output": "done"}
        )
        mock_urlopen.side_effect = [
            start_resp,
            running_resp_1,
            running_resp_2,
            done_resp,
        ]

        client = _make_client()
        result = client.invoke_orchestrator({"x": 1})

        assert result["runtimeStatus"] == "Completed"
        assert mock_urlopen.call_count == 4

    @patch("urllib.request.urlopen")
    def test_failed_status(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "inst-3", "statusQueryGetUri": "https://status/3"}
        )
        fail_resp = _urlopen_response(
            {"runtimeStatus": "Failed", "output": "something broke"}
        )
        mock_urlopen.side_effect = [start_resp, fail_resp]

        client = _make_client()
        with pytest.raises(ProcessingError, match="something broke"):
            client.invoke_orchestrator({"y": 2})

    @patch("urllib.request.urlopen")
    def test_timeout(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "inst-4", "statusQueryGetUri": "https://status/4"}
        )
        # Always return Running to force timeout
        running_resp = _urlopen_response({"runtimeStatus": "Running"})
        mock_urlopen.side_effect = [start_resp] + [running_resp] * 100

        client = _make_client(timeout=0.1, poll_interval=0.05)
        with pytest.raises(OrchestrationTimeoutError):
            client.invoke_orchestrator({"z": 3})

    def test_missing_endpoint(self):
        client = ProcessingClient(endpoint=None)
        with pytest.raises(ConfigurationError) as exc_info:
            client.invoke_orchestrator({"a": 1})
        assert exc_info.value.parameter == "endpoint"

    @patch("urllib.request.urlopen")
    def test_function_key_appended(self, mock_urlopen):
        start_resp = _urlopen_response({"runtimeStatus": "Completed"})
        mock_urlopen.side_effect = [start_resp]

        client = _make_client(key="my-secret-key")
        client.invoke_orchestrator({"b": 2})

        # Inspect the URL used in the POST request
        req_obj = mock_urlopen.call_args_list[0][0][0]
        assert "?code=my-secret-key" in req_obj.full_url

    @patch("urllib.request.urlopen")
    def test_no_status_uri_returns_start_response(self, mock_urlopen):
        # When no statusQueryGetUri is present, return the start response directly
        start_resp = _urlopen_response(
            {"runtimeStatus": "Completed", "output": "immediate"}
        )
        mock_urlopen.side_effect = [start_resp]

        client = _make_client()
        result = client.invoke_orchestrator({"c": 3})

        assert result["output"] == "immediate"
        assert mock_urlopen.call_count == 1


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


class TestGenerateThreadSummary:
    @patch("urllib.request.urlopen")
    def test_payload_has_thread_summary_flag(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "ts-1", "statusQueryGetUri": "https://status/ts-1"}
        )
        done_resp = _urlopen_response(
            {"runtimeStatus": "Completed", "output": "summary"}
        )
        mock_urlopen.side_effect = [start_resp, done_resp]

        client = _make_client()
        result = client.generate_thread_summary(user_id="u1", thread_id="t1")

        # Verify the POST body
        req_obj = mock_urlopen.call_args_list[0][0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["thread_summary_only"] is True
        assert body["user_id"] == "u1"
        assert body["thread_id"] == "t1"
        assert result["runtimeStatus"] == "Completed"


class TestExtractFacts:
    @patch("urllib.request.urlopen")
    def test_payload_has_extract_facts_flag(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "ef-1", "statusQueryGetUri": "https://status/ef-1"}
        )
        done_resp = _urlopen_response(
            {"runtimeStatus": "Completed", "output": "facts"}
        )
        mock_urlopen.side_effect = [start_resp, done_resp]

        client = _make_client()
        result = client.extract_facts(user_id="u1", thread_id="t1")

        req_obj = mock_urlopen.call_args_list[0][0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["extract_facts_only"] is True
        assert result["runtimeStatus"] == "Completed"


class TestGenerateUserSummary:
    @patch("urllib.request.urlopen")
    def test_payload_has_user_summary_flag_and_thread_ids(self, mock_urlopen):
        start_resp = _urlopen_response(
            {"id": "us-1", "statusQueryGetUri": "https://status/us-1"}
        )
        done_resp = _urlopen_response(
            {"runtimeStatus": "Completed", "output": "user_summary"}
        )
        mock_urlopen.side_effect = [start_resp, done_resp]

        client = _make_client()
        result = client.generate_user_summary(
            user_id="u1", thread_ids=["t1", "t2"]
        )

        req_obj = mock_urlopen.call_args_list[0][0][0]
        body = json.loads(req_obj.data.decode("utf-8"))
        assert body["user_summary_only"] is True
        assert body["thread_ids"] == ["t1", "t2"]
        assert result["runtimeStatus"] == "Completed"
