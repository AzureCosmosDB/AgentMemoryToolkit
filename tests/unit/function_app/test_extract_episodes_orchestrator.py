"""Unit tests for the episodic-extraction Durable orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from orchestrators import extract_episodes as ee_mod


def _user_function(builder):
    if hasattr(builder, "_function"):
        return builder._function.get_user_function().orchestrator_function
    return builder


def _make_context(payload):
    ctx = MagicMock()
    ctx.get_input.return_value = payload

    yielded_calls: list[tuple] = []

    def call_activity_with_retry(name, retry, activity_payload):
        yielded_calls.append((name, retry, activity_payload))
        return ("__call__", name, activity_payload)

    ctx.call_activity_with_retry.side_effect = call_activity_with_retry
    ctx._yielded_calls = yielded_calls
    return ctx


def _drive(gen, activity_results):
    yields = []
    iterator = iter(activity_results)
    try:
        sent = None
        while True:
            value = gen.send(sent)
            yields.append(value)
            sent = next(iterator)
    except StopIteration as stop:
        return stop.value, yields


class TestExtractEpisodesOrchestrator:
    def _orchestrator(self):
        return _user_function(ee_mod.ExtractEpisodesOrchestrator)

    @patch.object(ee_mod, "default_retry_options", return_value=MagicMock(name="retry"))
    def test_calls_activity_once_with_tenant_user_and_thread_and_returns_result(self, _retry):
        ctx = _make_context({"tenant_id": "acme", "user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)

        result, _ = _drive(gen, [{"episodes": 2}])

        assert [call[0] for call in ctx._yielded_calls] == ["ee_ExtractEpisodes"]
        assert ctx._yielded_calls[0][2] == {"tenant_id": "acme", "user_id": "u1", "thread_id": "t1"}
        assert result == {"episodes": 2}

    @pytest.mark.parametrize("missing", ["tenant_id", "user_id", "thread_id"])
    def test_missing_tenant_user_or_thread_id_raises(self, missing):
        payload = {"tenant_id": "acme", "user_id": "u1", "thread_id": "t1"}
        del payload[missing]
        with patch.object(ee_mod, "default_retry_options", return_value=MagicMock()):
            with pytest.raises(KeyError):
                next(self._orchestrator()(_make_context(payload)))


@patch.object(ee_mod, "get_pipeline")
def test_activity_calls_the_tenants_pipeline_and_returns_slim_payload(mock_get_pipeline):
    pipeline = MagicMock()
    pipeline.extract_episodes.return_value = {"episodes": 3}
    mock_get_pipeline.return_value = pipeline

    result = ee_mod.ee_ExtractEpisodes({"tenant_id": "acme", "user_id": "u1", "thread_id": "t1"})

    mock_get_pipeline.assert_called_once_with("acme")
    pipeline.extract_episodes.assert_called_once_with(
        user_id="u1",
        thread_id="t1",
        flush=False,
    )
    assert result == {"episodes": 3}
