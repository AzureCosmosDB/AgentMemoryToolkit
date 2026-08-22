"""Unit tests for the episodic-extraction Durable orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    def test_calls_activity_once_with_user_and_thread_and_returns_result(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)

        result, _ = _drive(gen, [{"episodes": 2}])

        assert [call[0] for call in ctx._yielded_calls] == ["ee_ExtractEpisodes"]
        assert ctx._yielded_calls[0][2] == {"tenant_id": None, "user_id": "u1", "thread_id": "t1"}
        assert result == {"episodes": 2}


@patch.object(ee_mod, "get_pipeline")
def test_activity_calls_pipeline_and_returns_slim_payload(mock_get_pipeline):
    pipeline = MagicMock()
    pipeline.extract_episodes.return_value = {"episodes": 3}
    mock_get_pipeline.return_value = pipeline

    result = ee_mod.ee_ExtractEpisodes({"user_id": "u1", "thread_id": "t1"})

    pipeline.extract_episodes.assert_called_once_with(
        user_id="u1",
        thread_id="t1",
        flush=False,
    )
    assert result == {"episodes": 3}
