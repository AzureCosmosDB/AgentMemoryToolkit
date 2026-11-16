"""Unit tests for the procedural-synthesis Durable orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from orchestrators import synthesize_procedural as sp_mod


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


class TestSynthesizeProceduralOrchestrator:
    def _orchestrator(self):
        return _user_function(sp_mod.SynthesizeProceduralOrchestrator)

    @patch.object(sp_mod, "default_retry_options", return_value=MagicMock(name="retry"))
    def test_calls_activity_once_with_user_and_force_and_returns_result(self, _retry):
        ctx = _make_context({"user_id": "u1", "force": True})
        gen = self._orchestrator()(ctx)

        result, _ = _drive(gen, [{"status": "synthesized", "procedures_created": 2}])

        assert [call[0] for call in ctx._yielded_calls] == ["sp_SynthesizeProcedural"]
        assert ctx._yielded_calls[0][2] == {"user_id": "u1", "force": True}
        assert result == {"status": "synthesized", "procedures_created": 2}


@pytest.mark.parametrize(
    ("payload", "pipeline_result", "expected"),
    [
        (
            {"user_id": "u1", "force": True},
            {"status": "synthesized", "procedures_created": 2, "procedures_updated": 1, "facts_retired": 1},
            {"status": "synthesized", "procedures_created": 2, "facts_retired": 1},
        ),
        (
            {"user_id": "u2", "force": False},
            {"status": "skipped", "procedures_created": None},
            {"status": "skipped", "procedures_created": 0, "facts_retired": 0},
        ),
    ],
    ids=["extra-fields-dropped", "missing-count-is-zero"],
)
@patch.object(sp_mod, "get_pipeline")
def test_activity_calls_pipeline_and_returns_slim_payload(mock_get_pipeline, payload, pipeline_result, expected):
    """Durable records activity outputs in the job history, so only the status and counts may come back."""
    pipeline = MagicMock()
    pipeline.synthesize_procedural.return_value = pipeline_result
    mock_get_pipeline.return_value = pipeline

    with patch.object(sp_mod.time, "monotonic", return_value=100.0):
        result = sp_mod.sp_SynthesizeProcedural(payload)

    # The pass stops starting model calls at the activity's time budget (240 s by default).
    pipeline.synthesize_procedural.assert_called_once_with(
        user_id=payload["user_id"], force=payload["force"], deadline=340.0
    )
    assert result == expected
    assert type(result["procedures_created"]) is int


@patch.object(sp_mod, "get_pipeline")
def test_activity_deadline_follows_the_configured_time_budget(mock_get_pipeline, monkeypatch):
    monkeypatch.setenv("ACTIVITY_TIME_BUDGET_SECONDS", "90")
    pipeline = MagicMock()
    pipeline.synthesize_procedural.return_value = {"status": "unchanged", "procedures_created": 0}
    mock_get_pipeline.return_value = pipeline

    with patch.object(sp_mod.time, "monotonic", return_value=100.0):
        sp_mod.sp_SynthesizeProcedural({"user_id": "u1"})

    assert pipeline.synthesize_procedural.call_args.kwargs["deadline"] == 190.0
