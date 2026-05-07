"""Unit tests for the three Durable Functions orchestrators.

Each orchestrator is a generator function wrapped by
``@bp.orchestration_trigger``. The decorator stores the original generator
on ``handle.orchestrator_function``; we drive that generator directly with a
mocked ``DurableOrchestrationContext`` so no Durable runtime, Cosmos, or
LLM is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from orchestrators import extract_memories as em_mod
from orchestrators import thread_summary as ts_mod
from orchestrators import user_summary as us_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_function(builder):
    """Return the original generator function for an orchestrator.

    Under the unit-test stub of ``azure.durable_functions`` (see
    ``tests/unit/conftest.py``), ``orchestration_trigger`` is a no-op
    decorator so the orchestrator IS the original generator function.
    Under the real SDK it is wrapped in a ``FunctionBuilder``; we unwrap
    it for completeness so the helper works either way.
    """
    if hasattr(builder, "_function"):
        return builder._function.get_user_function().orchestrator_function
    return builder


def _make_context(payload):
    ctx = MagicMock()
    ctx.get_input.return_value = payload

    yielded_calls: list[tuple] = []

    def call_activity_with_retry(name, retry, activity_payload):
        yielded_calls.append((name, retry, activity_payload))
        # Return a sentinel the generator will yield. The test driver intercepts
        # this and feeds the next pre-canned activity result back in.
        return ("__call__", name, activity_payload)

    ctx.call_activity_with_retry.side_effect = call_activity_with_retry
    ctx._yielded_calls = yielded_calls
    return ctx


def _drive(gen, activity_results):
    """Step through an orchestrator generator, feeding pre-canned results.

    Returns the generator's final return value (``StopIteration.value``)
    along with the list of yielded values for assertion convenience.
    """
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


# ---------------------------------------------------------------------------
# Shared env fixture: ensure MAX_BATCH_SIZE has a deterministic value across
# tests (other tests may set it via @patch.dict).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_SIZE", "20")
    yield


# ---------------------------------------------------------------------------
# ThreadSummaryOrchestrator
# ---------------------------------------------------------------------------


class TestThreadSummaryOrchestrator:
    def _orchestrator(self):
        return _user_function(ts_mod.ThreadSummaryOrchestrator)

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock(name="retry"))
    def test_happy_path_calls_two_activities_in_order(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"id": "sum-123"},  # ts_SummarizeThread
                {"id": "sum-123", "persisted": True},  # ts_PersistSummary
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == [
            "ts_SummarizeThread",
            "ts_PersistSummary",
        ]
        assert result == {"persisted": True, "summary_id": "sum-123"}

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_passes_user_and_thread_ids_to_each_activity(self, _retry):
        ctx = _make_context({"user_id": "alice", "thread_id": "T-9"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "s"}, {}])

        for _name, _retry_arg, payload in ctx._yielded_calls:
            assert payload["user_id"] == "alice"
        # Specific shape checks
        summarize_payload = ctx._yielded_calls[0][2]
        persist_payload = ctx._yielded_calls[1][2]

        assert summarize_payload == {"user_id": "alice", "thread_id": "T-9", "limit": 20}
        assert persist_payload == {
            "user_id": "alice",
            "thread_id": "T-9",
            "summary": {"id": "s"},
        }

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_summary_id_returned(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [{"id": "s1"}, {}])

        assert result["summary_id"] == "s1"
        assert result["persisted"] is True

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_non_dict_summary_yields_none_summary_id(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, ["not-a-dict", {}])
        assert result["summary_id"] is None

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        # Yield first; throw exception into the generator at the next yield.
        next(gen)
        with pytest.raises(RuntimeError, match="boom"):
            gen.throw(RuntimeError("boom"))

    def test_missing_user_id_raises(self):
        with patch.object(ts_mod, "default_retry_options", return_value=MagicMock()):
            ctx = _make_context({"thread_id": "t"})
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_uses_max_batch_size_from_env(self, _retry, monkeypatch):
        monkeypatch.setenv("MAX_BATCH_SIZE", "7")
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "s"}, {}])
        assert ctx._yielded_calls[0][2]["limit"] == 7


# ---------------------------------------------------------------------------
# ExtractMemoriesOrchestrator
# ---------------------------------------------------------------------------


class TestExtractMemoriesOrchestrator:
    def _orchestrator(self):
        return _user_function(em_mod.ExtractMemoriesOrchestrator)

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_only_when_reconcile_flag_absent(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {
                    "facts_count": 2,
                    "procedural_count": 1,
                    "episodic_count": 0,
                    "updated_count": 0,
                },
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == ["em_ExtractMemories"]
        assert result["persisted"] is True
        assert result["extracted"]["facts_count"] == 2
        assert result["reconciled"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_chains_reconcile_when_flag_true(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1", "reconcile": True})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"facts_count": 2, "procedural_count": 0, "episodic_count": 0, "updated_count": 0},
                {"kept": 0, "merged": 1, "contradicted": 0},
            ],
        )

        names = [c[0] for c in ctx._yielded_calls]
        assert names == ["em_ExtractMemories", "em_ReconcileMemories"]
        assert ctx._yielded_calls[1][2] == {"user_id": "u1"}
        assert result["reconciled"] == {"kept": 0, "merged": 1, "contradicted": 0}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_payload_carries_user_thread_and_limit(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": 5}])

        extract_payload = ctx._yielded_calls[0][2]
        assert extract_payload == {"user_id": "u", "thread_id": "t", "limit": 20}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        next(gen)
        with pytest.raises(ValueError, match="kaboom"):
            gen.throw(ValueError("kaboom"))

    def test_missing_thread_id_raises(self):
        with patch.object(em_mod, "default_retry_options", return_value=MagicMock()):
            ctx = _make_context({"user_id": "u"})
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)


# ---------------------------------------------------------------------------
# UserSummaryOrchestrator
# ---------------------------------------------------------------------------


class TestUserSummaryOrchestrator:
    def _orchestrator(self):
        return _user_function(us_mod.UserSummaryOrchestrator)

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_happy_path_calls_two_activities_in_order(self, _retry):
        ctx = _make_context({"user_id": "u1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"id": "user-sum-1"},  # us_GenerateUserSummary
                {"id": "user-sum-1", "persisted": True},  # us_PersistUserSummary
            ],
        )
        assert [c[0] for c in ctx._yielded_calls] == [
            "us_GenerateUserSummary",
            "us_PersistUserSummary",
        ]
        assert result == {
            "persisted": True,
            "user_summary_id": "user-sum-1",
        }

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_payloads_only_carry_user_id_limit_and_thread_ids(self, _retry):
        ctx = _make_context({"user_id": "alice"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "us"}, {}])

        gen_payload = ctx._yielded_calls[0][2]
        persist_payload = ctx._yielded_calls[1][2]

        assert gen_payload == {"user_id": "alice", "limit": 20, "thread_ids": None}
        assert persist_payload == {"user_id": "alice", "user_summary": {"id": "us"}}
        for payload in (gen_payload, persist_payload):
            assert "thread_id" not in payload

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_payload_passes_thread_ids_when_provided(self, _retry):
        ctx = _make_context({"user_id": "alice", "thread_ids": ["t1", "t2"]})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "us"}, {}])

        gen_payload = ctx._yielded_calls[0][2]
        assert gen_payload == {
            "user_id": "alice",
            "limit": 20,
            "thread_ids": ["t1", "t2"],
        }

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_user_summary_id_returned(self, _retry):
        ctx = _make_context({"user_id": "u"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [{"id": "us"}, {}])
        assert result["user_summary_id"] == "us"
        assert result["persisted"] is True

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_non_dict_user_summary_yields_none_id(self, _retry):
        ctx = _make_context({"user_id": "u"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [None, {}])
        assert result["user_summary_id"] is None

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"user_id": "u"})
        gen = self._orchestrator()(ctx)
        next(gen)
        with pytest.raises(RuntimeError):
            gen.throw(RuntimeError("activity failed"))

    def test_missing_user_id_raises(self):
        with patch.object(us_mod, "default_retry_options", return_value=MagicMock()):
            ctx = _make_context({})
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)
