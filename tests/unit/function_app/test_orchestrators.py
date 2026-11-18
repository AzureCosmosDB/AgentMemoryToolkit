"""Unit tests for the three Durable Functions orchestrators.

Each orchestrator is a generator function wrapped by
``@bp.orchestration_trigger``. The decorator stores the original generator
on ``handle.orchestrator_function``; we drive that generator directly with a
mocked ``DurableOrchestrationContext`` so no Durable runtime, Cosmos, or
LLM is required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from orchestrators import extract_episodes as ee_mod
from orchestrators import extract_memories as em_mod
from orchestrators import synthesize_procedural as sp_mod
from orchestrators import thread_summary as ts_mod
from orchestrators import user_summary as us_mod
from shared.instance_ids import orchestration_instance_id

from azure.cosmos.agent_memory.exceptions import NoSourceMemoriesError

TENANT = "acme"

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
    yielded_sub_orchestrators: list[tuple] = []
    sub_instance_ids: list = []

    def call_activity_with_retry(name, retry, activity_payload):
        yielded_calls.append((name, retry, activity_payload))
        # Return a sentinel the generator will yield. The test driver intercepts
        # this and feeds the next pre-canned activity result back in.
        return ("__call__", name, activity_payload)

    def call_sub_orchestrator_with_retry(name, retry, sub_payload, *_args, **kwargs):
        yielded_sub_orchestrators.append((name, retry, sub_payload))
        sub_instance_ids.append(kwargs.get("instance_id"))
        return ("__sub__", name, sub_payload)

    ctx.call_activity_with_retry.side_effect = call_activity_with_retry
    ctx.call_sub_orchestrator_with_retry.side_effect = call_sub_orchestrator_with_retry
    ctx._yielded_calls = yielded_calls
    ctx._yielded_sub_orchestrators = yielded_sub_orchestrators
    ctx._sub_instance_ids = sub_instance_ids
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


class _TimerContext:
    """DurableOrchestrationContext double for the user-summary wait/poll loop.

    Unlike the ``MagicMock``-based :func:`_make_context`, this exposes a real
    ``current_utc_datetime`` that advances by ``step_seconds`` on every read
    (so the wait budget can elapse deterministically) and records
    ``create_timer`` calls. Activity calls yield the same ``("__call__", ...)``
    sentinel that :func:`_drive` feeds pre-canned results into.
    """

    def __init__(self, payload, *, start=None, step_seconds=0):
        self._payload = payload
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._step = timedelta(seconds=step_seconds)
        self.activity_calls: list[tuple] = []
        self.timer_fire_times: list = []

    def get_input(self):
        return self._payload

    @property
    def current_utc_datetime(self):
        now = self._now
        self._now = now + self._step
        return now

    def call_activity_with_retry(self, name, retry, activity_payload):
        self.activity_calls.append((name, activity_payload))
        return ("__call__", name, activity_payload)

    def create_timer(self, fire_at):
        self.timer_fire_times.append(fire_at)
        return ("__timer__", fire_at)


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
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"id": "sum-123"},  # ts_Extract
                {"id": "sum-123", "persisted": True},  # ts_PersistSummary
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == [
            "ts_Extract",
            "ts_PersistSummary",
        ]
        assert result == {"persisted": True, "summary_id": "sum-123"}

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_passes_tenant_user_and_thread_ids_to_each_activity(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "alice", "thread_id": "T-9"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "s"}, {}])

        for _name, _retry_arg, payload in ctx._yielded_calls:
            assert payload["tenant_id"] == TENANT
            assert payload["user_id"] == "alice"
        # Specific shape checks
        summarize_payload = ctx._yielded_calls[0][2]
        persist_payload = ctx._yielded_calls[1][2]

        assert summarize_payload == {"tenant_id": TENANT, "user_id": "alice", "thread_id": "T-9", "limit": 20}
        assert persist_payload == {
            "tenant_id": TENANT,
            "user_id": "alice",
            "thread_id": "T-9",
            "summary": {"id": "s"},
        }

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_summary_id_returned(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [{"id": "s1"}, {"id": "s1"}])

        assert result["summary_id"] == "s1"
        assert result["persisted"] is True

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_non_dict_summary_yields_none_summary_id(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, ["not-a-dict", {}])
        assert result["summary_id"] is None

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        # Yield first; throw exception into the generator at the next yield.
        next(gen)
        with pytest.raises(RuntimeError, match="boom"):
            gen.throw(RuntimeError("boom"))

    @pytest.mark.parametrize("missing", ["tenant_id", "user_id"])
    def test_missing_tenant_or_user_id_raises(self, missing):
        with patch.object(ts_mod, "default_retry_options", return_value=MagicMock()):
            payload = {"tenant_id": TENANT, "user_id": "u", "thread_id": "t"}
            del payload[missing]
            ctx = _make_context(payload)
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)

    @patch.object(ts_mod, "default_retry_options", return_value=MagicMock())
    def test_uses_max_batch_size_from_env(self, _retry, monkeypatch):
        monkeypatch.setenv("MAX_BATCH_SIZE", "7")
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
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
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
                {
                    "fact_count": 2,
                    "episodic_count": 0,
                    "updated_count": 0,
                },
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == ["em_Extract", "em_Persist"]
        assert ctx._yielded_calls[1][2] == {
            "tenant_id": TENANT,
            "user_id": "u1",
            "extracted": {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
        }
        assert result["persisted"] is True
        assert result["extracted"]["fact_count"] == 2
        assert result["reconciled"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_chains_reconcile_when_flag_true(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "reconcile": True})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
                {"fact_count": 2, "episodic_count": 0, "updated_count": 0},
                {
                    "fact": {"kept": 0, "merged": 1, "conflicts": 0},
                    "episodic": {"kept": 1, "merged": 0, "conflicts": 0},
                },
                {"status": "synthesized", "version": 3},
            ],
        )

        names = [c[0] for c in ctx._yielded_calls]
        assert names == ["em_Extract", "em_Persist", "em_ReconcileMemories"]
        assert ctx._yielded_calls[2][2] == {"tenant_id": TENANT, "user_id": "u1"}
        assert [s[0] for s in ctx._yielded_sub_orchestrators] == [
            "SynthesizeProceduralOrchestrator",
        ]
        assert ctx._yielded_sub_orchestrators[0][2] == {"tenant_id": TENANT, "user_id": "u1", "force": False}
        assert result["reconciled"] == {
            "fact": {"kept": 0, "merged": 1, "conflicts": 0},
            "episodic": {"kept": 1, "merged": 0, "conflicts": 0},
        }
        assert result["procedural"] == {"status": "synthesized", "version": 3}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_procedural_failure_is_swallowed(self, _retry):
        """Procedural synthesis is best-effort; failure must not fail the orchestrator."""
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "reconcile": True})

        def boom_after_sub(name, retry, sub_payload, *args, **kwargs):
            ctx._yielded_sub_orchestrators.append((name, retry, sub_payload))
            return ("__sub_boom__", name, sub_payload)

        ctx.call_sub_orchestrator_with_retry.side_effect = boom_after_sub
        # We'll send activity results normally, then throw an exception into the
        # sub-orchestrator yield.
        gen = self._orchestrator()(ctx)
        # Yield 1: em_Extract
        gen.send(None)
        # Yield 2: em_Persist
        gen.send({"facts": [{"id": "f1"}], "episodic": [], "updates": []})
        # Yield 3: em_ReconcileMemories
        gen.send({"fact_count": 2, "episodic_count": 0, "updated_count": 0})
        # Yield 4: SynthesizeProceduralOrchestrator - throw an exception
        gen.send(
            {
                "fact": {"kept": 0, "merged": 1, "conflicts": 0},
                "episodic": {"kept": 1, "merged": 0, "conflicts": 0},
            }
        )
        try:
            gen.throw(RuntimeError("procedural blew up"))
        except StopIteration as stop:
            result = stop.value
        else:
            pytest.fail("orchestrator did not return after procedural exception")

        assert result["persisted"] is True
        assert result["reconciled"] == {
            "fact": {"kept": 0, "merged": 1, "conflicts": 0},
            "episodic": {"kept": 1, "merged": 0, "conflicts": 0},
        }
        assert result["procedural"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_procedural_not_called_when_reconcile_skipped(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"facts": [], "episodic": [], "updates": []},
                {"fact_count": 0, "episodic_count": 0, "updated_count": 0},
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == ["em_Extract", "em_Persist"]
        assert ctx._yielded_sub_orchestrators == []
        assert result["procedural"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_payload_carries_tenant_user_thread_without_recent_k_when_absent(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"fact_count": 0}])

        extract_payload = ctx._yielded_calls[0][2]
        assert extract_payload == {"tenant_id": TENANT, "user_id": "u", "thread_id": "t"}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_payload_carries_recent_k_when_provided(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t", "recent_k": 7})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"fact_count": 0}])

        extract_payload = ctx._yielded_calls[0][2]
        assert extract_payload == {"tenant_id": TENANT, "user_id": "u", "thread_id": "t", "recent_k": 7}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_output_flows_to_persist(self, _retry):
        extracted = {"facts": [{"id": "f1"}], "episodic": [], "updates": []}
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [extracted, {"fact_count": 1}])

        assert ctx._yielded_calls[1][2] == {"tenant_id": TENANT, "user_id": "u", "extracted": extracted}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        next(gen)
        with pytest.raises(ValueError, match="kaboom"):
            gen.throw(ValueError("kaboom"))

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_advance_watermark_after_persist_when_count_present(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "count": 42})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"fact_count": 0}, True])

        names = [c[0] for c in ctx._yielded_calls]
        assert names == ["em_Extract", "em_Persist", "em_AdvanceExtractWatermark"]
        assert ctx._yielded_calls[2][2] == {"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "count": 42}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_no_watermark_advance_when_count_absent(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"fact_count": 0}])
        names = [c[0] for c in ctx._yielded_calls]
        assert "em_AdvanceExtractWatermark" not in names

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_the_procedural_sub_orchestration_id_hashes_the_tenant_user_and_thread_with_the_count(self, _retry):
        """Two tenants can have the same user and thread, so the instance id hashes the tenant too."""
        instance_ids = []
        for tenant_id in ("acme", "globex"):
            payload = {"tenant_id": tenant_id, "user_id": "u1", "thread_id": "t1", "count": 42, "reconcile": True}
            ctx = _make_context(payload)
            _drive(self._orchestrator()(ctx), [{"facts": []}, {"fact_count": 0}, True, {}, {"status": "unchanged"}])
            instance_ids += ctx._sub_instance_ids

        assert instance_ids == [
            orchestration_instance_id("procedural", "acme", "u1", "t1", count=42),
            orchestration_instance_id("procedural", "globex", "u1", "t1", count=42),
        ]
        assert instance_ids[0] != instance_ids[1]

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_the_procedural_sub_orchestration_gets_no_fixed_id_without_a_count(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "reconcile": True})
        _drive(self._orchestrator()(ctx), [{"facts": []}, {"fact_count": 0}, {}, {"status": "unchanged"}])

        assert ctx._sub_instance_ids == [None]

    @pytest.mark.parametrize("missing", ["tenant_id", "user_id", "thread_id"])
    def test_missing_tenant_user_or_thread_id_raises(self, missing):
        with patch.object(em_mod, "default_retry_options", return_value=MagicMock()):
            payload = {"tenant_id": TENANT, "user_id": "u", "thread_id": "t"}
            del payload[missing]
            ctx = _make_context(payload)
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)


# ---------------------------------------------------------------------------
# Extract memory activities
# ---------------------------------------------------------------------------


class TestExtractMemoryActivities:
    def test_em_extract_uses_payload_recent_k(self):
        pipeline = MagicMock()
        pipeline.extract_memories_durable.return_value = {"facts": [], "episodic": [], "updates": []}

        with patch.object(em_mod, "get_pipeline", return_value=pipeline):
            result = em_mod.em_Extract({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "recent_k": 3})

        pipeline.extract_memories_durable.assert_called_once_with(user_id="u1", thread_id="t1", recent_k=3)
        assert result == {"facts": [], "episodic": [], "updates": []}

    def test_em_extract_falls_back_to_max_batch_size_when_recent_k_absent(self):
        pipeline = MagicMock()
        pipeline.extract_memories_durable.return_value = {"facts": [], "episodic": [], "updates": []}

        with patch.object(em_mod, "get_pipeline", return_value=pipeline):
            em_mod.em_Extract({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"})

        pipeline.extract_memories_durable.assert_called_once_with(user_id="u1", thread_id="t1", recent_k=20)

    def test_em_reconcile_memories_reconciles_fact_and_episodic(self):
        pipeline = MagicMock()
        pipeline.reconcile_memories.side_effect = [
            {"kept": 2, "merged": 1, "conflicts": 0},
            {"kept": 1, "merged": 0, "conflicts": 0},
        ]

        with (
            patch.object(em_mod, "get_pipeline", return_value=pipeline),
            patch("azure.cosmos.agent_memory.thresholds.get_dedup_pool_size", return_value=17),
            patch.object(em_mod.config, "get_activity_time_budget_seconds", return_value=45),
            patch.object(em_mod.time, "monotonic", return_value=1000.0),
        ):
            result = em_mod.em_ReconcileMemories({"tenant_id": TENANT, "user_id": "u1"})

        # Both passes share one deadline: the activity's time budget from when it started.
        assert pipeline.reconcile_memories.call_args_list == [
            call(user_id="u1", n=17, memory_type="fact", deadline=1045.0),
            call(user_id="u1", n=17, memory_type="episodic", deadline=1045.0),
        ]
        assert result == {
            "fact": {"kept": 2, "merged": 1, "conflicts": 0},
            "episodic": {"kept": 1, "merged": 0, "conflicts": 0},
        }

    def test_em_reconcile_memories_budget_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("ACTIVITY_TIME_BUDGET_SECONDS", "90")
        pipeline = MagicMock()
        pipeline.reconcile_memories.return_value = {}

        with (
            patch.object(em_mod, "get_pipeline", return_value=pipeline),
            patch.object(em_mod.time, "monotonic", return_value=10.0),
        ):
            em_mod.em_ReconcileMemories({"tenant_id": TENANT, "user_id": "u1"})

        assert {c.kwargs["deadline"] for c in pipeline.reconcile_memories.call_args_list} == {100.0}

    def test_em_advance_extract_watermark_stamps_counter(self):
        import asyncio

        from shared import cosmos_clients, counters

        container = MagicMock()
        with (
            patch.object(cosmos_clients, "get_counter_container_async", new=AsyncMock(return_value=container)),
            patch.object(counters, "advance_extract_watermark", new=AsyncMock()) as advance,
        ):
            result = asyncio.run(
                em_mod.em_AdvanceExtractWatermark({"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "count": 9})
            )

        assert result is True
        advance.assert_awaited_once_with(container, TENANT, "thread:u1:t1", "u1", 9)


class TestUserSummaryOrchestrator:
    def _orchestrator(self):
        return _user_function(us_mod.UserSummaryOrchestrator)

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_happy_path_calls_two_activities_in_order(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"id": "user-sum-1"},  # us_Extract
                {"id": "user-sum-1", "persisted": True},  # us_PersistUserSummary
            ],
        )
        assert [c[0] for c in ctx._yielded_calls] == [
            "us_Extract",
            "us_PersistUserSummary",
        ]
        assert result == {
            "persisted": True,
            "user_summary_id": "user-sum-1",
        }

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_payloads_only_carry_tenant_user_id_limit_and_thread_ids(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "alice"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "us"}, {}])

        gen_payload = ctx._yielded_calls[0][2]
        persist_payload = ctx._yielded_calls[1][2]

        assert gen_payload == {"tenant_id": TENANT, "user_id": "alice", "limit": 20, "thread_ids": None}
        assert persist_payload == {"tenant_id": TENANT, "user_id": "alice", "user_summary": {"id": "us"}}
        for payload in (gen_payload, persist_payload):
            assert "thread_id" not in payload

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_payload_passes_thread_ids_when_provided(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "alice", "thread_ids": ["t1", "t2"]})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"id": "us"}, {}])

        gen_payload = ctx._yielded_calls[0][2]
        assert gen_payload == {
            "tenant_id": TENANT,
            "user_id": "alice",
            "limit": 20,
            "thread_ids": ["t1", "t2"],
        }

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_user_summary_id_returned(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [{"id": "us"}, {"id": "us"}])
        assert result["user_summary_id"] == "us"
        assert result["persisted"] is True

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_non_dict_user_summary_yields_none_id(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [None, {}])
        assert result["user_summary_id"] is None

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"tenant_id": TENANT, "user_id": "u"})
        gen = self._orchestrator()(ctx)
        next(gen)
        with pytest.raises(RuntimeError):
            gen.throw(RuntimeError("activity failed"))

    @pytest.mark.parametrize("payload", [{"tenant_id": TENANT}, {"user_id": "u"}], ids=["no-user", "no-tenant"])
    def test_missing_tenant_or_user_id_raises(self, payload):
        with patch.object(us_mod, "default_retry_options", return_value=MagicMock()):
            ctx = _make_context(payload)
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)


class TestUserSummaryWaitForExtraction:
    """A brand-new user can cross the user-summary threshold before extraction
    has persisted anything (the change feed starts the two orchestrations
    independently). The orchestrator must wait on a replay-safe Durable timer
    and still produce the summary, instead of failing after the short
    activity-retry window."""

    def _orchestrator(self):
        return _user_function(us_mod.UserSummaryOrchestrator)

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_waits_then_persists_when_memories_arrive_late(self, _retry, monkeypatch):
        monkeypatch.setenv("USER_SUMMARY_WAIT_SECONDS", "120")
        monkeypatch.setenv("USER_SUMMARY_WAIT_INTERVAL_SECONDS", "10")
        # Static clock: the budget never elapses, so the loop ends only when
        # memories finally land.
        ctx = _TimerContext({"tenant_id": TENANT, "user_id": "u1"}, step_seconds=0)
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"status": "no_memories_yet"},  # us_Extract #1 - extraction not done
                None,  # create_timer #1
                {"status": "no_memories_yet"},  # us_Extract #2 - still not done
                None,  # create_timer #2
                {"id": "user-sum-1"},  # us_Extract #3 - memories landed
                {"id": "user-sum-1"},  # us_PersistUserSummary
            ],
        )
        assert result == {"persisted": True, "user_summary_id": "user-sum-1"}
        assert len(ctx.timer_fire_times) == 2  # waited twice before succeeding
        assert [name for name, _ in ctx.activity_calls] == [
            "us_Extract",
            "us_Extract",
            "us_Extract",
            "us_PersistUserSummary",
        ]

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_gives_up_after_budget_without_persisting(self, _retry, monkeypatch):
        monkeypatch.setenv("USER_SUMMARY_WAIT_SECONDS", "15")
        monkeypatch.setenv("USER_SUMMARY_WAIT_INTERVAL_SECONDS", "10")
        # Clock advances 10s per read, so the 15s budget elapses after one wait.
        ctx = _TimerContext({"tenant_id": TENANT, "user_id": "u1"}, step_seconds=10)
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"status": "no_memories_yet"},  # us_Extract #1
                None,  # create_timer #1
                {"status": "no_memories_yet"},  # us_Extract #2 - budget now exceeded
            ],
        )
        assert result == {"persisted": False, "user_summary_id": None, "skipped": "no_memories_yet"}
        assert "us_PersistUserSummary" not in [name for name, _ in ctx.activity_calls]

    @patch.object(us_mod, "default_retry_options", return_value=MagicMock())
    def test_persists_immediately_when_memories_already_present(self, _retry, monkeypatch):
        monkeypatch.setenv("USER_SUMMARY_WAIT_SECONDS", "120")
        ctx = _TimerContext({"tenant_id": TENANT, "user_id": "u1"}, step_seconds=0)
        gen = self._orchestrator()(ctx)
        result, _ = _drive(gen, [{"id": "us"}, {"id": "us"}])
        assert result == {"persisted": True, "user_summary_id": "us"}
        assert ctx.timer_fire_times == []  # no waiting when memories already exist


class TestUserSummaryExtractActivity:
    def test_returns_sentinel_when_no_source_memories(self):
        pipeline = MagicMock()
        pipeline.generate_user_summary_durable.side_effect = NoSourceMemoriesError("No memories found for user_id='u1'")
        with patch.object(us_mod, "get_pipeline", return_value=pipeline):
            result = us_mod.us_Extract({"tenant_id": TENANT, "user_id": "u1", "limit": 20, "thread_ids": None})
        assert result == {"status": "no_memories_yet"}

    def test_passes_through_summary_doc_when_memories_present(self):
        pipeline = MagicMock()
        pipeline.generate_user_summary_durable.return_value = {"id": "user_summary_u1", "type": "user_summary"}
        with patch.object(us_mod, "get_pipeline", return_value=pipeline):
            result = us_mod.us_Extract({"tenant_id": TENANT, "user_id": "u1", "limit": 20, "thread_ids": ["t1"]})
        assert result == {"id": "user_summary_u1", "type": "user_summary"}
        pipeline.generate_user_summary_durable.assert_called_once_with(user_id="u1", recent_k=20, thread_ids=["t1"])


# ---------------------------------------------------------------------------
# Tenants: every orchestration input and activity payload carries tenant_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "orchestrator", "extra_input", "results", "payloads_sent"),
    [
        pytest.param(
            ts_mod, "ThreadSummaryOrchestrator", {"thread_id": "t1"}, [{"id": "s"}, {"id": "s"}], 2, id="thread-summary"
        ),
        pytest.param(
            em_mod,
            "ExtractMemoriesOrchestrator",
            {"thread_id": "t1", "count": 4, "reconcile": True},
            [{"facts": []}, {"fact_count": 0}, True, {}, {"status": "unchanged"}],
            5,
            id="extract-memories",
        ),
        pytest.param(ee_mod, "ExtractEpisodesOrchestrator", {"thread_id": "t1"}, [{"episodes": 1}], 1, id="episodes"),
        pytest.param(
            sp_mod, "SynthesizeProceduralOrchestrator", {}, [{"status": "unchanged"}], 1, id="synthesize-procedural"
        ),
        pytest.param(us_mod, "UserSummaryOrchestrator", {}, [{"id": "us"}, {"id": "us"}], 2, id="user-summary"),
    ],
)
def test_every_activity_and_sub_orchestration_payload_carries_the_tenant(
    module, orchestrator, extra_input, results, payloads_sent, monkeypatch
):
    monkeypatch.setenv("PROCEDURAL_SYNTHESIS_AUTO", "true")
    with patch.object(module, "default_retry_options", return_value=MagicMock()):
        ctx = _make_context({"tenant_id": "globex", "user_id": "u1", **extra_input})
        _drive(_user_function(getattr(module, orchestrator))(ctx), results)

    sent = [payload for _, _, payload in ctx._yielded_calls + ctx._yielded_sub_orchestrators]
    assert len(sent) == payloads_sent
    assert all(payload["tenant_id"] == "globex" for payload in sent)


_PIPELINE_ACTIVITIES = [
    pytest.param(ts_mod.ts_Extract, ts_mod, {"thread_id": "t1", "limit": 5}, id="ts_Extract"),
    pytest.param(ts_mod.ts_PersistSummary, ts_mod, {"thread_id": "t1", "summary": {"id": "s"}}, id="ts_PersistSummary"),
    pytest.param(em_mod.em_Extract, em_mod, {"thread_id": "t1"}, id="em_Extract"),
    pytest.param(em_mod.em_Persist, em_mod, {"extracted": {"facts": []}}, id="em_Persist"),
    pytest.param(em_mod.em_ReconcileMemories, em_mod, {}, id="em_ReconcileMemories"),
    pytest.param(ee_mod.ee_ExtractEpisodes, ee_mod, {"thread_id": "t1"}, id="ee_ExtractEpisodes"),
    pytest.param(sp_mod.sp_SynthesizeProcedural, sp_mod, {}, id="sp_SynthesizeProcedural"),
    pytest.param(us_mod.us_Extract, us_mod, {"limit": 5}, id="us_Extract"),
    pytest.param(us_mod.us_PersistUserSummary, us_mod, {"user_summary": {"id": "us"}}, id="us_PersistUserSummary"),
]


@pytest.mark.parametrize(("activity", "module", "extra_payload"), _PIPELINE_ACTIVITIES)
def test_an_activity_runs_on_the_pipeline_of_the_tenant_in_its_payload(activity, module, extra_payload):
    pipelines = {"acme": MagicMock(name="acme-pipeline"), "globex": MagicMock(name="globex-pipeline")}

    with patch.object(module, "get_pipeline", side_effect=lambda tenant_id: pipelines[tenant_id]) as get_pipeline:
        activity({"tenant_id": "globex", "user_id": "u1", **extra_payload})

    get_pipeline.assert_called_once_with("globex")
    assert pipelines["globex"].method_calls
    assert not pipelines["acme"].method_calls
