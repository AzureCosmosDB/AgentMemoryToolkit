"""Unit tests for the three Durable Functions orchestrators.

Each orchestrator is a generator function wrapped by
``@bp.orchestration_trigger``. The decorator stores the original generator
on ``handle.orchestrator_function``; we drive that generator directly with a
mocked ``DurableOrchestrationContext`` so no Durable runtime, Cosmos, or
LLM is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

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
    yielded_sub_orchestrators: list[tuple] = []

    def call_activity_with_retry(name, retry, activity_payload):
        yielded_calls.append((name, retry, activity_payload))
        # Return a sentinel the generator will yield. The test driver intercepts
        # this and feeds the next pre-canned activity result back in.
        return ("__call__", name, activity_payload)

    def call_sub_orchestrator_with_retry(name, retry, sub_payload, *_args, **_kwargs):
        yielded_sub_orchestrators.append((name, retry, sub_payload))
        return ("__sub__", name, sub_payload)

    ctx.call_activity_with_retry.side_effect = call_activity_with_retry
    ctx.call_sub_orchestrator_with_retry.side_effect = call_sub_orchestrator_with_retry
    ctx._yielded_calls = yielded_calls
    ctx._yielded_sub_orchestrators = yielded_sub_orchestrators
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
        result, _ = _drive(gen, [{"id": "s1"}, {"id": "s1"}])

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
                {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
                {"facts": [{"id": "f1", "deduped": True}], "episodic": [], "updates": []},
                {
                    "fact_count": 2,
                    "episodic_count": 0,
                    "updated_count": 0,
                },
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == ["em_Extract", "em_Dedup", "em_Persist"]
        assert ctx._yielded_calls[1][2] == {
            "user_id": "u1",
            "extracted": {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
        }
        assert ctx._yielded_calls[2][2] == {
            "user_id": "u1",
            "extracted": {"facts": [{"id": "f1", "deduped": True}], "episodic": [], "updates": []},
        }
        assert result["persisted"] is True
        assert result["extracted"]["fact_count"] == 2
        assert result["reconciled"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_chains_reconcile_when_flag_true(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1", "reconcile": True})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
                {"facts": [{"id": "f1"}], "episodic": [], "updates": []},
                {"fact_count": 2, "episodic_count": 0, "updated_count": 0},
                {
                    "fact": {"kept": 0, "merged": 1, "contradicted": 0},
                    "episodic": {"kept": 1, "merged": 0, "contradicted": 0},
                },
                {"status": "synthesized", "version": 3},
            ],
        )

        names = [c[0] for c in ctx._yielded_calls]
        assert names == ["em_Extract", "em_Dedup", "em_Persist", "em_ReconcileMemories"]
        assert ctx._yielded_calls[3][2] == {"user_id": "u1", "full_rebuild": False}
        assert [s[0] for s in ctx._yielded_sub_orchestrators] == [
            "SynthesizeProceduralOrchestrator",
        ]
        assert ctx._yielded_sub_orchestrators[0][2] == {"user_id": "u1", "force": False}
        assert result["reconciled"] == {
            "fact": {"kept": 0, "merged": 1, "contradicted": 0},
            "episodic": {"kept": 1, "merged": 0, "contradicted": 0},
        }
        assert result["procedural"] == {"status": "synthesized", "version": 3}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_procedural_failure_is_swallowed(self, _retry):
        """Procedural synthesis is best-effort; failure must not fail the orchestrator."""
        ctx = _make_context({"user_id": "u1", "thread_id": "t1", "reconcile": True})

        def boom_after_sub(name, retry, sub_payload, *args, **kwargs):
            ctx._yielded_sub_orchestrators.append((name, retry, sub_payload))
            return ("__sub_boom__", name, sub_payload)

        ctx.call_sub_orchestrator_with_retry.side_effect = boom_after_sub
        # We'll send activity results normally, then throw an exception into the
        # sub-orchestrator yield.
        gen = self._orchestrator()(ctx)
        # Yield 1: em_Extract
        gen.send(None)
        # Yield 2: em_Dedup
        gen.send({"facts": [{"id": "f1"}], "episodic": [], "updates": []})
        # Yield 3: em_Persist
        gen.send({"facts": [{"id": "f1"}], "episodic": [], "updates": []})
        # Yield 4: em_ReconcileMemories
        gen.send({"fact_count": 2, "episodic_count": 0, "updated_count": 0})
        # Yield 5: SynthesizeProceduralOrchestrator — throw an exception
        gen.send(
            {
                "fact": {"kept": 0, "merged": 1, "contradicted": 0},
                "episodic": {"kept": 1, "merged": 0, "contradicted": 0},
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
            "fact": {"kept": 0, "merged": 1, "contradicted": 0},
            "episodic": {"kept": 1, "merged": 0, "contradicted": 0},
        }
        assert result["procedural"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_procedural_not_called_when_reconcile_skipped(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        result, _ = _drive(
            gen,
            [
                {"facts": [], "episodic": [], "updates": []},
                {"facts": [], "episodic": [], "updates": []},
                {"fact_count": 0, "episodic_count": 0, "updated_count": 0},
            ],
        )

        assert [c[0] for c in ctx._yielded_calls] == ["em_Extract", "em_Dedup", "em_Persist"]
        assert ctx._yielded_sub_orchestrators == []
        assert result["procedural"] is None

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_payload_carries_user_thread_without_recent_k_when_absent(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"facts": []}, {"fact_count": 0}])

        extract_payload = ctx._yielded_calls[0][2]
        assert extract_payload == {"user_id": "u", "thread_id": "t"}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_extract_payload_carries_recent_k_when_provided(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t", "recent_k": 7})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"facts": []}, {"fact_count": 0}])

        extract_payload = ctx._yielded_calls[0][2]
        assert extract_payload == {"user_id": "u", "thread_id": "t", "recent_k": 7}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_dedup_output_flows_to_persist(self, _retry):
        extracted = {"facts": [{"id": "f1"}], "episodic": [], "updates": []}
        deduped = {"facts": [{"id": "f1", "embedding": [0.1]}], "episodic": [], "updates": []}
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [extracted, deduped, {"fact_count": 1}])

        assert ctx._yielded_calls[1][2] == {"user_id": "u", "extracted": extracted}
        assert ctx._yielded_calls[2][2] == {"user_id": "u", "extracted": deduped}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_activity_failure_propagates(self, _retry):
        ctx = _make_context({"user_id": "u", "thread_id": "t"})
        gen = self._orchestrator()(ctx)
        next(gen)
        with pytest.raises(ValueError, match="kaboom"):
            gen.throw(ValueError("kaboom"))

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_advance_watermark_after_persist_when_count_present(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1", "count": 42})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"facts": []}, {"fact_count": 0}, True])

        names = [c[0] for c in ctx._yielded_calls]
        assert names == ["em_Extract", "em_Dedup", "em_Persist", "em_AdvanceExtractWatermark"]
        assert ctx._yielded_calls[3][2] == {"user_id": "u1", "thread_id": "t1", "count": 42}

    @patch.object(em_mod, "default_retry_options", return_value=MagicMock())
    def test_no_watermark_advance_when_count_absent(self, _retry):
        ctx = _make_context({"user_id": "u1", "thread_id": "t1"})
        gen = self._orchestrator()(ctx)
        _drive(gen, [{"facts": []}, {"facts": []}, {"fact_count": 0}])
        names = [c[0] for c in ctx._yielded_calls]
        assert "em_AdvanceExtractWatermark" not in names

    def test_missing_thread_id_raises(self):
        with patch.object(em_mod, "default_retry_options", return_value=MagicMock()):
            ctx = _make_context({"user_id": "u"})
            gen = self._orchestrator()(ctx)
            with pytest.raises(KeyError):
                next(gen)


# ---------------------------------------------------------------------------
# Extract memory activities
# ---------------------------------------------------------------------------


class TestExtractMemoryActivities:
    def test_em_extract_uses_payload_recent_k(self):
        pipeline = MagicMock()
        pipeline.extract_memories_dry.return_value = {"facts": [], "episodic": [], "updates": []}

        with patch.object(em_mod, "get_pipeline", return_value=pipeline):
            result = em_mod.em_Extract({"user_id": "u1", "thread_id": "t1", "recent_k": 3})

        pipeline.extract_memories_dry.assert_called_once_with(user_id="u1", thread_id="t1", recent_k=3)
        assert result == {"facts": [], "episodic": [], "updates": []}

    def test_em_extract_falls_back_to_max_batch_size_when_recent_k_absent(self):
        pipeline = MagicMock()
        pipeline.extract_memories_dry.return_value = {"facts": [], "episodic": [], "updates": []}

        with patch.object(em_mod, "get_pipeline", return_value=pipeline):
            em_mod.em_Extract({"user_id": "u1", "thread_id": "t1"})

        pipeline.extract_memories_dry.assert_called_once_with(user_id="u1", thread_id="t1", recent_k=20)

    def test_em_dedup_delegates_to_pipeline_and_returns_deduped_dict(self):
        extracted = {"facts": [{"id": "f1"}], "episodic": [], "updates": []}
        deduped = {"facts": [{"id": "f1", "embedding": [0.1]}], "episodic": [], "updates": []}
        pipeline = MagicMock()
        pipeline.dedup_extracted_memories.return_value = deduped

        with patch.object(em_mod, "get_pipeline", return_value=pipeline):
            result = em_mod.em_Dedup({"user_id": "u1", "extracted": extracted})

        pipeline.dedup_extracted_memories.assert_called_once_with(user_id="u1", extracted=extracted)
        assert result == deduped

    def test_em_dedup_falls_back_to_input_when_pipeline_returns_none(self):
        extracted = {"facts": [], "episodic": [], "updates": []}
        pipeline = MagicMock()
        pipeline.dedup_extracted_memories.return_value = None

        with patch.object(em_mod, "get_pipeline", return_value=pipeline):
            result = em_mod.em_Dedup({"user_id": "u1", "extracted": extracted})

        assert result is extracted

    def test_em_reconcile_memories_reconciles_fact_and_episodic(self):
        pipeline = MagicMock()
        pipeline.reconcile_memories.side_effect = [
            {"kept": 2, "merged": 1, "contradicted": 0},
            {"kept": 1, "merged": 0, "contradicted": 0},
        ]

        with (
            patch.object(em_mod, "get_pipeline", return_value=pipeline),
            patch("azure.cosmos.agent_memory.thresholds.get_dedup_pool_size", return_value=17),
        ):
            result = em_mod.em_ReconcileMemories({"user_id": "u1"})

        # full_rebuild defaults False when the change-feed didn't request a backstop.
        assert pipeline.reconcile_memories.call_args_list == [
            call(user_id="u1", n=17, memory_type="fact", full_rebuild=False),
            call(user_id="u1", n=17, memory_type="episodic", full_rebuild=False),
        ]
        assert result == {
            "fact": {"kept": 2, "merged": 1, "contradicted": 0},
            "episodic": {"kept": 1, "merged": 0, "contradicted": 0},
        }

    def test_em_reconcile_memories_forwards_full_rebuild(self):
        pipeline = MagicMock()
        pipeline.reconcile_memories.side_effect = [{"kept": 0}, {"kept": 0}]
        with (
            patch.object(em_mod, "get_pipeline", return_value=pipeline),
            patch("azure.cosmos.agent_memory.thresholds.get_dedup_pool_size", return_value=17),
        ):
            em_mod.em_ReconcileMemories({"user_id": "u1", "full_rebuild": True})

        assert pipeline.reconcile_memories.call_args_list == [
            call(user_id="u1", n=17, memory_type="fact", full_rebuild=True),
            call(user_id="u1", n=17, memory_type="episodic", full_rebuild=True),
        ]

    def test_em_advance_extract_watermark_stamps_counter(self):
        import asyncio

        from shared import cosmos_clients, counters

        container = MagicMock()
        with (
            patch.object(cosmos_clients, "get_counter_container_async", new=AsyncMock(return_value=container)),
            patch.object(counters, "advance_extract_watermark", new=AsyncMock()) as advance,
        ):
            result = asyncio.run(
                em_mod.em_AdvanceExtractWatermark({"user_id": "u1", "thread_id": "t1", "count": 9})
            )

        assert result is True
        advance.assert_awaited_once_with(container, "thread:u1:t1", "u1", "t1", 9)


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
        result, _ = _drive(gen, [{"id": "us"}, {"id": "us"}])
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
