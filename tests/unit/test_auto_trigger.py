"""Tests for the InProcess push_to_cosmos auto-trigger.

These tests set `FACT_EXTRACTION_EVERY_N=1` for per-turn extraction: each
turn flushed to Cosmos should immediately fire `process_thread` for the
in-process backend. The durable backend must remain a no-op (the
change-feed function app handles it).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory.auto_trigger import maybe_trigger_steps
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.processors import DurableFunctionProcessor, InProcessProcessor

TENANT = "acme"


class _FakeCounterContainer:
    """Minimal in-memory stand-in for the Cosmos counter container.

    Exercises the REAL increment / watermark-read / watermark-advance helpers
    end-to-end (no mocking of counter math), so a watermark/recent_k regression
    can't slip through behind constant mocks. It records the partition each
    read and patch asks for.
    """

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.partitions_read: list[list[str]] = []
        self.partitions_patched: list[list[str]] = []
        self._etag = 0

    def read_item(self, *, item, partition_key):
        self.partitions_read.append(partition_key)
        if item not in self.store:
            raise CosmosResourceNotFoundError(message="404")
        return dict(self.store[item])

    def create_item(self, *, body):
        self._etag += 1
        body = dict(body)
        body["_etag"] = f"e{self._etag}"
        self.store[body["id"]] = body
        return dict(body)

    def upsert_item(self, *, body, **_kwargs):
        self._etag += 1
        body = dict(body)
        body["_etag"] = f"e{self._etag}"
        self.store[body["id"]] = body
        return dict(body)

    def patch_item(self, *, item, partition_key, patch_operations):
        self.partitions_patched.append(partition_key)
        doc = self.store.setdefault(item, {"id": item})
        for op in patch_operations:
            doc[op["path"].lstrip("/")] = op["value"]
        return dict(doc)


def _connected(processor=None, **client_kwargs) -> CosmosMemoryClient:
    client = CosmosMemoryClient(use_default_credential=False, processor=processor, **client_kwargs)
    client._memories_container_client = MagicMock()
    client._turns_container_client = client._memories_container_client
    client._summaries_container_client = client._memories_container_client
    return client


def test_push_to_cosmos_fires_inprocess_trigger_per_turn(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
    monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    counter_container = MagicMock()
    client._counter_container_client = counter_container

    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = None
    pipeline.extract_memories.return_value = {"fact_count": 1}
    pipeline.reconcile_memories.return_value = {}
    client._processor._pipeline = pipeline

    with patch(
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
        return_value=(0, 1),
    ):
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    pipeline.extract_memories.assert_called_once_with("u1", "t1", recent_k=None)


def test_push_to_cosmos_durable_does_not_fire_trigger(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
    client = _connected(processor=DurableFunctionProcessor())
    client._counter_container_client = MagicMock()

    with patch(
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
        return_value=(0, 1),
    ) as inc:
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    inc.assert_not_called()


def test_push_to_cosmos_skips_trigger_when_thresholds_zero(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "0")
    monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
    monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
    monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    client._counter_container_client = MagicMock()

    with patch(
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
        return_value=(0, 1),
    ) as inc:
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    inc.assert_not_called()


def test_push_to_cosmos_swallows_trigger_failures(monkeypatch):
    """Auto-trigger errors must never propagate from push_to_cosmos."""
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")

    pipeline = MagicMock()
    pipeline.generate_thread_summary.side_effect = RuntimeError("boom")
    client = _connected(processor=InProcessProcessor(pipeline=pipeline))
    client._counter_container_client = MagicMock()

    with patch(
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
        return_value=(0, 1),
    ):
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()  # must not raise


def test_push_to_cosmos_skips_when_counter_container_unavailable(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    # Counter container handle stays None; lazy getter would normally try to
    # build one but will return None on failure.
    client._get_counter_container = MagicMock(return_value=None)

    pipeline = MagicMock()
    client._processor._pipeline = pipeline

    client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
    client.push_to_cosmos()

    pipeline.extract_memories.assert_not_called()


# ---------------------------------------------------------------------------
# Per-step trigger gating - each *_EVERY_N fires its own pipeline step
# independently. The InProcess backend mirrors the function-app
# split-orchestrator behavior so the two backends produce the same memory
# contents for the same chat history.
# ---------------------------------------------------------------------------


class TestPerStepAutoTrigger:
    def test_episode_zero_does_not_fire(self):
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_episodes = MagicMock()
        counter_container = _FakeCounterContainer()

        maybe_trigger_steps(
            processor,
            counter_container,
            TENANT,
            {("u1", "t1"): 1},
            thresholds={
                "FACT_EXTRACTION_EVERY_N": 0,
                "THREAD_SUMMARY_EVERY_N": 0,
                "EPISODE_EVAL_EVERY_N": 0,
                "USER_SUMMARY_EVERY_N": 0,
                "MEMORY_PROCESSOR_OWNER": "inprocess",
            },
        )

        processor.process_extract_episodes.assert_not_called()
        assert counter_container.store == {}

    def test_episode_fires_when_threshold_crossed(self):
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_episodes = MagicMock(return_value={})
        counter_container = _FakeCounterContainer()
        thresholds = {
            "FACT_EXTRACTION_EVERY_N": 0,
            "THREAD_SUMMARY_EVERY_N": 0,
            "EPISODE_EVAL_EVERY_N": 3,
            "USER_SUMMARY_EVERY_N": 0,
            "MEMORY_PROCESSOR_OWNER": "inprocess",
        }

        maybe_trigger_steps(processor, counter_container, TENANT, {("u1", "t1"): 2}, thresholds=thresholds)
        processor.process_extract_episodes.assert_not_called()

        maybe_trigger_steps(processor, counter_container, TENANT, {("u1", "t1"): 1}, thresholds=thresholds)

        processor.process_extract_episodes.assert_called_once_with(user_id="u1", thread_id="t1")

    def test_episode_failure_is_caught_and_other_steps_continue(self):
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_episodes = MagicMock(side_effect=RuntimeError("episode boom"))
        processor.process_thread_summary = MagicMock(return_value={})
        counter_container = _FakeCounterContainer()

        with patch("azure.cosmos.agent_memory._counters.stamp_failure_sync") as stamp:
            maybe_trigger_steps(
                processor,
                counter_container,
                TENANT,
                {("u1", "t1"): 1},
                thresholds={
                    "FACT_EXTRACTION_EVERY_N": 0,
                    "THREAD_SUMMARY_EVERY_N": 1,
                    "EPISODE_EVAL_EVERY_N": 1,
                    "USER_SUMMARY_EVERY_N": 0,
                    "MEMORY_PROCESSOR_OWNER": "inprocess",
                },
            )

        processor.process_extract_episodes.assert_called_once_with(user_id="u1", thread_id="t1")
        processor.process_thread_summary.assert_called_once_with(user_id="u1", thread_id="t1")
        stamp.assert_called_once()

    def test_extract_fires_independently_of_summary(self, monkeypatch):
        """N_facts=1 alone fires extract; summary/user-summary stay quiet."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "10")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "20")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        processor.process_thread_summary = MagicMock(return_value={})
        processor.process_user_summary = MagicMock()

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(0, 1),  # crosses 1 only
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once_with(user_id="u1", thread_id="t1")
        processor.process_thread_summary.assert_not_called()
        processor.process_user_summary.assert_not_called()

    def test_extract_fires_without_recent_k_or_watermark(self, monkeypatch):
        """Extraction now covers all un-extracted turns (gated per-turn by
        extracted_at) and batches internally, so the auto-trigger fires it with
        NO recent_k and tracks NO success-gated watermark."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(0, 1),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once_with(user_id="u1", thread_id="t1")

    def test_extract_failure_stamps_failure(self, monkeypatch):
        """A total extract failure is recorded via stamp_failure (telemetry); there
        is no watermark to (not) advance - per-batch failures are handled inside
        the pipeline, so this outer path only sees unexpected total failures."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(side_effect=RuntimeError("llm down"))

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with (
            patch(
                "azure.cosmos.agent_memory._counters.increment_counter_sync",
                return_value=(0, 1),
            ),
            patch(
                "azure.cosmos.agent_memory._counters.stamp_failure_sync",
            ) as stamp,
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        stamp.assert_called_once()

    def test_summary_fires_independently_when_threshold_crossed(self, monkeypatch):
        """N_summary=10 boundary fires summary; N_facts=0 prevents extract."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "0")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "10")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock()
        processor.process_thread_summary = MagicMock(return_value={})

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(9, 10),  # crosses 10 only
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_thread_summary.assert_called_once_with(user_id="u1", thread_id="t1")
        processor.process_extract_memories.assert_not_called()

    def test_user_summary_fires_at_user_threshold(self, monkeypatch):
        """The user-scoped counter is incremented separately from the thread counter."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "0")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "2")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_user_summary = MagicMock()

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        # Thread counter: (0,1) then (1,2); user counter: (1,2) crosses 2.
        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            side_effect=[(0, 1), (1, 2), (1, 2)],
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.add_local(user_id="u1", role="agent", thread_id="t1", content="ok")
            client.push_to_cosmos()

        processor.process_user_summary.assert_called_once_with(user_id="u1")


# ---------------------------------------------------------------------------
# Owner exclusivity - MEMORY_PROCESSOR_OWNER ensures only one of
# {SDK auto-trigger, FA change-feed processor} runs against a shared
# container, preventing double-extraction / double-dedup.
# ---------------------------------------------------------------------------


class TestProcessorOwner:
    def test_durable_owner_suppresses_sdk_trigger(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "durable")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock()

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(0, 1),
        ) as inc:
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        inc.assert_not_called()
        processor.process_extract_memories.assert_not_called()

    def test_inprocess_owner_allows_sdk_trigger(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "inprocess")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(0, 1),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once()


# ---------------------------------------------------------------------------
# Tenant - every counter the auto-trigger reads, writes or stamps is in the
# client's tenant, in the partition of the owner it counts.
# ---------------------------------------------------------------------------

_THREAD_AND_USER_STEPS = {
    "FACT_EXTRACTION_EVERY_N": 1,
    "THREAD_SUMMARY_EVERY_N": 0,
    "EPISODE_EVAL_EVERY_N": 0,
    "USER_SUMMARY_EVERY_N": 1,
    "DEDUP_EVERY_N": 0,
    "MEMORY_PROCESSOR_OWNER": "inprocess",
}


class TestCountersFollowTheTenant:
    def test_thread_and_user_counters_live_in_the_tenant_and_owner_partition(self):
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        processor.process_user_summary = MagicMock()
        counters = _FakeCounterContainer()

        maybe_trigger_steps(processor, counters, TENANT, {("u1", "t1"): 1}, thresholds=_THREAD_AND_USER_STEPS)

        assert {(doc["tenant_id"], doc["user_id"], doc["thread_id"]) for doc in counters.store.values()} == {
            (TENANT, "u1", "t1"),
            (TENANT, "u1", "__counters__"),
        }
        assert counters.partitions_read == [[TENANT, "u1"], [TENANT, "u1"]]

    def test_a_failed_step_is_stamped_in_the_tenant_and_owner_partition(self):
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(side_effect=RuntimeError("boom"))
        processor.process_user_summary = MagicMock(side_effect=RuntimeError("boom"))
        counters = _FakeCounterContainer()

        maybe_trigger_steps(processor, counters, TENANT, {("u1", "t1"): 1}, thresholds=_THREAD_AND_USER_STEPS)

        assert counters.partitions_patched == [[TENANT, "u1"], [TENANT, "u1"]]
        assert counters.store["thread:u1:t1"]["last_failure_reason"].startswith("process_extract_memories: ")
        assert counters.store["user:u1"]["last_failure_reason"].startswith("process_user_summary: ")

    def test_a_client_counts_the_turns_it_pushes_in_its_own_tenant(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("EPISODE_EVAL_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        client = _connected(processor=processor, tenant_id=TENANT)
        counters = client._counter_container_client = _FakeCounterContainer()

        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

        [counter] = counters.store.values()
        assert (counter["tenant_id"], counter["user_id"], counter["thread_id"]) == (TENANT, "u1", "t1")
        assert counters.partitions_read == [[TENANT, "u1"]]

    def test_a_client_hands_its_tenant_to_the_trigger(self):
        client = CosmosMemoryClient(use_default_credential=False, tenant_id=TENANT)
        processor, counters = MagicMock(), MagicMock()
        client._get_processor = MagicMock(return_value=processor)
        client._get_counter_container = MagicMock(return_value=counters)

        with patch("azure.cosmos.agent_memory.cosmos_memory_client.maybe_trigger_steps") as trigger:
            client._maybe_auto_trigger({("u1", "t1"): 1})

        assert trigger.call_args.args == (processor, counters, TENANT, {("u1", "t1"): 1})
