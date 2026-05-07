"""Unit tests for ``function_app/triggers/change_feed.py``.

Mocks the durable client's ``start_new`` and the counter container so the
trigger's pure-logic entry point :func:`process_changefeed_batch` can be
exercised without Azure Functions or Cosmos DB.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from azure.cosmos.exceptions import CosmosResourceNotFoundError
from triggers.change_feed import process_changefeed_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turn(user_id: str = "u1", thread_id: str = "t1", lsn: int | None = None) -> dict:
    doc: dict = {"type": "turn", "user_id": user_id, "thread_id": thread_id}
    if lsn is not None:
        doc["_lsn"] = lsn
    return doc


def _make_starter() -> MagicMock:
    starter = MagicMock()
    starter.start_new = AsyncMock()
    return starter


def _make_counter_container_starting_at(start_count: int = 0) -> MagicMock:
    """Return a counter container whose ``read_item`` answers from an in-mem
    dict and whose ``create_item``/``upsert_item`` mutate that same dict.

    Lets the test mimic a real Cosmos counter without needing the SDK.
    """
    state: dict[str, dict] = {}
    if start_count:
        # Seeded entries can be supplied by the test if needed; default empty.
        pass

    container = MagicMock()

    async def read_item(*, item, partition_key):
        if item not in state:
            raise CosmosResourceNotFoundError(message="404")
        return dict(state[item])

    async def create_item(*, body):
        state[body["id"]] = dict(body)
        return body

    async def upsert_item(*, body, **_kwargs):
        state[body["id"]] = dict(body)
        return body

    container.read_item = AsyncMock(side_effect=read_item)
    container.create_item = AsyncMock(side_effect=create_item)
    container.upsert_item = AsyncMock(side_effect=upsert_item)
    container._state = state  # exposed for assertions
    return container


# ---------------------------------------------------------------------------
# Filtering / disabled-mode behaviour
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_all_disabled_skips_everything():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch(
            [_turn(), _turn(), _turn(), _turn()],
            starter,
            counter_container=container,
        )
    )

    assert starter.start_new.await_count == 0
    assert container.read_item.await_count == 0


def test_unset_thresholds_apply_documented_defaults(monkeypatch):
    """When env vars are unset, thresholds fall back to documented defaults
    (fact=1, thread=10, user=20), not 0.

    Regression for the silent-no-op out-of-the-box deploy bug: a missing
    setting should NOT disable the orchestrator (only an explicit "0" does).
    """
    for name in ("THREAD_SUMMARY_EVERY_N", "FACT_EXTRACTION_EVERY_N", "USER_SUMMARY_EVERY_N"):
        monkeypatch.delenv(name, raising=False)

    starter = _make_starter()
    container = _make_counter_container_starting_at()

    # 1 turn => crosses fact extraction (n=1) only.
    asyncio.run(
        process_changefeed_batch(
            [_turn()],
            starter,
            counter_container=container,
        )
    )

    started_names = [c.args[0] for c in starter.start_new.await_args_list]
    assert "ExtractMemoriesOrchestrator" in started_names
    assert "ThreadSummaryOrchestrator" not in started_names
    assert "UserSummaryOrchestrator" not in started_names

    # 10 more turns from the same user/thread => crosses thread (n=10).
    starter.start_new.reset_mock()
    asyncio.run(
        process_changefeed_batch(
            [_turn() for _ in range(10)],
            starter,
            counter_container=container,
        )
    )
    started_names = [c.args[0] for c in starter.start_new.await_args_list]
    assert "ThreadSummaryOrchestrator" in started_names


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "20",
    },
    clear=False,
)
def test_non_turn_documents_are_filtered_out():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    docs = [
        {"type": "summary", "user_id": "u1", "thread_id": "t1"},
        {"type": "fact", "user_id": "u1", "thread_id": "t1"},
        {"type": "user_summary", "user_id": "u1", "thread_id": "__user_summary__"},
    ]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert starter.start_new.await_count == 0
    assert container.read_item.await_count == 0


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_sub_threshold_batch_does_not_start_orchestrator():
    """A batch of 3 turns with N=4 must NOT start any orchestrator."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch(
            [_turn() for _ in range(3)],
            starter,
            counter_container=container,
        )
    )

    assert starter.start_new.await_count == 0
    # We DID still increment the counter.
    assert container.create_item.await_count == 1
    assert container._state["thread:u1:t1"]["count"] == 3


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_turn_doc_missing_ids_is_skipped():
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [
        {"type": "turn", "user_id": "u1"},  # missing thread_id
        {"type": "turn", "thread_id": "t1"},  # missing user_id
    ]
    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))
    assert starter.start_new.await_count == 0
    assert container.read_item.await_count == 0


# ---------------------------------------------------------------------------
# Threshold crossing → deterministic instance IDs
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "20",
    },
    clear=False,
)
def test_thread_threshold_crossing_starts_summary_and_extract():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    # 4 turns from the same user/thread crosses N=4 once for both
    # thread-summary and extract orchestrators.
    asyncio.run(
        process_changefeed_batch(
            [_turn() for _ in range(4)],
            starter,
            counter_container=container,
        )
    )

    started = {(call.args[0], call.kwargs["instance_id"]) for call in starter.start_new.await_args_list}
    assert ("ThreadSummaryOrchestrator", "thread_summary:u1:t1:4") in started
    assert ("ExtractMemoriesOrchestrator", "extract:u1:t1:4") in started
    # User threshold 20 not crossed by 4 turns.
    assert not any(name == "UserSummaryOrchestrator" for name, _ in started)


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "20",
    },
    clear=False,
)
def test_user_threshold_crossing_starts_user_summary():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    # 20 turns spread across 4 threads: each thread crosses N=4 once;
    # user counter crosses N=20 once at count=20.
    docs: list[dict] = []
    for i in range(20):
        thread_id = f"t{i % 4}"  # 4 distinct threads, 5 turns each
        docs.append(_turn(user_id="u1", thread_id=thread_id))

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    started = {(call.args[0], call.kwargs["instance_id"]) for call in starter.start_new.await_args_list}
    # Exactly one user-summary, deterministic instance id at count=20.
    assert ("UserSummaryOrchestrator", "user_summary:u1:20") in started
    user_summary_starts = [n for n, _ in started if n == "UserSummaryOrchestrator"]
    assert len(user_summary_starts) == 1

    # Payload must include the contributing thread_ids so the orchestrator
    # can scope the user-summary query (avoids a full cross-partition scan).
    us_call = next(c for c in starter.start_new.await_args_list if c.args[0] == "UserSummaryOrchestrator")
    payload = us_call.kwargs["client_input"]
    assert payload["user_id"] == "u1"
    assert sorted(payload["thread_ids"]) == ["t0", "t1", "t2", "t3"]


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",  # disabled
        "FACT_EXTRACTION_EVERY_N": "4",  # enabled
        "USER_SUMMARY_EVERY_N": "0",  # disabled
    },
    clear=False,
)
def test_disabled_thread_summary_does_not_start_summary_orchestrator():
    """``EVERY_N=0`` for one orchestrator must not block the others."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch(
            [_turn() for _ in range(4)],
            starter,
            counter_container=container,
        )
    )

    started = [call.args[0] for call in starter.start_new.await_args_list]
    assert started == ["ExtractMemoriesOrchestrator"]


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "20",
    },
    clear=False,
)
def test_per_thread_grouping_is_correct():
    """Two distinct threads must each get their own counter document and
    independent threshold evaluation."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    docs = [
        _turn(user_id="u1", thread_id="t1"),
        _turn(user_id="u1", thread_id="t1"),
        _turn(user_id="u1", thread_id="t1"),
        _turn(user_id="u1", thread_id="t1"),  # t1 crosses 4
        _turn(user_id="u1", thread_id="t2"),
        _turn(user_id="u1", thread_id="t2"),  # t2 only at 2 — no cross
    ]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    started = {(call.args[0], call.kwargs["instance_id"]) for call in starter.start_new.await_args_list}
    # t1 crossed → summary + extract started for t1 only.
    assert ("ThreadSummaryOrchestrator", "thread_summary:u1:t1:4") in started
    assert ("ExtractMemoriesOrchestrator", "extract:u1:t1:4") in started
    # t2 below threshold — no orchestrators for t2.
    assert not any("t2" in iid for _, iid in started)


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_orchestrator_payload_includes_count_and_ids():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch(
            [_turn() for _ in range(4)],
            starter,
            counter_container=container,
        )
    )

    for call in starter.start_new.await_args_list:
        payload = call.kwargs["client_input"]
        assert payload["user_id"] == "u1"
        assert payload["thread_id"] == "t1"
        assert payload["count"] == 4


# ---------------------------------------------------------------------------
# LSN replay protection
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_lsn_replay_does_not_double_increment():
    """A replayed batch (same LSN) must not double-count nor re-fire orchestrators."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(lsn=10), _turn(lsn=11), _turn(lsn=12), _turn(lsn=13)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))
    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    # Layer 1 (LSN replay dedup): the counter must NOT double-increment.
    assert container._state["thread:u1:t1"]["count"] == 4

    # Layer 2 (deterministic instance ID): on replay the counter helper still
    # reports the same ``(old, new)`` it returned the first time, so the
    # threshold is "crossed" again — but with the IDENTICAL deterministic
    # instance id. Azure Durable Functions then dedups the duplicate
    # ``start_new`` server-side. We assert the determinism here.
    summary_starts = [c for c in starter.start_new.await_args_list if c.args[0] == "ThreadSummaryOrchestrator"]
    assert len(summary_starts) == 2  # same id sent twice — durable dedups
    assert all(c.kwargs["instance_id"] == "thread_summary:u1:t1:4" for c in summary_starts)


# ---------------------------------------------------------------------------
# MEMORY_PROCESSOR_OWNER exclusivity — the change-feed trigger must
# respect the owner env var the same way the SDK auto-trigger does, so a
# shared Cosmos container is processed by exactly one backend.
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "MEMORY_PROCESSOR_OWNER": "inprocess",
        "THREAD_SUMMARY_EVERY_N": "1",
        "FACT_EXTRACTION_EVERY_N": "1",
        "USER_SUMMARY_EVERY_N": "1",
    },
    clear=False,
)
def test_skips_when_owner_inprocess():
    """When the SDK owns processing, the FA must short-circuit:
    no counter writes, no orchestrator starts."""
    # Reset the module-level one-shot guard so the test sees the WARN path
    # deterministically (test isolation across runs).
    import triggers.change_feed as cf

    cf._warned_owner_skip = False

    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(lsn=1), _turn(lsn=2)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    # No counter reads/writes happened.
    container.read_item.assert_not_called()
    container.upsert_item.assert_not_called()
    container.create_item.assert_not_called()
    # No orchestrator starts happened.
    starter.start_new.assert_not_called()


@patch.dict(
    os.environ,
    {
        "MEMORY_PROCESSOR_OWNER": "durable",
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_runs_normally_when_owner_durable():
    """When the FA owns processing, the trigger must run the normal path."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(lsn=1), _turn(lsn=2)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    # Counter was written.
    assert container._state["thread:u1:t1"]["count"] == 2
    # Threshold (2) crossed — orchestrator started.
    summary_starts = [c for c in starter.start_new.await_args_list if c.args[0] == "ThreadSummaryOrchestrator"]
    assert len(summary_starts) == 1


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_runs_normally_when_owner_unset(monkeypatch):
    """When MEMORY_PROCESSOR_OWNER is unset, legacy behavior — FA still runs."""
    monkeypatch.delenv("MEMORY_PROCESSOR_OWNER", raising=False)

    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(lsn=1), _turn(lsn=2)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert container._state["thread:u1:t1"]["count"] == 2
    summary_starts = [c for c in starter.start_new.await_args_list if c.args[0] == "ThreadSummaryOrchestrator"]
    assert len(summary_starts) == 1


# ---------------------------------------------------------------------------
# Reconcile gating (DEDUP_EVERY_N parity with SDK auto-trigger)
# ---------------------------------------------------------------------------


def _extract_payload(call):
    if "client_input" in call.kwargs:
        return call.kwargs["client_input"]
    if len(call.args) >= 3:
        return call.args[2]
    return {}


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "1",
        "USER_SUMMARY_EVERY_N": "0",
        "DEDUP_EVERY_N": "5",
    },
    clear=False,
)
def test_reconcile_flag_set_only_when_n_facts_times_n_dedup_threshold_crosses():
    """Reconcile threshold = FACT_EXTRACTION_EVERY_N * DEDUP_EVERY_N
    (here 1 * 5 = 5). The change-feed signals reconcile via the
    ``reconcile`` flag on the orchestrator payload — never as a separate
    dispatch — so DEDUP_EVERY_N is honored on the FA path."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    # Batch of 4 turns: counter 0 -> 4. Extract crosses (n=1), reconcile (n=5) does not.
    asyncio.run(process_changefeed_batch([_turn() for _ in range(4)], starter, counter_container=container))
    extract_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractMemoriesOrchestrator"]
    assert len(extract_calls) == 1
    assert _extract_payload(extract_calls[0]).get("reconcile") is False

    # Next batch: counter 4 -> 5. Reconcile threshold crossed, so the same
    # extract dispatch carries reconcile=True.
    starter.start_new.reset_mock()
    asyncio.run(process_changefeed_batch([_turn()], starter, counter_container=container))
    extract_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractMemoriesOrchestrator"]
    assert len(extract_calls) == 1
    payload = _extract_payload(extract_calls[0])
    assert payload.get("reconcile") is True
    assert payload.get("user_id") == "u1"


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "1",
        "USER_SUMMARY_EVERY_N": "0",
        "DEDUP_EVERY_N": "0",
    },
    clear=False,
)
def test_dedup_every_n_zero_keeps_reconcile_flag_false():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(process_changefeed_batch([_turn() for _ in range(20)], starter, counter_container=container))
    extract_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractMemoriesOrchestrator"]
    assert extract_calls, "extract should still fire when n_facts > 0"
    assert all(_extract_payload(c).get("reconcile") is False for c in extract_calls)
