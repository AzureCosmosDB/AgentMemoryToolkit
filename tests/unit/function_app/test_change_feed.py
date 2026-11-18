"""Unit tests for ``function_app/triggers/change_feed.py``.

Mocks the durable client's ``start_new`` and the counter container so the
trigger's pure-logic entry point :func:`process_changefeed_batch` can be
exercised without Azure Functions or Cosmos DB.

Every turn names its tenant (``acme`` unless a test says otherwise): counters, instance ids and orchestrator
inputs are per ``(tenant_id, user_id[, thread_id])``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import triggers.change_feed as change_feed_module
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from shared.instance_ids import orchestration_instance_id
from triggers.change_feed import process_changefeed_batch

TENANT = "acme"


@pytest.fixture(autouse=True)
def _default_owner_durable(monkeypatch):
    """The change-feed trigger now requires ``MEMORY_PROCESSOR_OWNER=durable``
    to fire (default-deny). The legacy tests in this file all assume the
    trigger runs, so we set the env var by default; tests that need to
    exercise the skip path override it explicitly with their own
    ``@patch.dict`` or ``monkeypatch``.
    """
    monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "durable")
    change_feed_module._topology_validated = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turn(user_id: str = "u1", thread_id: str = "t1", lsn: int | None = None, tenant_id: str = TENANT) -> dict:
    doc: dict = {"type": "turn", "tenant_id": tenant_id, "user_id": user_id, "thread_id": thread_id}
    if lsn is not None:
        doc["_lsn"] = lsn
    return doc


class _Status:
    """What ``get_status`` returns: falsy for an instance that does not exist, as the Durable client's status is."""

    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def __bool__(self) -> bool:
        return self._exists


def _make_starter(existing: frozenset[str] | set[str] = frozenset()) -> MagicMock:
    """A durable client that knows the ``existing`` instance ids and, like the real one, every id it starts.

    ``get_status`` answers a truthy status for an id it knows and a falsy one for any other, so a test can run a batch
    twice and see the second run find the orchestrations of the first. ``instances`` is the set of ids it knows.
    """
    instances = set(existing)

    async def start_new(orchestration_name, instance_id=None, client_input=None):
        instances.add(instance_id)
        return instance_id

    async def get_status(instance_id):
        return _Status(instance_id in instances)

    starter = MagicMock()
    starter.start_new = AsyncMock(side_effect=start_new)
    starter.get_status = AsyncMock(side_effect=get_status)
    starter.instances = instances
    return starter


def _make_counter_container_starting_at(start_count: int = 0) -> MagicMock:
    """Return a counter container that keeps its documents per logical partition.

    ``read_item`` answers from an in-memory dict keyed by ``(tenant_id, user_id, id)``, that is, by the partition key
    ``[tenant_id, user_id]`` plus the document id, and ``create_item``/``upsert_item`` mutate that same dict, taking
    the partition from the document. A read in another partition than the write finds nothing, as in Cosmos.
    """
    state: dict[tuple[str, str, str], dict] = {}
    if start_count:
        # Seeded entries can be supplied by the test if needed; default empty.
        pass

    container = MagicMock()

    def slot(partition_key, item):
        tenant_id, user_id = partition_key
        return (tenant_id, user_id, item)

    async def read_item(*, item, partition_key):
        key = slot(partition_key, item)
        if key not in state:
            raise CosmosResourceNotFoundError(message="404")
        return dict(state[key])

    async def create_item(*, body):
        state[slot([body["tenant_id"], body["user_id"]], body["id"])] = dict(body)
        return body

    async def upsert_item(*, body, **_kwargs):
        state[slot([body["tenant_id"], body["user_id"]], body["id"])] = dict(body)
        return body

    async def patch_item(*, item, partition_key, patch_operations, filter_predicate=None):
        key = slot(partition_key, item)
        if key not in state:
            raise CosmosResourceNotFoundError(message="404")
        doc = state[key]
        if filter_predicate is not None:
            # Honor the monotonic watermark guard: apply only when the new
            # last_extract_count exceeds the stored one (else Cosmos returns 412).
            new_val = next(
                (op["value"] for op in patch_operations if op.get("path") == "/last_extract_count"),
                None,
            )
            cur = doc.get("last_extract_count")
            if new_val is not None and isinstance(cur, int) and cur >= new_val:
                raise CosmosHttpResponseError(status_code=412, message="precondition failed")
        for op in patch_operations:
            doc[op["path"].lstrip("/")] = op["value"]
        return dict(doc)

    container.read_item = AsyncMock(side_effect=read_item)
    container.create_item = AsyncMock(side_effect=create_item)
    container.upsert_item = AsyncMock(side_effect=upsert_item)
    container.patch_item = AsyncMock(side_effect=patch_item)
    container._state = state  # exposed for assertions
    return container


def _counter(container: MagicMock, counter_id: str, *, tenant_id: str = TENANT, user_id: str = "u1") -> dict:
    """The counter document ``counter_id`` of ``user_id`` in ``tenant_id``."""
    return container._state[(tenant_id, user_id, counter_id)]


def _thread_instance_id(
    kind: str,
    count: int,
    *,
    tenant_id: str = TENANT,
    user_id: str = "u1",
    thread_id: str = "t1",
    lsn: int | None = None,
) -> str:
    """The instance id of a thread-scoped orchestration: a hash of the tenant, user, thread and LSN, and the count."""
    return orchestration_instance_id(kind, tenant_id, user_id, thread_id, count=count, lsn=lsn)


def _user_instance_id(count: int, *, tenant_id: str = TENANT, user_id: str = "u1", lsn: int | None = None) -> str:
    """The instance id of a user summary: a hash of the tenant, user and batch LSN, and the count."""
    return orchestration_instance_id("user_summary", tenant_id, user_id, count=count, lsn=lsn)


def _fail_counter_reads(container: MagicMock, error: BaseException, *counters: tuple[str, str, str]) -> None:
    """Make reading any of the ``(tenant_id, user_id, counter_id)`` counters raise ``error``, as a failing call would.

    Every other counter is read as before, so the real counter code runs and only the named counters fail.
    """
    read_item = container.read_item.side_effect
    doomed = set(counters)

    async def failing_read_item(*, item, partition_key):
        tenant_id, user_id = partition_key
        if (tenant_id, user_id, item) in doomed:
            raise error
        return await read_item(item=item, partition_key=partition_key)

    container.read_item = AsyncMock(side_effect=failing_read_item)


def _make_topology_db(missing_name: str | None = None, earlier_key_name: str | None = None):
    containers: dict[str, MagicMock] = {}
    for name in (
        change_feed_module.config.TURNS_CONTAINER,
        change_feed_module.config.MEMORIES_CONTAINER,
        change_feed_module.config.SUMMARIES_CONTAINER,
        change_feed_module.config.COUNTERS_CONTAINER,
    ):
        container = MagicMock()
        if name == missing_name:
            container.read = AsyncMock(side_effect=CosmosResourceNotFoundError(message="missing"))
        else:
            paths = ["/user_id", "/thread_id"] if name == earlier_key_name else ["/tenant_id", "/user_id"]
            container.read = AsyncMock(return_value={"id": name, "partitionKey": {"paths": paths, "kind": "MultiHash"}})
        containers[name] = container

    db = MagicMock()
    db.get_container_client.side_effect = lambda name: containers[name]
    return db, containers


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------


def test_ensure_topology_succeeds_when_all_containers_exist():
    change_feed_module._topology_validated = False
    db, containers = _make_topology_db()

    with patch("shared.cosmos_clients.get_cosmos_database_async", new=AsyncMock(return_value=db)):
        asyncio.run(change_feed_module._ensure_topology())

    assert change_feed_module._topology_validated is True
    assert db.get_container_client.call_count == 4
    for container in containers.values():
        container.read.assert_awaited_once()


def test_ensure_topology_raises_on_missing_container():
    change_feed_module._topology_validated = False
    missing = change_feed_module.config.SUMMARIES_CONTAINER
    db, _containers = _make_topology_db(missing_name=missing)

    with patch("shared.cosmos_clients.get_cosmos_database_async", new=AsyncMock(return_value=db)):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(change_feed_module._ensure_topology())

    assert missing in str(exc_info.value)
    assert "does not exist" in str(exc_info.value)
    assert change_feed_module._topology_validated is False


@pytest.mark.parametrize(
    "earlier",
    [
        change_feed_module.config.TURNS_CONTAINER,
        change_feed_module.config.MEMORIES_CONTAINER,
        change_feed_module.config.SUMMARIES_CONTAINER,
        change_feed_module.config.COUNTERS_CONTAINER,
    ],
)
def test_ensure_topology_raises_on_a_container_with_the_earlier_key(earlier):
    change_feed_module._topology_validated = False
    db, _containers = _make_topology_db(earlier_key_name=earlier)

    with patch("shared.cosmos_clients.get_cosmos_database_async", new=AsyncMock(return_value=db)):
        with pytest.raises(RuntimeError, match=rf"Topology error: container '{earlier}' has partition key") as error:
            asyncio.run(change_feed_module._ensure_topology())

    assert "['/user_id', '/thread_id']" in str(error.value)
    assert "recreate" in str(error.value)
    assert change_feed_module._topology_validated is False


def test_ensure_topology_is_idempotent():
    change_feed_module._topology_validated = False
    db, containers = _make_topology_db()
    get_db = AsyncMock(return_value=db)

    async def run_twice():
        await change_feed_module._ensure_topology()
        await change_feed_module._ensure_topology()

    with patch("shared.cosmos_clients.get_cosmos_database_async", new=get_db):
        asyncio.run(run_twice())

    assert get_db.await_count == 1
    assert sum(container.read.await_count for container in containers.values()) == 4


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "2",
    },
    clear=False,
)
def test_a_batch_counts_nothing_when_a_container_has_the_earlier_key(monkeypatch):
    """The batch checks the topology before it counts: with a container keyed ``[/user_id, /thread_id]`` no counter is
    read or written and no orchestration starts, though these two turns would cross both thresholds."""
    monkeypatch.setattr(change_feed_module, "_topology_validated", False)
    memories = change_feed_module.config.MEMORIES_CONTAINER
    db, _containers = _make_topology_db(earlier_key_name=memories)
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    with patch("shared.cosmos_clients.get_cosmos_database_async", new=AsyncMock(return_value=db)):
        with pytest.raises(RuntimeError, match=rf"Topology error: container '{re.escape(memories)}' has partition key"):
            asyncio.run(process_changefeed_batch([_turn(), _turn()], starter, counter_container=container))

    container.read_item.assert_not_called()
    container.create_item.assert_not_called()
    container.upsert_item.assert_not_called()
    starter.start_new.assert_not_awaited()
    # A failed check is not remembered, so the next batch looks at the containers again.
    assert change_feed_module._topology_validated is False


# ---------------------------------------------------------------------------
# Filtering / disabled-mode behaviour
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
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
    (fact=2, thread=10, user=20), not 0.

    Regression for the silent-no-op out-of-the-box deploy bug: a missing
    setting should NOT disable the orchestrator (only an explicit "0" does).
    """
    for name in ("THREAD_SUMMARY_EVERY_N", "FACT_EXTRACTION_EVERY_N", "USER_SUMMARY_EVERY_N"):
        monkeypatch.delenv(name, raising=False)

    starter = _make_starter()
    container = _make_counter_container_starting_at()

    # 2 turns => crosses fact extraction (n=2) only.
    asyncio.run(
        process_changefeed_batch(
            [_turn(), _turn()],
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
        {"type": "thread_summary", "tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"},
        {"type": "fact", "tenant_id": TENANT, "user_id": "u1", "thread_id": "t1"},
        {"type": "user_summary", "tenant_id": TENANT, "user_id": "u1", "thread_id": "__user_summary__"},
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
    assert _counter(container, "thread:u1:t1")["count"] == 3


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
        {"type": "turn", "tenant_id": TENANT, "user_id": "u1"},  # missing thread_id
        {"type": "turn", "tenant_id": TENANT, "thread_id": "t1"},  # missing user_id
    ]
    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))
    assert starter.start_new.await_count == 0
    assert container.read_item.await_count == 0


_NO_TENANT = object()


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
@pytest.mark.parametrize(
    "tenant_id",
    [
        pytest.param(_NO_TENANT, id="missing"),
        pytest.param(None, id="null"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(7, id="not-a-string"),
        pytest.param("acme/eu", id="slash"),
        pytest.param("acme#eu", id="hash"),
        pytest.param("x" * 300, id="too-long"),
    ],
)
def test_a_turn_without_a_valid_tenant_is_skipped_with_a_warning_that_quotes_nothing(tenant_id, caplog):
    """The tenant is never guessed: a turn that names none, or an unusable one, is counted nowhere (not under the
    default tenant either) and the valid turns of the same batch still are."""
    caplog.set_level(logging.WARNING)
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    unusable = {"type": "turn", "user_id": "u2", "thread_id": "t2", "content": "private text"}
    if tenant_id is not _NO_TENANT:
        unusable["tenant_id"] = tenant_id

    asyncio.run(process_changefeed_batch([unusable, unusable, _turn(), _turn()], starter, counter_container=container))

    assert list(container._state) == [(TENANT, "u1", "thread:u1:t1")]
    assert [(c.args[0], c.kwargs["instance_id"]) for c in starter.start_new.await_args_list] == [
        ("ThreadSummaryOrchestrator", _thread_instance_id("thread_summary", 2))
    ]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["change-feed: turn doc missing a valid tenant_id, user_id or thread_id, skipping"] * 2


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
@pytest.mark.parametrize("owner", ["", "   ", 7, "a/b", "a#b", "o" * 300], ids=repr)
def test_a_turn_with_an_unsupported_owner_is_skipped_like_one_without_a_tenant(owner, caplog):
    """An owner the SDK would refuse is never counted; the valid turns of the same batch still are."""
    caplog.set_level(logging.WARNING)
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    unusable = {"type": "turn", "tenant_id": TENANT, "user_id": owner, "thread_id": "t2", "content": "private text"}

    asyncio.run(process_changefeed_batch([unusable, _turn(), _turn()], starter, counter_container=container))

    assert list(container._state) == [(TENANT, "u1", "thread:u1:t1")]
    assert [(c.args[0], c.kwargs["instance_id"]) for c in starter.start_new.await_args_list] == [
        ("ThreadSummaryOrchestrator", _thread_instance_id("thread_summary", 2))
    ]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["change-feed: turn doc missing a valid tenant_id, user_id or thread_id, skipping"]


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "1",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "2",
    },
    clear=False,
)
@pytest.mark.parametrize(
    "thread_id",
    [
        pytest.param(7, id="integer"),
        pytest.param(["t1"], id="list"),
        pytest.param(None, id="null"),
        pytest.param("", id="empty"),
        pytest.param("t1 ", id="trailing-space"),
        pytest.param(" t1", id="leading-space"),
        pytest.param("a/b", id="slash"),
        pytest.param("t\x7f1", id="control-character"),
        pytest.param("t\u20281", id="line-separator"),
        pytest.param("t\ud8001", id="lone-surrogate"),
        pytest.param("t" * 257, id="too-long"),
    ],
)
def test_a_turn_with_an_unsupported_thread_is_skipped_and_the_batch_still_completes(thread_id, caplog):
    """A thread id the SDK would refuse is skipped and the batch completes: an integer one used to make the instance
    id raise, which lost the counts of every tenant in the batch.

    The turn counts nowhere, not even in the counters of its owner, who has valid turns too. The thread summary
    threshold is 1, so counting the turn would start an orchestration for its thread at once.
    """
    caplog.set_level(logging.WARNING)
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    # First in the batch, so a failure on it would cost the turns after it.
    unusable = {"type": "turn", "tenant_id": TENANT, "user_id": "u1", "thread_id": thread_id, "content": "private text"}
    docs = [unusable, *(_turn(tenant_id=tenant_id) for tenant_id in ("acme", "globex", "acme", "globex"))]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert {key: doc["count"] for key, doc in container._state.items()} == {
        (tenant_id, "u1", counter_id): 2
        for tenant_id in ("acme", "globex")
        for counter_id in ("thread:u1:t1", "user:u1")
    }
    started = {c.kwargs["instance_id"]: c.kwargs["client_input"] for c in starter.start_new.await_args_list}
    assert set(started) == {
        _thread_instance_id("thread_summary", 2, tenant_id="acme"),
        _thread_instance_id("thread_summary", 2, tenant_id="globex"),
        _user_instance_id(2, tenant_id="acme"),
        _user_instance_id(2, tenant_id="globex"),
    }
    assert started[_user_instance_id(2, tenant_id="acme")]["thread_ids"] == ["t1"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["change-feed: turn doc missing a valid tenant_id, user_id or thread_id, skipping"]


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        pytest.param({"thread_id": "t1"}, "t1", id="plain"),
        pytest.param({"thread_id": "support chat: order 7"}, "support chat: order 7", id="space-and-colon-inside"),
        pytest.param({"thread_id": "ünï©ødé"}, "ünï©ødé", id="non-ascii"),
        pytest.param({"thread_id": "é" * 128}, "é" * 128, id="256-bytes"),
        pytest.param({"thread_id": "é" * 129}, None, id="258-bytes-in-129-characters"),
        pytest.param({}, None, id="missing"),
    ],
)
def test_thread_of_accepts_the_thread_ids_the_sdk_accepts_and_counts_the_limit_in_bytes(doc, expected):
    assert change_feed_module._thread_of(doc) == expected


# ---------------------------------------------------------------------------
# Threshold crossing → deterministic instance IDs
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "EPISODE_EVAL_EVERY_N": "4",
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
    assert ("ThreadSummaryOrchestrator", _thread_instance_id("thread_summary", 4)) in started
    assert ("ExtractMemoriesOrchestrator", _thread_instance_id("extract", 4)) in started
    assert ("ExtractEpisodesOrchestrator", _thread_instance_id("episode", 4)) in started
    # User threshold 20 not crossed by 4 turns.
    assert not any(name == "UserSummaryOrchestrator" for name, _ in started)


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_episode_threshold_crossing_starts_extract_episodes():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch(
            [_turn() for _ in range(4)],
            starter,
            counter_container=container,
        )
    )

    episode_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractEpisodesOrchestrator"]
    assert len(episode_calls) == 1
    call = episode_calls[0]
    assert call.kwargs["instance_id"] == _thread_instance_id("episode", 4)
    assert call.kwargs["client_input"] == {"tenant_id": TENANT, "user_id": "u1", "thread_id": "t1", "count": 4}


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
    assert ("UserSummaryOrchestrator", _user_instance_id(20)) in started
    user_summary_starts = [n for n, _ in started if n == "UserSummaryOrchestrator"]
    assert len(user_summary_starts) == 1

    # Payload carries the contributing thread_ids.
    us_call = next(c for c in starter.start_new.await_args_list if c.args[0] == "UserSummaryOrchestrator")
    payload = us_call.kwargs["client_input"]
    assert payload["tenant_id"] == TENANT
    assert payload["user_id"] == "u1"
    assert sorted(payload["thread_ids"]) == ["t0", "t1", "t2", "t3"]


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",  # disabled
        "FACT_EXTRACTION_EVERY_N": "4",  # enabled
        "EPISODE_EVAL_EVERY_N": "0",  # disabled
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
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "4",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_disabled_episode_does_not_start_extract_episodes():
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
    assert "ExtractMemoriesOrchestrator" in started
    assert "ExtractEpisodesOrchestrator" not in started


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "EPISODE_EVAL_EVERY_N": "4",
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
        _turn(user_id="u1", thread_id="t2"),  # t2 only at 2 - no cross
    ]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    started = {(call.args[0], call.kwargs["instance_id"]) for call in starter.start_new.await_args_list}
    # t1 crossed → summary, extract and episode started for t1 only; t2 is below every threshold.
    assert started == {
        ("ThreadSummaryOrchestrator", _thread_instance_id("thread_summary", 4)),
        ("ExtractMemoriesOrchestrator", _thread_instance_id("extract", 4)),
        ("ExtractEpisodesOrchestrator", _thread_instance_id("episode", 4)),
    }


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
        assert payload["tenant_id"] == TENANT
        assert payload["user_id"] == "u1"
        assert payload["thread_id"] == "t1"
        assert payload["count"] == 4


# ---------------------------------------------------------------------------
# Tenants: every scope is per (tenant_id, user_id[, thread_id])
# ---------------------------------------------------------------------------


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "EPISODE_EVAL_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "4",
    },
    clear=False,
)
def test_every_orchestration_is_started_with_a_hashed_instance_id_and_the_tenant_in_its_input():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch([_turn(tenant_id="globex") for _ in range(4)], starter, counter_container=container)
    )

    calls = starter.start_new.await_args_list
    started = {c.args[0]: (c.kwargs["instance_id"], c.kwargs["client_input"]) for c in calls}
    instance_ids = {name: instance_id for name, (instance_id, _) in started.items()}
    assert instance_ids == {
        "ThreadSummaryOrchestrator": _thread_instance_id("thread_summary", 4, tenant_id="globex"),
        "ExtractMemoriesOrchestrator": _thread_instance_id("extract", 4, tenant_id="globex"),
        "ExtractEpisodesOrchestrator": _thread_instance_id("episode", 4, tenant_id="globex"),
        "UserSummaryOrchestrator": _user_instance_id(4, tenant_id="globex"),
    }
    # An id names the kind and the count and hashes the rest: no tenant, user or thread in it.
    assert all(re.fullmatch(r"[a-z_]+:[0-9a-f]{32}:4", instance_id) for instance_id in instance_ids.values())
    assert all(client_input["tenant_id"] == "globex" for _, client_input in started.values())


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "4",
        "EPISODE_EVAL_EVERY_N": "4",
        "USER_SUMMARY_EVERY_N": "4",
    },
    clear=False,
)
def test_owners_with_very_long_ids_still_get_instance_ids_within_the_durable_limit():
    """Durable Functions takes at most 100 characters; ids of 200 characters each used to overflow it."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(user_id="u" * 200, thread_id="h" * 200, tenant_id="t" * 200) for _ in range(4)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    instance_ids = [c.kwargs["instance_id"] for c in starter.start_new.await_args_list]
    assert len(instance_ids) == 4
    assert all(re.fullmatch(r"[a-z_]+:[0-9a-f]{32}:4", instance_id) for instance_id in instance_ids)


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_the_log_of_a_start_names_the_owner_because_the_instance_id_does_not(caplog):
    caplog.set_level(logging.INFO)
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(process_changefeed_batch([_turn(), _turn()], starter, counter_container=container))

    instance_id = _thread_instance_id("thread_summary", 2)
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("change-feed: starting")] == [
        f"change-feed: starting ThreadSummaryOrchestrator instance={instance_id} tenant=acme user=u1 thread=t1"
    ]


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "4",
    },
    clear=False,
)
def test_the_same_user_and_thread_in_two_tenants_are_counted_separately():
    """Two turns per tenant never add up to the four that start an orchestration."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(tenant_id="acme"), _turn(tenant_id="globex"), _turn(tenant_id="acme"), _turn(tenant_id="globex")]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert starter.start_new.await_count == 0
    for tenant_id in ("acme", "globex"):
        assert _counter(container, "thread:u1:t1", tenant_id=tenant_id)["count"] == 2
        assert _counter(container, "user:u1", tenant_id=tenant_id)["count"] == 2


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_each_tenant_that_crosses_a_threshold_starts_its_own_orchestration():
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(tenant_id="acme") for _ in range(4)] + [_turn(tenant_id="globex") for _ in range(4)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    started = {c.kwargs["instance_id"]: c.kwargs["client_input"] for c in starter.start_new.await_args_list}
    assert started == {
        _thread_instance_id("thread_summary", 4, tenant_id="acme"): {
            "tenant_id": "acme",
            "user_id": "u1",
            "thread_id": "t1",
            "count": 4,
        },
        _thread_instance_id("thread_summary", 4, tenant_id="globex"): {
            "tenant_id": "globex",
            "user_id": "u1",
            "thread_id": "t1",
            "count": 4,
        },
    }


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "2",
    },
    clear=False,
)
def test_user_summary_thread_ids_are_collected_per_tenant():
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [
        _turn(tenant_id="acme", thread_id="t1"),
        _turn(tenant_id="globex", thread_id="t9"),
        _turn(tenant_id="acme", thread_id="t2"),
        _turn(tenant_id="globex", thread_id="t9"),
    ]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    started = {c.kwargs["instance_id"]: c.kwargs["client_input"] for c in starter.start_new.await_args_list}
    assert started == {
        _user_instance_id(2, tenant_id="acme"): {
            "tenant_id": "acme",
            "user_id": "u1",
            "count": 2,
            "thread_ids": ["t1", "t2"],
        },
        _user_instance_id(2, tenant_id="globex"): {
            "tenant_id": "globex",
            "user_id": "u1",
            "count": 2,
            "thread_ids": ["t9"],
        },
    }


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "100",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "100",
    },
    clear=False,
)
def test_counters_live_in_the_owner_partition_and_carry_the_tenant():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch([_turn(thread_id="t1"), _turn(thread_id="t2")], starter, counter_container=container)
    )

    assert {tuple(call.kwargs["partition_key"]) for call in container.read_item.await_args_list} == {(TENANT, "u1")}
    thread_counter = _counter(container, "thread:u1:t1")
    user_counter = _counter(container, "user:u1")
    assert (thread_counter["tenant_id"], thread_counter["user_id"], thread_counter["thread_id"]) == (TENANT, "u1", "t1")
    assert (user_counter["tenant_id"], user_counter["user_id"], user_counter["thread_id"]) == (
        TENANT,
        "u1",
        "__counters__",
    )
    assert set(container._state) == {
        (TENANT, "u1", "thread:u1:t1"),
        (TENANT, "u1", "thread:u1:t2"),
        (TENANT, "u1", "user:u1"),
    }


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
    assert _counter(container, "thread:u1:t1")["count"] == 4

    # Layer 2 (existing instance): on replay the counter helper still reports the same ``(old, new)`` it returned the
    # first time, so the threshold is "crossed" again - with the IDENTICAL instance id, whose instance the
    # first run started. The replay finds it and does not start it again.
    summary_starts = [c for c in starter.start_new.await_args_list if c.args[0] == "ThreadSummaryOrchestrator"]
    assert len(summary_starts) == 1
    assert summary_starts[0].kwargs["instance_id"] == _thread_instance_id("thread_summary", 4, lsn=13)


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "100",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "100",
    },
    clear=False,
)
def test_lsns_are_tracked_per_tenant_and_a_replay_counts_nothing_twice():
    """One batch can mix tenants; each tenant's counters record only the LSNs of its own turns."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [
        _turn(tenant_id="acme", lsn=10),
        _turn(tenant_id="globex", lsn=50),
        _turn(tenant_id="acme", lsn=11),
    ]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))
    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    for counter_id in ("thread:u1:t1", "user:u1"):
        acme = _counter(container, counter_id, tenant_id="acme")
        globex = _counter(container, counter_id, tenant_id="globex")
        assert (acme["count"], acme["last_batch_lsn"]) == (2, 11)
        assert (globex["count"], globex["last_batch_lsn"]) == (1, 50)


# ---------------------------------------------------------------------------
# A failing thread or owner does not stop the others in the batch; the batch fails at the end
# ---------------------------------------------------------------------------

_EVERY_TWO_TURNS = {
    "THREAD_SUMMARY_EVERY_N": "2",
    "FACT_EXTRACTION_EVERY_N": "0",
    "EPISODE_EVAL_EVERY_N": "0",
    "USER_SUMMARY_EVERY_N": "2",
}

# What ``_batch_of_three_threads`` leaves in the counters, and the instance ids of the orchestrations it starts: every
# thread crosses a threshold of 2, and so does each owner, so each of the five counters starts one orchestration.
_BATCH_COUNTS = {
    ("acme", "u1", "thread:u1:t1"): 2,
    ("acme", "u1", "thread:u1:t2"): 2,
    ("acme", "u1", "user:u1"): 4,
    ("globex", "u1", "thread:u1:t1"): 2,
    ("globex", "u1", "user:u1"): 2,
}
_BATCH_START_IDS = {
    _thread_instance_id("thread_summary", 2, tenant_id="acme", thread_id="t1"),
    _thread_instance_id("thread_summary", 2, tenant_id="acme", thread_id="t2"),
    _user_instance_id(4, tenant_id="acme"),
    _thread_instance_id("thread_summary", 2, tenant_id="globex", thread_id="t1"),
    _user_instance_id(2, tenant_id="globex"),
}


def _batch_of_three_threads() -> list[dict]:
    """Two turns each for acme's t1 and t2 and for globex's t1, all of owner u1; acme's thread comes first."""
    return [
        _turn(tenant_id=tenant_id, thread_id=thread_id)
        for tenant_id, thread_id in (("acme", "t1"), ("acme", "t2"), ("globex", "t1"))
        for _ in range(2)
    ]


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
@pytest.mark.parametrize(
    ("failing", "lost_start"),
    [
        pytest.param(
            (TENANT, "u1", "thread:u1:t1"), _thread_instance_id("thread_summary", 2, thread_id="t1"), id="thread"
        ),
        pytest.param((TENANT, "u1", "user:u1"), _user_instance_id(4), id="owner"),
    ],
)
def test_a_counter_that_cannot_be_updated_does_not_stop_the_rest_of_the_batch(failing, lost_start, caplog):
    """The first thread, or the first owner, cannot be updated: every other counter is still written and every other
    threshold crossing still starts its orchestration, and only then does the batch fail."""
    caplog.set_level(logging.INFO)
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    _fail_counter_reads(container, CosmosHttpResponseError(status_code=503, message="private text"), failing)
    summary = r"^Failed to update 1 counter\(s\): azure\.cosmos\.CosmosHttpResponseError\(status=503\) at "

    with pytest.raises(RuntimeError, match=summary) as failed:
        asyncio.run(process_changefeed_batch(_batch_of_three_threads(), starter, counter_container=container))

    assert {key: doc["count"] for key, doc in container._state.items()} == {
        key: count for key, count in _BATCH_COUNTS.items() if key != failing
    }
    assert {c.kwargs["instance_id"] for c in starter.start_new.await_args_list} == _BATCH_START_IDS - {lost_start}
    # The failure is logged once as its type and code location. Neither the log nor the error the batch raises quotes
    # the SDK error's text or chains to it.
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.exc_info is None
    assert record.getMessage().startswith(
        "change-feed: failed to update a counter (exc_type=CosmosHttpResponseError): "
        "azure.cosmos.CosmosHttpResponseError(status=503) at "
    )
    assert (failed.value.__cause__, failed.value.__context__) == (None, None)
    assert "private text" not in caplog.text + "".join(traceback.format_exception(failed.value))


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "2",
        "FACT_EXTRACTION_EVERY_N": "2",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_a_failure_after_a_thread_was_counted_does_not_stop_the_other_threads(monkeypatch):
    """The whole work of a thread is isolated, not only its counter update: acme's watermark read fails after its
    counter was written and its summary started, and globex is still extracted."""
    read_extract_watermark = change_feed_module.read_extract_watermark

    async def failing_for_acme(container, tenant_id, counter_id, user_id):
        if tenant_id == "acme":
            raise CosmosHttpResponseError(status_code=503, message="throttled")
        return await read_extract_watermark(container, tenant_id, counter_id, user_id)

    monkeypatch.setattr(change_feed_module, "read_extract_watermark", failing_for_acme)
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(tenant_id=tenant_id) for tenant_id in ("acme", "acme", "globex", "globex")]

    with pytest.raises(RuntimeError, match=r"^Failed to update 1 counter\(s\): "):
        asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert {key: doc["count"] for key, doc in container._state.items()} == {
        ("acme", "u1", "thread:u1:t1"): 2,
        ("globex", "u1", "thread:u1:t1"): 2,
    }
    assert [(c.args[0], c.kwargs["client_input"]["tenant_id"]) for c in starter.start_new.await_args_list] == [
        ("ThreadSummaryOrchestrator", "acme"),
        ("ThreadSummaryOrchestrator", "globex"),
        ("ExtractMemoriesOrchestrator", "globex"),
    ]


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
def test_failed_starts_and_failed_counters_of_one_batch_are_reported_in_one_error():
    starter = _make_starter()
    starter.start_new.side_effect = ConnectionError("starter unavailable")
    container = _make_counter_container_starting_at()
    _fail_counter_reads(
        container,
        CosmosHttpResponseError(status_code=503, message="throttled"),
        (TENANT, "u1", "thread:u1:t1"),
        ("globex", "u1", "user:u1"),
    )

    with pytest.raises(RuntimeError) as failed:
        asyncio.run(process_changefeed_batch(_batch_of_three_threads(), starter, counter_container=container))

    # Of the five orchestrations, three are still attempted (acme's t2 and owner, globex's t1), and none starts.
    assert starter.start_new.await_count == 3
    assert re.fullmatch(
        r"Failed to start 3 orchestration\(s\): ConnectionError at .+; "
        r"Failed to update 2 counter\(s\): azure\.cosmos\.CosmosHttpResponseError\(status=503\) at .+",
        str(failed.value),
    )


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
@pytest.mark.parametrize(
    "cancelled", [(TENANT, "u1", "thread:u1:t1"), (TENANT, "u1", "user:u1")], ids=["thread", "owner"]
)
def test_a_cancellation_during_a_counter_update_is_not_swallowed(cancelled):
    """Only an ``Exception`` is recorded and isolated per thread and owner: a ``CancelledError`` stops the batch."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    _fail_counter_reads(container, asyncio.CancelledError(), cancelled)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_changefeed_batch(_batch_of_three_threads(), starter, counter_container=container))


# ---------------------------------------------------------------------------
# A retried batch repeats only what failed
# ---------------------------------------------------------------------------


def _batch_of_three_threads_with_lsns() -> list[dict]:
    """``_batch_of_three_threads`` with the ``_lsn`` the change feed gives each turn: 100 to 105 in listed order."""
    docs = _batch_of_three_threads()
    for lsn, doc in enumerate(docs, start=100):
        doc["_lsn"] = lsn
    return docs


# The ids of the five orchestrations that batch starts. Each names the highest LSN of the turns it counted.
_LSN_BATCH_START_IDS = {
    _thread_instance_id("thread_summary", 2, tenant_id="acme", thread_id="t1", lsn=101),
    _thread_instance_id("thread_summary", 2, tenant_id="acme", thread_id="t2", lsn=103),
    _user_instance_id(4, tenant_id="acme", lsn=103),
    _thread_instance_id("thread_summary", 2, tenant_id="globex", thread_id="t1", lsn=105),
    _user_instance_id(2, tenant_id="globex", lsn=105),
}


def _counts(container: MagicMock) -> dict[tuple[str, str, str], int]:
    return {key: doc["count"] for key, doc in container._state.items()}


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
def test_a_replayed_batch_starts_no_orchestration_whose_instance_exists():
    """The host runs a failed batch again with the same documents. Every counter reports the threshold crossing it
    reported the first time, and each orchestration it names is found and left alone, although Durable Functions
    would accept the id of an instance that has finished and run it again."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = _batch_of_three_threads_with_lsns()

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))
    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert starter.start_new.await_count == 5
    assert starter.instances == _LSN_BATCH_START_IDS
    assert _counts(container) == _BATCH_COUNTS


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
def test_a_retry_starts_only_the_orchestrations_the_failed_attempt_did_not_start():
    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = _batch_of_three_threads_with_lsns()
    refused = _thread_instance_id("thread_summary", 2, tenant_id="acme", thread_id="t2", lsn=103)
    start = starter.start_new.side_effect

    async def start_but_refuse_one(orchestration_name, instance_id=None, client_input=None):
        if instance_id == refused:
            raise ConnectionError("starter unavailable")
        return await start(orchestration_name, instance_id=instance_id, client_input=client_input)

    starter.start_new.side_effect = start_but_refuse_one
    with pytest.raises(RuntimeError, match=r"^Failed to start 1 orchestration\(s\): "):
        asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))
    assert starter.instances == _LSN_BATCH_START_IDS - {refused}

    # The retry: the starter works again.
    starter.start_new.side_effect = start
    starter.start_new.reset_mock()
    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    assert [c.kwargs["instance_id"] for c in starter.start_new.await_args_list] == [refused]
    assert starter.instances == _LSN_BATCH_START_IDS
    assert _counts(container) == _BATCH_COUNTS


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "4",
        "FACT_EXTRACTION_EVERY_N": "0",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    },
    clear=False,
)
def test_a_count_reached_again_after_the_counter_expired_starts_a_new_orchestration():
    """A counter is deleted 90 days after its last turn, so a thread that was idle that long counts from 1 again and
    reaches the same count in a later batch. The instance of the first time still exists, and the later batch must
    not take it for its own."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(
        process_changefeed_batch([_turn(lsn=lsn) for lsn in range(10, 14)], starter, counter_container=container)
    )
    container._state.clear()
    asyncio.run(
        process_changefeed_batch([_turn(lsn=lsn) for lsn in range(500, 504)], starter, counter_container=container)
    )

    assert [c.kwargs["instance_id"] for c in starter.start_new.await_args_list] == [
        _thread_instance_id("thread_summary", 4, lsn=13),
        _thread_instance_id("thread_summary", 4, lsn=503),
    ]


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
def test_a_status_that_cannot_be_read_fails_the_batch_and_starts_nothing():
    """An orchestration whose instance cannot be looked up is not started blind, which could run a finished one again:
    the batch fails, and the host retries it."""
    starter = _make_starter()
    starter.get_status.side_effect = ConnectionError("status endpoint unavailable")
    container = _make_counter_container_starting_at()

    with pytest.raises(RuntimeError, match=r"^Failed to start 5 orchestration\(s\): ConnectionError at "):
        asyncio.run(process_changefeed_batch(_batch_of_three_threads_with_lsns(), starter, counter_container=container))

    starter.start_new.assert_not_awaited()
    assert _counts(container) == _BATCH_COUNTS


def _conflict_with_a_status_code() -> Exception:
    conflict = Exception("conflict")
    conflict.status_code = 409
    return conflict


@patch.dict(os.environ, _EVERY_TWO_TURNS, clear=False)
@pytest.mark.parametrize(
    "conflict",
    [
        pytest.param(_conflict_with_a_status_code(), id="status-code"),
        pytest.param(Exception("An Orchestration instance with the status Pending already exists."), id="message"),
    ],
)
def test_an_instance_that_is_starting_is_skipped_not_an_error(conflict):
    """The status of an instance that has not started running yet reads as missing, and Durable Functions then refuses
    to start it a second time. That refusal is a skip, not a failed batch."""
    starter = _make_starter()
    starter.start_new.side_effect = conflict
    container = _make_counter_container_starting_at()

    asyncio.run(process_changefeed_batch(_batch_of_three_threads_with_lsns(), starter, counter_container=container))

    assert starter.start_new.await_count == 5
    assert _counts(container) == _BATCH_COUNTS


def test_the_trigger_retries_a_failed_batch_a_bounded_number_of_times():
    policy = change_feed_module.on_memory_change.retry_policy

    assert policy["strategy"] == "exponential_backoff"
    # The host takes the count as a string, and -1 would retry for ever.
    assert 1 <= int(policy["max_retry_count"]) <= 10


# ---------------------------------------------------------------------------
# MEMORY_PROCESSOR_OWNER exclusivity - the change-feed trigger must
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
    assert _counter(container, "thread:u1:t1")["count"] == 2
    # Threshold (2) crossed - orchestrator started.
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
def test_skips_when_owner_unset(monkeypatch):
    """Default-deny: the FA must NOT fire when MEMORY_PROCESSOR_OWNER is unset.

    This is the protection against the day-one footgun where a customer
    deploys the Function App next to an existing SDK install without
    configuring the env var - without this guard both backends would
    race on the same writes.
    """
    monkeypatch.delenv("MEMORY_PROCESSOR_OWNER", raising=False)

    starter = _make_starter()
    container = _make_counter_container_starting_at()
    docs = [_turn(lsn=1), _turn(lsn=2)]

    asyncio.run(process_changefeed_batch(docs, starter, counter_container=container))

    # Trigger short-circuited before touching the counter or starting orchestrators.
    container.read_item.assert_not_called()
    container.upsert_item.assert_not_called()
    container.create_item.assert_not_called()
    starter.start_new.assert_not_called()


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
    ``reconcile`` flag on the orchestrator payload - never as a separate
    dispatch - so DEDUP_EVERY_N is honored on the FA path."""
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    # Batch of 4 turns: counter 0 -> 4. Extract crosses (n=1), reconcile (n=5) does not.
    asyncio.run(process_changefeed_batch([_turn() for _ in range(4)], starter, counter_container=container))
    extract_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractMemoriesOrchestrator"]
    assert len(extract_calls) == 1
    first_payload = _extract_payload(extract_calls[0])
    assert first_payload.get("reconcile") is False
    # Bootstrap: no watermark yet -> recent_k spans all 4 turns so far.
    assert first_payload.get("recent_k") == 4

    # Simulate the extract orchestrator's em_AdvanceExtractWatermark activity
    # advancing the watermark to the processed count (4) after batch 1.
    from shared.counters import advance_extract_watermark, thread_counter_id

    asyncio.run(advance_extract_watermark(container, TENANT, thread_counter_id("u1", "t1"), "u1", 4))

    # Next batch: counter 4 -> 5. Reconcile threshold crossed, so the same
    # extract dispatch carries reconcile=True, and recent_k = 5 - watermark(4) = 1.
    starter.start_new.reset_mock()
    asyncio.run(process_changefeed_batch([_turn()], starter, counter_container=container))
    extract_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractMemoriesOrchestrator"]
    assert len(extract_calls) == 1
    payload = _extract_payload(extract_calls[0])
    assert payload.get("reconcile") is True
    assert payload.get("tenant_id") == TENANT
    assert payload.get("user_id") == "u1"
    assert payload.get("recent_k") == 1


def test_advance_extract_watermark_is_monotonic():
    """A later, out-of-order run must not regress the extract watermark below a
    higher value a concurrent run already wrote (that would re-extract turns)."""
    from shared.counters import advance_extract_watermark, increment_counter_by, thread_counter_id

    container = _make_counter_container_starting_at()
    cid = thread_counter_id("u1", "t1")
    asyncio.run(increment_counter_by(container, TENANT, cid, "u1", "t1", 12))

    asyncio.run(advance_extract_watermark(container, TENANT, cid, "u1", 8))
    assert _counter(container, cid)["last_extract_count"] == 8

    # An out-of-order run with a lower count is a no-op (Cosmos 412, swallowed).
    asyncio.run(advance_extract_watermark(container, TENANT, cid, "u1", 4))
    assert _counter(container, cid)["last_extract_count"] == 8

    # A higher count still advances.
    asyncio.run(advance_extract_watermark(container, TENANT, cid, "u1", 12))
    assert _counter(container, cid)["last_extract_count"] == 12


@patch.dict(
    os.environ,
    {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "1",
        "EPISODE_EVAL_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
        "DEDUP_EVERY_N": "0",
    },
    clear=False,
)
def test_the_extract_watermark_is_read_from_the_tenant_of_the_turns():
    from shared.counters import advance_extract_watermark, thread_counter_id

    starter = _make_starter()
    container = _make_counter_container_starting_at()
    first = [_turn(tenant_id=tenant_id) for tenant_id in ("acme", "globex") for _ in range(4)]
    asyncio.run(process_changefeed_batch(first, starter, counter_container=container))
    # Only acme's extraction has gone through its 4 turns.
    asyncio.run(advance_extract_watermark(container, "acme", thread_counter_id("u1", "t1"), "u1", 4))

    starter.start_new.reset_mock()
    asyncio.run(
        process_changefeed_batch(
            [_turn(tenant_id="acme"), _turn(tenant_id="globex")], starter, counter_container=container
        )
    )

    payloads = [_extract_payload(c) for c in starter.start_new.await_args_list]
    assert {payload["tenant_id"]: payload["recent_k"] for payload in payloads} == {"acme": 1, "globex": 5}


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
def test_extract_payload_recent_k_uses_max_threshold_and_batch_delta():
    starter = _make_starter()
    container = _make_counter_container_starting_at()

    asyncio.run(process_changefeed_batch([_turn() for _ in range(3)], starter, counter_container=container))

    extract_calls = [c for c in starter.start_new.await_args_list if c.args[0] == "ExtractMemoriesOrchestrator"]
    assert len(extract_calls) == 1
    payload = _extract_payload(extract_calls[0])
    assert payload["recent_k"] == 3


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


@patch.dict(os.environ, {"MEMORY_PROCESSOR_OWNER": "inprocess"}, clear=False)
def test_topology_probe_skipped_when_owner_gate_closed(monkeypatch):
    """Owner gate must close BEFORE topology probe; otherwise SDK-only
    deployments pay container reads + risk transient-error spinloops on
    every cold start."""
    probe = AsyncMock()
    monkeypatch.setattr(change_feed_module, "_ensure_topology", probe)

    starter = _make_starter()
    asyncio.run(process_changefeed_batch([_turn()], starter))

    probe.assert_not_awaited()
    starter.start_new.assert_not_awaited()
