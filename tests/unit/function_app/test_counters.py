"""Unit tests for ``function_app/shared/counters.py``.

Covers:
* ``crosses_threshold`` boundary semantics (the truth table from the spec).
* ``increment_counter_by`` first-write path, happy path, ETag retry path,
  ETag exhaustion (skip), and LSN-based replay detection.

The Cosmos container is mocked - no Azure dependency is required at test time.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from shared.counters import (
    crosses_threshold,
    increment_counter_by,
    thread_counter_id,
    user_counter_id,
)

# ---------------------------------------------------------------------------
# crosses_threshold
# ---------------------------------------------------------------------------


class TestCrossesThreshold:
    """Boundary-case truth table for ``crosses_threshold`` with N=4.

    Cases come straight from the spec brief: 0→4, 3→4, 3→5, 4→8, 6→9.
    """

    @pytest.mark.parametrize(
        "old,new,n,expected",
        [
            # The spec-mandated boundary cases ---------------------------------
            (0, 4, 4, True),  # crossed at 4
            (3, 4, 4, True),  # crossed at 4
            (3, 5, 4, True),  # crossed at 4
            (4, 8, 4, True),  # crossed at 8
            (6, 9, 4, True),  # crossed at 8
            # Sub-threshold (no crossing) --------------------------------------
            (0, 3, 4, False),
            (4, 7, 4, False),
            (5, 5, 4, False),  # no progress at all
            (8, 11, 4, False),
            # Other thresholds (sanity) ----------------------------------------
            (0, 20, 20, True),
            (19, 20, 20, True),
            (0, 19, 20, False),
            (40, 60, 20, True),
            # Multiple-bucket jumps cross at least once ------------------------
            (0, 9, 4, True),
            (1, 100, 4, True),
        ],
    )
    def test_truth_table(self, old, new, n, expected):
        assert crosses_threshold(old, new, n) is expected

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError):
            crosses_threshold(0, 5, 0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            crosses_threshold(0, 5, -1)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def test_counter_id_helpers():
    assert thread_counter_id("u1", "t1") == "thread:u1:t1"
    assert user_counter_id("u1") == "user:u1"


# ---------------------------------------------------------------------------
# increment_counter_by - fixtures
# ---------------------------------------------------------------------------


def _make_container() -> MagicMock:
    """Return a mock Cosmos container with async methods stubbed."""
    container = MagicMock()
    container.read_item = AsyncMock()
    container.upsert_item = AsyncMock()
    container.create_item = AsyncMock()
    return container


def _http_error(status_code: int) -> CosmosHttpResponseError:
    err = CosmosHttpResponseError(message=f"http {status_code}")
    err.status_code = status_code
    return err


# ---------------------------------------------------------------------------
# increment_counter_by - first-write path
# ---------------------------------------------------------------------------


class TestIncrementFirstWrite:
    def test_creates_doc_when_missing(self):
        container = _make_container()
        container.read_item.side_effect = CosmosResourceNotFoundError(message="404")

        old, new = asyncio.run(increment_counter_by(container, "thread:u1:t1", "u1", "t1", 3))

        assert (old, new) == (0, 3)
        assert container.create_item.await_count == 1
        body = container.create_item.await_args.kwargs["body"]
        assert body["id"] == "thread:u1:t1"
        assert body["count"] == 3
        assert body["last_batch_old_count"] == 0
        assert container.upsert_item.await_count == 0

    def test_create_conflict_then_succeeds(self):
        """409 on create → second attempt re-reads (now found) → upserts."""
        container = _make_container()
        # first attempt: 404 then 409
        # second attempt: read finds existing doc with count=3 → upsert succeeds
        container.read_item.side_effect = [
            CosmosResourceNotFoundError(message="404"),
            {"id": "thread:u1:t1", "count": 3, "_etag": "etag1"},
        ]
        container.create_item.side_effect = [_http_error(409)]

        old, new = asyncio.run(increment_counter_by(container, "thread:u1:t1", "u1", "t1", 2))

        assert (old, new) == (3, 5)
        assert container.create_item.await_count == 1
        assert container.upsert_item.await_count == 1


# ---------------------------------------------------------------------------
# increment_counter_by - happy / etag-retry / replay paths
# ---------------------------------------------------------------------------


class TestIncrementExisting:
    def test_happy_path_upserts(self):
        container = _make_container()
        container.read_item.return_value = {
            "id": "thread:u1:t1",
            "count": 5,
            "_etag": "etag-A",
        }

        old, new = asyncio.run(
            increment_counter_by(
                container,
                "thread:u1:t1",
                "u1",
                "t1",
                4,
                batch_max_lsn=42,
            )
        )

        assert (old, new) == (5, 9)
        assert container.upsert_item.await_count == 1
        upsert_body = container.upsert_item.await_args.kwargs["body"]
        assert upsert_body["count"] == 9
        assert upsert_body["last_batch_lsn"] == 42
        assert upsert_body["last_batch_old_count"] == 5

    def test_etag_conflict_retries_then_succeeds(self):
        container = _make_container()
        # First read+upsert collides; second read+upsert succeeds.
        container.read_item.side_effect = [
            {"id": "thread:u1:t1", "count": 5, "_etag": "etag-A"},
            {"id": "thread:u1:t1", "count": 7, "_etag": "etag-B"},
        ]
        container.upsert_item.side_effect = [_http_error(412), None]

        old, new = asyncio.run(increment_counter_by(container, "thread:u1:t1", "u1", "t1", 2))

        # We restart from the *latest* read - count went 7 → 9.
        assert (old, new) == (7, 9)
        assert container.read_item.await_count == 2
        assert container.upsert_item.await_count == 2

    def test_etag_conflict_exhausted_raises(self):
        """After MAX_RETRIES failed attempts the helper RAISES so the change-feed
        batch retries (at-least-once redelivery + LSN replay protection make
        this safe). Silently returning ``(old, old)`` would advance the lease
        without ever firing the orchestrator the increment was supposed to
        trigger - a permanent threshold-miss bug."""
        from azure.cosmos.exceptions import CosmosHttpResponseError

        container = _make_container()
        container.read_item.return_value = {
            "id": "thread:u1:t1",
            "count": 5,
            "_etag": "etag-A",
        }
        container.upsert_item.side_effect = _http_error(412)

        with pytest.raises(CosmosHttpResponseError) as exc_info:
            asyncio.run(increment_counter_by(container, "thread:u1:t1", "u1", "t1", 1))

        assert exc_info.value.status_code == 412
        assert container.upsert_item.await_count == 3  # MAX_RETRIES

    def test_lsn_replay_returns_cached_without_writing(self):
        """When the doc already records ``last_batch_lsn == batch_max_lsn``
        we treat the change-feed batch as a duplicate delivery: return the
        cached ``(pre_batch_count, current_count)`` and DO NOT write."""
        container = _make_container()
        container.read_item.return_value = {
            "id": "thread:u1:t1",
            "count": 9,  # current
            "last_batch_lsn": 100,  # this batch was already applied
            "last_batch_old_count": 5,  # pre-batch count
            "_etag": "etag-A",
        }

        old, new = asyncio.run(
            increment_counter_by(
                container,
                "thread:u1:t1",
                "u1",
                "t1",
                4,
                batch_max_lsn=100,
            )
        )

        assert (old, new) == (5, 9)
        assert container.upsert_item.await_count == 0
        assert container.create_item.await_count == 0

    def test_different_lsn_does_not_trigger_replay(self):
        """A different LSN is treated as a fresh batch and increments normally."""
        container = _make_container()
        container.read_item.return_value = {
            "id": "thread:u1:t1",
            "count": 9,
            "last_batch_lsn": 100,
            "last_batch_old_count": 5,
            "_etag": "etag-A",
        }

        old, new = asyncio.run(
            increment_counter_by(
                container,
                "thread:u1:t1",
                "u1",
                "t1",
                4,
                batch_max_lsn=101,
            )
        )

        assert (old, new) == (9, 13)
        assert container.upsert_item.await_count == 1

    def test_out_of_order_replay_is_noop(self):
        """When the redelivered batch's LSN is *less than* the stored LSN
        (lease re-balance / host crash redelivering an old batch after a
        newer one already landed), the increment is a no-op - return
        (current, current) so threshold-crossing logic doesn't fire a
        spurious extract/dedup. We use ``>=`` not
        ``==`` for replay detection."""
        container = _make_container()
        container.read_item.return_value = {
            "id": "thread:u1:t1",
            "count": 10,
            "last_batch_lsn": 200,
            "last_batch_old_count": 5,
            "_etag": "etag-A",
        }

        old, new = asyncio.run(
            increment_counter_by(
                container,
                "thread:u1:t1",
                "u1",
                "t1",
                5,
                batch_max_lsn=100,
            )
        )

        assert (old, new) == (10, 10)
        assert container.upsert_item.await_count == 0
        assert container.create_item.await_count == 0
