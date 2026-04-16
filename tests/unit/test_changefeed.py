"""Unit tests for the change feed trigger helpers and batch logic in function_app.py."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Add azure_functions directory to path so we can import function_app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "azure_functions"))

from azure.core import MatchConditions

from function_app import (
    _parse_threshold,
    crosses_threshold,
    increment_counter_by,
    process_changefeed_batch,
)

# =====================================================================
# crosses_threshold
# =====================================================================


class TestCrossesThreshold:
    """Tests for the crosses_threshold helper."""

    def test_single_crossing(self):
        # old=7, new=11, N=5 → 7//5=1, 11//5=2 → True
        assert crosses_threshold(7, 11, 5) is True

    def test_multiple_crossings_collapsed(self):
        # old=3, new=17, N=5 → 3//5=0, 17//5=3 → True (but still one fire)
        assert crosses_threshold(3, 17, 5) is True

    def test_no_crossing(self):
        # old=6, new=9, N=5 → 6//5=1, 9//5=1 → False
        assert crosses_threshold(6, 9, 5) is False

    def test_exact_boundary(self):
        # old=9, new=10, N=5 → 9//5=1, 10//5=2 → True
        assert crosses_threshold(9, 10, 5) is True

    def test_n_equals_1(self):
        # Every increment crosses the threshold
        assert crosses_threshold(0, 1, 1) is True
        assert crosses_threshold(5, 6, 1) is True

    def test_count_zero_to_zero(self):
        # No change, no crossing
        assert crosses_threshold(0, 0, 5) is False

    def test_count_zero_to_below_n(self):
        # old=0, new=3, N=5 → 0//5=0, 3//5=0 → False
        assert crosses_threshold(0, 3, 5) is False

    def test_count_zero_to_exactly_n(self):
        # old=0, new=5, N=5 → 0//5=0, 5//5=1 → True
        assert crosses_threshold(0, 5, 5) is True

    def test_large_batch_multiple_thresholds(self):
        # old=0, new=25, N=10 → 0//10=0, 25//10=2 → True
        assert crosses_threshold(0, 25, 10) is True


# =====================================================================
# increment_counter_by
# =====================================================================


class TestIncrementCounterBy:
    """Tests for the increment_counter_by function with mocked Cosmos container."""

    def _make_mock_container(self, existing_doc=None, etag_conflict_times=0):
        """Create a mock Cosmos container client.

        Args:
            existing_doc: If set, read_item returns this doc. If None, raises 404.
            etag_conflict_times: Number of times upsert should raise 412 before succeeding.
        """
        container = AsyncMock()
        if existing_doc is not None:
            container.read_item.return_value = existing_doc
        else:
            from azure.cosmos.exceptions import CosmosResourceNotFoundError

            container.read_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="Not found")

        if etag_conflict_times > 0:
            from azure.cosmos.exceptions import CosmosHttpResponseError

            conflict_exc = CosmosHttpResponseError(status_code=412, message="Precondition failed")
            conflict_exc.status_code = 412

            call_count = {"n": 0}
            original_doc = existing_doc

            async def upsert_side_effect(body, **kwargs):
                call_count["n"] += 1
                if call_count["n"] <= etag_conflict_times:
                    raise conflict_exc
                return body

            async def read_side_effect(item, partition_key):
                if call_count["n"] > 0 and original_doc is not None:
                    # After a conflict, return updated doc
                    updated = dict(original_doc)
                    updated["count"] = original_doc["count"] + 1
                    updated["_etag"] = "new-etag"
                    return updated
                if original_doc is not None:
                    return original_doc
                from azure.cosmos.exceptions import CosmosResourceNotFoundError

                raise CosmosResourceNotFoundError(status_code=404, message="Not found")

            container.upsert_item.side_effect = upsert_side_effect
            container.read_item.side_effect = read_side_effect
        else:
            container.upsert_item.return_value = None

        return container

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_successful_increment(self, mock_get_container):
        existing = {
            "id": "thread_counter_alice_chat42",
            "user_id": "alice",
            "thread_id": "chat42",
            "count": 7,
            "_etag": "etag-1",
            "created_at": "2026-04-09T00:00:00+00:00",
        }
        container = self._make_mock_container(existing_doc=existing)
        mock_get_container.return_value = container

        old, new = await increment_counter_by(
            "thread_counter_alice_chat42",
            "alice",
            "chat42",
            3,
            batch_max_lsn=42,
        )
        assert old == 7
        assert new == 10
        container.upsert_item.assert_called_once()
        call_kwargs = container.upsert_item.call_args
        assert call_kwargs[1]["etag"] == "etag-1"
        assert call_kwargs[1]["match_condition"] == MatchConditions.IfNotModified
        body = call_kwargs[1]["body"]
        assert body["thread_id"] == "chat42"
        assert body["count"] == 10
        assert body["last_batch_lsn"] == 42
        assert body["last_batch_old_count"] == 7

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_first_time_creation(self, mock_get_container):
        container = self._make_mock_container(existing_doc=None)
        container.create_item = AsyncMock(return_value=None)
        mock_get_container.return_value = container

        old, new = await increment_counter_by(
            "thread_counter_bob_chat1",
            "bob",
            "chat1",
            5,
            batch_max_lsn=10,
        )
        assert old == 0
        assert new == 5
        container.create_item.assert_called_once()
        call_body = container.create_item.call_args[1].get("body") or container.create_item.call_args[0][0]
        assert call_body["count"] == 5
        assert call_body["thread_id"] == "chat1"
        assert call_body["last_batch_lsn"] == 10
        assert call_body["last_batch_old_count"] == 0
        assert "type" not in call_body

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_etag_conflict_retry(self, mock_get_container):
        existing = {
            "id": "ctr",
            "user_id": "alice",
            "thread_id": "chat1",
            "count": 10,
            "_etag": "old-etag",
            "created_at": "2026-04-09T00:00:00+00:00",
        }
        container = self._make_mock_container(existing_doc=existing, etag_conflict_times=1)
        mock_get_container.return_value = container

        old, new = await increment_counter_by("ctr", "alice", "chat1", 2)
        # After retry, it reads the updated doc (count=11) and increments by 2
        assert old == 11
        assert new == 13
        assert container.upsert_item.call_count == 2


# =====================================================================
# on_memory_change batch logic
# =====================================================================


class TestOnMemoryChange:
    """Tests for the process_changefeed_batch core logic."""

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "5",
            "FACT_EXTRACTION_EVERY_N": "3",
            "USER_SUMMARY_EVERY_N": "10",
        },
    )
    async def test_filters_non_turn_documents(self, mock_increment):
        docs = [
            {"type": "summary", "user_id": "alice", "thread_id": "t1", "_lsn": 1},
            {"type": "fact", "user_id": "alice", "thread_id": "t1", "_lsn": 2},
            {"type": "user_summary", "user_id": "alice", "thread_id": "__user_summary__", "_lsn": 3},
        ]
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        mock_increment.assert_not_called()
        starter.start_new.assert_not_called()

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "5",
            "FACT_EXTRACTION_EVERY_N": "0",
            "USER_SUMMARY_EVERY_N": "0",
        },
    )
    async def test_groups_by_thread_scope(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 10},
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 11},
            {"type": "turn", "user_id": "alice", "thread_id": "t2", "_lsn": 12},
        ]

        # No threshold crossings
        mock_increment.return_value = (1, 2)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        # User counters are disabled in this test, so only the two thread scopes increment.
        assert mock_increment.call_count == 2
        calls = mock_increment.call_args_list
        thread_call_args = {c[0][0]: c[0][3] for c in calls if c[0][0].startswith("thread_")}
        assert thread_call_args["thread_counter_alice_t1"] == 2
        assert thread_call_args["thread_counter_alice_t2"] == 1
        # Verify LSN is passed
        thread_call_lsn = {c[0][0]: c[1].get("batch_max_lsn") for c in calls if c[0][0].startswith("thread_")}
        assert thread_call_lsn["thread_counter_alice_t1"] == 11
        assert thread_call_lsn["thread_counter_alice_t2"] == 12

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "5",
            "FACT_EXTRACTION_EVERY_N": "0",
            "USER_SUMMARY_EVERY_N": "0",
        },
    )
    async def test_starts_orchestration_on_threshold_crossing(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 20},
        ]

        # Simulate threshold crossing: old=4, new=5, N=5 → 4//5=0, 5//5=1 → True
        mock_increment.return_value = (4, 5)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        starter.start_new.assert_called_once_with(
            "memory_orchestrator",
            instance_id="ts_alice_t1_1",
            client_input={
                "thread_summary_only": True,
                "user_id": "alice",
                "thread_id": "t1",
            },
        )

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "0",
            "FACT_EXTRACTION_EVERY_N": "0",
            "USER_SUMMARY_EVERY_N": "0",
        },
    )
    async def test_all_disabled_skips_processing(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 1},
        ]
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        mock_increment.assert_not_called()
        starter.start_new.assert_not_called()

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "0",
            "FACT_EXTRACTION_EVERY_N": "0",
            "USER_SUMMARY_EVERY_N": "10",
        },
    )
    async def test_user_summary_threshold(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 30},
            {"type": "turn", "user_id": "alice", "thread_id": "t2", "_lsn": 31},
        ]

        # User counter crosses threshold
        mock_increment.return_value = (9, 11)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        # Should start user summary orchestration with deterministic instance ID
        starter.start_new.assert_called_once_with(
            "memory_orchestrator",
            instance_id="us_alice_1",
            client_input={
                "user_summary_only": True,
                "user_id": "alice",
            },
        )

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "5",
            "FACT_EXTRACTION_EVERY_N": "0",
            "USER_SUMMARY_EVERY_N": "0",
        },
    )
    async def test_orchestration_failure_raises_after_batch(self, mock_increment):
        """When starter.start_new() fails, errors are collected and re-raised."""
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 40},
            {"type": "turn", "user_id": "bob", "thread_id": "t2", "_lsn": 41},
        ]

        # Both cross the threshold
        mock_increment.return_value = (4, 5)
        starter = AsyncMock()
        starter.start_new.side_effect = RuntimeError("task hub unavailable")

        with pytest.raises(RuntimeError, match="Failed to start 2 orchestration"):
            await process_changefeed_batch(docs, starter)

        # Both orchestration starts were attempted despite failures
        assert starter.start_new.call_count == 2

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "5",
            "FACT_EXTRACTION_EVERY_N": "3",
            "USER_SUMMARY_EVERY_N": "0",
        },
    )
    async def test_fact_extraction_threshold(self, mock_increment):
        """Fact extraction orchestration fires when its threshold is crossed."""
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 50},
        ]

        # Crosses fact threshold (N=3) but not thread summary (N=5)
        mock_increment.return_value = (2, 3)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        starter.start_new.assert_called_once_with(
            "memory_orchestrator",
            instance_id="ef_alice_t1_1",
            client_input={
                "extract_facts_only": True,
                "user_id": "alice",
                "thread_id": "t1",
            },
        )


# =====================================================================
# _parse_threshold
# =====================================================================


class TestParseThreshold:
    """Tests for the _parse_threshold helper."""

    @patch.dict(os.environ, {"TEST_THRESHOLD": "10"})
    def test_valid_integer(self):
        assert _parse_threshold("TEST_THRESHOLD") == 10

    @patch.dict(os.environ, {"TEST_THRESHOLD": "0"})
    def test_zero(self):
        assert _parse_threshold("TEST_THRESHOLD") == 0

    @patch.dict(os.environ, {}, clear=False)
    def test_missing_defaults_to_zero(self):
        # Ensure the key is not in env
        os.environ.pop("TEST_MISSING_VAR", None)
        assert _parse_threshold("TEST_MISSING_VAR") == 0

    @patch.dict(os.environ, {"TEST_THRESHOLD": ""})
    def test_empty_string_defaults_to_zero(self):
        assert _parse_threshold("TEST_THRESHOLD") == 0

    @patch.dict(os.environ, {"TEST_THRESHOLD": "not_a_number"})
    def test_invalid_string_defaults_to_zero(self):
        assert _parse_threshold("TEST_THRESHOLD") == 0

    @patch.dict(os.environ, {"TEST_THRESHOLD": "5 # comment"})
    def test_trailing_comment_defaults_to_zero(self):
        assert _parse_threshold("TEST_THRESHOLD") == 0


# =====================================================================
# increment_counter_by – first-creation 409 race
# =====================================================================


class TestIncrementCounterByRace:
    """Tests for the 409-conflict handling on first counter creation."""

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_create_conflict_retries_with_read(self, mock_get_container):
        """When create_item returns 409, we retry via the read-modify-write path."""
        from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

        container = AsyncMock()

        call_count = {"n": 0}

        async def read_side_effect(item, partition_key):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise CosmosResourceNotFoundError(status_code=404, message="Not found")
            # On retry, return the doc created by the other instance
            return {
                "id": item,
                "user_id": "alice",
                "thread_id": "t1",
                "count": 3,
                "_etag": "other-etag",
                "created_at": "2026-04-10T00:00:00+00:00",
            }

        container.read_item.side_effect = read_side_effect

        conflict_exc = CosmosHttpResponseError(status_code=409, message="Conflict")
        conflict_exc.status_code = 409
        container.create_item.side_effect = conflict_exc
        container.upsert_item = AsyncMock(return_value=None)

        mock_get_container.return_value = container

        old, new = await increment_counter_by("thread_counter_alice_t1", "alice", "t1", 2)

        # On retry: reads count=3, adds 2 → new=5
        assert old == 3
        assert new == 5
        container.create_item.assert_called_once()
        container.upsert_item.assert_called_once()
        upsert_kwargs = container.upsert_item.call_args[1]
        assert upsert_kwargs["match_condition"] == MatchConditions.IfNotModified


# =====================================================================
# increment_counter_by – LSN replay detection
# =====================================================================


class TestIncrementCounterByReplay:
    """Tests for LSN-based replay detection in increment_counter_by."""

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_replay_detected_returns_cached_result(self, mock_get_container):
        """When batch_max_lsn matches stored last_batch_lsn, skip increment."""
        existing = {
            "id": "thread_counter_alice_t1",
            "user_id": "alice",
            "thread_id": "t1",
            "count": 8,
            "last_batch_lsn": 42,
            "last_batch_old_count": 5,
            "_etag": "etag-1",
            "created_at": "2026-04-09T00:00:00+00:00",
        }
        container = AsyncMock()
        container.read_item.return_value = existing
        mock_get_container.return_value = container

        old, new = await increment_counter_by(
            "thread_counter_alice_t1",
            "alice",
            "t1",
            3,
            batch_max_lsn=42,
        )

        # Returns cached (pre-batch count, current count) without writing
        assert old == 5
        assert new == 8
        container.upsert_item.assert_not_called()
        container.create_item.assert_not_called()

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_no_replay_when_lsn_differs(self, mock_get_container):
        """When batch_max_lsn differs from stored, process normally."""
        existing = {
            "id": "thread_counter_alice_t1",
            "user_id": "alice",
            "thread_id": "t1",
            "count": 8,
            "last_batch_lsn": 42,
            "last_batch_old_count": 5,
            "_etag": "etag-1",
            "created_at": "2026-04-09T00:00:00+00:00",
        }
        container = AsyncMock()
        container.read_item.return_value = existing
        container.upsert_item.return_value = None
        mock_get_container.return_value = container

        old, new = await increment_counter_by(
            "thread_counter_alice_t1",
            "alice",
            "t1",
            3,
            batch_max_lsn=99,
        )

        # Normal increment: 8 + 3 = 11
        assert old == 8
        assert new == 11
        container.upsert_item.assert_called_once()
        body = container.upsert_item.call_args[1]["body"]
        assert body["last_batch_lsn"] == 99
        assert body["last_batch_old_count"] == 8

    @pytest.mark.asyncio
    @patch("function_app._get_cosmos_counter_container", new_callable=AsyncMock)
    async def test_no_replay_check_when_lsn_not_provided(self, mock_get_container):
        """When batch_max_lsn is None, skip replay detection (backward compat)."""
        existing = {
            "id": "ctr",
            "user_id": "alice",
            "thread_id": "t1",
            "count": 5,
            "last_batch_lsn": 42,
            "_etag": "etag-1",
            "created_at": "2026-04-09T00:00:00+00:00",
        }
        container = AsyncMock()
        container.read_item.return_value = existing
        container.upsert_item.return_value = None
        mock_get_container.return_value = container

        old, new = await increment_counter_by("ctr", "alice", "t1", 2)

        # No replay detection, processes normally
        assert old == 5
        assert new == 7
        container.upsert_item.assert_called_once()

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {
            "THREAD_SUMMARY_EVERY_N": "5",
            "FACT_EXTRACTION_EVERY_N": "0",
            "USER_SUMMARY_EVERY_N": "0",
        },
    )
    async def test_replay_still_fires_threshold(self, mock_increment):
        """On replay, increment_counter_by returns cached (old, new) so thresholds re-fire."""
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1", "_lsn": 42},
        ]

        # Simulate replay returning cached values that cross threshold
        mock_increment.return_value = (4, 5)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)

        # Threshold still fires → orchestration attempted (with deterministic ID)
        starter.start_new.assert_called_once_with(
            "memory_orchestrator",
            instance_id="ts_alice_t1_1",
            client_input={
                "thread_summary_only": True,
                "user_id": "alice",
                "thread_id": "t1",
            },
        )
