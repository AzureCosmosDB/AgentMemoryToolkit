"""Unit tests for the change feed trigger helpers and batch logic in function_app.py."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add azure_functions directory to path so we can import function_app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "azure_functions"))

from function_app import crosses_threshold, increment_counter_by, process_changefeed_batch


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
        container = MagicMock()
        if existing_doc is not None:
            container.read_item.return_value = existing_doc
        else:
            from azure.cosmos.exceptions import CosmosResourceNotFoundError

            container.read_item.side_effect = CosmosResourceNotFoundError(
                status_code=404, message="Not found"
            )

        if etag_conflict_times > 0:
            from azure.cosmos.exceptions import CosmosHttpResponseError

            conflict_exc = CosmosHttpResponseError(
                status_code=412, message="Precondition failed"
            )
            conflict_exc.status_code = 412

            call_count = {"n": 0}
            original_doc = existing_doc

            def upsert_side_effect(body, **kwargs):
                call_count["n"] += 1
                if call_count["n"] <= etag_conflict_times:
                    raise conflict_exc
                return body

            def read_side_effect(item, partition_key):
                if call_count["n"] > 0 and original_doc is not None:
                    # After a conflict, return updated doc
                    updated = dict(original_doc)
                    updated["count"] = original_doc["count"] + 1
                    updated["_etag"] = "new-etag"
                    return updated
                if original_doc is not None:
                    return original_doc
                from azure.cosmos.exceptions import CosmosResourceNotFoundError

                raise CosmosResourceNotFoundError(
                    status_code=404, message="Not found"
                )

            container.upsert_item.side_effect = upsert_side_effect
            container.read_item.side_effect = read_side_effect
        else:
            container.upsert_item.return_value = None

        return container

    @patch("function_app._get_counters_container")
    def test_successful_increment(self, mock_get_container):
        existing = {"id": "thread_counter_alice_chat42", "user_id": "alice", "count": 7, "_etag": "etag-1"}
        container = self._make_mock_container(existing_doc=existing)
        mock_get_container.return_value = container

        old, new = increment_counter_by("thread_counter_alice_chat42", "alice", 3)
        assert old == 7
        assert new == 10
        container.upsert_item.assert_called_once()
        call_kwargs = container.upsert_item.call_args
        assert call_kwargs[1]["etag"] == "etag-1"
        assert call_kwargs[1]["match_condition"] == "IfMatch"

    @patch("function_app._get_counters_container")
    def test_first_time_creation(self, mock_get_container):
        container = self._make_mock_container(existing_doc=None)
        mock_get_container.return_value = container

        old, new = increment_counter_by("thread_counter_bob_chat1", "bob", 5)
        assert old == 0
        assert new == 5
        container.upsert_item.assert_called_once()
        call_body = container.upsert_item.call_args[1].get("body") or container.upsert_item.call_args[0][0]
        assert call_body["count"] == 5

    @patch("function_app._get_counters_container")
    def test_etag_conflict_retry(self, mock_get_container):
        existing = {"id": "ctr", "user_id": "alice", "count": 10, "_etag": "old-etag"}
        container = self._make_mock_container(existing_doc=existing, etag_conflict_times=1)
        mock_get_container.return_value = container

        old, new = increment_counter_by("ctr", "alice", 2)
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
    @patch("function_app.increment_counter_by")
    @patch.dict(os.environ, {
        "THREAD_SUMMARY_EVERY_N": "5",
        "FACT_EXTRACTION_EVERY_N": "3",
        "USER_SUMMARY_EVERY_N": "10",
    })
    async def test_filters_non_turn_documents(self, mock_increment):
        docs = [
            {"type": "summary", "user_id": "alice", "thread_id": "t1"},
            {"type": "fact", "user_id": "alice", "thread_id": "t1"},
            {"type": "user_summary", "user_id": "alice", "thread_id": "__user_summary__"},
        ]
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        mock_increment.assert_not_called()
        starter.start_new.assert_not_called()

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by")
    @patch.dict(os.environ, {
        "THREAD_SUMMARY_EVERY_N": "5",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    })
    async def test_groups_by_thread_scope(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1"},
            {"type": "turn", "user_id": "alice", "thread_id": "t1"},
            {"type": "turn", "user_id": "alice", "thread_id": "t2"},
        ]

        # No threshold crossings
        mock_increment.return_value = (1, 2)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        # 2 thread-scope increments + 1 user-scope increment = 3 total
        assert mock_increment.call_count == 3
        calls = mock_increment.call_args_list
        thread_call_args = {c[0][0]: c[0][2] for c in calls if c[0][0].startswith("thread_")}
        assert thread_call_args["thread_counter_alice_t1"] == 2
        assert thread_call_args["thread_counter_alice_t2"] == 1
        # User-scope call: alice with total 3 turns
        user_call_args = {c[0][0]: c[0][2] for c in calls if c[0][0].startswith("user_")}
        assert user_call_args["user_counter_alice"] == 3

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by")
    @patch.dict(os.environ, {
        "THREAD_SUMMARY_EVERY_N": "5",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    })
    async def test_starts_orchestration_on_threshold_crossing(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1"},
        ]

        # Simulate threshold crossing: old=4, new=5, N=5 → 4//5=0, 5//5=1 → True
        mock_increment.return_value = (4, 5)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        starter.start_new.assert_called_once_with(
            "memory_orchestrator",
            client_input={
                "thread_summary_only": True,
                "user_id": "alice",
                "thread_id": "t1",
            },
        )

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by")
    @patch.dict(os.environ, {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "0",
    })
    async def test_all_disabled_skips_processing(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1"},
        ]
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        mock_increment.assert_not_called()
        starter.start_new.assert_not_called()

    @pytest.mark.asyncio
    @patch("function_app.increment_counter_by")
    @patch.dict(os.environ, {
        "THREAD_SUMMARY_EVERY_N": "0",
        "FACT_EXTRACTION_EVERY_N": "0",
        "USER_SUMMARY_EVERY_N": "10",
    })
    async def test_user_summary_threshold(self, mock_increment):
        docs = [
            {"type": "turn", "user_id": "alice", "thread_id": "t1"},
            {"type": "turn", "user_id": "alice", "thread_id": "t2"},
        ]

        # User counter crosses threshold
        mock_increment.return_value = (9, 11)
        starter = AsyncMock()

        await process_changefeed_batch(docs, starter)
        # Should start user summary orchestration
        starter.start_new.assert_called_once_with(
            "memory_orchestrator",
            client_input={
                "user_summary_only": True,
                "user_id": "alice",
            },
        )
