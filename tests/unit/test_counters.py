"""Tests for ``agent_memory_toolkit._counters``.

Covers counter-doc construction (LSN preservation, failure breadcrumbs)
and ``stamp_failure_sync`` — the helpers that let SDK and FA share a
counter container without trampling each other (Round 4 fixes #8 and #10).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_memory_toolkit._counters import _build_counter_doc, stamp_failure_sync


class TestBuildCounterDoc:
    def test_first_write_initializes_lsn_none(self):
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            user_id="u",
            thread_id="t",
            new_count=1,
            old_count=0,
            existing=None,
        )
        assert doc["count"] == 1
        assert doc["last_batch_lsn"] is None
        assert "last_failure_at" not in doc

    def test_preserves_fa_lsn(self):
        """SDK writes must not clobber an LSN seeded by the FA-side helper.

        The FA's ``increment_counter_by`` uses ``last_batch_lsn`` to dedup
        change-feed redelivery; setting it back to ``None`` here would
        invalidate the FA's monotonicity assumption on the next replay.
        """
        existing = {"last_batch_lsn": 42, "count": 5, "created_at": "2024-01-01T00:00:00Z"}
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            user_id="u",
            thread_id="t",
            new_count=6,
            old_count=5,
            existing=existing,
        )
        assert doc["last_batch_lsn"] == 42
        assert doc["created_at"] == "2024-01-01T00:00:00Z"

    def test_preserves_failure_breadcrumbs(self):
        existing = {
            "last_batch_lsn": 7,
            "count": 1,
            "last_failure_at": "2024-06-01T00:00:00Z",
            "last_failure_reason": "AI Foundry 401",
        }
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            user_id="u",
            thread_id="t",
            new_count=2,
            old_count=1,
            existing=existing,
        )
        assert doc["last_failure_at"] == "2024-06-01T00:00:00Z"
        assert doc["last_failure_reason"] == "AI Foundry 401"


class TestStampFailureSync:
    def test_writes_failure_fields_via_patch_item(self):
        """Round 5 fix #2: stamp_failure_sync must use ``patch_item`` so it
        can never lose-update concurrent count increments.
        """
        container = MagicMock()

        stamp_failure_sync(container, "thread:u:t", "u", "t", "boom")

        # No read+upsert race: patch_item is the only write.
        container.read_item.assert_not_called()
        container.upsert_item.assert_not_called()
        container.patch_item.assert_called_once()

        kwargs = container.patch_item.call_args.kwargs
        assert kwargs["item"] == "thread:u:t"
        assert kwargs["partition_key"] == ["u", "t"]
        ops = kwargs["patch_operations"]
        op_paths = {op["path"]: op for op in ops}
        assert op_paths["/last_failure_reason"]["value"] == "boom"
        assert op_paths["/last_failure_reason"]["op"] == "add"
        assert "/last_failure_at" in op_paths

    def test_swallows_patch_errors(self):
        """Failure stamping is best-effort breadcrumbing — must never raise."""
        container = MagicMock()
        container.patch_item.side_effect = RuntimeError("cosmos down")
        stamp_failure_sync(container, "thread:u:t", "u", "t", "boom")
        # Did not propagate — that's the contract.

    def test_truncates_long_reason(self):
        container = MagicMock()
        long_reason = "x" * 1000
        stamp_failure_sync(container, "thread:u:t", "u", "t", long_reason)
        kwargs = container.patch_item.call_args.kwargs
        ops = {op["path"]: op for op in kwargs["patch_operations"]}
        assert len(ops["/last_failure_reason"]["value"]) == 500


class TestBuildCounterDocOwnerStamping:
    """Round 5 fix #6: ``last_owner`` is stamped advisory metadata so a
    second backend (or a future operator audit) can see who last wrote.
    """

    def test_stamps_last_owner_when_provided(self):
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            user_id="u",
            thread_id="t",
            new_count=1,
            old_count=0,
            existing=None,
            owner="inprocess",
        )
        assert doc["last_owner"] == "inprocess"

    def test_preserves_existing_owner_when_none_passed(self):
        existing = {"last_owner": "durable", "count": 1, "last_batch_lsn": 5}
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            user_id="u",
            thread_id="t",
            new_count=2,
            old_count=1,
            existing=existing,
            owner=None,
        )
        assert doc["last_owner"] == "durable"

    def test_overwrites_existing_owner_when_explicit(self):
        existing = {"last_owner": "durable", "count": 1}
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            user_id="u",
            thread_id="t",
            new_count=2,
            old_count=1,
            existing=existing,
            owner="inprocess",
        )
        assert doc["last_owner"] == "inprocess"
