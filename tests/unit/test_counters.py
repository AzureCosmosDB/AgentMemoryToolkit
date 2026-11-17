"""Tests for ``azure.cosmos.agent_memory._counters``.

Covers counter-doc construction (LSN preservation, failure breadcrumbs),
``stamp_failure_sync`` - the helpers that let SDK and FA share a counter
container without trampling each other - and the ``[tenant_id, user_id]``
partition every counter lives in.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory._counters import (
    USER_COUNTER_THREAD_ID,
    _build_counter_doc,
    increment_counter_async,
    increment_counter_sync,
    stamp_failure_async,
    stamp_failure_sync,
    thread_counter_id,
    user_counter_id,
)

TENANT = "acme"


class TestBuildCounterDoc:
    def test_first_write_initializes_lsn_none(self):
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            tenant_id=TENANT,
            user_id="u",
            thread_id="t",
            new_count=1,
            old_count=0,
            existing=None,
        )
        assert doc["count"] == 1
        assert doc["last_batch_lsn"] is None
        assert "last_failure_at" not in doc

    def test_carries_the_tenant(self):
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            tenant_id=TENANT,
            user_id="u",
            thread_id="t",
            new_count=1,
            old_count=0,
            existing=None,
        )
        assert (doc["tenant_id"], doc["user_id"], doc["thread_id"]) == (TENANT, "u", "t")

    def test_preserves_fa_lsn(self):
        """SDK writes must not clobber an LSN seeded by the FA-side helper.

        The FA's ``increment_counter_by`` uses ``last_batch_lsn`` to dedup
        change-feed redelivery; setting it back to ``None`` here would
        invalidate the FA's monotonicity assumption on the next replay.
        """
        existing = {"last_batch_lsn": 42, "count": 5, "created_at": "2024-01-01T00:00:00Z"}
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            tenant_id=TENANT,
            user_id="u",
            thread_id="t",
            new_count=6,
            old_count=5,
            existing=existing,
        )
        assert doc["last_batch_lsn"] == 42
        assert doc["created_at"] == "2024-01-01T00:00:00Z"

    def test_preserves_fa_last_batch_old_count(self):
        """SDK writes must preserve the FA's cached pre-batch counter value.

        ``last_batch_old_count`` pairs with ``last_batch_lsn`` for cached
        replay of an FA-driven batch. If the SDK overwrote it with its
        own ``old_count`` (which reflects only the SDK's local delta), a
        change-feed redelivery would replay an inconsistent ``(old, new)``
        pair and re-fire orchestrators against already-summarized data.
        """
        existing = {
            "count": 15,
            "last_batch_lsn": 5,
            "last_batch_old_count": 8,
            "created_at": "2024-01-01T00:00:00Z",
        }
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            tenant_id=TENANT,
            user_id="u",
            thread_id="t",
            new_count=20,
            old_count=15,
            existing=existing,
        )
        assert doc["last_batch_lsn"] == 5
        assert doc["last_batch_old_count"] == 8

    def test_preserves_failure_breadcrumbs(self):
        existing = {
            "last_batch_lsn": 7,
            "count": 1,
            "last_failure_at": "2024-06-01T00:00:00Z",
            "last_failure_reason": "AI Foundry 401",
        }
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            tenant_id=TENANT,
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
        """``stamp_failure_sync`` must use ``patch_item`` so it can never
        lose-update concurrent count increments.
        """
        container = MagicMock()

        stamp_failure_sync(container, TENANT, "thread:u:t", "u", "boom")

        # No read+upsert race: patch_item is the only write.
        container.read_item.assert_not_called()
        container.upsert_item.assert_not_called()
        container.patch_item.assert_called_once()

        kwargs = container.patch_item.call_args.kwargs
        assert kwargs["item"] == "thread:u:t"
        assert kwargs["partition_key"] == [TENANT, "u"]
        ops = kwargs["patch_operations"]
        op_paths = {op["path"]: op for op in ops}
        assert op_paths["/last_failure_reason"]["value"] == "boom"
        assert op_paths["/last_failure_reason"]["op"] == "add"
        assert "/last_failure_at" in op_paths

    def test_swallows_patch_errors(self):
        """Failure stamping is best-effort breadcrumbing - must never raise."""
        container = MagicMock()
        container.patch_item.side_effect = RuntimeError("cosmos down")
        stamp_failure_sync(container, TENANT, "thread:u:t", "u", "boom")
        # Did not propagate - that's the contract.

    def test_truncates_long_reason(self):
        container = MagicMock()
        long_reason = "x" * 1000
        stamp_failure_sync(container, TENANT, "thread:u:t", "u", long_reason)
        kwargs = container.patch_item.call_args.kwargs
        ops = {op["path"]: op for op in kwargs["patch_operations"]}
        assert len(ops["/last_failure_reason"]["value"]) == 500


class TestBuildCounterDocOwnerStamping:
    """``last_owner`` is stamped as advisory metadata so a second backend
    (or a future operator audit) can see who last wrote the counter.
    """

    def test_stamps_last_owner_when_provided(self):
        doc = _build_counter_doc(
            counter_id="thread:u:t",
            tenant_id=TENANT,
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
            tenant_id=TENANT,
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
            tenant_id=TENANT,
            user_id="u",
            thread_id="t",
            new_count=2,
            old_count=1,
            existing=existing,
            owner="inprocess",
        )
        assert doc["last_owner"] == "inprocess"


class _OwnerPartitions:
    """A counter container that keeps documents per ``[tenant_id, user_id]`` partition, as Cosmos does with that key.

    A document lands in the partition its own ``tenant_id`` and ``user_id`` name, and reads, patches and writes are
    answered from the partition they ask for, so a counter written without its tenant cannot be read back.
    """

    def __init__(self) -> None:
        self.docs: dict[tuple[tuple[Any, Any], str], dict[str, Any]] = {}
        self.partitions_read: list[list[str]] = []
        self.partitions_patched: list[list[str]] = []
        self._etag = 0

    def read_item(self, *, item, partition_key):
        self.partitions_read.append(partition_key)
        try:
            return dict(self.docs[(tuple(partition_key), item)])
        except KeyError:
            raise CosmosResourceNotFoundError(message="404") from None

    def create_item(self, *, body):
        return self._write(body)

    def upsert_item(self, *, body, **_kwargs):
        return self._write(body)

    def patch_item(self, *, item, partition_key, patch_operations):
        self.partitions_patched.append(partition_key)
        doc = self.docs[(tuple(partition_key), item)]
        for op in patch_operations:
            doc[op["path"].lstrip("/")] = op["value"]
        return dict(doc)

    def _write(self, body):
        self._etag += 1
        stored = {**body, "_etag": f"e{self._etag}"}
        self.docs[((body.get("tenant_id"), body.get("user_id")), body["id"])] = stored
        return dict(stored)


class _AsyncOwnerPartitions(_OwnerPartitions):
    async def read_item(self, *, item, partition_key):
        return super().read_item(item=item, partition_key=partition_key)

    async def create_item(self, *, body):
        return super().create_item(body=body)

    async def upsert_item(self, *, body, **kwargs):
        return super().upsert_item(body=body, **kwargs)

    async def patch_item(self, *, item, partition_key, patch_operations):
        return super().patch_item(item=item, partition_key=partition_key, patch_operations=patch_operations)


class TestCountersLiveInTheOwnersPartition:
    def test_a_counter_is_read_and_written_in_the_tenant_and_user_partition(self):
        container = _OwnerPartitions()

        first = increment_counter_sync(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 2)
        second = increment_counter_sync(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 1)

        assert (first, second) == ((0, 2), (2, 3))
        assert container.partitions_read == [[TENANT, "u1"], [TENANT, "u1"]]
        [stored] = container.docs.values()
        assert (stored["id"], stored["tenant_id"], stored["user_id"], stored["thread_id"], stored["count"]) == (
            "thread:u1:t1",
            TENANT,
            "u1",
            "t1",
            3,
        )

    def test_the_thread_counters_and_the_user_counter_of_one_owner_share_the_partition(self):
        container = _OwnerPartitions()

        increment_counter_sync(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 1)
        increment_counter_sync(container, TENANT, thread_counter_id("u1", "t2"), "u1", "t2", 1)
        increment_counter_sync(container, TENANT, user_counter_id("u1"), "u1", USER_COUNTER_THREAD_ID, 2)

        assert {partition for partition, _ in container.docs} == {(TENANT, "u1")}
        assert {item for _, item in container.docs} == {"thread:u1:t1", "thread:u1:t2", "user:u1"}
        assert container.docs[((TENANT, "u1"), "user:u1")]["thread_id"] == "__counters__"

    def test_the_same_owner_in_another_tenant_counts_on_its_own(self):
        container = _OwnerPartitions()

        increment_counter_sync(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 5)
        other = increment_counter_sync(container, "globex", thread_counter_id("u1", "t1"), "u1", "t1", 1)

        assert other == (0, 1)
        assert {partition for partition, _ in container.docs} == {(TENANT, "u1"), ("globex", "u1")}

    def test_a_failure_is_stamped_in_the_tenant_and_user_partition(self):
        container = _OwnerPartitions()
        increment_counter_sync(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 1)

        stamp_failure_sync(container, TENANT, thread_counter_id("u1", "t1"), "u1", "boom")

        assert container.partitions_patched == [[TENANT, "u1"]]
        assert container.docs[((TENANT, "u1"), "thread:u1:t1")]["last_failure_reason"] == "boom"

    async def test_async_counters_live_in_the_tenant_and_user_partition(self):
        container = _AsyncOwnerPartitions()

        first = await increment_counter_async(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 2)
        second = await increment_counter_async(container, TENANT, thread_counter_id("u1", "t1"), "u1", "t1", 1)
        other = await increment_counter_async(container, "globex", thread_counter_id("u1", "t1"), "u1", "t1", 1)
        await stamp_failure_async(container, TENANT, thread_counter_id("u1", "t1"), "u1", "boom")

        assert (first, second, other) == ((0, 2), (2, 3), (0, 1))
        assert container.partitions_read == [[TENANT, "u1"], [TENANT, "u1"], ["globex", "u1"]]
        assert container.partitions_patched == [[TENANT, "u1"]]
        stored = container.docs[((TENANT, "u1"), "thread:u1:t1")]
        assert (stored["tenant_id"], stored["user_id"], stored["thread_id"], stored["count"]) == (TENANT, "u1", "t1", 3)
