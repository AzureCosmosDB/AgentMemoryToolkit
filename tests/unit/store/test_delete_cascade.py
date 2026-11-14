"""Deleting a memory deletes the records folded into it."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.exceptions import CosmosOperationError, MemoryTypeMismatchError
from azure.cosmos.agent_memory.store import MemoryStore


def _containers(memories):
    return {ContainerKey.TURNS: MagicMock(), ContainerKey.MEMORIES: memories, ContainerKey.SUMMARIES: MagicMock()}


def _memory(memory_type="fact", memory_id="m1"):
    return {"id": memory_id, "type": memory_type, "user_id": "u1", "thread_id": "t1", "_etag": "e1"}


def _row(row_id, thread_id="t1"):
    return {"id": row_id, "type": "fact", "thread_id": thread_id}


def _memories(doc, absorbed=None, fail=False):
    """A memories container holding ``doc``, whose folded-record lookup answers ``absorbed`` (rows by the id the
    query names) or fails."""
    absorbed = absorbed or {}

    def rows(*, parameters, **_kwargs):
        if fail:
            raise RuntimeError("cosmos unavailable")
        return list(absorbed.get({p["name"]: p["value"] for p in parameters}["@memory_id"], []))

    memories = MagicMock()
    memories.read_item.return_value = doc
    memories.query_items.side_effect = rows
    return memories


def _deleted(memories):
    return [call.kwargs["item"] for call in memories.delete_item.call_args_list]


def test_deleting_a_fact_first_deletes_the_duplicates_merged_into_it():
    memories = _memories(_memory(), absorbed={"m1": [_row("copy_1", "t2")]})

    MemoryStore(containers=_containers(memories)).delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    assert _deleted(memories) == ["copy_1", "m1"]
    lookup = memories.query_items.call_args_list[0].kwargs
    params = {p["name"]: p["value"] for p in lookup["parameters"]}
    assert params["@user_id"] == "u1" and params["@memory_id"] == "m1"
    assert {v for k, v in params.items() if k.startswith("@reason_")} == {"combined", "duplicate", "procedure"}
    assert lookup["enable_cross_partition_query"] is True
    copy_delete, memory_delete = memories.delete_item.call_args_list
    assert copy_delete.kwargs["partition_key"] == ["u1", "t2"]
    assert memory_delete.kwargs["partition_key"] == ["u1", "t1"]


def test_deleting_a_procedure_deletes_the_statements_it_absorbed_and_their_copies():
    memories = _memories(
        _memory("procedural", "p1"),
        absorbed={"p1": [_row("said_1", "t2")], "said_1": [_row("said_1_copy", "t3")]},
    )

    MemoryStore(containers=_containers(memories)).delete("p1", user_id="u1", thread_id="t1", memory_type="procedural")

    assert _deleted(memories) == ["said_1_copy", "said_1", "p1"]


@pytest.mark.parametrize("memory_type", ["episodic", "turn", "thread_summary", "user_summary"])
def test_other_memory_types_are_deleted_without_a_lookup(memory_type):
    target = _memories(_memory(memory_type))
    containers = {ContainerKey.TURNS: target, ContainerKey.MEMORIES: target, ContainerKey.SUMMARIES: target}

    MemoryStore(containers=containers).delete("m1", user_id="u1", thread_id="t1", memory_type=memory_type)

    assert _deleted(target) == ["m1"]
    target.query_items.assert_not_called()


def test_a_copy_already_gone_does_not_fail_the_delete():
    memories = _memories(_memory(), absorbed={"m1": [_row("copy_1")]})

    def delete_item(*, item, **_kwargs):
        if item == "copy_1":
            raise CosmosResourceNotFoundError(status_code=404, message="not found")

    memories.delete_item.side_effect = delete_item

    MemoryStore(containers=_containers(memories)).delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    assert _deleted(memories) == ["copy_1", "m1"]


def test_a_failed_lookup_keeps_the_memory_for_a_retry():
    memories = _memories(_memory(), fail=True)

    with pytest.raises(CosmosOperationError):
        MemoryStore(containers=_containers(memories)).delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    memories.delete_item.assert_not_called()


def test_a_failed_delete_of_a_folded_record_keeps_the_memory_for_a_retry():
    memories = _memories(_memory(), absorbed={"m1": [_row("copy_1")]})
    memories.delete_item.side_effect = RuntimeError("throttled")

    with pytest.raises(CosmosOperationError):
        MemoryStore(containers=_containers(memories)).delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    assert _deleted(memories) == ["copy_1"]


def test_a_type_mismatch_deletes_nothing():
    memories = _memories(_memory("fact"), absorbed={"m1": [_row("copy_1")]})

    with pytest.raises(MemoryTypeMismatchError):
        MemoryStore(containers=_containers(memories)).delete(
            "m1", user_id="u1", thread_id="t1", memory_type="procedural"
        )

    memories.query_items.assert_not_called()
    memories.delete_item.assert_not_called()


class _Memories:
    """An in-memory memories container: point reads and deletes by ``[user_id, thread_id]``, and the folded-record
    lookup answered from the query parameters."""

    def __init__(self, *docs):
        self.docs = {(d["user_id"], d["thread_id"], d["id"]): d for d in docs}
        self.failing_id = None

    def read_item(self, *, item, partition_key):
        try:
            return self.docs[(*partition_key, item)]
        except KeyError:
            raise CosmosResourceNotFoundError(status_code=404, message="not found") from None

    def query_items(self, *, parameters, **_kwargs):
        params = {p["name"]: p["value"] for p in parameters}
        reasons = {v for k, v in params.items() if k.startswith("@reason_")}
        return [
            {key: d[key] for key in ("id", "type", "thread_id")}
            for d in self.docs.values()
            if d["user_id"] == params["@user_id"]
            and d.get("superseded_by") == params["@memory_id"]
            and d.get("supersede_reason") in reasons
        ]

    def delete_item(self, *, item, partition_key, **_kwargs):
        if item == self.failing_id:
            raise RuntimeError("throttled")
        try:
            del self.docs[(*partition_key, item)]
        except KeyError:
            raise CosmosResourceNotFoundError(status_code=404, message="not found") from None


def _stored(doc_id, thread_id, user_id="u1", **fields):
    return {"id": doc_id, "user_id": user_id, "thread_id": thread_id, "type": "fact", **fields}


def test_only_what_was_folded_into_the_memory_is_deleted_and_a_retry_finishes_the_job():
    memories = _Memories(
        _stored("m1", "t1"),
        _stored("dup", "t2", superseded_by="m1", supersede_reason="duplicate"),
        _stored("src", "t3", superseded_by="dup", supersede_reason="combined"),
        _stored("older", "t4", superseded_by="m1", supersede_reason="update"),
        _stored("lost", "t5", superseded_by="m1", supersede_reason="contradict"),
        _stored("theirs", "t2", user_id="u2", superseded_by="m1", supersede_reason="duplicate"),
    )
    store = MemoryStore(containers=_containers(memories))

    memories.failing_id = "dup"
    with pytest.raises(CosmosOperationError):
        store.delete("m1", user_id="u1", thread_id="t1", memory_type="fact")
    assert ("u1", "t1", "m1") in memories.docs and ("u1", "t3", "src") not in memories.docs

    memories.failing_id = None
    store.delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    assert set(memories.docs) == {("u1", "t4", "older"), ("u1", "t5", "lost"), ("u2", "t2", "theirs")}
