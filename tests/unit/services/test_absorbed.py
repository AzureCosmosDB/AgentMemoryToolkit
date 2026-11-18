"""The walk that finds and deletes the records folded into a memory."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory.aio.services import pipeline as aio_pipeline
from azure.cosmos.agent_memory.models import MemoryRecordBase
from azure.cosmos.agent_memory.services import pipeline as sync_pipeline
from azure.cosmos.agent_memory.services._absorbed import (
    ABSORBED_REASONS,
    absorbed_query,
    adelete_absorbed,
    delete_absorbed,
)
from azure.cosmos.agent_memory.services._combine import COMBINE_REASON

# m1 absorbed a and b; a absorbed c; a bad record points c back at a. The records come from different threads, which
# does not matter: they all sit in the owner's partition.
FOLDED = {
    "m1": [{"id": "a", "type": "fact"}, {"id": "b", "type": "fact"}],
    "a": [{"id": "c", "type": "fact"}],
    "c": [{"id": "a", "type": "fact"}],
}
OWNER_PARTITION = ["acme", "u1"]


def _rows(*, query, parameters, **_kwargs):
    params = {p["name"]: p["value"] for p in parameters}
    assert (params["@tenant_id"], params["@user_id"]) == ("acme", "u1")
    return list(FOLDED.get(params["@memory_id"], []))


def _deleted(container):
    return {call.kwargs["item"]: call.kwargs["partition_key"] for call in container.delete_item.call_args_list}


def _async_container():
    def query_items(**kwargs):
        async def iterate():
            for row in _rows(**kwargs):
                yield row

        return iterate()

    container = MagicMock()
    container.query_items = MagicMock(side_effect=query_items)
    container.delete_item = AsyncMock()
    return container


def test_the_query_names_only_duplicate_combined_and_procedure_retirements():
    query, parameters = absorbed_query("acme", "u1", "m1")

    params = {p["name"]: p["value"] for p in parameters}
    assert "c.tenant_id = @tenant_id AND c.user_id = @user_id" in query
    assert (params["@tenant_id"], params["@user_id"]) == ("acme", "u1")
    assert "c.superseded_by = @memory_id" in query and params["@memory_id"] == "m1"
    assert sorted(v for k, v in params.items() if k.startswith("@reason_")) == ["combined", "duplicate", "procedure"]


def test_everything_folded_in_is_deleted_once_children_before_parents():
    container = MagicMock()
    container.query_items.side_effect = _rows

    found = delete_absorbed(container, "acme", "u1", "m1")

    assert sorted(row["id"] for row in found) == ["a", "b", "c"]
    deleted = [call.kwargs["item"] for call in container.delete_item.call_args_list]
    assert sorted(deleted) == ["a", "b", "c"] and deleted.index("c") < deleted.index("a")


def test_every_record_is_looked_up_and_deleted_in_the_owners_partition():
    container = MagicMock()
    container.query_items.side_effect = _rows

    delete_absorbed(container, "acme", "u1", "m1")

    queried = [call.kwargs["partition_key"] for call in container.query_items.call_args_list]
    assert queried and all(key == OWNER_PARTITION for key in queried)
    assert not any("enable_cross_partition_query" in call.kwargs for call in container.query_items.call_args_list)
    assert _deleted(container) == {"a": OWNER_PARTITION, "b": OWNER_PARTITION, "c": OWNER_PARTITION}


def test_a_record_already_gone_is_skipped_and_other_failures_raise():
    container = MagicMock()
    container.query_items.side_effect = _rows
    container.delete_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="not found")

    assert len(delete_absorbed(container, "acme", "u1", "m1")) == 3

    container.delete_item.side_effect = RuntimeError("throttled")
    with pytest.raises(RuntimeError):
        delete_absorbed(container, "acme", "u1", "m1")


def test_nothing_folded_in_means_nothing_is_deleted():
    container = MagicMock()
    container.query_items.return_value = []

    assert delete_absorbed(container, "acme", "u1", "m1") == []
    container.delete_item.assert_not_called()


def _reasons_the_pipeline_retires_with(module) -> set[str]:
    nodes = ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8")))
    return {
        keyword.value.value
        for node in nodes
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_mark_superseded"
        for keyword in node.keywords
        if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant)
    }


@pytest.mark.parametrize("module", [sync_pipeline, aio_pipeline], ids=["sync", "async"])
def test_the_cascade_follows_exactly_the_retirements_that_leave_a_copy_behind(module):
    """A record retired as a duplicate, a combined member or a taught fact holds what its survivor holds, so deleting
    the survivor deletes it; a record an earlier version retired as contradicted or updated is an earlier version and
    stays."""
    written = _reasons_the_pipeline_retires_with(module)
    # What ``supersede_reason`` accepts: the vocabulary the pipelines, the models and the cascade share.
    accepted = set(get_args(get_args(MemoryRecordBase.model_fields["supersede_reason"].annotation)[0]))

    assert written == {"duplicate", "combined", "procedure"}
    assert written <= accepted and set(ABSORBED_REASONS) <= accepted
    assert set(ABSORBED_REASONS) == written
    assert COMBINE_REASON in ABSORBED_REASONS
    assert accepted - set(ABSORBED_REASONS) == {"contradict", "update"}


async def test_the_async_walk_matches_the_sync_one():
    container = _async_container()

    found = await adelete_absorbed(container, "acme", "u1", "m1")

    assert sorted(row["id"] for row in found) == ["a", "b", "c"]
    deleted = [call.kwargs["item"] for call in container.delete_item.call_args_list]
    assert sorted(deleted) == ["a", "b", "c"] and deleted.index("c") < deleted.index("a")
    queried = [call.kwargs["partition_key"] for call in container.query_items.call_args_list]
    assert queried and all(key == OWNER_PARTITION for key in queried)
    assert _deleted(container) == {"a": OWNER_PARTITION, "b": OWNER_PARTITION, "c": OWNER_PARTITION}


async def test_the_async_walk_skips_a_record_already_gone_and_raises_other_failures():
    container = _async_container()
    container.delete_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="not found")

    assert len(await adelete_absorbed(container, "acme", "u1", "m1")) == 3

    container.delete_item.side_effect = RuntimeError("throttled")
    with pytest.raises(RuntimeError):
        await adelete_absorbed(container, "acme", "u1", "m1")
