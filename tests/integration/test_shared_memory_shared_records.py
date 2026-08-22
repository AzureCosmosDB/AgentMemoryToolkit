"""End-to-end tests for live shared coordination records (Layer 6b).

Covers put/get round-trip, ETag optimistic concurrency (compare-and-swap +
read-modify-write retry), the ``read_only`` guard, and scope-gated access.
These use only Cosmos (no embeddings / LLM), so they run whenever the account
is reachable.
"""

from __future__ import annotations

import uuid

import pytest

from azure.cosmos.agent_memory.exceptions import MemoryConflictError, SharedRecordReadOnlyError, ValidationError
from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def test_put_then_get_round_trips(make_ctx, shared_client):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    key = "plan-" + uuid.uuid4().hex[:6]

    put = shared_client.put_shared_record(
        ctx=author, scope_key=team, key=key, content="phase 1", data={"steps": ["a", "b"]}
    )
    assert put.get("_etag")

    got = shared_client.get_shared_record(ctx=author, scope_key=team, key=key)
    assert got["content"] == "phase 1"
    assert got["data"] == {"steps": ["a", "b"]}


def test_compare_and_swap_rejects_stale_etag(make_ctx, shared_client):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    key = "queue-" + uuid.uuid4().hex[:6]

    first = shared_client.put_shared_record(ctx=author, scope_key=team, key=key, data={"n": 0})
    stale_etag = first["_etag"]

    # A concurrent writer advances the record (new etag).
    shared_client.compare_and_swap_shared_record(
        ctx=author, scope_key=team, key=key, record={"data": {"n": 1}}, etag=stale_etag
    )

    # The original holder now has a stale etag -> conflict.
    with pytest.raises(MemoryConflictError):
        shared_client.compare_and_swap_shared_record(
            ctx=author, scope_key=team, key=key, record={"data": {"n": 99}}, etag=stale_etag
        )

    assert shared_client.get_shared_record(ctx=author, scope_key=team, key=key)["data"] == {"n": 1}


def test_update_shared_record_read_modify_write(make_ctx, shared_client):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    key = "counter-" + uuid.uuid4().hex[:6]
    shared_client.put_shared_record(ctx=author, scope_key=team, key=key, data={"n": 0})

    def bump(draft: dict) -> dict:
        draft["data"] = {"n": int(draft.get("data", {}).get("n", 0)) + 1}
        return draft

    for _ in range(3):
        shared_client.update_shared_record(ctx=author, scope_key=team, key=key, mutator=bump)

    assert shared_client.get_shared_record(ctx=author, scope_key=team, key=key)["data"] == {"n": 3}


def test_read_only_record_cannot_be_mutated(make_ctx, shared_client):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    key = "config-" + uuid.uuid4().hex[:6]
    shared_client.put_shared_record(ctx=author, scope_key=team, key=key, content="frozen", read_only=True)

    with pytest.raises(SharedRecordReadOnlyError):
        shared_client.put_shared_record(ctx=author, scope_key=team, key=key, content="tampered")
    with pytest.raises(SharedRecordReadOnlyError):
        shared_client.update_shared_record(ctx=author, scope_key=team, key=key, mutator=lambda d: d)

    assert shared_client.get_shared_record(ctx=author, scope_key=team, key=key)["content"] == "frozen"


def test_shared_record_read_is_scope_gated(make_ctx, shared_client):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    key = "handoff-" + uuid.uuid4().hex[:6]
    shared_client.put_shared_record(ctx=author, scope_key=team, key=key, content="secret handoff")

    outsider = make_ctx("user:bob")  # not in the group
    with pytest.raises(ValidationError):
        shared_client.get_shared_record(ctx=outsider, scope_key=team, key=key)
    with pytest.raises(ValidationError):
        shared_client.put_shared_record(ctx=outsider, scope_key=team, key=key, content="intrusion")
