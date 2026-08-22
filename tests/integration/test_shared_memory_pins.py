"""End-to-end selective-injection (pin) tests (Layer 5 / Phase 7).

Covers pinning a memory and a scope to an agent, priority-ordered listing,
unpinning, permission gating on the agent scope, and blending pinned memories
into a search via ``include_pins``.
"""

from __future__ import annotations

import uuid

import pytest

from azure.cosmos.agent_memory.exceptions import ValidationError
from tests.conftest import INTEGRATION_ENABLED
from tests.integration.conftest import contents

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _agent_ctx(make_ctx, agent="copilot", **kw):
    """A context that owns ``agent:<agent>`` (so it may manage that agent's pins)."""
    return make_ctx("user:alice", agent_id=f"agent:{agent}", **kw)


def test_pin_list_and_unpin_a_memory(make_ctx, write_fact, shared_client, foundry_live):
    ctx = _agent_ctx(make_ctx)
    memory_id = write_fact(ctx, "user:alice", f"PIN-{uuid.uuid4().hex[:6]} remember to use tabs not spaces")

    shared_client.pin_memory("copilot", memory_id, injection_mode="direct", priority=80, ctx=ctx)
    pins = shared_client.list_pins("copilot", ctx=ctx)
    assert any(p.get("resource") == memory_id for p in pins)
    assert pins[0]["injection_mode"] == "direct"

    assert shared_client.unpin_memory("copilot", memory_id, ctx=ctx) is True
    assert all(p.get("resource") != memory_id for p in shared_client.list_pins("copilot", ctx=ctx))


def test_pins_are_listed_highest_priority_first(make_ctx, write_fact, shared_client, foundry_live):
    ctx = _agent_ctx(make_ctx)
    low = write_fact(ctx, "user:alice", f"PINLOW-{uuid.uuid4().hex[:6]} minor preference")
    high = write_fact(ctx, "user:alice", f"PINHIGH-{uuid.uuid4().hex[:6]} critical directive")

    shared_client.pin_memory("copilot", low, priority=10, ctx=ctx)
    shared_client.pin_memory("copilot", high, priority=90, ctx=ctx)

    pins = shared_client.list_pins("copilot", ctx=ctx)
    ordered = [p.get("resource") for p in pins if p.get("resource") in {low, high}]
    assert ordered[0] == high and ordered[-1] == low


def test_pin_requires_permission_on_agent_scope(make_ctx, write_fact, shared_client, foundry_live):
    owner = _agent_ctx(make_ctx)
    memory_id = write_fact(owner, "user:alice", f"PINPERM-{uuid.uuid4().hex[:6]} some fact")

    # A caller that does not own agent:copilot (and is not admin) cannot manage its pins.
    intruder = make_ctx("user:mallory")  # no agent_id, no roles
    with pytest.raises(ValidationError):
        shared_client.pin_memory("copilot", memory_id, ctx=intruder)


def test_include_pins_blends_pinned_memory_into_search(
    make_ctx, write_fact, shared_client, search_until, foundry_live
):
    ctx = _agent_ctx(make_ctx)
    # A memory whose wording will NOT match the query terms, so only the pin surfaces it.
    marker = f"PINBLEND-{uuid.uuid4().hex[:6]} zzznonmatching lexical token qqzz"
    memory_id = write_fact(ctx, "user:alice", marker)
    shared_client.pin_memory("copilot", memory_id, injection_mode="direct", priority=95, ctx=ctx)

    def _pinned_present(results: list[dict]) -> bool:
        return any(marker in c for c in contents(results))

    # Unrelated query; the pinned memory is injected regardless of vector relevance.
    results = search_until(
        ctx,
        "completely unrelated weather forecast query",
        _pinned_present,
        agent_id="copilot",
        include_pins=True,
    )
    assert _pinned_present(results), "an explicitly pinned memory must be blended into the search results"
