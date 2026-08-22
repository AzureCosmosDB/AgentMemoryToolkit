"""End-to-end scoped reconciliation tests (Layer 6 / Phase 5).

Reconciliation runs *within* a placement scope: contradictory facts in a team
scope are resolved (loser superseded) without touching a contradicting fact in
a different (private) scope. Requires the live LLM (contradiction detection).
"""

from __future__ import annotations

import uuid

import pytest

from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _active_facts_in_scope(shared_client, tenant_id: str, scope_key: str) -> list[dict]:
    """Return non-superseded facts in a scope (partition-scoped, no vector needed)."""
    query = (
        "SELECT * FROM c WHERE c.tenant_id = @t AND c.scope_key = @s AND c.type = 'fact' "
        "AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
    )
    return list(
        shared_client._memories_container_client.query_items(
            query=query,
            parameters=[{"name": "@t", "value": tenant_id}, {"name": "@s", "value": scope_key}],
            enable_cross_partition_query=True,
        )
    )


def test_reconcile_resolves_contradiction_within_team_scope(
    make_ctx, write_fact, shared_client, foundry_live
):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])

    tag = uuid.uuid4().hex[:6]
    write_fact(author, team, f"RECON-{tag} The production API rate limit is 100 requests per minute.")
    write_fact(author, team, f"RECON-{tag} The production API rate limit is 500 requests per minute.")

    before = _active_facts_in_scope(shared_client, author.tenant_id, team)
    assert len([f for f in before if tag in f.get("content", "")]) == 2

    result = shared_client.reconcile(scope_key=team, ctx=author)
    assert result["contradicted"] >= 1, f"expected a contradiction to be detected, got {result}"

    after = [f for f in _active_facts_in_scope(shared_client, author.tenant_id, team) if tag in f.get("content", "")]
    assert len(after) == 1, "exactly one of the contradictory team facts must remain active"


def test_reconcile_is_isolated_per_scope(make_ctx, write_fact, shared_client, foundry_live):
    """A team-scope reconcile must not supersede a contradicting fact in a private scope."""
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])

    tag = uuid.uuid4().hex[:6]
    write_fact(author, team, f"ISO-{tag} The office WiFi password rotates every 30 days.")
    write_fact(author, team, f"ISO-{tag} The office WiFi password rotates every 90 days.")
    # A contradicting statement in alice's PRIVATE scope - different placement.
    private_id = write_fact(author, "user:alice", f"ISO-{tag} The office WiFi password rotates every 7 days.")

    shared_client.reconcile(scope_key=team, ctx=author)

    private_active = [
        f for f in _active_facts_in_scope(shared_client, author.tenant_id, "user:alice") if tag in f.get("content", "")
    ]
    assert any(
        f["id"] == private_id for f in private_active
    ), "the private-scope fact must be untouched by a team reconcile"


def test_reconcile_requires_write_permission_on_scope(make_ctx, write_fact, shared_client, foundry_live):
    from azure.cosmos.agent_memory.exceptions import ValidationError

    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    write_fact(author, team, f"PERM-{uuid.uuid4().hex[:6]} a team fact")

    outsider = make_ctx("user:bob")  # not in the team
    with pytest.raises(ValidationError):
        shared_client.reconcile(scope_key=team, ctx=outsider)
