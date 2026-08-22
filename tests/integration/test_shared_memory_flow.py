"""One cohesive end-to-end shared-memory flow.

Exercises the shared-memory code paths together, in a single isolated tenant:
team write -> scoped reconcile -> teammate retrieval -> promote to org ->
org-wide retrieval -> cross-tenant denial. Requires the live LLM (reconcile).
"""

from __future__ import annotations

import uuid

import pytest

from tests.conftest import INTEGRATION_ENABLED
from tests.integration.conftest import contents

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _has(results: list[dict], needle: str) -> bool:
    return any(needle in c for c in contents(results))


def test_team_reconcile_promote_org_read_flow(
    make_ctx, write_fact, shared_client, search_until, tenant, foundry_live
):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    tag = uuid.uuid4().hex[:6]

    # The author reconciles within the team (a write) and promotes to org, so it holds
    # explicit member roles on both scopes; readers below stay unprivileged.
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member", f"org:{tenant}:member"])
    teammate = make_ctx("user:carol", groups=[gid])
    org_peer = make_ctx("user:zoe")  # same tenant, not in the team
    outsider = make_ctx("user:eve", tenant_id="it-other-" + uuid.uuid4().hex[:8])

    # 1) The team accrues two contradictory norms plus one durable standard.
    write_fact(author, team, f"FLOW-{tag} Our SLA response time target is 4 hours.")
    write_fact(author, team, f"FLOW-{tag} Our SLA response time target is 2 hours.")
    standard = f"FLOW-{tag} All services must emit OpenTelemetry traces."
    standard_id = write_fact(author, team, standard)

    # 2) Scoped reconcile collapses the contradiction inside the team scope.
    result = shared_client.reconcile(scope_key=team, ctx=author)
    assert result["contradicted"] >= 1

    # 3) A teammate can retrieve the surviving shared knowledge.
    teammate_hits = search_until(teammate, "services must emit opentelemetry traces", lambda r: _has(r, standard))
    assert _has(teammate_hits, standard)

    # 4) Promote the durable standard org-wide.
    org = f"org:{author.tenant_id}"
    shared_client.promote(standard_id, team, org, author)

    # 5) Any tenant member (outside the team) now reads it via the org scope.
    org_hits = search_until(org_peer, "services must emit opentelemetry traces", lambda r: _has(r, standard))
    assert _has(org_hits, standard)

    # 6) A different tenant sees none of it (hard isolation).
    outsider_hits = shared_client.session(outsider).search_cosmos(
        "services must emit opentelemetry traces", top_k=10
    )
    assert not _has(outsider_hits, standard)
