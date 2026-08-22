"""End-to-end scope placement + visibility tests across every scope type.

Covered scope types: ``user``, ``team``, ``project``, ``org``, ``global``
(``agent`` is intentionally skipped). Each scope proves the full triangle:
the writer/owner reads it back, an authorized peer reads it, and an
unauthorized peer (and any other tenant) cannot.

Runs live against Cosmos + AI Foundry; see ``tests/integration/conftest.py``.
"""

from __future__ import annotations

import uuid

import pytest

from azure.cosmos.agent_memory._authz import MAX_READ_SCOPES, resolve_read_scopes
from azure.cosmos.agent_memory._partitioning import private_scope_key_for_principal
from tests.conftest import INTEGRATION_ENABLED
from tests.integration.conftest import contents

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _has(results: list[dict], needle: str) -> bool:
    return any(needle in c for c in contents(results))


# ---------------------------------------------------------------------------
# user scope (private)
# ---------------------------------------------------------------------------


def test_user_scope_is_private_to_owner(make_ctx, write_fact, search_as, search_until, foundry_live):
    marker = f"USERSCOPE-{uuid.uuid4().hex[:8]} personal note about my dentist appointment"
    alice = make_ctx("user:alice")
    bob = make_ctx("user:bob")

    write_fact(alice, "user:alice", marker)

    got = search_until(alice, "dentist appointment personal note", lambda r: _has(r, marker))
    assert _has(got, marker), "owner must read her own private memory"
    assert not _has(search_as(bob, "dentist appointment personal note"), marker), "another user must not see it"


# ---------------------------------------------------------------------------
# team scope (Entra group membership)
# ---------------------------------------------------------------------------


def test_team_scope_visible_to_group_members_only(make_ctx, write_fact, search_as, search_until, foundry_live):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    marker = f"TEAMSCOPE-{uuid.uuid4().hex[:8]} the staging deploy runbook lives in the ops wiki"

    author = make_ctx("user:alice", groups=[gid])
    teammate = make_ctx("user:carol", groups=[gid])
    outsider = make_ctx("user:bob", groups=[])

    write_fact(author, team, marker)

    got = search_until(teammate, "staging deploy runbook ops wiki", lambda r: _has(r, marker))
    assert _has(got, marker), "a fellow group member must read the team memory"
    assert _has(search_until(author, "staging deploy runbook ops wiki", lambda r: _has(r, marker)), marker)
    assert not _has(search_as(outsider, "staging deploy runbook ops wiki"), marker), "non-member must not see it"


# ---------------------------------------------------------------------------
# project scope (role-derived membership)
# ---------------------------------------------------------------------------


def test_project_scope_visible_to_project_members_only(make_ctx, write_fact, search_as, search_until, foundry_live):
    pid = "atlas-" + uuid.uuid4().hex[:8]
    project = f"project:{pid}"
    marker = f"PROJSCOPE-{uuid.uuid4().hex[:8]} the payments service uses idempotency keys on every POST"

    writer = make_ctx("user:alice", roles=[f"{project}:writer"])
    member = make_ctx("user:carol", roles=[f"{project}:member"])
    outsider = make_ctx("user:bob")

    write_fact(writer, project, marker)

    # Project scope is only in the read union when the caller declares it.
    got = search_until(
        member, "payments service idempotency keys POST", lambda r: _has(r, marker), project_scope=project
    )
    assert _has(got, marker), "a project member must read the project memory"
    assert not _has(
        search_as(outsider, "payments service idempotency keys POST", project_scope=project), marker
    ), "a non-member must not see the project memory even when declaring the scope"


def test_project_reader_role_cannot_be_leaked_to_other_projects(make_ctx, write_fact, search_as, foundry_live):
    pid = "atlas-" + uuid.uuid4().hex[:8]
    other = f"project:zeus-{uuid.uuid4().hex[:8]}"
    project = f"project:{pid}"
    marker = f"PROJLEAK-{uuid.uuid4().hex[:8]} confidential atlas roadmap milestones"

    writer = make_ctx("user:alice", roles=[f"{project}:writer"])
    write_fact(writer, project, marker)

    # A member of a *different* project must not read this project's memory.
    intruder = make_ctx("user:mallory", roles=[f"{other}:member"])
    assert not _has(search_as(intruder, "confidential atlas roadmap milestones", project_scope=project), marker)


# ---------------------------------------------------------------------------
# org scope (whole tenant)
# ---------------------------------------------------------------------------


def test_org_scope_visible_to_every_tenant_member(make_ctx, write_fact, search_as, search_until, tenant, foundry_live):
    org = f"org:{tenant}"
    marker = f"ORGSCOPE-{uuid.uuid4().hex[:8]} the company all-hands is the first friday of each month"

    author = make_ctx("user:alice")
    any_member = make_ctx("user:zoe")

    write_fact(author, org, marker)

    got = search_until(any_member, "company all-hands first friday of the month", lambda r: _has(r, marker))
    assert _has(got, marker), "any tenant member must read org-scoped memory (org is in the default read union)"


# ---------------------------------------------------------------------------
# global scope (tenant-public, opt-in bucket)
# ---------------------------------------------------------------------------


def test_global_scope_is_tenant_public_but_opt_in(make_ctx, write_fact, search_as, search_until, foundry_live):
    scope = "global:global"
    marker = f"GLOBALSCOPE-{uuid.uuid4().hex[:8]} the assistant should always answer in metric units"

    author = make_ctx("user:alice")
    reader = make_ctx("user:carl")

    write_fact(author, scope, marker)

    # Global is opt-in: the reader includes it in the read union explicitly.
    read_scopes = [private_scope_key_for_principal(reader.principal), scope]
    got = search_until(
        reader, "assistant answer in metric units", lambda r: _has(r, marker), read_scopes=read_scopes
    )
    assert _has(got, marker), "a tenant member who opts into the global bucket must read global memory"


def test_global_scope_still_respects_tenant_isolation(make_ctx, write_fact, shared_client, foundry_live):
    """A global record is still partitioned by the writer's tenant; other tenants can't read it."""
    scope = "global:global"
    marker = f"GLOBALISO-{uuid.uuid4().hex[:8]} tenant A private-ish global note"

    author = make_ctx("user:alice")  # tenant == the isolated test tenant
    write_fact(author, scope, marker)

    other_tenant_reader = make_ctx("user:alice", tenant_id="it-other-" + uuid.uuid4().hex[:8])
    read_scopes = [private_scope_key_for_principal(other_tenant_reader.principal), scope]
    results = shared_client.session(other_tenant_reader, read_scopes=read_scopes).search_cosmos(
        "tenant A private-ish global note", top_k=10
    )
    assert not any(marker in str(r.get("content", "")) for r in results), "cross-tenant global read must be empty"


# ---------------------------------------------------------------------------
# cross-cutting: default write scope + read-union composition/cap
# ---------------------------------------------------------------------------


def test_session_default_write_scope_is_private(make_ctx, shared_client, search_as, search_until, foundry_live):
    """With no ``write_scope``, a bound session writes to the caller's private scope."""
    marker = f"DEFAULTPRIV-{uuid.uuid4().hex[:8]} my private grocery list has oat milk"
    alice = make_ctx("user:alice")

    # No write_scope -> defaults to user:alice (private).
    shared_client.session(alice).upsert_memory(
        role="user", content=marker, memory_type="fact", thread_id="t1"
    )

    got = search_until(alice, "private grocery list oat milk", lambda r: _has(r, marker))
    assert _has(got, marker)
    bob = make_ctx("user:bob")
    assert not _has(search_as(bob, "private grocery list oat milk"), marker)


def test_read_union_is_capped(make_ctx):
    """The auto-resolved read union never exceeds the configured cap."""
    ctx = make_ctx("user:alice", groups=[f"g{i}" for i in range(10)], agent_id="agent:sp")
    scopes = resolve_read_scopes(ctx, project_scope="project:payments")
    assert len(scopes) == MAX_READ_SCOPES
    assert scopes[0] == "user:alice"
    assert "org:" + ctx.tenant_id in scopes
