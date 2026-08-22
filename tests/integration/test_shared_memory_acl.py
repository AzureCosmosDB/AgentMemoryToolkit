"""End-to-end inline-ACL enforcement tests.

Proves record read authorization - the inline ``acl.read`` resolved against the caller's
Entra-derived subjects - is enforced live: restricted reads are hidden by the pre-rank
Cosmos predicate and cross-tenant reads are impossible. Write / forget / annotate are
authorized at the placement-scope level (``resolve_scope_access``), not per record, so
those decisions are asserted against the scope rather than an inline ACL.
"""

from __future__ import annotations

import uuid

import pytest

from azure.cosmos.agent_memory._authz import can, resolve_scope_access
from tests.conftest import INTEGRATION_ENABLED
from tests.integration.conftest import contents

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _has(results: list[dict], needle: str) -> bool:
    return any(needle in c for c in contents(results))


# ---------------------------------------------------------------------------
# Read enforcement (the primary security boundary) via search
# ---------------------------------------------------------------------------


def test_restricted_acl_hides_record_from_plain_team_member(
    make_ctx, write_with_acl, search_as, search_until, foundry_live
):
    """A team-placed record whose ACL is narrowed to one lead is invisible to other members."""
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    marker = f"RESTRICTED-{uuid.uuid4().hex[:8]} the acquisition of northwind closes next quarter"

    author = make_ctx("user:lead", groups=[gid])
    write_with_acl(author, team, marker, {"read": ["user:lead"]})

    lead = make_ctx("user:lead", groups=[gid])
    member = make_ctx("user:carol", groups=[gid])

    got = search_until(lead, "acquisition of northwind closes next quarter", lambda r: _has(r, marker))
    assert _has(got, marker), "the named lead must still read the restricted record"
    assert not _has(
        search_as(member, "acquisition of northwind closes next quarter"), marker
    ), "a plain team member must not read a record whose acl.read is narrowed to the lead"


def test_user_scope_tenant_isolation_via_search(make_ctx, write_fact, shared_client, search_until, foundry_live):
    marker = f"TENANTISO-{uuid.uuid4().hex[:8]} my private banking pin hint is the dog's birthday"
    alice = make_ctx("user:alice")
    write_fact(alice, "user:alice", marker)

    # Same principal string, *different* tenant -> hard isolation boundary.
    other = make_ctx("user:alice", tenant_id="it-other-" + uuid.uuid4().hex[:8])
    results = shared_client.session(other).search_cosmos("private banking pin hint dog birthday", top_k=10)
    assert not _has(results, marker), "a different tenant must never read another tenant's memory"

    # Sanity: the owner in the correct tenant *can* read it (guards against a false-negative).
    assert _has(search_until(alice, "private banking pin hint dog birthday", lambda r: _has(r, marker)), marker)


# ---------------------------------------------------------------------------
# Write / forget / annotate / admin decisions (scope-level authorization)
# ---------------------------------------------------------------------------


def test_read_gating_and_scope_level_write_forget(make_ctx, write_fact, read_record, foundry_live):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid])
    memory_id = write_fact(author, team, f"WRITEGATE-{uuid.uuid4().hex[:6]} team norm: code review within 24h")

    doc = read_record(tenant_id=author.tenant_id, scope_key=team, thread_id="t1", memory_id=memory_id)
    acl = doc["acl"]
    member = make_ctx("user:carol", groups=[gid])
    outsider = make_ctx("user:bob")

    # Inline ACL governs reads only: the team subject reads, an outsider does not.
    assert can(member, acl), "team members can read the team-placed record"
    assert not can(outsider, acl), "a non-member cannot read"

    # Write / forget are scope-level: plain membership is read-only, an explicit team
    # writer/member role is required to write or forget into the team scope.
    assert resolve_scope_access(member, [team], "read").allowed_scopes == {team}
    assert resolve_scope_access(member, [team], "write").allowed_scopes == set()
    assert resolve_scope_access(outsider, [team], "read").allowed_scopes == set()
    writer = make_ctx("user:dave", roles=[f"{team}:writer"])
    assert resolve_scope_access(writer, [team], "write").allowed_scopes == {team}
    assert resolve_scope_access(writer, [team], "forget").allowed_scopes == {team}


def test_annotate_is_scope_level_for_members(make_ctx):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    # annotate follows write authorization at the scope level: a team member/writer may
    # annotate; a plain (read-only) member and an outsider may not.
    member_role = make_ctx("user:svc", roles=[f"{team}:member"])
    assert resolve_scope_access(member_role, [team], "annotate").allowed_scopes == {team}
    plain = make_ctx("user:carol", groups=[gid])
    assert resolve_scope_access(plain, [team], "annotate").allowed_scopes == set()
    outsider = make_ctx("user:bob")
    assert resolve_scope_access(outsider, [team], "annotate").allowed_scopes == set()


def test_admin_role_short_circuits_read_and_scope_write(make_ctx, write_with_acl, read_record, foundry_live):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid])
    memory_id = write_with_acl(
        author, team, f"ADMIN-{uuid.uuid4().hex[:6]} locked record", {"read": ["user:nobody"]}
    )
    acl = read_record(tenant_id=author.tenant_id, scope_key=team, thread_id="t1", memory_id=memory_id)["acl"]

    admin = make_ctx("user:root", roles=["memory:admin"])
    assert can(admin, acl), "an admin reads any record in its tenant"
    assert resolve_scope_access(admin, [team], "write").allowed_scopes == {team}
    assert resolve_scope_access(admin, [team], "forget").allowed_scopes == {team}
    # A non-admin whose subjects are not in acl.read stays denied.
    assert not can(make_ctx("user:carol", groups=[gid]), acl)
