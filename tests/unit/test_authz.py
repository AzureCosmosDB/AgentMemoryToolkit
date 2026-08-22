from __future__ import annotations

from unittest.mock import MagicMock

from azure.cosmos.agent_memory._authz import build_acl_predicate, caller_subjects, can, resolve_scope_access
from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.models import MemoryAcl
from azure.cosmos.agent_memory.store import MemoryStore


def _containers(*, memories=None):
    return {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }


def test_resolver_allows_own_user_scope_without_grants():
    ctx = SecurityContext(tenant_id="default", principal="user:alice")

    result = resolve_scope_access(ctx, ["user:alice"], "read")

    assert result.allowed_scopes == {"user:alice"}
    assert result.decisions["user:alice"].reason == "own_scope"


def test_resolver_denies_unrelated_scope_without_membership():
    ctx = SecurityContext(tenant_id="tenant-a", principal="user:alice")

    result = resolve_scope_access(ctx, ["team:eng"], "read")

    assert result.allowed_scopes == set()
    assert result.decisions["team:eng"].reason == "deny_default"
    assert {"name": "@authz_tenant_id", "value": "tenant-a"} in result.predicate.parameters


def test_resolver_role_default_admin_write_short_circuits_without_acl():
    ctx = SecurityContext(tenant_id="acme", principal="user:admin", roles=["tenant:admin"])

    result = resolve_scope_access(ctx, ["team:eng"], "write")

    assert result.allowed_scopes == {"team:eng"}
    assert result.decisions["team:eng"].reason == "role_default_admin"


def test_caller_subjects_and_can_use_inline_acl():
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", agent_id="agent:copilot", groups=["eng"])

    assert {"user:alice", "agent:copilot", "team:eng", "org:acme", "tenant:*"} <= caller_subjects(ctx)
    assert can(ctx, MemoryAcl(read=["team:eng"]))
    assert can(ctx, {"read": ["user:alice"]})
    assert not can(ctx, MemoryAcl(read=["user:lead"]))


def test_acl_predicate_has_scope_union_and_array_contains_subjects():
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", groups=["eng"])

    predicate = build_acl_predicate(ctx, ["user:alice", "team:eng"])

    assert "c.tenant_id = @authz_tenant_id" in predicate.sql
    assert "c.scope_key IN" in predicate.sql
    assert "ARRAY_CONTAINS(c.acl.read" in predicate.sql
    params = {p["name"]: p["value"] for p in predicate.parameters}
    assert params["@authz_tenant_id"] == "acme"
    assert "user:alice" in params.values()
    assert "team:eng" in params.values()


def test_project_membership_role_grants_matching_acl_subject():
    """A ``project:<id>:member`` role must satisfy the default project ACL subject.

    Without this, project placement passes but the inline ACL (``read=[project:atlas]``)
    never matches any caller subject, so project reads silently return nothing. Writes
    are authorized at the scope level, not via the inline ACL.
    """
    acl = MemoryAcl(read=["project:atlas"])

    member = SecurityContext(tenant_id="acme", principal="user:m", roles=["project:atlas:member"])
    assert can(member, acl)
    assert resolve_scope_access(member, ["project:atlas"], "write").allowed_scopes == {"project:atlas"}
    # The read predicate must include the derived project subject for the ARRAY_CONTAINS.
    predicate = build_acl_predicate(member, ["project:atlas"])
    assert "project:atlas" in {p["value"] for p in predicate.parameters}


def test_project_reader_role_is_read_only_and_writer_can_write():
    acl = MemoryAcl(read=["project:atlas"])

    reader = SecurityContext(tenant_id="acme", principal="user:r", roles=["scope:project:atlas:reader"])
    assert can(reader, acl)
    assert resolve_scope_access(reader, ["project:atlas"], "read").allowed_scopes == {"project:atlas"}
    assert resolve_scope_access(reader, ["project:atlas"], "write").allowed_scopes == set()
    assert resolve_scope_access(reader, ["project:atlas"], "forget").allowed_scopes == set()

    writer = SecurityContext(tenant_id="acme", principal="user:w", roles=["project:atlas:writer"])
    assert can(writer, acl)
    assert resolve_scope_access(writer, ["project:atlas"], "write").allowed_scopes == {"project:atlas"}

    outsider = SecurityContext(tenant_id="acme", principal="user:o")
    assert not can(outsider, acl)
    assert resolve_scope_access(outsider, ["project:atlas"], "read").allowed_scopes == set()


def test_project_membership_role_does_not_leak_across_projects():
    acl_other = MemoryAcl(read=["project:zeus"])
    member = SecurityContext(tenant_id="acme", principal="user:m", roles=["project:atlas:member"])

    assert not can(member, acl_other)
    assert resolve_scope_access(member, ["project:zeus"], "read").allowed_scopes == set()


def test_scope_suffixed_admin_role_is_not_tenant_admin():
    """A per-scope ``<scope>:admin`` role must not confer tenant-wide admin.

    Otherwise a project/team admin could read and forget records anywhere in the tenant.
    """
    ctx = SecurityContext(tenant_id="acme", principal="user:x", roles=["project:atlas:admin"])

    assert not can(ctx, MemoryAcl(read=["user:nobody"]))
    assert resolve_scope_access(ctx, ["team:secret"], "read").allowed_scopes == set()
    assert resolve_scope_access(ctx, ["team:secret"], "forget").allowed_scopes == set()


def test_bare_scope_shaped_role_does_not_grant_membership():
    """A role string shaped like a scope must not become a membership subject.

    Only the ``<scope>:reader/member/writer`` grammar grants access; a raw ``team:secret``
    role must not read or write ``team:secret``.
    """
    ctx = SecurityContext(tenant_id="acme", principal="user:x", roles=["team:secret"])

    assert "team:secret" not in caller_subjects(ctx)
    assert not can(ctx, MemoryAcl(read=["team:secret"]))
    assert resolve_scope_access(ctx, ["team:secret"], "read").allowed_scopes == set()
    assert resolve_scope_access(ctx, ["team:secret"], "write").allowed_scopes == set()


def test_org_member_can_read_but_not_write_tenant_scope():
    """Tenant membership grants org-wide READ only; org writes need an explicit role."""
    member = SecurityContext(tenant_id="acme", principal="user:m", groups=["eng"])
    assert resolve_scope_access(member, ["org:acme"], "read").allowed_scopes == {"org:acme"}
    assert resolve_scope_access(member, ["org:acme"], "write").allowed_scopes == set()
    assert resolve_scope_access(member, ["org:acme"], "forget").allowed_scopes == set()

    writer = SecurityContext(tenant_id="acme", principal="user:w", roles=["org:acme:writer"])
    assert resolve_scope_access(writer, ["org:acme"], "write").allowed_scopes == {"org:acme"}


def test_store_search_pushes_acl_predicate_before_ranking():
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", groups=["eng"])

    result = store.search(search_terms="hello", scopes=["user:alice", "team:eng", "team:secret"], ctx=ctx)

    assert result == []
    assert memories.query_items.call_count == 2
    queries = [call.kwargs["query"] for call in memories.query_items.call_args_list]
    assert all("ARRAY_CONTAINS(c.acl.read" in query for query in queries)
    assert all("VectorDistance" in query for query in queries)


def test_membership_keys_off_entra_groups():
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", groups=["eng-gid"])
    res = resolve_scope_access(ctx, ["team:eng-gid", "team:sales-gid", "org:acme"], "read")
    assert res.allowed_scopes == {"team:eng-gid", "org:acme"}


def test_agent_owns_its_own_agent_scope_in_delegated_mode():
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", agent_id="agent:copilot-sp")
    res = resolve_scope_access(ctx, ["agent:copilot-sp"], "write")
    assert "agent:copilot-sp" in res.allowed_scopes


def test_resolve_read_scopes_composition_and_cap():
    from azure.cosmos.agent_memory._authz import MAX_READ_SCOPES, resolve_read_scopes

    ctx = SecurityContext(
        tenant_id="acme",
        principal="user:alice",
        agent_id="agent:copilot-sp",
        groups=["g1", "g2", "g3", "g4", "g5"],
    )
    scopes = resolve_read_scopes(ctx, project_scope="project:payments-api")
    assert len(scopes) == MAX_READ_SCOPES == 5
    assert scopes[0] == "user:alice"
    assert "agent:copilot-sp" in scopes
    assert "project:payments-api" in scopes
    assert "org:acme" in scopes
    assert sum(1 for s in scopes if s.startswith("team:")) == 1
