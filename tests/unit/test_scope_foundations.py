from __future__ import annotations

import pydantic
import pytest

from azure.cosmos.agent_memory import SecurityContext
from azure.cosmos.agent_memory.models import MemoryAcl, MemoryRecordBase, MemoryScopeType


def test_memory_scope_type_values():
    assert {scope.value for scope in MemoryScopeType} == {"user", "agent", "team", "project", "org", "global"}
    assert MemoryScopeType.global_scope.value == "global"


def test_memory_acl_dedupes_subjects():
    acl = MemoryAcl(read=["team:eng", "team:eng", "user:lead"])

    assert acl.read == ["team:eng", "user:lead"]


def test_scope_key_derives_when_scope_type_and_scope_id_are_set():
    rec = MemoryRecordBase(content="remember this", scope_type="team", scope_id="eng")

    assert rec.scope_type == "team"
    assert rec.scope_id == "eng"
    assert rec.scope_key == "team:eng"


def test_record_without_scope_fields_leaves_scope_unset():
    rec = MemoryRecordBase(content="remember this")

    assert rec.scope_type is None
    assert rec.scope_id is None
    assert rec.scope_key is None


def test_scope_key_is_not_overwritten_when_supplied():
    rec = MemoryRecordBase(
        content="remember this",
        scope_type=MemoryScopeType.project,
        scope_id="repo",
        scope_key="custom:repo",
    )

    assert rec.scope_key == "custom:repo"


def test_user_id_constructor_input_maps_to_scope_not_stored():
    rec = MemoryRecordBase(user_id="u1", content="old construction maps to scope")
    doc = rec.to_doc()

    assert rec.tenant_id == "default"
    assert rec.scope_type == "user"
    assert rec.scope_id == "u1"
    assert rec.scope_key == "user:u1"
    assert "user_id" not in doc
    assert "visibility" not in doc
    assert "owner" not in doc


def test_scope_rejects_unknown_values():
    with pytest.raises(pydantic.ValidationError, match="scope_type"):
        MemoryRecordBase(content="bad", scope_type="domain")


def test_security_context_defaults():
    ctx = SecurityContext()

    assert ctx.tenant_id == "default"
    assert ctx.principal is None
    assert ctx.user_tenant_id is None
    assert ctx.roles == []


def test_security_context_from_user_id_factory():
    ctx = SecurityContext.from_user_id("alice")

    assert ctx.tenant_id == "default"
    assert ctx.principal == "user:alice"
    assert ctx.user_tenant_id is None
    assert ctx.roles == []


def test_security_context_roles_are_not_shared_between_instances():
    ctx1 = SecurityContext()
    ctx2 = SecurityContext()

    ctx1.roles.append("team:eng")

    assert ctx2.roles == []


# ---------------------------------------------------------------------------
# §4.10: SecurityContext gains groups + agent_id; session auto-resolves read_scopes
# ---------------------------------------------------------------------------


def test_security_context_groups_and_agent_id():
    ctx = SecurityContext(
        tenant_id="acme",
        principal="user:alice",
        agent_id="agent:copilot-sp",
        groups=["eng-gid", "platform-gid"],
        roles=["reviewer"],
    )
    assert ctx.agent_id == "agent:copilot-sp"
    assert ctx.groups == ["eng-gid", "platform-gid"]
    # defaults are empty, not shared across instances
    other = SecurityContext(principal="user:bob")
    assert other.groups == []
    assert other.agent_id is None


def test_session_auto_resolves_read_scopes_from_context():
    from azure.cosmos.agent_memory import CosmosMemoryClient
    from azure.cosmos.agent_memory._authz import resolve_read_scopes

    client = CosmosMemoryClient(use_default_credential=False)
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", agent_id="agent:copilot-sp", groups=["eng-gid"])
    bound = client.session(ctx)
    assert bound.read_scopes == resolve_read_scopes(ctx)
    assert "user:alice" in bound.read_scopes
    assert "agent:copilot-sp" in bound.read_scopes
    assert "org:acme" in bound.read_scopes
    assert "team:eng-gid" in bound.read_scopes


def test_bound_add_local_refuses_unauthorized_write_scope():
    """The buffer path must be gated too: add_local into a foreign scope is refused so an
    unauthorized record never reaches the buffer that push_to_cosmos later flushes."""
    from azure.cosmos.agent_memory import CosmosMemoryClient
    from azure.cosmos.agent_memory.exceptions import ValidationError

    client = CosmosMemoryClient(use_default_credential=False)
    mallory = SecurityContext(tenant_id="acme", principal="user:mallory")
    bound = client.session(mallory, write_scope="team:secret")

    with pytest.raises(ValidationError, match="write permission denied"):
        bound.add_local(role="user", content="exfiltrate", memory_type="fact")
    assert client.local_memory == [], "nothing may enter the shared buffer on a denied write"


def test_bound_add_local_allows_own_scope_and_buffers():
    from azure.cosmos.agent_memory import CosmosMemoryClient

    client = CosmosMemoryClient(use_default_credential=False)
    alice = SecurityContext(tenant_id="acme", principal="user:alice")
    bound = client.session(alice)  # default write_scope = own user scope

    bound.add_local(role="user", content="mine", memory_type="fact")
    assert len(client.local_memory) == 1
    assert client.local_memory[0]["scope_key"] == "user:alice"


def test_bound_add_local_agent_scope_is_readable_by_the_agent():
    """A buffered write in an agent session must be readable by that agent: the record's
    ACL is re-derived for the agent scope, not left as the user_id-derived default."""
    from azure.cosmos.agent_memory import CosmosMemoryClient
    from azure.cosmos.agent_memory._authz import can

    client = CosmosMemoryClient(use_default_credential=False)
    ctx = SecurityContext(tenant_id="acme", principal="agent:svc", agent_id="agent:svc")
    bound = client.session(ctx)  # default write_scope = agent:svc

    bound.add_local(role="agent", content="operational note", memory_type="fact")
    record = client.local_memory[-1]
    assert record["scope_key"] == "agent:svc"
    assert record["acl"]["read"] == ["agent:svc"], "read ACL must follow the agent write scope"
    assert can(ctx, record["acl"]), "the agent must be able to read its own buffered write"


def test_bound_add_local_team_scope_is_readable_by_teammates():
    """A buffered write into a team scope must be readable by other team members, not just
    the author's private user scope."""
    from azure.cosmos.agent_memory import CosmosMemoryClient
    from azure.cosmos.agent_memory._authz import can

    client = CosmosMemoryClient(use_default_credential=False)
    author = SecurityContext(tenant_id="acme", principal="user:alice", roles=["team:eng:writer"])
    bound = client.session(author, write_scope="team:eng")

    bound.add_local(role="user", content="team norm", memory_type="fact")
    record = client.local_memory[-1]
    assert record["scope_key"] == "team:eng"
    assert record["acl"]["read"] == ["team:eng"]
    teammate = SecurityContext(tenant_id="acme", principal="user:carol", groups=["eng"])
    assert can(teammate, record["acl"]), "a teammate must be able to read the team-scoped buffered write"


def test_bound_session_supports_agent_principal():
    """App-only ``agent:`` principals may open a session (regression: the strict user-id
    helper must not be called unconditionally in the bound ctor)."""
    from azure.cosmos.agent_memory import CosmosMemoryClient

    client = CosmosMemoryClient(use_default_credential=False)
    bound = client.session(SecurityContext(tenant_id="acme", principal="agent:svc"))
    assert bound.write_scope == "agent:svc"
    assert bound.user_id == "svc"


def test_resolve_read_scopes_does_not_double_prefix_team_groups():
    from azure.cosmos.agent_memory._authz import resolve_read_scopes

    ctx = SecurityContext(tenant_id="acme", principal="user:alice", groups=["team:eng", "sales"])
    scopes = resolve_read_scopes(ctx)
    assert "team:eng" in scopes
    assert "team:sales" in scopes
    assert "team:team:eng" not in scopes


def test_normalize_scope_keys_drops_none_and_blank_entries():
    from azure.cosmos.agent_memory.store._search_helpers import normalize_scope_keys

    # A None in the list must be dropped, not coerced to the literal "None" scope key.
    assert normalize_scope_keys(["team:eng", None, "  ", "team:eng"]) == ["team:eng"]
    assert normalize_scope_keys(None) is None


def test_agent_scope_key_does_not_double_prefix():
    from azure.cosmos.agent_memory._pins import agent_scope_key

    assert agent_scope_key("copilot") == "agent:copilot"
    assert agent_scope_key("agent:copilot") == "agent:copilot"


def test_caller_subjects_and_can_support_group_acl():
    from azure.cosmos.agent_memory._authz import caller_subjects, can

    ctx = SecurityContext(tenant_id="acme", principal="user:alice", agent_id="agent:copilot-sp", groups=["eng"])

    assert {"user:alice", "agent:copilot-sp", "org:acme", "tenant:*", "team:eng"} <= caller_subjects(ctx)
    assert can(ctx, MemoryAcl(read=["team:eng"]))
    assert not can(
        SecurityContext(tenant_id="acme", principal="user:bob", groups=["sales"]), MemoryAcl(read=["team:eng"])
    )
    assert not can(ctx, MemoryAcl(read=["user:lead"]))


def test_default_acl_for_user_scope_reads_owner_only():
    from azure.cosmos.agent_memory._partitioning import default_acl_for_scope

    acl = default_acl_for_scope("user:u1", principal="user:u1", agent_id="agent:bot")

    # Inline ACLs are read-visibility only; write/forget are authorized at the scope level.
    assert acl.read == ["user:u1"]
    assert acl.model_dump() == {"read": ["user:u1"]}


def test_default_acl_for_team_scope_reads_team():
    from azure.cosmos.agent_memory._partitioning import default_acl_for_scope

    acl = default_acl_for_scope("team:eng", principal="user:lead")

    assert acl.read == ["team:eng"]
    assert acl.model_dump() == {"read": ["team:eng"]}


def test_default_acl_for_global_scope_reads_tenant_wildcard():
    from azure.cosmos.agent_memory._partitioning import default_acl_for_scope

    acl = default_acl_for_scope("global:global", principal="user:admin")

    assert acl.read == ["tenant:*"]
    assert acl.model_dump() == {"read": ["tenant:*"]}


def test_source_fields_normalize_to_provenance_not_top_level():
    rec = MemoryRecordBase(
        user_id="u1",
        content="generated",
        source="summary",
        source_memory_ids=["m1"],
        prompt_id="p.prompty",
        prompt_version="v1",
    )
    doc = rec.to_doc()

    assert doc["provenance"] == {
        "source": "summary",
        "source_ids": ["m1"],
        "prompt_id": "p.prompty",
        "prompt_version": "v1",
    }
    assert "source_memory_ids" not in doc
    assert "prompt_id" not in doc


def test_session_stamps_agent_id_as_provenance_on_writes():
    from unittest.mock import MagicMock

    from azure.cosmos.agent_memory import CosmosMemoryClient

    client = CosmosMemoryClient(use_default_credential=False)
    client.upsert_memory = MagicMock(return_value="mem-1")  # type: ignore[method-assign]
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", agent_id="agent:copilot-sp")
    bound = client.session(ctx)

    bound.upsert_memory(role="user", content="hi", memory_type="fact")

    _, kwargs = client.upsert_memory.call_args
    assert kwargs["agent_id"] == "agent:copilot-sp"
    assert kwargs["scope_key"] == "user:alice"
    assert kwargs["tenant_id"] == "acme"
