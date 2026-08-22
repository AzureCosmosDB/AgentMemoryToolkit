"""Authorization helpers for inline memory ACLs and scope placement."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from azure.cosmos.agent_memory._partitioning import scope_key_for_user
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.logging import get_logger
from azure.cosmos.agent_memory.models import MemoryAcl

logger = get_logger(__name__)

# Inline ACLs govern read visibility only. Write / share / assign / forget / annotate
# authorization is decided at the placement-scope level by :func:`resolve_scope_access`
# (see ``_decide_scope``); records do not carry an enforced per-record write ACL.
PermissionAction = Literal["read", "write", "share", "assign", "annotate", "forget"]
_ADMIN_ROLES: frozenset[str] = frozenset({"admin", "tenant:admin", "memory:admin", "amt:admin"})

# Cap on the number of scopes a single read fans out over. ``None`` disables it.
MAX_READ_SCOPES: int = 5


@dataclass(frozen=True)
class ScopeAccessDecision:
    scope_key: str
    allowed: bool
    reason: str
    restricted_allowed: bool = False


@dataclass(frozen=True)
class AuthzQueryPredicate:
    """Parameterized SQL fragment for Cosmos read pre-filtering."""

    sql: str
    parameters: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AuthzResolution:
    allowed_scopes: set[str]
    predicate: AuthzQueryPredicate
    decisions: dict[str, ScopeAccessDecision]
    denied_resources: set[str] = field(default_factory=set)


def resolve_read_scopes(
    ctx: SecurityContext,
    *,
    project_scope: str | None = None,
    cap: int | None = MAX_READ_SCOPES,
) -> list[str]:
    """Build the bounded placement scopes a caller's reads may union over."""
    scopes: list[str] = []
    principal = ctx.principal
    if principal and (principal.startswith("user:") or principal.startswith("agent:")):
        scopes.append(principal)
    if ctx.agent_id and ctx.agent_id.startswith("agent:"):
        scopes.append(ctx.agent_id)
    if project_scope:
        scopes.append(project_scope)
    scopes.append(f"org:{ctx.tenant_id}")
    for group in ctx.groups:
        if group:
            group_str = str(group)
            scopes.append(group_str if group_str.startswith("team:") else f"team:{group_str}")
    deduped = _dedupe_scopes(scopes)
    if cap is not None and cap >= 0:
        return deduped[:cap]
    return deduped


def caller_subjects(ctx: SecurityContext) -> set[str]:
    """Return all inline-ACL subjects represented by ``ctx``."""
    subjects: set[str] = {"tenant:*", f"org:{ctx.tenant_id}"}
    if ctx.principal:
        subjects.add(ctx.principal)
    if ctx.agent_id:
        subjects.add(ctx.agent_id)
    for group in ctx.groups:
        if group:
            group_str = str(group)
            subjects.add(group_str if group_str.startswith("team:") else f"team:{group_str}")
    for role in ctx.roles:
        normalized = str(role).strip().lower()
        if normalized:
            subjects.add(f"role:{normalized}")
    return subjects


_MEMBERSHIP_SUFFIXES: frozenset[str] = frozenset({"reader", "member", "writer"})


def _scope_membership_subjects(ctx: SecurityContext) -> set[str]:
    """Derive bare scope subjects (e.g. ``project:atlas``) from membership roles for reads.

    A role of the form ``<scope>:<suffix>`` or ``scope:<scope>:<suffix>`` (suffix in
    reader/member/writer) grants the caller the bare ``<scope>`` subject so that inline
    read ACLs written for a shared placement scope - notably ``project`` - match, exactly
    as ``org`` / ``team`` / ``global`` already do through :func:`caller_subjects`. This
    governs read visibility only; write authorization is decided by :func:`_decide_scope`.
    """
    subjects: set[str] = set()
    for raw in ctx.roles:
        role = str(raw).strip().lower()
        if not role:
            continue
        if role.startswith("scope:"):
            role = role[len("scope:") :]
        parts = role.split(":")
        if len(parts) < 2:
            continue
        if parts[-1] not in _MEMBERSHIP_SUFFIXES:
            continue
        scope = ":".join(parts[:-1])
        if scope:
            subjects.add(scope)
    return subjects


def can(ctx: SecurityContext, record_acl: Mapping[str, Any] | MemoryAcl | None) -> bool:
    """Return True when the caller may READ a record given its inline ACL.

    Inline ACLs are a read-visibility mechanism only. Write / share / assign / forget /
    annotate authorization is enforced at the placement-scope level via
    :func:`resolve_scope_access`, not per record.
    """
    if _is_admin(ctx):
        return True
    acl = _coerce_acl(record_acl)
    subjects = caller_subjects(ctx) | _scope_membership_subjects(ctx)
    return bool(subjects & set(acl.read))


def build_acl_predicate(ctx: SecurityContext, scope_keys: Iterable[str]) -> AuthzQueryPredicate:
    """Build a Cosmos SQL pre-filter for placement scope plus inline read ACL."""
    scopes = _dedupe_scopes(scope_keys)
    params: list[dict[str, Any]] = [{"name": "@authz_tenant_id", "value": ctx.tenant_id}]
    if not scopes:
        return AuthzQueryPredicate("c.tenant_id = @authz_tenant_id AND false", params)
    scope_params: list[str] = []
    for index, scope_key in enumerate(scopes):
        name = f"@authz_scope_{index}"
        scope_params.append(name)
        params.append({"name": name, "value": scope_key})
    subject_terms: list[str] = []
    for index, subject in enumerate(sorted(caller_subjects(ctx) | _scope_membership_subjects(ctx))):
        name = f"@authz_subject_{index}"
        subject_terms.append(f"ARRAY_CONTAINS(c.acl.read, {name})")
        params.append({"name": name, "value": subject})
    subject_sql = " OR ".join(subject_terms) if subject_terms else "false"
    scope_sql = f"c.scope_key IN ({', '.join(scope_params)})"
    return AuthzQueryPredicate(
        f"c.tenant_id = @authz_tenant_id AND {scope_sql} AND ({subject_sql})",
        params,
    )


def resolve_scope_access(
    ctx: SecurityContext,
    requested_scope_keys: Iterable[str],
    action: PermissionAction,
) -> AuthzResolution:
    """Narrow requested scope keys to those the caller may access (placement membership).

    This only filters candidate partitions; ``build_acl_predicate`` and ``can`` enforce
    the actual record-level ACL permissions.
    """
    requested = _dedupe_scopes(requested_scope_keys)
    allowed: set[str] = set()
    decisions: dict[str, ScopeAccessDecision] = {}
    for scope_key in requested:
        decision = _decide_scope(ctx, scope_key, action)
        decisions[scope_key] = decision
        if decision.allowed:
            allowed.add(scope_key)
            logger.debug(
                "authz scope allow action=%s scope_hash=%s reason=%s", action, _safe_hash(scope_key), decision.reason
            )
        else:
            logger.info(
                "authz scope deny action=%s scope_hash=%s reason=%s", action, _safe_hash(scope_key), decision.reason
            )
    return AuthzResolution(
        allowed_scopes=allowed,
        predicate=build_acl_predicate(ctx, allowed),
        decisions=decisions,
    )


def authorize_scope_write(ctx: SecurityContext | None, scope_key: str | None, user_id: str) -> None:
    """Refuse a write into a non-owner placement scope without write authorization.

    The gate is applied to the *effective* target scope - the explicit ``scope_key`` or,
    when absent, the record's own ``user:<user_id>`` scope. Without a context, only the
    caller's own user scope is writable (the pre-existing single-user trust model). With a
    context, the resolver decides: the caller's own principal scope (``user:``/``agent:``)
    is always allowed, and any other scope requires a member/writer role or admin. Own-scope
    is derived from the trusted ``ctx.principal``, never from the caller-supplied
    ``user_id``, so a request cannot claim another principal's private scope. Shared by the
    store write path and the bound-session buffer path so both enforce one contract.
    """
    effective_scope = scope_key or scope_key_for_user(user_id)
    if ctx is None:
        if effective_scope != scope_key_for_user(user_id):
            raise ValidationError(
                f"writing to scope {effective_scope!r} requires a SecurityContext with write permission"
            )
        return
    if effective_scope not in resolve_scope_access(ctx, [effective_scope], "write").allowed_scopes:
        raise ValidationError(
            f"write permission denied for scope_key={effective_scope!r} and principal {ctx.principal!r}"
        )


def _decide_scope(ctx: SecurityContext, scope_key: str, action: PermissionAction) -> ScopeAccessDecision:
    if _is_admin(ctx):
        return ScopeAccessDecision(scope_key, True, "role_default_admin")
    if scope_key in _own_scopes(ctx):
        return ScopeAccessDecision(scope_key, True, "own_scope")
    # Membership subjects (org / team / tenant / group) grant READ visibility only.
    # Writing into a shared scope requires an explicit member/writer role (below) or
    # admin, so tenant-wide membership never confers write/forget on ``org:<tenant>``.
    if action == "read" and scope_key in caller_subjects(ctx):
        return ScopeAccessDecision(scope_key, True, "subject_member")
    roles = {str(role).strip().lower() for role in ctx.roles if str(role).strip()}
    scope_lower = scope_key.lower()
    if action == "read" and (
        f"{scope_lower}:reader" in roles
        or f"{scope_lower}:member" in roles
        or f"{scope_lower}:writer" in roles
        or f"scope:{scope_lower}:reader" in roles
        or f"scope:{scope_lower}:member" in roles
        or f"scope:{scope_lower}:writer" in roles
    ):
        return ScopeAccessDecision(scope_key, True, "role_scope_reader")
    if action in {"write", "share", "assign", "forget", "annotate"} and (
        f"{scope_lower}:member" in roles
        or f"{scope_lower}:writer" in roles
        or f"scope:{scope_lower}:member" in roles
        or f"scope:{scope_lower}:writer" in roles
    ):
        return ScopeAccessDecision(scope_key, True, "role_scope_writer")
    if action == "read" and _is_public_scope(scope_key):
        return ScopeAccessDecision(scope_key, True, "tenant_public")
    return ScopeAccessDecision(scope_key, False, "deny_default")


def _coerce_acl(record_acl: Mapping[str, Any] | MemoryAcl | None) -> MemoryAcl:
    if isinstance(record_acl, MemoryAcl):
        return record_acl
    if isinstance(record_acl, Mapping):
        return MemoryAcl.model_validate(record_acl)
    return MemoryAcl()


def _dedupe_scopes(scopes: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for scope in scopes:
        key = str(scope).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _is_admin(ctx: SecurityContext) -> bool:
    roles = {role.strip().lower() for role in ctx.roles if role and role.strip()}
    return bool(roles & _ADMIN_ROLES)


def _own_scopes(ctx: SecurityContext) -> set[str]:
    scopes: set[str] = set()
    principal = ctx.principal
    if principal and (principal.startswith("user:") or principal.startswith("agent:")):
        scopes.add(principal)
    if ctx.agent_id and ctx.agent_id.startswith("agent:"):
        scopes.add(ctx.agent_id)
    return scopes


def _is_public_scope(scope_key: str) -> bool:
    return scope_key == "global:global" or scope_key.startswith("global:")


def _safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
