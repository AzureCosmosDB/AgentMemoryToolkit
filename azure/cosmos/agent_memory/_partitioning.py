"""Shared helpers for Cosmos DB hierarchical partition keys."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.models import MemoryAcl

DEFAULT_TENANT_ID = "default"
DEFAULT_SCOPE_TYPE = "user"
# User summaries are user-scoped records addressed by this stable third-level marker
# under PK [tenant_id, user:<user_id>, USER_SUMMARY_THREAD_ID].
USER_SUMMARY_THREAD_ID = "__user_summary__"
COUNTER_THREAD_ID = "__counters__"
SHARED_RECORD_THREAD_ID = "__shared_state__"

# Request-scoped tenant for the background pipeline. The user-scoped read/write helpers
# below (``scope_values_for_user`` / ``user_scope_parameters``) historically assumed the
# single-tenant default; in the managed/multi-tenant path the tenant comes from the
# trusted SecurityContext. Rather than thread ``tenant_id`` through every internal
# pipeline helper (extraction, episodic cursors, summaries, procedural), the public
# pipeline entry points set this ContextVar via :func:`tenant_scope`, and the two choke
# points resolve it. ContextVars are task-local, so this is safe under async concurrency.
_current_tenant_id: ContextVar[Optional[str]] = ContextVar("amt_current_tenant_id", default=None)


def resolve_tenant_id(tenant_id: Optional[str] = None) -> str:
    """Resolve the effective tenant: explicit arg > request-scoped context > default."""
    return tenant_id or _current_tenant_id.get() or DEFAULT_TENANT_ID


@contextlib.contextmanager
def tenant_scope(tenant_id: Optional[str]) -> Iterator[None]:
    """Bind the request-scoped tenant for the duration of a pipeline operation.

    A falsy ``tenant_id`` is a no-op that inherits the current context, so nesting
    (e.g. a bound session that sets the tenant, calling a client method that also
    opens ``tenant_scope(None)``) never clears an already-bound tenant.
    """
    if not tenant_id:
        yield
        return
    token = _current_tenant_id.set(tenant_id)
    try:
        yield
    finally:
        _current_tenant_id.reset(token)


def scope_key_for_user(user_id: str) -> str:
    """Return the Phase 1 user-scope partition component for ``user_id``."""
    return f"user:{user_id}"


def scope_values_for_user(user_id: str, tenant_id: Optional[str] = None) -> tuple[str, str, str, str]:
    """Return ``(tenant_id, scope_type, scope_id, scope_key)`` for a user-scoped record."""
    tenant = resolve_tenant_id(tenant_id)
    return tenant, DEFAULT_SCOPE_TYPE, user_id, scope_key_for_user(user_id)


def scope_values_for_scope_key(scope_key: str, tenant_id: Optional[str] = None) -> tuple[str, str, str, str]:
    """Return ``(tenant_id, scope_type, scope_id, scope_key)`` for an explicit scope key."""
    key = str(scope_key).strip()
    if not key or ":" not in key:
        raise ValidationError("scope_key must be in '<type>:<id>' form")
    scope_type, scope_id = key.split(":", 1)
    if not scope_type or not scope_id:
        raise ValidationError("scope_key must be in '<type>:<id>' form")
    return resolve_tenant_id(tenant_id), scope_type, scope_id, key


def private_scope_key_for_principal(principal: Optional[str]) -> str:
    """Return the default private/owner scope for a trusted principal.

    ``user:<id>`` (delegated) and ``agent:<id>`` (app-only) principals both own a private
    scope equal to the principal itself.
    """
    if not principal:
        raise ValidationError("SecurityContext.principal is required for session binding")
    if principal in ("user:", "agent:") or not (principal.startswith("user:") or principal.startswith("agent:")):
        raise ValidationError(
            "SecurityContext.principal must be a user:<id> or agent:<id> principal for the default private scope"
        )
    return principal


def user_id_from_principal(principal: Optional[str]) -> str:
    """Return the user id portion of a ``user:<id>`` principal.

    Only delegated ``user:<id>`` principals map to a user id. App-only ``agent:<id>``
    principals own an ``agent:<id>`` private scope (see
    :func:`private_scope_key_for_principal`), not a user scope, so they are rejected here
    rather than being silently coerced into a ``user:<agent-id>`` scope.
    """
    scope = private_scope_key_for_principal(principal)
    if not scope.startswith("user:"):
        raise ValidationError("user_id_from_principal requires a user:<id> principal")
    return scope.split(":", 1)[1]


def partition_key_for_scope_thread(scope_key: str, thread_id: str, tenant_id: Optional[str] = None) -> list[str]:
    """Return ``[/tenant_id, /scope_key, /thread_id]`` values for an explicit scope."""
    tenant, _, _, key = scope_values_for_scope_key(scope_key, tenant_id)
    return [tenant, key, thread_id]


def partition_key_for_user_thread(user_id: str, thread_id: str, tenant_id: Optional[str] = None) -> list[str]:
    """Return ``[/tenant_id, /scope_key, /thread_id]`` values for user-scoped data."""
    tenant, _, _, scope_key = scope_values_for_user(user_id, tenant_id)
    return [tenant, scope_key, thread_id]


def user_scope_predicate(user_param: str = "@user_id") -> str:
    """Return the SQL predicate for the default user scope.

    ``user_param`` is accepted for source compatibility but records no longer
    store ``user_id``; callers bind user ids to ``scope_key=user:<id>``.
    """
    del user_param
    return "c.tenant_id = @tenant_id AND c.scope_key = @scope_key"


def user_scope_parameters(
    user_id: str, user_param: str = "@user_id", tenant_id: Optional[str] = None
) -> list[dict[str, str]]:
    """Return tenant/scope parameters for ``user_scope_predicate``.

    ``tenant_id`` defaults to :data:`DEFAULT_TENANT_ID` for single-tenant callers; the
    managed/multi-tenant path passes the tenant from the trusted ``SecurityContext``.
    """
    del user_param
    return [
        {"name": "@tenant_id", "value": resolve_tenant_id(tenant_id)},
        {"name": "@scope_key", "value": scope_key_for_user(user_id)},
    ]


def _dedupe_subjects(subjects: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for subject in subjects:
        stripped = str(subject).strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            out.append(stripped)
    return out


def default_acl_for_scope(
    scope_key: str, *, principal: Optional[str] = None, agent_id: Optional[str] = None
) -> MemoryAcl:
    """Return the default inline read-visibility ACL for a target placement scope.

    Only ``read`` is populated: writes are authorized at the scope level, not via an
    inline ACL (see :class:`~azure.cosmos.agent_memory.models.MemoryAcl`). ``principal`` /
    ``agent_id`` are accepted for signature stability and recorded in provenance by the
    caller; they no longer seed a write/forget subject list.
    """
    key = str(scope_key).strip()
    if not key:
        raise ValidationError("scope_key cannot be empty")
    read = ["tenant:*"] if key.startswith("global:") else [key]
    return MemoryAcl(read=_dedupe_subjects(read))


def ensure_scope_fields(
    doc: dict[str, Any],
    *,
    tenant_id: Optional[str] = None,
    scope_key: Optional[str] = None,
    principal: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> dict[str, Any]:
    """Populate tenant/scope, acl, and provenance fields for a write document."""
    if scope_key is not None:
        tenant, scope_type, scope_id, key = scope_values_for_scope_key(scope_key, tenant_id or doc.get("tenant_id"))
    else:
        existing_key = doc.get("scope_key")
        if existing_key:
            tenant, scope_type, scope_id, key = scope_values_for_scope_key(
                existing_key, tenant_id or doc.get("tenant_id")
            )
        else:
            user_id = doc.get("user_id")
            if not user_id:
                raise ValidationError(
                    "ensure_scope_fields requires a scope_key or a legacy user_id to place the document; "
                    "an unscoped write cannot be partitioned or ACL-stamped"
                )
            tenant, scope_type, scope_id, key = scope_values_for_user(str(user_id), tenant_id or doc.get("tenant_id"))
    doc["tenant_id"] = tenant
    doc["scope_type"] = scope_type
    doc["scope_id"] = scope_id
    doc["scope_key"] = key
    effective_principal = principal or doc.get("principal") or doc.get("owner")
    effective_agent = agent_id or doc.get("agent_id")
    existing_acl = doc.get("acl") if isinstance(doc.get("acl"), dict) else None
    if not existing_acl or not existing_acl.get("read"):
        doc["acl"] = default_acl_for_scope(key, principal=effective_principal, agent_id=effective_agent).model_dump(
            mode="json"
        )
    provenance = dict(doc.get("provenance") or {})
    if effective_agent and not provenance.get("agent_id"):
        provenance["agent_id"] = effective_agent
    if effective_principal and not provenance.get("created_by"):
        provenance["created_by"] = effective_principal
    if doc.get("source") and not provenance.get("source"):
        provenance["source"] = doc.get("source")
    if doc.get("source_memory_ids") and not provenance.get("source_ids"):
        provenance["source_ids"] = list(doc.get("source_memory_ids") or [])
    if doc.get("prompt_id") and not provenance.get("prompt_id"):
        provenance["prompt_id"] = doc.get("prompt_id")
    if doc.get("prompt_version") and not provenance.get("prompt_version"):
        provenance["prompt_version"] = doc.get("prompt_version")
    doc["provenance"] = provenance
    # These are write-time input fields, not persisted columns: they have been folded
    # into scope_key / provenance above, so strip them from the stored document.
    for input_only in (
        "user_id",
        "visibility",
        "owner",
        "agent_id",
        "source",
        "source_memory_ids",
        "prompt_id",
        "prompt_version",
        "principal",
    ):
        doc.pop(input_only, None)
    return doc


def ensure_user_scope_fields(doc: dict[str, Any]) -> dict[str, Any]:
    """Populate Phase 1 tenant/scope fields for today's user-scoped documents."""
    return ensure_scope_fields(doc)


def scope_fields_from_doc(doc: dict[str, Any], *, fallback_user_id: Optional[str] = None) -> dict[str, Any]:
    """Return scope fields from a source doc, falling back to its/private user scope."""
    source = dict(doc)
    if fallback_user_id and not source.get("scope_key"):
        source["scope_key"] = scope_key_for_user(fallback_user_id)
    scoped = ensure_scope_fields(source)
    return {
        "tenant_id": scoped["tenant_id"],
        "scope_type": scoped["scope_type"],
        "scope_id": scoped["scope_id"],
        "scope_key": scoped["scope_key"],
        "acl": scoped.get("acl"),
        "provenance": scoped.get("provenance") or {},
    }


def inherit_scope_fields(
    doc: dict[str, Any], source: dict[str, Any], *, fallback_user_id: Optional[str] = None
) -> dict[str, Any]:
    """Stamp ``doc`` with source tenant/scope fields and merge provenance."""
    inherited = scope_fields_from_doc(source, fallback_user_id=fallback_user_id)
    provenance = {**(inherited.get("provenance") or {}), **(doc.get("provenance") or {})}
    doc.update(inherited)
    doc["provenance"] = provenance
    return doc
