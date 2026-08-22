"""Synchronous Cosmos DB memory store primitives."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos.agent_memory._authz import (
    PermissionAction,
    authorize_scope_write,
    can,
    resolve_scope_access,
)
from azure.cosmos.agent_memory._container_routing import (
    _CONTAINER_FOR_TYPE,
    USER_SCOPED_MEMORIES_TYPES,
    ContainerKey,
    container_key_for_type,
)
from azure.cosmos.agent_memory._curation import (
    CURATION_STATUS_APPROVED,
    PERSONAL_SCOPE_TYPE,
    is_pii_or_secret_flagged,
    utc_now_iso,
)
from azure.cosmos.agent_memory._partitioning import (
    SHARED_RECORD_THREAD_ID,
    USER_SUMMARY_THREAD_ID,
    default_acl_for_scope,
    ensure_scope_fields,
    ensure_user_scope_fields,
    partition_key_for_scope_thread,
    partition_key_for_user_thread,
    private_scope_key_for_principal,
    scope_key_for_user,
    scope_values_for_scope_key,
    user_scope_parameters,
    user_scope_predicate,
)
from azure.cosmos.agent_memory._pins import (
    PIN_RECORD_TYPE,
    PIN_THREAD_ID,
    agent_scope_key,
    apply_injection_mode,
    classify_pin_resource,
    normalize_injection_mode,
    normalize_priority,
    pin_id,
)
from azure.cosmos.agent_memory._pins import (
    utc_now_iso as pin_utc_now_iso,
)
from azure.cosmos.agent_memory._query_builder import _QueryBuilder
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory._utils import (
    _build_memory_query_builder,
    _coerce_datetime_iso,
    compute_content_hash,
    extract_keywords,
    new_id,
    normalize_created_at_iso,
)
from azure.cosmos.agent_memory.exceptions import (
    ConfigurationError,
    CosmosOperationError,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryTypeMismatchError,
    SharedRecordReadOnlyError,
    ValidationError,
)
from azure.cosmos.agent_memory.logging import get_logger
from azure.cosmos.agent_memory.models import MemoryRecord, ProcedureKind
from azure.cosmos.agent_memory.store._search_helpers import (
    MEMORY_PROJECTION,
    add_salience_filter,
    add_tag_filters,
    add_tenant_scope_filter,
    build_search_sql,
    coerce_embedding,
    format_episodic_context,
    merge_ranked_results,
    normalize_scope_keys,
    query_scope,
    require_search_terms,
    top_literal,
)
from azure.cosmos.agent_memory.thresholds import default_ttl_for

logger = get_logger(__name__)

_MEMORIES_TYPES: tuple[str, ...] = ("fact", "episodic", "procedural")
_SHARED_STATE_TYPE = "shared_state"

# Explicit turn-document projection used by get_thread(). The raw conversation
# log is the only place embeddings are stored on turns (when
# enable_turn_embeddings=True), so we project every turn field *except*
# ``embedding`` to keep the vector off the wire and out of the result.
_TURN_PROJECTION_FIELDS: tuple[str, ...] = (
    "id",
    "thread_id",
    "tenant_id",
    "scope_type",
    "scope_id",
    "scope_key",
    "role",
    "type",
    "content",
    "metadata",
    "acl",
    "provenance",
    "created_at",
    "tags",
    "ttl",
)
_TURN_PROJECTION: str = ", ".join(f"c.{field}" for field in _TURN_PROJECTION_FIELDS)


def _validated_memories_types(memory_types: Optional[list[str]]) -> list[str]:
    types = list(memory_types) if memory_types else list(_MEMORIES_TYPES)
    invalid = [t for t in types if t not in _MEMORIES_TYPES]
    if invalid:
        raise ValueError(f"memory_types must be a subset of {list(_MEMORIES_TYPES)}; got {invalid}")
    return types


def _validate_taggable_type(memory_type: str) -> None:
    if memory_type not in _MEMORIES_TYPES:
        raise ValueError(f"memory_type for tag mutation must be one of {list(_MEMORIES_TYPES)}; got {memory_type!r}")


def _wrap_cosmos_exception(exc: BaseException, *, message: str) -> CosmosOperationError:
    """Wrap a Cosmos SDK exception with a contextual message."""
    return CosmosOperationError(message)


def _is_precondition_failed(exc: BaseException) -> bool:
    """Return True for Cosmos 412 / If-Match failures, including fakes."""
    return exc.__class__.__name__ == "CosmosAccessConditionFailedError" or getattr(exc, "status_code", None) == 412


class MemoryStore:
    """Typed CRUD and query primitives over the Cosmos DB containers."""

    def __init__(
        self,
        *,
        containers: dict[ContainerKey, Any],
        embeddings_client: Any = None,
        enable_turn_embeddings: bool = False,
    ) -> None:
        self._containers = containers
        self._turns_container = containers[ContainerKey.TURNS]
        self._memories_container = containers[ContainerKey.MEMORIES]
        self._summaries_container = containers[ContainerKey.SUMMARIES]
        self._embeddings_client = embeddings_client
        self._enable_turn_embeddings = enable_turn_embeddings

    @property
    def container(self) -> Any:
        """Return the memories Cosmos container client."""
        return self._memories_container

    def _prepare_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Return a write-ready document with type defaults applied."""
        body = ensure_user_scope_fields(dict(doc))
        if body.get("ttl") is None:
            body.pop("ttl", None)
            ttl = default_ttl_for(body.get("type"))
            if ttl is not None:
                body["ttl"] = ttl
        return body

    def _container_for_type(self, memory_type: str) -> Any:
        """Return the container that owns documents of ``memory_type``."""
        return self._containers[container_key_for_type(memory_type)]

    def read_item(self, item_id: str, partition_key: Any, *, container_key: ContainerKey) -> dict[str, Any]:
        """Point-read a document from the explicitly selected split container."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return self._containers[container_key].read_item(item=item_id, partition_key=partition_key)
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=item_id) from exc
        except Exception as exc:
            raise CosmosOperationError(f"read_item failed for {item_id}: {exc}") from exc

    def query(
        self,
        sql: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        *,
        container_key: ContainerKey,
        partition_key: Any = None,
        cross_partition: bool = False,
    ) -> list[dict[str, Any]]:
        """Run a query against the explicitly selected split container."""
        return self._query_items(
            query=sql,
            parameters=parameters,
            partition_key=partition_key,
            cross_partition=cross_partition,
            operation="query",
            container=self._containers[container_key],
        )

    def _query_items(
        self,
        *,
        query: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Any = None,
        cross_partition: bool = False,
        operation: str,
        container: Any = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"query": query, "parameters": parameters or None}
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        if cross_partition:
            kwargs["enable_cross_partition_query"] = True
        target = container if container is not None else self._memories_container
        try:
            return list(target.query_items(**kwargs))
        except Exception as exc:
            raise CosmosOperationError(f"{operation} failed: {exc}") from exc

    def _resolve_read_authz(self, ctx: SecurityContext, scope_keys: list[str]):
        return resolve_scope_access(ctx, scope_keys, "read")

    def _resolve_scope_action(self, ctx: SecurityContext, scope_key: str, action: PermissionAction):
        return resolve_scope_access(ctx, [scope_key], action)

    def _can_scope_action(self, ctx: SecurityContext, scope_key: str, action: PermissionAction) -> bool:
        return scope_key in self._resolve_scope_action(ctx, scope_key, action).allowed_scopes

    def _can_scope_any_action(
        self, ctx: SecurityContext, scope_key: str, actions: tuple[PermissionAction, ...]
    ) -> bool:
        return any(self._can_scope_action(ctx, scope_key, action) for action in actions)

    def _authorize_write(self, ctx: SecurityContext | None, scope_key: str | None, user_id: str) -> None:
        """Refuse a write into a non-owner placement scope (see ``authorize_scope_write``)."""
        authorize_scope_write(ctx, scope_key, user_id)

    def _shared_scopes_need_ctx(
        self, ctx: SecurityContext | None, scope_keys: list[str], user_id: str | None
    ) -> bool:
        """True when a read names a non-owner scope without a SecurityContext (fail closed).

        Naming a shared/foreign ``scope_key`` requires a context so the inline read-ACL
        pre-filter is applied. Without one, only the caller's own ``user:<user_id>`` scope
        may be read; anything else returns no rows rather than fanning out unfiltered.
        """
        if not scope_keys or ctx is not None:
            return False
        own_scope = scope_key_for_user(user_id) if user_id else None
        return any(scope_key != own_scope for scope_key in scope_keys)

    def _require_pin_agent_permission(self, ctx: SecurityContext, agent_scope: str) -> None:
        if self._can_scope_any_action(ctx, agent_scope, ("assign", "write")):
            return
        raise ValidationError(
            "pin_memory/unpin_memory requires 'assign' or 'write' permission on "
            f"agent scope {agent_scope!r} for principal {ctx.principal!r}"
        )

    def _require_list_pins_permission(self, ctx: SecurityContext, agent_scope: str) -> None:
        if self._can_scope_any_action(ctx, agent_scope, ("assign", "write", "read")):
            return
        raise ValidationError(
            f"list_pins requires read/write/assign permission on agent scope {agent_scope!r} "
            f"for principal {ctx.principal!r}"
        )

    def _find_pin_memory_candidates(self, *, ctx: SecurityContext, memory_id: str) -> list[dict[str, Any]]:
        query = "SELECT TOP 10 * FROM c WHERE c.tenant_id = @tenant_id AND c.id = @memory_id"
        parameters = [{"name": "@tenant_id", "value": ctx.tenant_id}, {"name": "@memory_id", "value": memory_id}]
        rows: list[dict[str, Any]] = []
        for container in (self._memories_container, self._summaries_container, self._turns_container):
            rows.extend(
                self._query_items(
                    query=query,
                    parameters=parameters,
                    cross_partition=True,
                    operation="pin memory lookup",
                    container=container,
                )
            )
        return rows

    def _read_pinned_memory(
        self,
        *,
        ctx: SecurityContext,
        memory_id: str,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> dict[str, Any] | None:
        candidates = self._find_pin_memory_candidates(ctx=ctx, memory_id=memory_id)
        if not candidates:
            raise MemoryNotFoundError(memory_id=memory_id)
        allowed_types = set(memory_types or [])
        for candidate in candidates:
            memory_type = str(candidate.get("type") or "")
            if allowed_types and memory_type not in allowed_types:
                continue
            if not include_superseded and candidate.get("superseded_by"):
                continue
            scope_key = str(candidate.get("scope_key") or "")
            thread_id = str(candidate.get("thread_id") or "")
            if not scope_key or not thread_id:
                continue
            resolution = self._resolve_read_authz(ctx, [scope_key])
            if scope_key not in resolution.allowed_scopes:
                continue
            qb = _QueryBuilder()
            qb.add_filter("c.tenant_id", "@tenant_id", ctx.tenant_id)
            qb.add_filter("c.scope_key", "@scope_key", scope_key)
            qb.add_filter("c.id", "@memory_id", memory_id)
            qb.add_condition(resolution.predicate.sql, resolution.predicate.parameters)
            if not include_superseded:
                qb.add_is_null_or_undefined("c.superseded_by")
            sql = f"SELECT TOP 1 {MEMORY_PROJECTION} FROM c{qb.build_where()}"
            rows = self._query_items(
                query=sql,
                parameters=qb.get_parameters(),
                partition_key=partition_key_for_scope_thread(scope_key, thread_id, ctx.tenant_id),
                operation="pinned memory read",
                container=self._container_for_type(memory_type),
            )
            if rows:
                return rows[0]
        return None

    def _read_pinned_scope(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        memory_types: list[str],
        limit: int,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        resolution = self._resolve_read_authz(ctx, [scope_key])
        if scope_key not in resolution.allowed_scopes:
            return []
        top = top_literal(limit, name="pin scope limit")
        qb = _QueryBuilder()
        qb.add_filter("c.tenant_id", "@tenant_id", ctx.tenant_id)
        qb.add_filter("c.scope_key", "@scope_key", scope_key)
        qb.add_condition(resolution.predicate.sql, resolution.predicate.parameters)
        if memory_types:
            qb.add_in_filter("c.type", "@pin_type_", memory_types)
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")
        sql = f"SELECT TOP {top} {MEMORY_PROJECTION} FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        return self.query(
            sql,
            qb.get_parameters(),
            container_key=ContainerKey.MEMORIES,
            partition_key=[ctx.tenant_id, scope_key],
            cross_partition=False,
        )

    def pin_memory(
        self,
        agent_id: str,
        resource: str,
        injection_mode: str = "summary",
        priority: int = 50,
        ctx: SecurityContext | None = None,
        resource_type: str | None = None,
    ) -> dict[str, Any]:
        """Bind a memory id or scope key to an agent for selective injection.

        Creating a pin requires ``assign`` or ``write`` on ``agent:<agent_id>`` and
        ``read`` on the pinned memory's scope (or on the pinned scope itself). Pass
        ``resource_type`` (``"memory"`` or ``"scope"``) to disambiguate a scope-shaped
        custom memory id; otherwise it is inferred from ``resource``.
        """
        if ctx is None:
            raise ValidationError("ctx is required for pin_memory")
        mode = normalize_injection_mode(injection_mode)
        priority_value = normalize_priority(priority)
        resource_type, normalized_resource = classify_pin_resource(resource, resource_type)
        agent_scope = agent_scope_key(agent_id)
        self._require_pin_agent_permission(ctx, agent_scope)
        resource_scope_key: str | None = None
        if resource_type == "scope":
            resource_scope_key = normalized_resource
            resolution = self._resolve_read_authz(ctx, [resource_scope_key])
            if resource_scope_key not in resolution.allowed_scopes:
                raise ValidationError(f"pin_memory requires read permission on scope {resource_scope_key!r}")
        else:
            memory = self._read_pinned_memory(ctx=ctx, memory_id=normalized_resource, include_superseded=True)
            if memory is None:
                raise ValidationError(f"pin_memory requires read permission on memory {normalized_resource!r}")
            candidates = self._find_pin_memory_candidates(ctx=ctx, memory_id=normalized_resource)
            for candidate in candidates:
                if candidate.get("id") == normalized_resource:
                    resource_scope_key = candidate.get("scope_key")
                    break
        tenant, scope_type, scope_id, normalized_agent_scope = scope_values_for_scope_key(agent_scope, ctx.tenant_id)
        now = pin_utc_now_iso()
        body = {
            "id": pin_id(
                tenant_id=tenant,
                agent_id=str(agent_id).strip(),
                resource_type=resource_type,
                resource=normalized_resource,
            ),
            "type": PIN_RECORD_TYPE,
            "tenant_id": tenant,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "scope_key": normalized_agent_scope,
            "thread_id": PIN_THREAD_ID,
            "role": "system",
            "content": "",
            "metadata": {},
            "acl": default_acl_for_scope(
                normalized_agent_scope, principal=ctx.principal, agent_id=str(agent_id).strip()
            ).model_dump(mode="json"),
            "provenance": {"created_by": ctx.principal, "agent_id": str(agent_id).strip()},
            "resource": normalized_resource,
            "resource_type": resource_type,
            "resource_scope_key": resource_scope_key,
            "injection_mode": mode,
            "priority": priority_value,
            "created_at": now,
            "updated_at": now,
        }
        response = self._memories_container.upsert_item(body=self._prepare_doc(body))
        return response if isinstance(response, dict) else body

    def unpin_memory(
        self,
        agent_id: str,
        resource: str,
        ctx: SecurityContext | None = None,
        resource_type: str | None = None,
    ) -> bool:
        """Remove a selective-injection pin for an agent/resource binding."""
        if ctx is None:
            raise ValidationError("ctx is required for unpin_memory")
        resource_type, normalized_resource = classify_pin_resource(resource, resource_type)
        agent_scope = agent_scope_key(agent_id)
        self._require_pin_agent_permission(ctx, agent_scope)
        item_id = pin_id(
            tenant_id=ctx.tenant_id,
            agent_id=str(agent_id).strip(),
            resource_type=resource_type,
            resource=normalized_resource,
        )
        try:
            self._memories_container.delete_item(
                item=item_id,
                partition_key=partition_key_for_scope_thread(agent_scope, PIN_THREAD_ID, ctx.tenant_id),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "CosmosResourceNotFoundError":
                return False
            raise CosmosOperationError(f"unpin_memory failed for pin {item_id!r}: {exc}") from exc

    def list_pins(self, agent_id: str, ctx: SecurityContext | None = None) -> list[dict[str, Any]]:
        """List selective-injection pins for ``agent_id``, highest priority first."""
        if ctx is None:
            raise ValidationError("ctx is required for list_pins")
        agent_scope = agent_scope_key(agent_id)
        self._require_list_pins_permission(ctx, agent_scope)
        query = (
            "SELECT * FROM c WHERE c.tenant_id = @tenant_id AND c.scope_key = @scope_key "
            "AND c.thread_id = @thread_id AND c.type = @type"
        )
        rows = self._query_items(
            query=query,
            parameters=[
                {"name": "@tenant_id", "value": ctx.tenant_id},
                {"name": "@scope_key", "value": agent_scope},
                {"name": "@thread_id", "value": PIN_THREAD_ID},
                {"name": "@type", "value": PIN_RECORD_TYPE},
            ],
            partition_key=partition_key_for_scope_thread(agent_scope, PIN_THREAD_ID, ctx.tenant_id),
            operation="list_pins query",
            container=self._memories_container,
        )
        return sorted(rows, key=lambda pin: (-int(pin.get("priority") or 0), str(pin.get("created_at") or "")))

    def resolve_pinned_memories(
        self,
        *,
        agent_id: str,
        ctx: SecurityContext,
        memory_types: list[str] | None = None,
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Resolve readable pinned memories for context injection within ``top_k``."""
        remaining = top_literal(top_k, name="top_k")
        types = list(memory_types or ["fact"])
        pins = self.list_pins(agent_id, ctx)
        resolved: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for pin in pins:
            if remaining <= 0:
                break
            resource_type = pin.get("resource_type")
            resource = str(pin.get("resource") or "")
            docs: list[dict[str, Any]] = []
            if resource_type == "memory":
                doc = self._read_pinned_memory(
                    ctx=ctx,
                    memory_id=resource,
                    memory_types=types,
                    include_superseded=include_superseded,
                )
                if doc is not None:
                    docs = [doc]
            elif resource_type == "scope":
                docs = self._read_pinned_scope(
                    ctx=ctx,
                    scope_key=resource,
                    memory_types=types,
                    limit=remaining,
                    include_superseded=include_superseded,
                )
            for doc in docs:
                doc_id = str(doc.get("id") or "")
                if not doc_id or doc_id in seen_ids:
                    continue
                resolved.append(apply_injection_mode(doc, pin))
                seen_ids.add(doc_id)
                remaining -= 1
                if remaining <= 0:
                    break
        return resolved

    def _require_promote_permission(self, ctx: SecurityContext, to_scope: str) -> None:
        if self._can_scope_action(ctx, to_scope, "write"):
            return
        raise ValidationError(
            f"promote requires write permission on target scope {to_scope!r} for principal {ctx.principal!r}"
        )

    def _read_memory_from_scope(self, memory_id: str, from_scope: str, tenant_id: str) -> dict[str, Any]:
        query = (
            "SELECT TOP 1 * FROM c WHERE c.tenant_id = @tenant_id AND c.scope_key = @scope_key AND c.id = @memory_id"
        )
        parameters = [
            {"name": "@tenant_id", "value": tenant_id},
            {"name": "@scope_key", "value": from_scope},
            {"name": "@memory_id", "value": memory_id},
        ]
        for container in (self._memories_container, self._summaries_container, self._turns_container):
            rows = self._query_items(
                query=query,
                parameters=parameters,
                cross_partition=True,
                operation="promotion source query",
                container=container,
            )
            if rows:
                return rows[0]
        raise MemoryNotFoundError(memory_id=memory_id)

    def promote(
        self,
        memory_id: str,
        from_scope: str,
        to_scope: str,
        ctx: SecurityContext,
    ) -> dict[str, Any]:
        """Copy a memory into ``to_scope`` after share/assign authorization."""
        self._require_promote_permission(ctx, to_scope)
        if from_scope not in self._resolve_scope_action(ctx, from_scope, "read").allowed_scopes:
            raise ValidationError(
                f"promote requires read permission on source scope {from_scope!r} for principal {ctx.principal!r}"
            )
        source_doc = self._read_memory_from_scope(memory_id, from_scope, ctx.tenant_id)
        if source_doc.get("tenant_id") != ctx.tenant_id:
            raise ValidationError("cannot promote a memory outside the caller tenant")
        if not can(ctx, source_doc.get("acl")):
            raise ValidationError(
                f"promote requires read access to source memory {memory_id!r} in scope {from_scope!r}"
            )
        tenant, scope_type, scope_id, scope_key = scope_values_for_scope_key(to_scope, ctx.tenant_id)
        now = utc_now_iso()
        promoted = {k: v for k, v in source_doc.items() if not k.startswith("_")}
        suffix_seed = f"{memory_id}|{from_scope}|{to_scope}"
        promoted["id"] = f"{memory_id}_promoted_{hashlib.sha256(suffix_seed.encode()).hexdigest()[:12]}"
        promoted["tenant_id"] = tenant
        promoted["scope_type"] = scope_type
        promoted["scope_id"] = scope_id
        promoted["scope_key"] = scope_key
        promoted["acl"] = default_acl_for_scope(scope_key, principal=ctx.principal, agent_id=ctx.agent_id).model_dump(
            mode="json"
        )
        provenance = dict(promoted.get("provenance") or {})
        provenance["created_by"] = provenance.get("created_by") or ctx.principal
        if ctx.agent_id and not provenance.get("agent_id"):
            provenance["agent_id"] = ctx.agent_id
        provenance["source"] = "promotion"
        source_ids = list(dict.fromkeys([*(provenance.get("source_ids") or []), memory_id]))
        provenance["source_ids"] = source_ids
        promoted["provenance"] = provenance
        promoted["updated_at"] = now
        promoted.setdefault("supersedes_ids", source_doc.get("supersedes_ids") or [])
        metadata = dict(promoted.get("metadata") or {})
        metadata.update(
            {
                "promoted_from_memory_id": memory_id,
                "promoted_from_scope": from_scope,
                "promoted_to_scope": to_scope,
                "promoted_by": ctx.principal,
                "promoted_at": now,
                "curation_status": CURATION_STATUS_APPROVED,
            }
        )
        promoted["metadata"] = metadata
        if promoted.get("type") != "procedural":
            promoted["status"] = CURATION_STATUS_APPROVED
        return self.upsert_memory(promoted)

    def list_promotion_candidates(
        self,
        ctx: SecurityContext,
        *,
        from_scope: str | None = None,
        min_confidence: float | None = None,
        top: int = 100,
    ) -> list[dict[str, Any]]:
        """List private records carrying advisory scope hints for manual review."""
        scope = from_scope or private_scope_key_for_principal(ctx.principal)
        read_resolution = self._resolve_scope_action(ctx, scope, "read")
        if scope not in read_resolution.allowed_scopes:
            raise ValidationError(f"list_promotion_candidates requires read permission on source scope {scope!r}")
        top_sql = top_literal(top, name="top")
        filters = [
            "c.tenant_id = @tenant_id",
            "c.scope_key = @scope_key",
            "c.scope_type = @user_scope_type",
            "IS_DEFINED(c.metadata.suggested_scope_type)",
            "c.metadata.suggested_scope_type != @personal_scope",
            "IS_DEFINED(c.metadata.scope_confidence)",
            "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))",
        ]
        parameters: list[dict[str, Any]] = [
            {"name": "@tenant_id", "value": ctx.tenant_id},
            {"name": "@scope_key", "value": scope},
            {"name": "@user_scope_type", "value": "user"},
            {"name": "@personal_scope", "value": PERSONAL_SCOPE_TYPE},
        ]
        if min_confidence is not None:
            filters.append("c.metadata.scope_confidence >= @min_confidence")
            parameters.append({"name": "@min_confidence", "value": float(min_confidence)})
        query = f"SELECT TOP {top_sql} * FROM c WHERE " + " AND ".join(filters) + " ORDER BY c.created_at DESC"
        rows = self._query_items(
            query=query,
            parameters=parameters,
            cross_partition=True,
            operation="list_promotion_candidates query",
            container=self._memories_container,
        )
        return rows

    def approve_promotion(
        self,
        memory_id: str,
        *,
        from_scope: str,
        to_scope: str,
        ctx: SecurityContext,
    ) -> dict[str, Any]:
        """Approve one queued candidate by promoting it into the target scope."""
        return self.promote(memory_id, from_scope, to_scope, ctx)

    def auto_promote_candidates(
        self,
        ctx: SecurityContext,
        *,
        target_scopes_by_type: dict[str, str],
        confidence_threshold: float = 0.95,
        allow_memory_types: set[str] | None = None,
        from_scope: str | None = None,
        top: int = 100,
    ) -> list[dict[str, Any]]:
        """Auto-promote eligible hinted records; unsafe records are skipped with reasons."""
        candidates = self.list_promotion_candidates(
            ctx,
            from_scope=from_scope,
            min_confidence=confidence_threshold,
            top=top,
        )
        source_scope = from_scope or private_scope_key_for_principal(ctx.principal)
        allow_types = allow_memory_types or {"fact", "episodic", "procedural"}
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            hint_type = str(metadata.get("suggested_scope_type") or "")
            target_scope = target_scopes_by_type.get(hint_type)
            confidence = float(metadata.get("scope_confidence") or 0.0)
            if confidence < confidence_threshold:
                results.append({"memory_id": candidate.get("id"), "status": "skipped", "reason": "low_confidence"})
                continue
            if candidate.get("type") not in allow_types:
                results.append(
                    {"memory_id": candidate.get("id"), "status": "skipped", "reason": "type_not_allowlisted"}
                )
                continue
            if not target_scope:
                results.append({"memory_id": candidate.get("id"), "status": "skipped", "reason": "no_target_scope"})
                continue
            if is_pii_or_secret_flagged(candidate):
                results.append({"memory_id": candidate.get("id"), "status": "skipped", "reason": "pii_or_secret"})
                continue
            has_write = self._can_scope_action(ctx, target_scope, "write")
            if not has_write:
                results.append({"memory_id": candidate.get("id"), "status": "skipped", "reason": "permission_denied"})
                continue
            promoted = self.promote(str(candidate["id"]), source_scope, target_scope, ctx)
            results.append({"memory_id": candidate.get("id"), "status": "promoted", "promoted": promoted})
        return results

    @staticmethod
    def _validate_shared_record_key(key: str) -> str:
        record_key = str(key).strip()
        if not record_key:
            raise ValidationError("shared record key cannot be empty")
        return record_key

    @staticmethod
    def _shared_user_id(ctx: SecurityContext) -> str:
        if ctx.principal and ctx.principal.startswith("user:") and ctx.principal != "user:":
            return ctx.principal.split(":", 1)[1]
        return ctx.principal or "shared"

    def _require_shared_scope_access(self, ctx: SecurityContext, scope_key: str, action: PermissionAction) -> None:
        scope_values_for_scope_key(scope_key, ctx.tenant_id)
        resolution = self._resolve_scope_action(ctx, scope_key, action)
        if scope_key not in resolution.allowed_scopes:
            raise ValidationError(
                f"{action} permission denied for shared record scope_key={scope_key!r} and principal {ctx.principal!r}"
            )

    def _shared_record_pk(self, ctx: SecurityContext, scope_key: str) -> list[str]:
        return partition_key_for_scope_thread(scope_key, SHARED_RECORD_THREAD_ID, ctx.tenant_id)

    def _read_shared_record_no_auth(self, *, ctx: SecurityContext, scope_key: str, key: str) -> dict[str, Any]:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        record_key = self._validate_shared_record_key(key)
        try:
            doc = self._memories_container.read_item(
                item=record_key,
                partition_key=self._shared_record_pk(ctx, scope_key),
            )
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=record_key, thread_id=SHARED_RECORD_THREAD_ID) from exc
        except Exception as exc:
            raise _wrap_cosmos_exception(
                exc, message=f"get_shared_record read failed for key {record_key!r}: {exc}"
            ) from exc
        if doc.get("tenant_id") != ctx.tenant_id or doc.get("scope_key") != scope_key:
            raise MemoryNotFoundError(memory_id=record_key, thread_id=SHARED_RECORD_THREAD_ID)
        if doc.get("thread_id") != SHARED_RECORD_THREAD_ID or doc.get("type") != _SHARED_STATE_TYPE:
            raise MemoryTypeMismatchError(memory_id=record_key, expected=_SHARED_STATE_TYPE, actual=doc.get("type"))
        return doc

    @staticmethod
    def _without_cosmos_system_fields(doc: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in doc.items() if not k.startswith("_")}

    def _prepare_shared_record_body(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        record: dict[str, Any],
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tenant, scope_type, scope_id, normalized_scope = scope_values_for_scope_key(scope_key, ctx.tenant_id)
        now = utc_now_iso()
        body = self._without_cosmos_system_fields(dict(record))
        body.update(
            {
                "id": self._validate_shared_record_key(key),
                "type": _SHARED_STATE_TYPE,
                "tenant_id": tenant,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "scope_key": normalized_scope,
                "thread_id": SHARED_RECORD_THREAD_ID,
                "role": body.get("role") or "system",
                "content": body.get("content") or "",
                "metadata": body.get("metadata") or {},
                "data": body.get("data") or {},
                "read_only": bool(body.get("read_only", (current or {}).get("read_only", False))),
                "acl": body.get("acl")
                or (current or {}).get("acl")
                or default_acl_for_scope(normalized_scope, principal=ctx.principal, agent_id=ctx.agent_id).model_dump(
                    mode="json"
                ),
                "provenance": {
                    **((current or {}).get("provenance") or {}),
                    **(body.get("provenance") or {}),
                    "created_by": (
                        (body.get("provenance") or {}).get("created_by")
                        or ((current or {}).get("provenance") or {}).get("created_by")
                        or ctx.principal
                    ),
                    **({"agent_id": ctx.agent_id} if ctx.agent_id else {}),
                },
                "created_at": body.get("created_at") or (current or {}).get("created_at") or now,
                "updated_at": now,
            }
        )
        return self._prepare_doc(body)

    def put_shared_record(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        content: str = "",
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        read_only: bool = False,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Create or replace a small live shared coordination record.

        These records are for plans, task queues, and handoff state. Keep them
        small and section-owned; use ``read_only=True`` for reference/config
        blocks that agents may read but tools must not mutate later.
        """
        self._require_shared_scope_access(ctx, scope_key, "write")
        current: dict[str, Any] | None = None
        try:
            current = self._read_shared_record_no_auth(ctx=ctx, scope_key=scope_key, key=key)
        except MemoryNotFoundError:
            current = None
        if current and current.get("read_only"):
            raise SharedRecordReadOnlyError(f"shared record {key!r} is read-only")
        body = self._prepare_shared_record_body(
            ctx=ctx,
            scope_key=scope_key,
            key=key,
            record={
                "content": content,
                "data": data or {},
                "metadata": metadata or {},
                "read_only": read_only,
                "provenance": {"created_by": created_by or ctx.principal},
            },
            current=current,
        )
        try:
            response = self._memories_container.upsert_item(body=body)
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"put_shared_record failed for key {key!r}: {exc}") from exc
        return response if isinstance(response, dict) else body

    def get_shared_record(self, *, ctx: SecurityContext, scope_key: str, key: str) -> dict[str, Any]:
        """Read one shared coordination record after read authorization."""
        self._require_shared_scope_access(ctx, scope_key, "read")
        return self._read_shared_record_no_auth(ctx=ctx, scope_key=scope_key, key=key)

    def compare_and_swap_shared_record(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        record: dict[str, Any],
        etag: str,
    ) -> dict[str, Any]:
        """Replace a shared record iff ``etag`` still matches the Cosmos document."""
        from azure.core import MatchConditions

        if not etag:
            raise ValidationError("etag is required for compare_and_swap_shared_record")
        self._require_shared_scope_access(ctx, scope_key, "write")
        current = self._read_shared_record_no_auth(ctx=ctx, scope_key=scope_key, key=key)
        if current.get("read_only"):
            raise SharedRecordReadOnlyError(f"shared record {key!r} is read-only")
        body = self._prepare_shared_record_body(ctx=ctx, scope_key=scope_key, key=key, record=record, current=current)
        try:
            response = self._memories_container.replace_item(
                item=body["id"],
                body=body,
                match_condition=MatchConditions.IfNotModified,
                etag=etag,
            )
        except Exception as exc:
            if _is_precondition_failed(exc):
                raise MemoryConflictError(f"shared record {key!r} was modified by another writer") from exc
            raise _wrap_cosmos_exception(
                exc, message=f"compare_and_swap_shared_record failed for key {key!r}: {exc}"
            ) from exc
        return response if isinstance(response, dict) else body

    def update_shared_record(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        mutator: Any,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Read-modify-write a shared record, retrying bounded ETag conflicts."""
        if not callable(mutator):
            raise ValidationError("mutator must be callable")
        if max_retries < 1:
            raise ValidationError("max_retries must be at least 1")
        self._require_shared_scope_access(ctx, scope_key, "write")
        last_conflict: MemoryConflictError | None = None
        for _ in range(max_retries):
            current = self._read_shared_record_no_auth(ctx=ctx, scope_key=scope_key, key=key)
            if current.get("read_only"):
                raise SharedRecordReadOnlyError(f"shared record {key!r} is read-only")
            draft = self._without_cosmos_system_fields(dict(current))
            mutated = mutator(draft)
            if mutated is None:
                mutated = draft
            if not isinstance(mutated, dict):
                raise ValidationError("mutator must return a dict or None")
            try:
                return self.compare_and_swap_shared_record(
                    ctx=ctx,
                    scope_key=scope_key,
                    key=key,
                    record=mutated,
                    etag=str(current.get("_etag") or ""),
                )
            except MemoryConflictError as exc:
                last_conflict = exc
                continue
        raise MemoryConflictError(
            f"shared record {key!r} update conflicted after {max_retries} attempts"
        ) from last_conflict

    def upsert_memory(self, record: dict[str, Any]) -> dict[str, Any]:
        """Upsert a pre-built Cosmos memory document and return the stored body."""
        body = self._prepare_doc(record)
        memory_type = body.get("type")
        if memory_type not in _CONTAINER_FOR_TYPE:
            raise ValueError(
                f"upsert_memory: record id={body.get('id')!r} has invalid type={memory_type!r}. "
                f"Set 'type' to one of {sorted(_CONTAINER_FOR_TYPE)} before calling upsert_memory."
            )
        container = self._container_for_type(memory_type)
        try:
            response = container.upsert_item(body=body)
        except Exception as exc:
            raise _wrap_cosmos_exception(
                exc, message=f"upsert_memory upsert failed for record {body.get('id')}: {exc}"
            ) from exc
        logger.info("upsert_memory id=%s role=%s type=%s", body.get("id"), body.get("role"), body.get("type"))
        return response if isinstance(response, dict) else body

    def add(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        ttl: Optional[int] = None,
        salience: Optional[float] = None,
        embedding: Optional[list[float]] = None,
        embed: Optional[bool] = None,
        created_at: Optional[str | datetime] = None,
        tenant_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        ctx: Optional[SecurityContext] = None,
    ) -> str:
        """Add a memory document to Cosmos DB and return its id."""
        self._authorize_write(ctx, scope_key, user_id)
        kwargs: dict[str, Any] = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "memory_type": memory_type,
            "metadata": metadata or {},
        }
        if thread_id is not None:
            kwargs["thread_id"] = thread_id
        if tags is not None:
            kwargs["tags"] = tags
        if ttl is not None:
            kwargs["ttl"] = ttl
        if salience is not None:
            kwargs["salience"] = salience
        if created_at is not None:
            kwargs["created_at"] = normalize_created_at_iso(created_at)
        if memory_type != "turn":
            kwargs.setdefault("content_hash", compute_content_hash(content))
            provenance = dict(kwargs.get("provenance") or {})
            provenance.setdefault("prompt_id", "manual:add")
            kwargs["provenance"] = provenance
            kwargs.setdefault("id", new_id(memory_type))
            meta = kwargs.get("metadata") or {}
            if memory_type == "fact":
                meta.setdefault("category", "unclassified:manual")
            elif memory_type == "episodic":
                kwargs.setdefault("title", content[:80] or "Manual episode")
                kwargs.setdefault("events", [])
                kwargs.setdefault("participants", [])
                kwargs.setdefault("lessons", [])
                kwargs.setdefault("source_turn_ids", [])
            elif memory_type == "procedural":
                kwargs.setdefault("source_fact_ids", ["manual"])
                # Manually-added procedures still satisfy the required procedural fields:
                # default to a behavioral policy (no steps required) and derive name /
                # summary / retrieval_text from the content. Pipeline-synthesized
                # procedures set these explicitly.
                kwargs.setdefault("name", (content[:80] or "Manual procedure"))
                kwargs.setdefault("summary", content)
                kwargs.setdefault("retrieval_text", content)
                kwargs.setdefault("procedure_kind", ProcedureKind.behavioral_policy.value)
            kwargs["metadata"] = meta
        provenance = dict(kwargs.get("provenance") or {})
        provenance.setdefault("created_by", f"user:{user_id}")
        if agent_id is not None:
            provenance["agent_id"] = agent_id
        kwargs["provenance"] = provenance
        record = MemoryRecord(**kwargs)
        body = record.to_doc()

        if embed is None:
            embed = memory_type != "turn" or self._enable_turn_embeddings
        if embedding is not None:
            body["embedding"] = embedding
        elif embed and content and self._embeddings_client is not None:
            try:
                body["embedding"] = self._embeddings_client.generate(content)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "upsert_memory: embedding generation failed for %s (%s); proceeding without embedding",
                    record.id,
                    exc,
                )

        body = ensure_scope_fields(
            body, tenant_id=tenant_id, scope_key=scope_key, principal=f"user:{user_id}", agent_id=agent_id
        )
        body = self._prepare_doc(body)
        try:
            container = self._container_for_type(memory_type)
            container.upsert_item(body=body)
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("upsert_memory id=%s role=%s type=%s", record.id, role, memory_type)
        return record.id

    def push(self, local_memory: list[dict[str, Any]], batch_size: int = 25) -> None:
        """Upsert all local memory records to Cosmos DB in sequential batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        logger.info(
            "push_to_cosmos count=%d batch_size=%d",
            len(local_memory),
            batch_size,
        )
        records = [dict(m) for m in local_memory]
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            bodies = [dict(r) for r in batch]

            to_embed_idx: list[int] = []
            to_embed_text: list[str] = []
            for i, body in enumerate(bodies):
                embeddable_type = body.get("type") != "turn" or self._enable_turn_embeddings
                if embeddable_type and body.get("content") and not body.get("embedding"):
                    to_embed_idx.append(i)
                    to_embed_text.append(body["content"])
            if to_embed_text and self._embeddings_client is not None:
                try:
                    vectors = self._embeddings_client.generate_batch(to_embed_text)
                    for i, vec in zip(to_embed_idx, vectors):
                        bodies[i]["embedding"] = vec
                        local_memory[start + i]["embedding"] = vec
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "push_to_cosmos: batch embedding generation failed (%s); "
                        "proceeding without embeddings for %d records",
                        exc,
                        len(to_embed_text),
                    )

            bodies = [self._prepare_doc(body) for body in bodies]
            for body in bodies:
                memory_type = body.get("type")
                if memory_type not in _CONTAINER_FOR_TYPE:
                    raise ValueError(
                        f"push: record id={body.get('id')!r} has invalid type={memory_type!r}. "
                        f"Set 'type' to one of {sorted(_CONTAINER_FOR_TYPE)} on every local "
                        f"memory before calling push_to_cosmos."
                    )
            for record, body in zip(batch, bodies):
                container = self._container_for_type(body.get("type"))
                try:
                    container.upsert_item(body=body)
                except Exception as exc:
                    record_id = record.get("id") if isinstance(record, dict) else getattr(record, "id", None)
                    raise _wrap_cosmos_exception(exc, message=f"Upsert failed for record {record_id}: {exc}") from exc
        logger.info("Upserted batch of %d records", len(records))

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from the MEMORIES container with optional filters."""
        types = _validated_memories_types(memory_types)
        logger.debug(
            "get_memories filters: memory_id=%s user_id=%s thread_id=%s role=%s types=%s recent_k=%s",
            memory_id,
            user_id,
            thread_id,
            role,
            types,
            recent_k,
        )

        qb = _build_memory_query_builder(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            memory_types=types,
            min_confidence=min_confidence,
        )

        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        where = qb.build_where()
        parameters = qb.get_parameters()

        if recent_k is not None:
            parameters.append({"name": "@recent_k", "value": recent_k})
            query = f"SELECT TOP @recent_k * FROM c{where} ORDER BY c._ts DESC"
        else:
            query = f"SELECT * FROM c{where}"

        logger.debug("get_memories query: %s", query)
        items = self._query_items(
            query=query,
            parameters=parameters or None,
            cross_partition=True,
            operation="get_memories query",
            container=self._memories_container,
        )

        if recent_k is not None:
            items.reverse()
        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]
        if not items:
            logger.warning("get_memories returned empty results")
        return items

    def update(
        self,
        memory_id: str,
        *,
        user_id: str,
        thread_id: str,
        memory_type: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory document via point read in the container for ``memory_type``."""
        import random
        import time

        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosResourceNotFoundError,
        )

        container = self._container_for_type(memory_type)
        max_attempts = 5
        attempts = 0
        while True:
            try:
                doc = container.read_item(
                    item=memory_id,
                    partition_key=partition_key_for_user_thread(user_id, thread_id),
                )
            except CosmosResourceNotFoundError as exc:
                raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
            except Exception as exc:
                raise _wrap_cosmos_exception(exc, message=f"update read failed for {memory_id}: {exc}") from exc

            actual_type = doc.get("type")
            if actual_type != memory_type:
                raise MemoryTypeMismatchError(memory_id=memory_id, expected=memory_type, actual=actual_type)

            if content is not None:
                doc["content"] = content
            if role is not None:
                doc["role"] = role
            if metadata is not None:
                doc["metadata"] = metadata
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()

            kwargs: dict[str, Any] = {"item": doc["id"], "body": doc}
            if etag := doc.get("_etag"):
                kwargs.update(match_condition=MatchConditions.IfNotModified, etag=etag)
            try:
                container.replace_item(**kwargs)
                logger.info("Updated record %s", memory_id)
                return
            except CosmosAccessConditionFailedError as exc:
                attempts += 1
                if attempts >= max_attempts:
                    raise MemoryConflictError(
                        f"update conflicted after {max_attempts} attempts for memory_id={memory_id!r}"
                    ) from exc
                base = 0.02 * (2 ** (attempts - 1))
                time.sleep(base + random.uniform(0, base))
            except Exception as exc:
                raise _wrap_cosmos_exception(exc, message=f"update replace failed for {memory_id}: {exc}") from exc

    def delete(
        self,
        memory_id: str,
        *,
        user_id: str,
        thread_id: str,
        memory_type: str,
    ) -> None:
        """Delete a memory document from the container for ``memory_type``.

        Reads the doc first to verify its ``type`` matches ``memory_type`` and
        then issues the delete with ``If-Match`` on the read ETag, so a
        concurrent type mutation between read and delete is rejected (412)
        rather than silently dropping the wrong document.
        """
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosResourceNotFoundError,
        )

        container = self._container_for_type(memory_type)
        try:
            doc = container.read_item(item=memory_id, partition_key=partition_key_for_user_thread(user_id, thread_id))
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"delete read failed for {memory_id}: {exc}") from exc

        actual_type = doc.get("type")
        if actual_type != memory_type:
            raise MemoryTypeMismatchError(memory_id=memory_id, expected=memory_type, actual=actual_type)

        kwargs: dict[str, Any] = {"item": memory_id, "partition_key": partition_key_for_user_thread(user_id, thread_id)}
        if etag := doc.get("_etag"):
            kwargs.update(match_condition=MatchConditions.IfNotModified, etag=etag)
        try:
            container.delete_item(**kwargs)
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
        except CosmosAccessConditionFailedError as exc:
            raise MemoryConflictError(
                f"delete conflicted for memory_id={memory_id!r} - document was modified after the type check"
            ) from exc
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"delete failed for {memory_id}: {exc}") from exc

        logger.info("Deleted record %s", memory_id)

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        recent_k: Optional[int] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread (turns) sorted oldest first."""
        qb = _QueryBuilder()
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT {_TURN_PROJECTION} FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        logger.debug("get_thread query: %s", query)
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_thread query",
            container=self._turns_container,
        )
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    def get_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active thread summaries for ``(user_id, thread_id)``, newest first."""
        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "thread_summary")
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_is_null_or_undefined("c.superseded_by")
        parameters = qb.get_parameters()
        if recent_k is not None:
            parameters.append({"name": "@recent_k", "value": recent_k})
            sql = f"SELECT TOP @recent_k * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        else:
            sql = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        return self._query_items(
            query=sql,
            parameters=parameters,
            partition_key=partition_key_for_user_thread(user_id, thread_id),
            operation="get_thread_summary query",
            container=self._summaries_container,
        )

    def get_episodes(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active episodic memories for ``user_id``, newest first."""
        if not user_id:
            raise ValidationError("user_id is required for get_episodes")
        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "episodic")
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_is_null_or_undefined("c.superseded_by")
        parameters = qb.get_parameters()
        if recent_k is not None:
            parameters.append({"name": "@recent_k", "value": recent_k})
            sql = f"SELECT TOP @recent_k * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        else:
            sql = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        partition_key, cross_partition = query_scope(user_id, thread_id)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
        )

    def get_user_summary(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the user's summary document from Cosmos DB, or ``None`` if absent."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return self._summaries_container.read_item(
                item=f"user_summary_{user_id}",
                partition_key=partition_key_for_user_thread(user_id, USER_SUMMARY_THREAD_ID),
            )
        except CosmosResourceNotFoundError:
            return None
        except Exception as exc:
            raise CosmosOperationError(f"get_user_summary read failed: {exc}") from exc

    def list_tags(
        self,
        user_id: str,
        *,
        thread_id: Optional[str] = None,
        prefix: Optional[str] = None,
        include_sys: bool = False,
        include_superseded: bool = False,
    ) -> list[str]:
        """Return sorted distinct tags for a user from the MEMORIES container."""
        query = f"SELECT VALUE c.tags FROM c WHERE {user_scope_predicate()} AND ARRAY_LENGTH(c.tags) > 0"
        parameters = user_scope_parameters(user_id)
        if thread_id is not None:
            query += " AND c.thread_id = @thread_id"
            parameters.append({"name": "@thread_id", "value": thread_id})
        if not include_superseded:
            query += " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"

        prefix_norm = prefix.strip().lower() if prefix else None
        partition_key, cross_partition = query_scope(user_id, thread_id)
        rows = self._query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
            cross_partition=cross_partition,
            operation="list_tags query",
            container=self._memories_container,
        )
        tags: set[str] = set()
        for row in rows:
            values = row.get("tags", []) if isinstance(row, dict) else row
            for tag in values or []:
                tag_value = str(tag).strip().lower()
                if not tag_value:
                    continue
                if not include_sys and tag_value.startswith("sys:"):
                    continue
                if prefix_norm is not None and not tag_value.startswith(prefix_norm):
                    continue
                tags.add(tag_value)
        return sorted(tags)

    def _mutate_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
        *,
        add: bool,
    ) -> None:
        import random
        import time

        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosResourceNotFoundError,
        )

        _validate_taggable_type(memory_type)
        container = self._container_for_type(memory_type)
        normalized = {t.strip().lower() for t in tags if t and t.strip()}
        max_attempts = 5
        attempts = 0
        while True:
            try:
                doc = container.read_item(
                    item=memory_id,
                    partition_key=partition_key_for_user_thread(user_id, thread_id),
                )
            except CosmosResourceNotFoundError as exc:
                raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
            existing_tags = set(doc.get("tags", []))
            if add:
                existing_tags.update(normalized)
            else:
                existing_tags.difference_update(normalized)
            doc["tags"] = sorted(existing_tags)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()

            kwargs: dict[str, Any] = {"item": memory_id, "body": doc}
            if etag := doc.get("_etag"):
                kwargs.update(match_condition=MatchConditions.IfNotModified, etag=etag)
            try:
                container.replace_item(**kwargs)
                return
            except CosmosAccessConditionFailedError as exc:
                attempts += 1
                if attempts >= max_attempts:
                    raise MemoryConflictError(
                        f"Tag update conflicted after {max_attempts} attempts for memory_id={memory_id!r}"
                    ) from exc
                base = 0.02 * (2 ** (attempts - 1))
                time.sleep(base + random.uniform(0, base))

    def add_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Add tags to an existing memory document."""
        self._mutate_tags(memory_id, user_id, thread_id, memory_type, tags, add=True)

    def remove_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Remove tags from an existing memory document."""
        self._mutate_tags(memory_id, user_id, thread_id, memory_type, tags, add=False)

    def mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: str,
    ) -> bool:
        """Set supersession audit fields using ETag protection when available."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosHttpResponseError,
        )

        etag = old_doc.get("_etag")
        new_doc = {
            **old_doc,
            "superseded_by": superseder_id,
            "supersede_reason": reason,
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        target_container = self._container_for_type(old_doc.get("type"))
        try:
            if etag:
                target_container.replace_item(
                    item=new_doc["id"],
                    body=new_doc,
                    match_condition=MatchConditions.IfNotModified,
                    etag=etag,
                )
            else:
                target_container.upsert_item(body=new_doc)
            return True
        except CosmosAccessConditionFailedError as exc:
            logger.warning(
                "supersede skipped (concurrent writer won) id=%s superseder=%s",
                old_doc.get("id"),
                superseder_id,
                extra={"operation": "mark_superseded"},
            )
            del exc
            return False
        except CosmosHttpResponseError as exc:
            logger.warning(
                "supersede failed id=%s superseder=%s status=%s: %s",
                old_doc.get("id"),
                superseder_id,
                getattr(exc, "status_code", None),
                exc,
            )
            return False

    def get_procedural_prompt(self, user_id: str) -> Optional[str]:
        """Return the active synthesized procedural prompt for a user."""
        qb = _QueryBuilder()
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")
        qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT TOP 1 c.content, c.version FROM c{qb.build_where()} ORDER BY c.version DESC"
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_procedural_prompt query",
            container=self._memories_container,
        )
        if not items:
            return None
        return items[0].get("content")

    def get_procedural_history(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return synthesized procedural docs for a user, newest first."""
        if limit <= 0:
            return []

        qb = _QueryBuilder()
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")

        query = f"SELECT * FROM c{qb.build_where()} ORDER BY c.version DESC"
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_procedural_history query",
            container=self._memories_container,
        )

        def _is_active(doc: dict[str, Any]) -> bool:
            return not doc.get("superseded_by")

        items.sort(
            key=lambda doc: (
                1 if _is_active(doc) else 0,
                int(doc.get("version") or 0),
                int(doc.get("_ts") or 0),
            ),
            reverse=True,
        )
        return items[:limit]

    def get_procedural_memories(
        self,
        user_id: str,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve active procedural memories for a user."""
        qb = _QueryBuilder()
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_procedural_memories query",
            container=self._memories_container,
        )

        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]
        if priority is not None:
            items = [i for i in items if i.get("metadata", {}).get("priority") == priority]
        if category is not None:
            items = [i for i in items if i.get("metadata", {}).get("category") == category]
        return items

    def search(
        self,
        search_terms: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        tenant_id: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        ctx: Optional[SecurityContext] = None,
        *,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search memories using vector similarity with optional full-text hybrid ranking."""
        terms = require_search_terms(search_terms, query)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)
        scope_keys = normalize_scope_keys(scopes)
        if self._shared_scopes_need_ctx(ctx, scope_keys, user_id):
            logger.info("search denied: shared scope requested without a SecurityContext")
            return []
        authz_resolution = None
        if ctx is not None and scope_keys:
            tenant_id = ctx.tenant_id
            authz_resolution = self._resolve_read_authz(ctx, scope_keys)
            scope_keys = [scope_key for scope_key in scope_keys if scope_key in authz_resolution.allowed_scopes]
            if not scope_keys:
                return []

        def run_one(scope_key: str | None) -> list[dict[str, Any]]:
            scoped_user_id = None if scope_key is not None else user_id
            qb = _build_memory_query_builder(
                memory_id=memory_id,
                user_id=scoped_user_id,
                role=role,
                memory_types=memory_types,
                thread_id=thread_id,
                min_confidence=min_confidence,
            )
            if scope_key is not None:
                add_tenant_scope_filter(qb, tenant_id=tenant_id, scope_key=scope_key)
            if authz_resolution is not None:
                qb.add_condition(authz_resolution.predicate.sql, authz_resolution.predicate.parameters)
            add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
            qb.add_time_range(
                "c.created_at",
                after=_coerce_datetime_iso(created_after),
                before=_coerce_datetime_iso(created_before),
                after_param="@created_after",
                before_param="@created_before",
            )
            add_salience_filter(qb, min_salience)

            sql = build_search_sql(
                qb=qb,
                top=top,
                keyword_count=len(keywords),
                include_superseded=include_superseded,
            )
            parameters = qb.get_parameters()
            parameters.append({"name": "@embedding", "value": query_vector})
            for i, kw in enumerate(keywords):
                parameters.append({"name": f"@kw{i}", "value": kw})

            partition_key, cross_partition = query_scope(scoped_user_id, thread_id, tenant_id, scope_key)
            if thread_id is not None and (not memory_types or set(memory_types) & USER_SCOPED_MEMORIES_TYPES):
                partition_key, cross_partition = None, True
            logger.debug("MemoryStore.search query: %s", sql)
            return self.query(
                sql,
                parameters,
                container_key=ContainerKey.MEMORIES,
                partition_key=partition_key,
                cross_partition=cross_partition,
            )

        if not scope_keys or len(scope_keys) == 1:
            return run_one(scope_keys[0] if scope_keys else None)
        rows: list[dict[str, Any]] = []
        for scope_key in scope_keys:
            rows.extend(run_one(scope_key))
        return merge_ranked_results(rows, top_k=top)

    def get_memory_history(
        self,
        memory_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Return a memory's superseded predecessors, most-recently-superseded first.

        AMT supersedes rather than deletes: when a fact is updated or
        contradicted, the prior document is retained with ``superseded_by``
        pointing at its replacement (see :meth:`mark_superseded`). This walks
        that chain backwards from *memory_id* so callers can reason about how a
        fact evolved - knowledge updates, preference reversals, relocations -
        instead of seeing only the current value.

        The document identified by *memory_id* is not itself included; the return
        value is everything it superseded, transitively, bounded by *max_depth*
        to guard against cycles. Each entry carries the ``superseded_at`` /
        ``supersede_reason`` audit fields so callers can order and explain the
        transitions.
        """
        if not memory_id:
            raise ValidationError("memory_id is required")
        if not user_id:
            raise ValidationError("user_id is required")
        partition_key, cross_partition = query_scope(user_id, thread_id)
        history: list[dict[str, Any]] = []
        seen: set[str] = {memory_id}
        frontier: list[str] = [memory_id]
        for _ in range(max(1, max_depth)):
            id_params = [{"name": f"@sid{i}", "value": sid} for i, sid in enumerate(frontier)]
            placeholders = ", ".join(param["name"] for param in id_params)
            parameters: list[dict[str, Any]] = [*id_params, *user_scope_parameters(user_id)]
            where = f"c.superseded_by IN ({placeholders}) AND {user_scope_predicate()}"
            if thread_id is not None:
                where += " AND c.thread_id = @thread_id"
                parameters.append({"name": "@thread_id", "value": thread_id})
            sql = f"SELECT {MEMORY_PROJECTION} FROM c WHERE {where}"
            rows = self.query(
                sql,
                parameters,
                container_key=ContainerKey.MEMORIES,
                partition_key=partition_key,
                cross_partition=cross_partition,
            )
            frontier = []
            for doc in rows:
                doc_id = doc.get("id")
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                history.append(doc)
                frontier.append(doc_id)
            if not frontier:
                break
        history.sort(key=lambda d: d.get("superseded_at") or d.get("created_at") or "", reverse=True)
        return history

    def search_turns(
        self,
        search_terms: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        tenant_id: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        ctx: Optional[SecurityContext] = None,
        *,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search raw conversation turns using vector similarity with hybrid ranking."""
        if not user_id and not scopes:
            raise ValidationError("user_id is required unless scopes is provided for search_turns")
        terms = require_search_terms(search_terms, query)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)
        scope_keys = normalize_scope_keys(scopes)
        if self._shared_scopes_need_ctx(ctx, scope_keys, user_id):
            logger.info("search_turns denied: shared scope requested without a SecurityContext")
            return []
        authz_resolution = None
        if ctx is not None and scope_keys:
            tenant_id = ctx.tenant_id
            authz_resolution = self._resolve_read_authz(ctx, scope_keys)
            scope_keys = [scope_key for scope_key in scope_keys if scope_key in authz_resolution.allowed_scopes]
            if not scope_keys:
                return []

        def build_query(scope_key: str | None) -> tuple[str, list[dict[str, Any]], Any, bool]:
            scoped_user_id = None if scope_key is not None else user_id
            qb = _QueryBuilder()
            qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(scoped_user_id))
            if scope_key is not None:
                add_tenant_scope_filter(qb, tenant_id=tenant_id, scope_key=scope_key)
            if authz_resolution is not None:
                qb.add_condition(authz_resolution.predicate.sql, authz_resolution.predicate.parameters)
            qb.add_filter("c.thread_id", "@thread_id", thread_id)
            qb.add_filter("c.role", "@role", role)
            add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
            qb.add_time_range(
                "c.created_at",
                after=_coerce_datetime_iso(created_after),
                before=_coerce_datetime_iso(created_before),
                after_param="@created_after",
                before_param="@created_before",
            )
            sql = build_search_sql(qb=qb, top=top, keyword_count=len(keywords), include_superseded=False)
            parameters = qb.get_parameters()
            parameters.append({"name": "@embedding", "value": query_vector})
            for i, kw in enumerate(keywords):
                parameters.append({"name": f"@kw{i}", "value": kw})
            partition_key, cross_partition = query_scope(scoped_user_id, thread_id, tenant_id, scope_key)
            return sql, parameters, partition_key, cross_partition

        if not scope_keys or len(scope_keys) == 1:
            sql, parameters, partition_key, cross_partition = build_query(scope_keys[0] if scope_keys else None)
            logger.debug("MemoryStore.search_turns query: %s", sql)
            return self.query(
                sql,
                parameters,
                container_key=ContainerKey.TURNS,
                partition_key=partition_key,
                cross_partition=cross_partition,
            )
        rows: list[dict[str, Any]] = []
        for scope_key in scope_keys:
            sql, parameters, partition_key, cross_partition = build_query(scope_key)
            rows.extend(
                self.query(
                    sql,
                    parameters,
                    container_key=ContainerKey.TURNS,
                    partition_key=partition_key,
                    cross_partition=cross_partition,
                )
            )
        return merge_ranked_results(rows, top_k=top)

    def search_summaries(
        self,
        search_terms: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        tenant_id: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        ctx: Optional[SecurityContext] = None,
        *,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Vector-search summaries, optionally as an explicit union of scope keys."""
        if not user_id and not scopes:
            raise ValidationError("user_id is required unless scopes is provided for search_summaries")
        terms = require_search_terms(search_terms, query)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)
        scope_keys = normalize_scope_keys(scopes)
        if self._shared_scopes_need_ctx(ctx, scope_keys, user_id):
            logger.info("search_summaries denied: shared scope requested without a SecurityContext")
            return []
        authz_resolution = None
        if ctx is not None and scope_keys:
            tenant_id = ctx.tenant_id
            authz_resolution = self._resolve_read_authz(ctx, scope_keys)
            scope_keys = [scope_key for scope_key in scope_keys if scope_key in authz_resolution.allowed_scopes]
            if not scope_keys:
                return []

        def build_query(scope_key: str | None) -> tuple[str, list[dict[str, Any]], Any, bool]:
            scoped_user_id = None if scope_key is not None else user_id
            qb = _QueryBuilder()
            qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(scoped_user_id))
            if scope_key is not None:
                add_tenant_scope_filter(qb, tenant_id=tenant_id, scope_key=scope_key)
            if authz_resolution is not None:
                qb.add_condition(authz_resolution.predicate.sql, authz_resolution.predicate.parameters)
            qb.add_filter("c.thread_id", "@thread_id", thread_id)
            add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
            sql = build_search_sql(qb=qb, top=top, keyword_count=len(keywords), include_superseded=False)
            parameters = qb.get_parameters()
            parameters.append({"name": "@embedding", "value": query_vector})
            for i, kw in enumerate(keywords):
                parameters.append({"name": f"@kw{i}", "value": kw})
            partition_key, cross_partition = query_scope(scoped_user_id, thread_id, tenant_id, scope_key)
            return sql, parameters, partition_key, cross_partition

        if not scope_keys or len(scope_keys) == 1:
            sql, parameters, partition_key, cross_partition = build_query(scope_keys[0] if scope_keys else None)
            logger.debug("MemoryStore.search_summaries query: %s", sql)
            return self.query(
                sql,
                parameters,
                container_key=ContainerKey.SUMMARIES,
                partition_key=partition_key,
                cross_partition=cross_partition,
            )
        rows: list[dict[str, Any]] = []
        for scope_key in scope_keys:
            sql, parameters, partition_key, cross_partition = build_query(scope_key)
            rows.extend(
                self.query(
                    sql,
                    parameters,
                    container_key=ContainerKey.SUMMARIES,
                    partition_key=partition_key,
                    cross_partition=cross_partition,
                )
            )
        return merge_ranked_results(rows, top_k=top)

    def search_episodic(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
        thread_id: Optional[str] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        started_after: Optional[str | datetime] = None,
        started_before: Optional[str | datetime] = None,
        ended_after: Optional[str | datetime] = None,
        ended_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across episodic memories for a user.

        Temporal arguments are filters only; relevance ranking is vector/FTS-only.
        """
        if not user_id:
            raise ValidationError("user_id is required for search_episodic")
        terms = require_search_terms(search_terms)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)

        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "episodic")
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
        qb.add_time_range(
            "c.started_at",
            after=_coerce_datetime_iso(started_after),
            before=_coerce_datetime_iso(started_before),
            after_param="@started_after",
            before_param="@started_before",
        )
        qb.add_time_range(
            "c.ended_at",
            after=_coerce_datetime_iso(ended_after),
            before=_coerce_datetime_iso(ended_before),
            after_param="@ended_after",
            before_param="@ended_before",
        )
        add_salience_filter(qb, min_salience)

        sql = build_search_sql(
            qb=qb,
            top=top,
            keyword_count=len(keywords),
            include_superseded=include_superseded,
        )
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, thread_id)
        logger.debug("MemoryStore.search_episodic query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
        )

    def build_episodic_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Build formatted context of relevant past experiences."""
        memories = self.search_episodic(user_id, query, top_k=top_k)
        return format_episodic_context(memories)

    def retrieve_procedures(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        *,
        procedure_kind: Optional[str] = None,
        status: Optional[str] = "active",
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic search across procedural memories for a user."""
        if not user_id:
            raise ValidationError("user_id is required for retrieve_procedures")
        terms = require_search_terms(search_terms)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)

        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "procedural")
        qb.add_filter("c.scope_key", "@scope_key", scope_key_for_user(user_id))
        qb.add_filter("c.procedure_kind", "@procedure_kind", procedure_kind)
        qb.add_filter("c.status", "@status", status)

        sql = build_search_sql(
            qb=qb,
            top=top,
            keyword_count=len(keywords),
            include_superseded=include_superseded,
        )
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, None)
        logger.debug("MemoryStore.retrieve_procedures query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
        )

    def _embed(self, text: str) -> list[float]:
        if self._embeddings_client is None:
            raise ConfigurationError(
                "An embeddings_client is required for retrieval search",
                parameter="embeddings_client",
            )
        for method_name in ("generate", "embed_one"):
            method = getattr(self._embeddings_client, method_name, None)
            if callable(method):
                return coerce_embedding(method(text))
        raise ConfigurationError(
            "embeddings_client must expose generate or embed_one",
            parameter="embeddings_client",
        )
