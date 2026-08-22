"""Shared helpers for selective memory injection pins."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from azure.cosmos.agent_memory._partitioning import scope_values_for_scope_key
from azure.cosmos.agent_memory.exceptions import ValidationError

PIN_RECORD_TYPE = "memory_pin"
PIN_THREAD_ID = "__pins__"
InjectionMode = Literal["direct", "summary", "reference"]
_RESOURCE_TYPES = frozenset({"memory", "scope"})
_SCOPE_TYPES = frozenset({"user", "agent", "team", "project", "org", "global"})
_INJECTION_MODES = frozenset({"direct", "summary", "reference"})


def agent_scope_key(agent_id: str) -> str:
    agent = str(agent_id).strip()
    if not agent:
        raise ValidationError("agent_id cannot be empty")
    return agent if agent.startswith("agent:") else f"agent:{agent}"


def normalize_injection_mode(injection_mode: str) -> InjectionMode:
    mode = str(injection_mode or "").strip().lower()
    if mode not in _INJECTION_MODES:
        raise ValidationError("injection_mode must be one of: direct, summary, reference")
    return mode  # type: ignore[return-value]


def normalize_priority(priority: int) -> int:
    try:
        return int(priority)
    except (TypeError, ValueError) as exc:
        raise ValidationError("priority must be an integer") from exc


def classify_pin_resource(resource: str, resource_type: str | None = None) -> tuple[str, str]:
    """Classify a pin target as a ``memory`` id or a ``scope`` key.

    When ``resource_type`` is given it is authoritative (disambiguating a scope-shaped
    custom memory id such as ``user:123`` that would otherwise auto-classify as a scope);
    otherwise the type is inferred from the ``<type>:<id>`` prefix.
    """
    value = str(resource).strip()
    if not value:
        raise ValidationError("resource cannot be empty")
    if resource_type is not None:
        rt = str(resource_type).strip().lower()
        if rt not in _RESOURCE_TYPES:
            raise ValidationError("resource_type must be memory or scope")
        if rt == "scope":
            scope_values_for_scope_key(value)
        return rt, value
    prefix, _, _ = value.partition(":")
    if prefix not in _SCOPE_TYPES:
        return "memory", value
    try:
        scope_values_for_scope_key(value)
    except ValidationError:
        return "memory", value
    return "scope", value


def pin_id(*, tenant_id: str, agent_id: str, resource_type: str, resource: str) -> str:
    if resource_type not in _RESOURCE_TYPES:
        raise ValidationError("resource_type must be memory or scope")
    digest = hashlib.sha256(
        "\x1f".join([str(tenant_id), str(agent_id), resource_type, str(resource)]).encode("utf-8")
    ).hexdigest()[:32]
    return f"pin_{digest}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_injection_mode(memory: dict[str, Any], pin: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``memory`` annotated and projected for the pin mode."""
    mode = normalize_injection_mode(str(pin.get("injection_mode") or "summary"))
    projected = dict(memory)
    metadata = dict(projected.get("metadata") or {})
    metadata["pin"] = {
        "id": pin.get("id"),
        "agent_id": pin.get("agent_id"),
        "resource": pin.get("resource"),
        "resource_type": pin.get("resource_type"),
        "injection_mode": mode,
        "priority": pin.get("priority"),
    }
    metadata["injection_mode"] = mode
    projected["metadata"] = metadata
    projected["pinned"] = True
    projected["pin_id"] = pin.get("id")
    projected["pin_priority"] = pin.get("priority")
    projected["injection_mode"] = mode
    if mode == "summary":
        summary = metadata.get("summary") or projected.get("summary") or projected.get("title")
        if summary:
            projected["content"] = str(summary)
        else:
            # No summary/title to inject: fall back to a reference rather than leaking the
            # full content, which would defeat the token-budget / exposure intent of
            # summary mode.
            reference = projected.get("content_ref") or metadata.get("content_ref") or projected.get("id")
            projected["content"] = f"Pinned memory reference: {reference}"
            projected["injection_mode"] = "reference"
            metadata["injection_mode"] = "reference"
            projected["metadata"] = metadata
    elif mode == "reference":
        reference = projected.get("content_ref") or metadata.get("content_ref") or projected.get("id")
        projected["content"] = f"Pinned memory reference: {reference}"
    return projected
