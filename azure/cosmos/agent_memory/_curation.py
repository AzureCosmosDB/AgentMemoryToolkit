"""Shared helpers for memory curation and promotion hints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.models import MemoryScopeType

CURATION_STATUS_DRAFT = "draft"
CURATION_STATUS_CANDIDATE = "candidate"
CURATION_STATUS_APPROVED = "approved"
CURATION_STATUS_DEPRECATED = "deprecated"
CURATION_STATUS_VALUES = frozenset(
    {
        CURATION_STATUS_DRAFT,
        CURATION_STATUS_CANDIDATE,
        CURATION_STATUS_APPROVED,
        CURATION_STATUS_DEPRECATED,
    }
)

# Scope hints are advisory. suggested_scope_type is one of the full scope enum
# {user, agent, team, project, org, global}; "user" means the personal/private
# default (no promotion). Only the shared scopes below are promotion candidates.
# tenant_id is an auth boundary, never a scope hint.
_SHARED_SCOPE_TYPES = frozenset({"agent", "team", "project", "org", "global"})
PERSONAL_SCOPE_TYPE = "user"
_PII_SECRET_METADATA_KEYS = frozenset(
    {
        "pii",
        "contains_pii",
        "has_pii",
        "secret",
        "contains_secret",
        "has_secret",
        "sensitive",
    }
)
_PII_SECRET_TAGS = frozenset({"pii", "secret", "sys:pii", "sys:secret", "sensitive", "sys:sensitive"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_curation_status(value: Any) -> str | None:
    if value is None:
        return None
    status = str(value).strip().lower()
    if not status:
        return None
    if status not in CURATION_STATUS_VALUES:
        valid = ", ".join(sorted(CURATION_STATUS_VALUES))
        raise ValidationError(f"curation status must be one of {{{valid}}}, got {value!r}")
    return status


def normalize_scope_hint(source: Mapping[str, Any]) -> tuple[str, float] | None:
    raw_type = source.get("suggested_scope_type")
    if raw_type is None:
        return None
    try:
        scope_type = MemoryScopeType(raw_type).value
    except ValueError as exc:
        valid = ", ".join(s.value for s in MemoryScopeType)
        raise ValidationError(f"suggested_scope_type must be one of {{{valid}}}, got {raw_type!r}") from exc
    raw_confidence = source.get("scope_confidence")
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise ValidationError("scope_confidence is required when suggested_scope_type is supplied") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValidationError(f"scope_confidence must be between 0.0 and 1.0, got {raw_confidence!r}")
    return scope_type, confidence


def is_promotable_scope_hint(scope_type: str | None) -> bool:
    """True when a hint points at a shared scope (i.e. not the personal ``user`` default)."""
    return scope_type in _SHARED_SCOPE_TYPES


def apply_scope_hint_metadata(metadata: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    hint = normalize_scope_hint(source)
    if hint is None:
        return metadata
    scope_type, confidence = hint
    metadata["suggested_scope_type"] = scope_type
    metadata["scope_confidence"] = confidence
    return metadata


def is_pii_or_secret_flagged(doc: Mapping[str, Any]) -> bool:
    """Return True when a record is *advisory*-flagged as PII/secret via metadata or tags.

    This inspects only the advisory ``metadata`` flags / ``classification`` and ``tags``
    that a producer (LLM prompt, adapter, or upstream classifier) set - it deliberately
    does NOT scan ``content``. Content-level secret/PII detection is intentionally
    deferred to the promotion-workflow PR, where it will gate the human-review candidate
    list (the point at which a record is actually proposed for a shared scope). Treat a
    ``False`` result as "not flagged", never as "proven safe to broadcast".
    """
    metadata = doc.get("metadata")
    if isinstance(metadata, Mapping):
        for key in _PII_SECRET_METADATA_KEYS:
            if bool(metadata.get(key)):
                return True
        classification = str(metadata.get("classification") or metadata.get("data_classification") or "").lower()
        if classification in {"pii", "secret", "credential", "credentials"}:
            return True
    tags = doc.get("tags")
    if isinstance(tags, list):
        normalized = {str(tag).strip().lower() for tag in tags}
        if normalized & _PII_SECRET_TAGS:
            return True
    return False
