"""Pydantic data models for the Agent Memory Toolkit.

Six concrete record types - :class:`TurnRecord`, :class:`ThreadSummaryRecord`,
:class:`UserSummaryRecord`, :class:`FactRecord`, :class:`EpisodicRecord`,
:class:`ProceduralRecord` - share :class:`MemoryRecordBase` and serialize
to/from Cosmos DB-compatible dicts.

``MemoryRecord`` is exported as the union type for return-signature use and
also acts as a back-compat constructor that dispatches on the ``memory_type``
field to the matching subclass.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from azure.cosmos.agent_memory.logging import get_logger

logger = get_logger(__name__)


class MemoryRole(str, Enum):
    """Allowed roles for a memory record."""

    user = "user"
    agent = "agent"
    tool = "tool"
    system = "system"


class MemoryType(str, Enum):
    """Allowed memory types stored in Cosmos DB."""

    turn = "turn"
    thread_summary = "thread_summary"
    fact = "fact"
    user_summary = "user_summary"
    procedural = "procedural"
    episodic = "episodic"
    shared_state = "shared_state"


class ProcedureKind(str, Enum):
    """The kind of atomic procedural knowledge a procedure captures."""

    behavioral_policy = "behavioral_policy"
    workflow = "workflow"
    decision_rule = "decision_rule"
    tool_usage = "tool_usage"
    recovery_strategy = "recovery_strategy"


class MemoryScopeType(str, Enum):
    """Typed owner scope for memory records."""

    user = "user"
    agent = "agent"
    team = "team"
    project = "project"
    org = "org"
    global_scope = "global"


class MemoryCurationStatus(str, Enum):
    """Curation lifecycle for shareable memory records."""

    draft = "draft"
    candidate = "candidate"
    approved = "approved"
    deprecated = "deprecated"


class ProcedureStatus(str, Enum):
    """Lifecycle state of a procedure. Only ``active`` procedures are compiled/injected."""

    candidate = "candidate"
    active = "active"
    deprecated = "deprecated"
    rejected = "rejected"


class ProcedureSourceKind(str, Enum):
    """Where a procedure came from - gates whether it may become active policy."""

    explicit_user_instruction = "explicit_user_instruction"
    observed_user_preference = "observed_user_preference"
    organization_policy = "organization_policy"
    episode_distillation = "episode_distillation"
    document_content = "document_content"
    agent_inference = "agent_inference"


class ProcedureSourceAuthority(str, Enum):
    """How authoritative a procedure's source is, for conflict resolution and gating."""

    mandatory = "mandatory"
    high = "high"
    medium = "medium"
    low = "low"


# Source kinds trusted enough to auto-activate a procedure as behavioral policy.
# Untrusted kinds (document_content, agent_inference) stay ``candidate`` until
# validated or explicitly approved - this stops an imperative sentence lifted
# from a document, or a one-off agent guess, from silently becoming agent policy.
TRUSTED_PROCEDURE_SOURCE_KINDS: frozenset[ProcedureSourceKind] = frozenset(
    {
        ProcedureSourceKind.explicit_user_instruction,
        ProcedureSourceKind.observed_user_preference,
        ProcedureSourceKind.organization_policy,
    }
)


class MemoryAcl(BaseModel):
    """Inline, group-aware read-visibility list for a memory record.

    Only ``read`` is enforced: the Cosmos read pre-filter (``build_acl_predicate``) and
    :func:`~azure.cosmos.agent_memory._authz.can` honor ``acl.read``. Write / forget /
    annotate authorization is decided at the placement-scope level via
    ``resolve_scope_access`` (scope membership + role), not per record, so there is no
    inline write ACL to keep in sync.
    """

    read: list[str] = Field(default_factory=list)

    @field_validator("read", mode="before")
    @classmethod
    def _validate_subjects(cls, v: Any, info: ValidationInfo) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(f"acl.{info.field_name} must be a list of subject strings")
        subjects: list[str] = []
        seen: set[str] = set()
        for subject in v:
            if not isinstance(subject, str):
                raise ValueError(f"acl.{info.field_name} entries must be strings")
            stripped = subject.strip()
            if not stripped:
                raise ValueError(f"acl.{info.field_name} entries cannot be empty")
            if stripped not in seen:
                seen.add(stripped)
                subjects.append(stripped)
        return subjects


class MemoryProvenance(BaseModel):
    """Grouped write/source provenance for a memory record."""

    application: Optional[str] = None
    agent_id: Optional[str] = None
    created_by: Optional[str] = None
    source: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None

    @field_validator("application", "agent_id", "created_by", "source", "prompt_id", "prompt_version", mode="before")
    @classmethod
    def _validate_optional_strings(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError(f"provenance.{info.field_name} must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError(f"provenance.{info.field_name} cannot be empty")
        if info.field_name == "prompt_version" and not PROMPT_VERSION_PATTERN.match(stripped):
            raise ValueError(f"provenance.prompt_version must match {PROMPT_VERSION_PATTERN.pattern!r}, got {v!r}")
        return stripped

    @field_validator("source_ids", mode="before")
    @classmethod
    def _validate_source_ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("provenance.source_ids must be a list of strings")
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            if not isinstance(item, str):
                raise ValueError("provenance.source_ids entries must be strings")
            stripped = item.strip()
            if not stripped:
                continue
            if stripped not in seen:
                seen.add(stripped)
                out.append(stripped)
        return out


def _uuid4_str() -> str:
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_:./-]{0,99}$")
MAX_TAGS_PER_RECORD = 50
CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PROMPT_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

_COSMOS_SYSTEM_FIELDS = ("_rid", "_self", "_attachments", "_ts")

_INTERNAL_VALIDATION_CONTEXT: dict[str, Any] = {"internal": True}


def _is_internal_context(info: Optional[ValidationInfo]) -> bool:
    """True if the validation was initiated through an internal pathway."""
    if info is None:
        return False
    ctx = getattr(info, "context", None)
    if not isinstance(ctx, dict):
        return False
    return bool(ctx.get("internal"))


class MemoryRecordBase(BaseModel):
    """Shared fields and validators for every memory record subtype.

    Concrete subclasses lock ``memory_type`` to a single :class:`MemoryType`
    value via a :class:`typing.Literal`, add per-type required fields, and
    layer on subtype-specific validators.  Use :meth:`from_doc` to parse a
    raw Cosmos document into the correct subclass via the ``type``
    discriminator.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        extra="ignore",
    )

    id: str = Field(default_factory=_uuid4_str)
    thread_id: str = Field(default_factory=_uuid4_str)
    tenant_id: str = "default"
    scope_type: Optional[MemoryScopeType] = None
    scope_id: Optional[str] = None
    scope_key: Optional[str] = None
    role: Optional[MemoryRole] = None
    memory_type: MemoryType = Field(alias="type", default=MemoryType.turn)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    acl: MemoryAcl = Field(default_factory=MemoryAcl)
    provenance: MemoryProvenance = Field(default_factory=MemoryProvenance)
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    ttl: Optional[int] = None
    salience: Optional[float] = None
    confidence: Optional[float] = None
    content_hash: Optional[str] = None
    status: Optional[MemoryCurationStatus] = None
    superseded_by: Optional[str] = None
    supersede_reason: Optional[Literal["duplicate", "contradict", "update"]] = None
    superseded_at: Optional[str] = None
    supersedes_ids: list[str] = Field(default_factory=list)
    last_used_at: Optional[str] = None
    use_count: int = 0
    version: Optional[int] = None
    source_fact_ids: list[str] = Field(default_factory=list)
    source_episodic_ids: list[str] = Field(default_factory=list)

    _etag: Optional[str] = PrivateAttr(default=None)

    _ID_PREFIX: ClassVar[Optional[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        """Normalize write inputs before validation.

        Maps the ``user_id`` write parameter onto a user scope and folds top-level
        provenance fields into the ``provenance`` block. (``owner`` / ``visibility``
        were dropped from the model; unknown keys are ignored by config.)
        """
        if not isinstance(data, dict):
            return data
        values = dict(data)
        user_id = values.pop("user_id", None)
        if user_id and not values.get("scope_key"):
            values.setdefault("scope_type", MemoryScopeType.user)
            values.setdefault("scope_id", str(user_id).strip())
            values.setdefault("scope_key", f"user:{str(user_id).strip()}")
        provenance = dict(values.get("provenance") or {})
        provenance_map = {
            "agent_id": "agent_id",
            "source": "source",
            "source_memory_ids": "source_ids",
            "prompt_id": "prompt_id",
            "prompt_version": "prompt_version",
        }
        for top_key, prov_key in provenance_map.items():
            value = values.pop(top_key, None)
            if value is not None and not provenance.get(prov_key):
                provenance[prov_key] = value
        if provenance:
            values["provenance"] = provenance
        return values

    @field_validator("tenant_id", "scope_id", "scope_key", mode="before")
    @classmethod
    def _validate_optional_non_empty_strings(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} cannot be empty")
        return stripped

    @field_validator("scope_type", mode="before")
    @classmethod
    def _validate_scope_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryScopeType(v)
            except ValueError:
                valid = ", ".join(s.value for s in MemoryScopeType)
                raise ValueError(f"scope_type must be one of {{{valid}}}, got '{v}'")
        return v

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryRole(v)
            except ValueError:
                valid = ", ".join(r.value for r in MemoryRole)
                raise ValueError(f"role must be one of {{{valid}}}, got '{v}'")
        return v

    @field_validator("memory_type", mode="before")
    @classmethod
    def _validate_memory_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryType(v)
            except ValueError:
                valid = ", ".join(t.value for t in MemoryType)
                raise ValueError(f"type must be one of {{{valid}}}, got '{v}'")
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, v: Any, info: ValidationInfo) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("tags must be a list of strings")
        internal = _is_internal_context(info)
        normalized: list[str] = []
        for tag in v:
            tag = str(tag).strip().lower()
            if not tag:
                continue
            if not TAG_PATTERN.match(tag):
                raise ValueError(f"Invalid tag format: '{tag}'. Must match [a-z0-9][a-z0-9_:./-]{{0,99}}")
            if not internal and tag.startswith("sys:"):
                raise ValueError(
                    f"Invalid tag '{tag}': the 'sys:' namespace is reserved for "
                    "internal pipeline writes and cannot be supplied by user code."
                )
            normalized.append(tag)
        deduped = sorted(set(normalized))
        if len(deduped) > MAX_TAGS_PER_RECORD:
            raise ValueError(f"too many tags: {len(deduped)} exceeds the {MAX_TAGS_PER_RECORD}-tag cap")
        return deduped

    @field_validator("salience", mode="before")
    @classmethod
    def _validate_salience(cls, v: Any) -> Any:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"salience must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, v: Any) -> Any:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, str) or not CONTENT_HASH_PATTERN.match(v):
            raise ValueError(f"content_hash must be 32 lowercase hex chars, got {v!r}")
        return v

    @field_validator("status", mode="before", check_fields=False)
    @classmethod
    def _validate_curation_status(cls, v: Any) -> Any:
        if v is None or isinstance(v, ProcedureStatus):
            return v
        if isinstance(v, str):
            try:
                return MemoryCurationStatus(v)
            except ValueError:
                try:
                    return ProcedureStatus(v)
                except ValueError:
                    valid = ", ".join(s.value for s in MemoryCurationStatus)
                    raise ValueError(f"status must be one of {{{valid}}}, got '{v}'")
        return v

    @field_validator("use_count", mode="before")
    @classmethod
    def _validate_use_count(cls, v: Any) -> Any:
        if v is None:
            return 0
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"use_count must be a non-negative integer, got {v!r}")
        return v

    @model_validator(mode="after")
    def _derive_scope_key(self) -> "MemoryRecordBase":
        if self.scope_key is None and self.scope_type is not None and self.scope_id is not None:
            scope_type_value = (
                self.scope_type.value if isinstance(self.scope_type, MemoryScopeType) else str(self.scope_type)
            )
            self.scope_key = f"{scope_type_value}:{self.scope_id}"
        return self

    @model_validator(mode="after")
    def _validate_id_prefix(self) -> "MemoryRecordBase":
        prefix = self.__class__._ID_PREFIX
        if prefix is not None and not self.id.startswith(prefix):
            raise ValueError(f"{self.__class__.__name__}.id must start with '{prefix}', got '{self.id}'")
        return self

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "MemoryRecord":
        """Parse a raw Cosmos document into the matching subclass.

        Discriminates on the document's ``type`` field. Cosmos system fields
        (``_rid``, ``_self``, ``_attachments``, ``_ts``) are dropped on
        parse; ``_etag`` is preserved on the resulting instance for
        ETag-protected writes via :attr:`etag`.

        Falls back to :class:`MemoryRecordBase` for unknown type values so
        that legacy documents do not break reads.
        """
        if not isinstance(doc, dict):
            raise TypeError(f"from_doc expects a dict, got {type(doc).__name__}")
        cleaned = {k: v for k, v in doc.items() if k not in _COSMOS_SYSTEM_FIELDS}
        etag = doc.get("_etag")
        type_value = cleaned.get("type") or cleaned.get("memory_type")
        target_cls = _TYPE_TO_CLASS.get(str(type_value), cls)
        instance = target_cls.model_validate(cleaned)
        if isinstance(etag, str):
            instance._etag = etag
        return instance

    def to_doc(self) -> dict[str, Any]:
        """Serialize to a dict suitable for ``container.upsert_item()``.

        The ``memory_type`` field is emitted as ``"type"`` (the wire name).
        ``None`` values and empty collections that match their defaults are
        omitted so documents stay compact and match the historical shape
        expected by the Cosmos queries elsewhere in this package.
        """
        raw = self.model_dump(mode="json", by_alias=True)
        return _strip_unset_optional(raw)

    @property
    def etag(self) -> Optional[str]:
        """Cosmos ``_etag`` for the most recent read, if any."""
        return self._etag

    def __getitem__(self, key: str) -> Any:
        """Dict-style access shim for callers transitioning from raw docs."""
        data = self.to_doc()
        if key in data:
            return data[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``.get()`` shim for callers transitioning from raw docs."""
        return self.to_doc().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.to_doc()


def _strip_unset_optional(data: dict[str, Any]) -> dict[str, Any]:
    """Drop optional fields that match their unset defaults.

    Keeps the wire shape lean and matches the historical Cosmos document
    layout: keys like ``embedding``, ``salience``, etc. are
    only emitted when populated. ``tags`` and ``metadata`` are always
    emitted because callers and Cosmos queries treat them as ever-present.
    """
    drop_when_none = {
        "scope_type",
        "scope_id",
        "scope_key",
        "embedding",
        "updated_at",
        "ttl",
        "salience",
        "confidence",
        "content_hash",
        "status",
        "superseded_by",
        "supersede_reason",
        "superseded_at",
        "last_used_at",
        "version",
    }
    drop_when_empty_list: set[str] = set()
    drop_when_zero = {"use_count"}

    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in drop_when_none and value is None:
            continue
        if key in drop_when_empty_list and (value is None or value == []):
            continue
        if key in drop_when_zero and (value is None or value == 0):
            continue
        if key == "provenance" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if v is not None and v != []}
        out[key] = value
    return out


class TurnRecord(MemoryRecordBase):
    """A single conversation turn - raw user/agent/tool/system message."""

    memory_type: Literal[MemoryType.turn] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.turn
    )
    role: MemoryRole  # type: ignore[assignment]

    @model_validator(mode="after")
    def _forbid_llm_fields(self) -> "TurnRecord":
        forbidden = {
            "salience": self.salience,
            "content_hash": self.content_hash,
            "provenance.prompt_id": self.provenance.prompt_id,
            "provenance.prompt_version": self.provenance.prompt_version,
        }
        for name, value in forbidden.items():
            if value is not None:
                raise ValueError(f"TurnRecord rejects '{name}': raw turns are not LLM-derived")
        return self


class ThreadSummaryRecord(MemoryRecordBase):
    """LLM-generated summary of a single thread's turns."""

    memory_type: Literal[MemoryType.thread_summary] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.thread_summary
    )
    content_hash: Optional[str] = None
    salience: float = 1.0

    _ID_PREFIX: ClassVar[Optional[str]] = "summary_"

    @model_validator(mode="after")
    def _require_prompt_provenance(self) -> "ThreadSummaryRecord":
        if not self.provenance.prompt_id:
            raise ValueError("ThreadSummaryRecord requires provenance.prompt_id")
        if not self.provenance.prompt_version:
            self.provenance.prompt_version = "v1"
        return self


class UserSummaryRecord(MemoryRecordBase):
    """LLM-generated cross-thread roll-up summary for a single user."""

    memory_type: Literal[MemoryType.user_summary] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.user_summary
    )
    content_hash: Optional[str] = None
    salience: float = 1.0

    _ID_PREFIX: ClassVar[Optional[str]] = "user_summary_"

    @model_validator(mode="after")
    def _require_prompt_provenance(self) -> "UserSummaryRecord":
        if not self.provenance.prompt_id:
            raise ValueError("UserSummaryRecord requires provenance.prompt_id")
        if not self.provenance.prompt_version:
            self.provenance.prompt_version = "v1"
        return self

    @model_validator(mode="after")
    def _require_thread_ids_in_metadata(self) -> "UserSummaryRecord":
        thread_ids = self.metadata.get("thread_ids") if isinstance(self.metadata, dict) else None
        if thread_ids is None:
            raise ValueError("UserSummaryRecord requires metadata.thread_ids")
        if not isinstance(thread_ids, list):
            raise ValueError("metadata.thread_ids must be a list of thread ids")
        return self


_FACT_ALLOWED_CATEGORIES = {
    "preference",
    "requirement",
    "biographical",
    "other",
}


class FactRecord(MemoryRecordBase):
    """A semantic fact about the user or environment, extracted by an LLM."""

    memory_type: Literal[MemoryType.fact] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.fact
    )
    content_hash: str
    salience: float = 0.5
    confidence: float = 0.5

    _ID_PREFIX: ClassVar[Optional[str]] = "fact_"

    @model_validator(mode="after")
    def _require_prompt_provenance(self) -> "FactRecord":
        if not self.provenance.prompt_id:
            raise ValueError("FactRecord requires provenance.prompt_id")
        if not self.provenance.prompt_version:
            self.provenance.prompt_version = "v1"
        return self

    @model_validator(mode="after")
    def _require_category(self) -> "FactRecord":
        meta = self.metadata if isinstance(self.metadata, dict) else None
        if not meta or not meta.get("category"):
            raise ValueError("FactRecord requires metadata.category")
        category = meta.get("category")
        if category not in _FACT_ALLOWED_CATEGORIES and not str(category).startswith("unclassified"):
            logger.debug(
                "FactRecord metadata.category=%r is outside the standard vocabulary",
                category,
            )
        return self


class EpisodeEvent(BaseModel):
    sequence: int
    description: str
    occurred_at: Optional[str] = None
    source_turn_ids: list[str] = Field(default_factory=list)


class EpisodeOutcome(BaseModel):
    status: Literal["successful", "partially_successful", "failed", "abandoned", "unknown"]
    description: str


class EpisodicRecord(MemoryRecordBase):
    """A specific past experience captured as an episode timeline."""

    memory_type: Literal[MemoryType.episodic] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.episodic
    )
    title: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    participants: list[str] = Field(default_factory=list)
    events: list[EpisodeEvent] = Field(default_factory=list)
    outcome: Optional[EpisodeOutcome] = None
    lessons: list[str] = Field(default_factory=list)
    source_turn_ids: list[str] = Field(default_factory=list)
    content_hash: str

    _ID_PREFIX: ClassVar[Optional[str]] = "ep_"

    @model_validator(mode="after")
    def _require_prompt_provenance(self) -> "EpisodicRecord":
        if not self.provenance.prompt_id:
            raise ValueError("EpisodicRecord requires provenance.prompt_id")
        if not self.provenance.prompt_version:
            self.provenance.prompt_version = "v1"
        return self

    @model_validator(mode="after")
    def _validate_time_order(self) -> "EpisodicRecord":
        if self.started_at is not None and self.ended_at is not None:
            started = datetime.fromisoformat(self.started_at.strip().replace("Z", "+00:00"))
            ended = datetime.fromisoformat(self.ended_at.strip().replace("Z", "+00:00"))
            # Normalize naive values to UTC so a mixed naive/tz-aware pair compares
            # safely instead of raising TypeError.
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if ended.tzinfo is None:
                ended = ended.replace(tzinfo=timezone.utc)
            if ended < started:
                raise ValueError("ended_at must not be before started_at")
        return self


class ProcedureStep(BaseModel):
    """One ordered step in a procedure's execution."""

    sequence: int
    instruction: str
    expected_result: Optional[str] = None
    on_failure: Optional[str] = None
    tool_name: Optional[str] = None


class ProceduralRecord(MemoryRecordBase):
    """An atomic, reusable procedural memory: a behavioral policy or task skill.

    Procedural memory answers "what should I do, and how should I do it" - stored
    as many small, independently retrievable units (a policy/skill library), not
    as one compiled system prompt. The runtime personalized system prompt is a
    deterministic projection of the active, in-scope procedures (built by
    ``build_procedural_context``); it is not stored as a record here.
    """

    memory_type: Literal[MemoryType.procedural] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.procedural
    )

    name: str
    summary: str
    retrieval_text: str

    procedure_kind: ProcedureKind

    activation_conditions: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    steps: list[ProcedureStep] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)

    status: ProcedureStatus = ProcedureStatus.candidate
    priority: int = 0
    utility_score: float = 0.5
    successful_uses: int = 0
    failed_uses: int = 0

    source_kind: ProcedureSourceKind = ProcedureSourceKind.agent_inference
    source_authority: ProcedureSourceAuthority = ProcedureSourceAuthority.low
    source_turn_ids: list[str] = Field(default_factory=list)

    content_hash: Optional[str] = None
    version: int = 1

    _ID_PREFIX: ClassVar[Optional[str]] = "proc_"

    @field_validator("version", mode="before")
    @classmethod
    def _validate_version(cls, v: Any) -> Any:
        if v is None:
            return 1
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"ProceduralRecord.version must be a positive integer, got {v!r}")
        return v

    @field_validator("utility_score", mode="before")
    @classmethod
    def _clamp_utility(cls, v: Any) -> Any:
        if v is None:
            return 0.5
        try:
            value = float(v)
        except (TypeError, ValueError):
            return 0.5
        if not math.isfinite(value):
            return 0.5
        return min(1.0, max(0.0, value))

    @model_validator(mode="after")
    def _require_executable_steps(self) -> "ProceduralRecord":
        # A workflow/recovery_strategy asserts a multi-step method; requiring at
        # least one step keeps such procedures executable rather than empty.
        # ``procedure_kind`` is a plain string here (model_config use_enum_values).
        if self.procedure_kind in (ProcedureKind.workflow, ProcedureKind.recovery_strategy) and not self.steps:
            raise ValueError(f"ProceduralRecord of kind {self.procedure_kind} requires at least one step")
        return self


class SharedStateRecord(MemoryRecordBase):
    """Small live coordination record updated in place by agents.

    Use for section-owned plans, task queues, and handoff context; keep the
    payload small to avoid hot-record contention and prefer separate records
    when different agents own different sections.
    """

    memory_type: Literal[MemoryType.shared_state] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.shared_state
    )
    role: Optional[MemoryRole] = MemoryRole.system
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = False


_TYPE_TO_CLASS: dict[str, type[MemoryRecordBase]] = {
    MemoryType.turn.value: TurnRecord,
    MemoryType.thread_summary.value: ThreadSummaryRecord,
    MemoryType.user_summary.value: UserSummaryRecord,
    MemoryType.fact.value: FactRecord,
    MemoryType.episodic.value: EpisodicRecord,
    MemoryType.procedural.value: ProceduralRecord,
    MemoryType.shared_state.value: SharedStateRecord,
}


_TypedMemoryRecord = Annotated[
    Union[
        TurnRecord,
        ThreadSummaryRecord,
        UserSummaryRecord,
        FactRecord,
        EpisodicRecord,
        ProceduralRecord,
    ],
    Field(discriminator="memory_type"),
]


class _MemoryRecordFactory:
    """Backward-compatible constructor that dispatches on ``memory_type``.

    ``MemoryRecord(...)`` builds the concrete subclass matching the
    requested ``memory_type`` so existing callers keep working without
    knowing about the new per-type classes.  ``MemoryRecord.from_doc(...)``
    is the standard discriminator entry point for raw dicts.
    """

    def __call__(self, **kwargs: Any) -> "MemoryRecord":
        type_value = kwargs.get("memory_type") or kwargs.get("type") or MemoryType.turn
        if isinstance(type_value, MemoryType):
            type_str = type_value.value
        else:
            type_str = str(type_value)
        target = _TYPE_TO_CLASS.get(type_str, MemoryRecordBase)
        return target(**kwargs)

    @staticmethod
    def from_doc(doc: dict[str, Any]) -> "MemoryRecord":
        return MemoryRecordBase.from_doc(doc)


MemoryRecord = _MemoryRecordFactory()

TYPED_RECORD_CLASSES: tuple[type[MemoryRecordBase], ...] = (
    TurnRecord,
    ThreadSummaryRecord,
    UserSummaryRecord,
    FactRecord,
    EpisodicRecord,
    ProceduralRecord,
)


def construct_internal(cls: type[MemoryRecordBase], data: dict[str, Any]) -> MemoryRecordBase:
    """Validate ``data`` into ``cls`` with the internal validation context.

    Internal-context construction allows ``sys:*`` tags written by the
    pipeline itself; user-facing construction continues to reject them.
    """
    return cls.model_validate(data, context=_INTERNAL_VALIDATION_CONTEXT)


class SearchResult(BaseModel):
    """A memory record returned from a similarity or keyword search."""

    record: MemoryRecordBase
    score: Optional[float] = None


class OrchestrationResult(BaseModel):
    """Response envelope for Durable Functions orchestration calls."""

    runtime_status: str
    output: Optional[Any] = None
    custom_status: Optional[Any] = None
    instance_id: Optional[str] = None


__all__ = [
    "MemoryRole",
    "MemoryType",
    "MemoryRecord",
    "MemoryRecordBase",
    "MemoryAcl",
    "MemoryProvenance",
    "TurnRecord",
    "ThreadSummaryRecord",
    "UserSummaryRecord",
    "FactRecord",
    "EpisodeEvent",
    "EpisodeOutcome",
    "EpisodicRecord",
    "ProcedureStep",
    "ProcedureKind",
    "ProcedureStatus",
    "ProcedureSourceKind",
    "ProcedureSourceAuthority",
    "TRUSTED_PROCEDURE_SOURCE_KINDS",
    "ProceduralRecord",
    "TYPED_RECORD_CLASSES",
    "TAG_PATTERN",
    "MAX_TAGS_PER_RECORD",
    "CONTENT_HASH_PATTERN",
    "SearchResult",
    "OrchestrationResult",
    "construct_internal",
]
