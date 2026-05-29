"""Pydantic data models for the Agent Memory Toolkit.

Six concrete record types — :class:`TurnRecord`, :class:`ThreadSummaryRecord`,
:class:`UserSummaryRecord`, :class:`FactRecord`, :class:`EpisodicRecord`,
:class:`ProceduralRecord` — share :class:`MemoryRecordBase` and serialize
to/from Cosmos DB-compatible dicts.

``MemoryRecord`` is exported as the union type for return-signature use and
also acts as a back-compat constructor that dispatches on the ``memory_type``
field to the matching subclass.
"""

from __future__ import annotations

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

from agent_memory_toolkit.logging import get_logger

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
    summary = "summary"
    fact = "fact"
    user_summary = "user_summary"
    procedural = "procedural"
    episodic = "episodic"


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
    user_id: str
    thread_id: str = Field(default_factory=_uuid4_str)
    role: Optional[MemoryRole] = None
    memory_type: MemoryType = Field(alias="type", default=MemoryType.turn)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    agent_id: Optional[str] = None
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    ttl: Optional[int] = None
    salience: Optional[float] = None
    confidence: Optional[float] = None
    content_hash: Optional[str] = None
    superseded_by: Optional[str] = None
    supersede_reason: Optional[Literal["duplicate", "contradict", "update"]] = None
    superseded_at: Optional[str] = None
    supersedes_ids: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    last_used_at: Optional[str] = None
    use_count: int = 0
    version: Optional[int] = None
    source_fact_ids: list[str] = Field(default_factory=list)
    source_episodic_ids: list[str] = Field(default_factory=list)

    _etag: Optional[str] = PrivateAttr(default=None)

    _ID_PREFIX: ClassVar[Optional[str]] = None

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

    @field_validator("prompt_version", mode="before")
    @classmethod
    def _validate_prompt_version(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, str) or not PROMPT_VERSION_PATTERN.match(v):
            raise ValueError(f"prompt_version must match {PROMPT_VERSION_PATTERN.pattern!r}, got {v!r}")
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

    @classmethod
    def from_cosmos_dict(cls, doc: dict[str, Any]) -> "MemoryRecord":
        """Back-compat alias for :meth:`from_doc`."""
        return cls.from_doc(doc)

    def to_doc(self) -> dict[str, Any]:
        """Serialize to a dict suitable for ``container.upsert_item()``.

        The ``memory_type`` field is emitted as ``"type"`` (the wire name).
        ``None`` values and empty collections that match their defaults are
        omitted so documents stay compact and match the historical shape
        expected by the Cosmos queries elsewhere in this package.
        """
        raw = self.model_dump(mode="json", by_alias=True)
        return _strip_unset_optional(raw)

    def to_cosmos_dict(self) -> dict[str, Any]:
        """Back-compat alias for :meth:`to_doc`."""
        return self.to_doc()

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
    layout: keys like ``embedding``, ``agent_id``, ``salience``, etc. are
    only emitted when populated. ``tags`` and ``metadata`` are always
    emitted because callers and Cosmos queries treat them as ever-present.
    """
    drop_when_none = {
        "embedding",
        "agent_id",
        "updated_at",
        "ttl",
        "salience",
        "confidence",
        "content_hash",
        "superseded_by",
        "supersede_reason",
        "superseded_at",
        "prompt_id",
        "prompt_version",
        "last_used_at",
        "version",
        "scope_type",
        "scope_value",
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
        out[key] = value
    return out


class TurnRecord(MemoryRecordBase):
    """A single conversation turn — raw user/agent/tool/system message."""

    memory_type: Literal[MemoryType.turn] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.turn
    )
    role: MemoryRole  # type: ignore[assignment]

    @model_validator(mode="after")
    def _forbid_llm_fields(self) -> "TurnRecord":
        forbidden = {
            "salience": self.salience,
            "content_hash": self.content_hash,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
        }
        for name, value in forbidden.items():
            if value is not None:
                raise ValueError(f"TurnRecord rejects '{name}': raw turns are not LLM-derived")
        return self


class ThreadSummaryRecord(MemoryRecordBase):
    """LLM-generated summary of a single thread's turns."""

    memory_type: Literal[MemoryType.summary] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.summary
    )
    content_hash: Optional[str] = None
    salience: float = 1.0
    prompt_id: str
    prompt_version: str = "v1"

    _ID_PREFIX: ClassVar[Optional[str]] = "summary_"


class UserSummaryRecord(MemoryRecordBase):
    """LLM-generated cross-thread roll-up summary for a single user."""

    memory_type: Literal[MemoryType.user_summary] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.user_summary
    )
    content_hash: Optional[str] = None
    salience: float = 1.0
    prompt_id: str
    prompt_version: str = "v1"

    _ID_PREFIX: ClassVar[Optional[str]] = "user_summary_"

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
    "decision",
    "biographical",
    "temporal",
    "relational",
    "action_item",
}


class FactRecord(MemoryRecordBase):
    """A semantic fact about the user or environment, extracted by an LLM."""

    memory_type: Literal[MemoryType.fact] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.fact
    )
    content_hash: str
    salience: float = 0.5
    confidence: float = 0.5
    prompt_id: str
    prompt_version: str = "v1"

    _ID_PREFIX: ClassVar[Optional[str]] = "fact_"

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


_EPISODIC_ALLOWED_VALENCES = {"positive", "negative", "neutral", "mixed"}


class EpisodicRecord(MemoryRecordBase):
    """A specific past experience: situation → action → outcome → lesson."""

    memory_type: Literal[MemoryType.episodic] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.episodic
    )
    content_hash: str
    confidence: float = 0.5
    scope_type: Optional[str] = None
    scope_value: Optional[str] = None
    prompt_id: str
    prompt_version: str = "v1"

    _ID_PREFIX: ClassVar[Optional[str]] = "ep_"

    @model_validator(mode="after")
    def _require_episodic_metadata(self) -> "EpisodicRecord":
        meta = self.metadata if isinstance(self.metadata, dict) else None
        if not meta:
            raise ValueError(
                "EpisodicRecord requires metadata.lesson, metadata.scope_type, "
                "metadata.scope_value, and metadata.outcome_valence"
            )
        missing = [k for k in ("lesson", "scope_type", "scope_value", "outcome_valence") if not meta.get(k)]
        if missing:
            raise ValueError(f"EpisodicRecord missing required metadata field(s): {missing}")
        valence = meta.get("outcome_valence")
        if valence not in _EPISODIC_ALLOWED_VALENCES:
            raise ValueError(
                f"metadata.outcome_valence must be one of {sorted(_EPISODIC_ALLOWED_VALENCES)}, got {valence!r}"
            )
        # Mirror metadata.scope_* to top-level fields so queries that filter
        # on the indexed top-level columns keep working.
        if not self.scope_type:
            object.__setattr__(self, "scope_type", meta.get("scope_type"))
        if not self.scope_value:
            object.__setattr__(self, "scope_value", meta.get("scope_value"))
        return self


class ProceduralRecord(MemoryRecordBase):
    """Synthesized agent self-knowledge: the active personalized system prompt."""

    memory_type: Literal[MemoryType.procedural] = Field(  # type: ignore[assignment]
        alias="type", default=MemoryType.procedural
    )
    content_hash: Optional[str] = None
    prompt_id: str
    prompt_version: str = "v1"
    version: int = 1
    source_fact_ids: list[str] = Field(default_factory=list)
    source_episodic_ids: list[str] = Field(default_factory=list)

    _ID_PREFIX: ClassVar[Optional[str]] = "proc_"

    @field_validator("version", mode="before")
    @classmethod
    def _validate_version(cls, v: Any) -> Any:
        if v is None:
            return 1
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"ProceduralRecord.version must be a positive integer, got {v!r}")
        return v

    @model_validator(mode="after")
    def _require_sources(self) -> "ProceduralRecord":
        if not self.source_fact_ids and not self.source_episodic_ids:
            raise ValueError(
                "ProceduralRecord requires at least one of source_fact_ids or source_episodic_ids to be non-empty"
            )
        return self


_TYPE_TO_CLASS: dict[str, type[MemoryRecordBase]] = {
    MemoryType.turn.value: TurnRecord,
    MemoryType.summary.value: ThreadSummaryRecord,
    MemoryType.user_summary.value: UserSummaryRecord,
    MemoryType.fact.value: FactRecord,
    MemoryType.episodic.value: EpisodicRecord,
    MemoryType.procedural.value: ProceduralRecord,
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

    @staticmethod
    def from_cosmos_dict(doc: dict[str, Any]) -> "MemoryRecord":
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
    "TurnRecord",
    "ThreadSummaryRecord",
    "UserSummaryRecord",
    "FactRecord",
    "EpisodicRecord",
    "ProceduralRecord",
    "TYPED_RECORD_CLASSES",
    "TAG_PATTERN",
    "MAX_TAGS_PER_RECORD",
    "CONTENT_HASH_PATTERN",
    "SearchResult",
    "OrchestrationResult",
    "construct_internal",
]
