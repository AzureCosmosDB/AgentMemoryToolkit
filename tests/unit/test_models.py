"""Unit tests for azure.cosmos.agent_memory.models.

Each typed subclass (TurnRecord / ThreadSummaryRecord / UserSummaryRecord /
FactRecord / EpisodicRecord / ProceduralRecord) gets its own focused class
of tests covering required fields, id-prefix enforcement, and the per-type
metadata invariants. The legacy ``MemoryRecord`` callable factory is
exercised at the boundaries (discriminated construction, from-doc parsing,
back-compat dict access).
"""

from __future__ import annotations

import uuid
from typing import Any

import pydantic
import pytest

from azure.cosmos.agent_memory.models import (
    EpisodeEvent,
    EpisodeOutcome,
    EpisodicRecord,
    FactRecord,
    MemoryRecord,
    MemoryRecordBase,
    MemoryType,
    OrchestrationResult,
    ProceduralRecord,
    ProcedureApplies,
    ProcedureEvidence,
    ProcedureKind,
    ProcedureSourceKind,
    ProcedureStatus,
    ProcedureStep,
    SearchResult,
    ThreadSummaryRecord,
    TurnRecord,
    UserSummaryRecord,
    construct_internal,
)

# ---------------------------------------------------------------------------
# Helpers - minimal valid kwargs per typed subclass
# ---------------------------------------------------------------------------


_HEX32 = "a" * 32


def _fact_kwargs(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "fact_" + _HEX32,
        "user_id": "u1",
        "content": "User prefers tea.",
        "content_hash": _HEX32,
        "metadata": {"category": "preference"},
        "prompt_id": "extract_memories.prompty",
    }
    base.update(overrides)
    return base


def _episodic_kwargs(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "ep_" + _HEX32,
        "user_id": "u1",
        "content": "Paris trip planning worked because the team planned early.",
        "title": "Paris trip planning",
        "started_at": "2026-01-01T09:00:00+00:00",
        "ended_at": "2026-01-02T10:00:00+00:00",
        "participants": ["user", "agent"],
        "events": [
            {
                "sequence": 1,
                "description": "User asked for Paris trip planning help.",
                "occurred_at": "2026-01-01T09:00:00+00:00",
                "source_turn_ids": ["turn-1"],
            },
            EpisodeEvent(sequence=2, description="Agent recommended booking early.", source_turn_ids=["turn-2"]),
        ],
        "outcome": {"status": "successful", "description": "The trip plan was completed."},
        "lessons": ["Plan early."],
        "source_turn_ids": ["turn-1", "turn-2"],
        "content_hash": _HEX32,
        "prompt_id": "extract_memories.prompty",
    }
    base.update(overrides)
    return base


def _thread_summary_kwargs(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "summary_" + _HEX32,
        "user_id": "u1",
        "thread_id": "t1",
        "content": "Thread summary.",
        "prompt_id": "summarize.prompty",
    }
    base.update(overrides)
    return base


def _user_summary_kwargs(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "user_summary_" + _HEX32,
        "user_id": "u1",
        "thread_id": "__user_summary__",
        "content": "User-level rollup.",
        "metadata": {"thread_ids": ["t1", "t2"]},
        "prompt_id": "user_summary.prompty",
    }
    base.update(overrides)
    return base


def _procedural_kwargs(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "proc_u1_1",
        "user_id": "u1",
        "content": "Be concise.",
        "name": "Be concise",
        "summary": "Prefer concise responses.",
        "retrieval_text": "Respond concisely when answering the user.",
        "procedure_kind": "instruction",
        "version": 1,
        "prompt_id": "extract_procedure.prompty",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Discriminated factory dispatch
# ---------------------------------------------------------------------------


def test_factory_default_dispatches_to_turn_record():
    rec = MemoryRecord(user_id="u1", role="user", content="hello")
    assert isinstance(rec, TurnRecord)
    assert rec.memory_type == "turn"


def test_factory_dispatches_to_fact_record():
    rec = MemoryRecord(memory_type="fact", **_fact_kwargs())
    assert isinstance(rec, FactRecord)
    assert rec.memory_type == "fact"


def test_factory_accepts_type_alias_alongside_memory_type():
    rec = MemoryRecord(type="episodic", **{k: v for k, v in _episodic_kwargs().items() if k != "memory_type"})
    assert isinstance(rec, EpisodicRecord)


def test_factory_dispatches_to_all_subclasses():
    cases = [
        ("thread_summary", _thread_summary_kwargs(), ThreadSummaryRecord),
        ("user_summary", _user_summary_kwargs(), UserSummaryRecord),
        ("fact", _fact_kwargs(), FactRecord),
        ("episodic", _episodic_kwargs(), EpisodicRecord),
        ("procedural", _procedural_kwargs(), ProceduralRecord),
    ]
    for mt, kwargs, expected_cls in cases:
        rec = MemoryRecord(memory_type=mt, **kwargs)
        assert isinstance(rec, expected_cls), f"{mt} should dispatch to {expected_cls.__name__}"


def test_factory_rejects_unknown_memory_type():
    # Unknown type falls back to MemoryRecordBase which itself rejects unknown
    # enum values via pydantic's enum coercion.
    with pytest.raises(pydantic.ValidationError):
        MemoryRecord(user_id="u1", role="user", content="c", memory_type="bad")


# ---------------------------------------------------------------------------
# TurnRecord
# ---------------------------------------------------------------------------


class TestTurnRecord:
    def test_defaults(self):
        rec = TurnRecord(user_id="u1", role="user", content="hello")
        uuid.UUID(rec.id)
        uuid.UUID(rec.thread_id)
        assert rec.memory_type == "turn"
        assert rec.metadata == {}
        assert rec.embedding is None
        assert rec.agent_id is None
        assert rec.updated_at is None
        assert rec.salience is None
        assert rec.confidence is None
        assert rec.content_hash is None

    @pytest.mark.parametrize("role", ["user", "agent", "tool", "system"])
    def test_valid_roles(self, role):
        rec = TurnRecord(user_id="u", role=role, content="c")
        assert rec.role == role

    def test_invalid_role(self):
        with pytest.raises(pydantic.ValidationError, match="role"):
            TurnRecord(user_id="u", role="invalid_role", content="c")

    def test_to_doc_uses_type_key(self):
        rec = TurnRecord(user_id="u", role="user", content="c")
        d = rec.to_doc()
        assert d["type"] == "turn"
        assert "memory_type" not in d

    def test_to_doc_omits_none_optional(self):
        rec = TurnRecord(user_id="u", role="user", content="c")
        d = rec.to_doc()
        assert "embedding" not in d
        assert "agent_id" not in d
        assert "updated_at" not in d
        assert "salience" not in d
        assert "content_hash" not in d

    def test_rejects_salience(self):
        with pytest.raises(pydantic.ValidationError, match="salience"):
            TurnRecord(user_id="u", role="user", content="c", salience=0.5)

    def test_rejects_content_hash(self):
        with pytest.raises(pydantic.ValidationError, match="content_hash"):
            TurnRecord(user_id="u", role="user", content="c", content_hash=_HEX32)

    def test_rejects_prompt_id(self):
        with pytest.raises(pydantic.ValidationError, match="prompt_id"):
            TurnRecord(user_id="u", role="user", content="c", prompt_id="x.prompty")

    def test_requires_role(self):
        with pytest.raises(pydantic.ValidationError, match="role"):
            TurnRecord(user_id="u", content="c")  # type: ignore[call-arg]


class TestRoleOptionalOnDerivedTypes:
    """role is required only on TurnRecord; LLM-derived types accept it as optional."""

    def test_fact_constructs_without_role(self):
        rec = FactRecord(**_fact_kwargs())
        assert rec.role is None

    def test_episodic_constructs_without_role(self):
        rec = EpisodicRecord(**_episodic_kwargs())
        assert rec.role is None

    def test_procedural_constructs_without_role(self):
        rec = ProceduralRecord(**_procedural_kwargs())
        assert rec.role is None

    def test_thread_summary_constructs_without_role(self):
        rec = ThreadSummaryRecord(**_thread_summary_kwargs())
        assert rec.role is None

    def test_user_summary_constructs_without_role(self):
        rec = UserSummaryRecord(**_user_summary_kwargs())
        assert rec.role is None

    def test_fact_accepts_role_when_supplied(self):
        rec = FactRecord(**_fact_kwargs(role="system"))
        assert rec.role == "system"


# ---------------------------------------------------------------------------
# FactRecord
# ---------------------------------------------------------------------------


class TestFactRecord:
    def test_minimal_valid(self):
        rec = FactRecord(**_fact_kwargs())
        assert rec.memory_type == "fact"
        assert rec.confidence == 0.5
        assert rec.salience == 0.5

    def test_requires_content_hash(self):
        kwargs = _fact_kwargs()
        kwargs.pop("content_hash")
        with pytest.raises(pydantic.ValidationError, match="content_hash"):
            FactRecord(**kwargs)

    def test_requires_metadata_category(self):
        kwargs = _fact_kwargs(metadata={})
        with pytest.raises(pydantic.ValidationError, match="metadata.category"):
            FactRecord(**kwargs)

    def test_id_must_start_with_fact_prefix(self):
        with pytest.raises(pydantic.ValidationError, match="id must start with 'fact_'"):
            FactRecord(**_fact_kwargs(id="bad-id"))

    def test_requires_prompt_id(self):
        kwargs = _fact_kwargs()
        kwargs.pop("prompt_id")
        with pytest.raises(pydantic.ValidationError, match="prompt_id"):
            FactRecord(**kwargs)


class TestFactStructuredFields:
    """The fields ``extract_memories-v3`` adds, normalized whatever the model wrote."""

    def test_a_fact_without_them_keeps_the_v2_shape(self):
        doc = FactRecord(**_fact_kwargs(assertion="bogus", entities=[])).to_doc()

        assert not {"assertion", "subject", "attribute", "entities"} & set(doc)
        assert doc["source_turn_ids"] == []

    def test_they_are_normalized(self):
        doc = FactRecord(
            **_fact_kwargs(
                assertion="decided",
                subject="  The User ",
                attribute="Home   City",
                entities=["Priya", "priya", "the assistant", " Portland ", 7, ""],
                source_turn_ids=["turn-1"],
            )
        ).to_doc()

        assert (doc["assertion"], doc["subject"], doc["attribute"]) == ("decided", "user", "home city")
        assert doc["entities"] == ["Priya", "Portland"]
        assert doc["source_turn_ids"] == ["turn-1"]

    def test_subjects_and_entities_are_bounded_and_aliased(self):
        rec = FactRecord(
            **_fact_kwargs(
                subject="The Agent",
                attribute="x" * 500,
                entities=[f"entity {i}" for i in range(30)],
            )
        )

        assert rec.subject == "assistant"
        assert len(rec.attribute) == 120
        assert rec.entities == [f"entity {i}" for i in range(12)]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("stated", "stated"),
            ("decided", "decided"),
            ("suggested", "suggested"),
            ("observed", "observed"),
            ("directive", "directive"),
            ("correction", "correction"),
            ("Stated", None),
            ("claimed", None),
            (3, None),
            (None, None),
        ],
    )
    def test_assertion_keeps_only_a_known_value(self, value, expected):
        rec = FactRecord(**_fact_kwargs(assertion=value))

        assert rec.assertion == expected
        assert rec.to_doc().get("assertion") == expected

    @pytest.mark.parametrize("assertion", ["directive", "correction"])
    def test_only_the_user_directs_or_corrects_the_assistant(self, assertion):
        agent = FactRecord(**_fact_kwargs(assertion=assertion, metadata={"category": "other", "source": "agent"}))
        tagged = construct_internal(FactRecord, _fact_kwargs(assertion=assertion, tags=["sys:agent-fact"]))
        user = FactRecord(**_fact_kwargs(assertion=assertion, metadata={"category": "other", "source": "user"}))

        assert (agent.assertion, tagged.assertion, user.assertion) == ("stated", "stated", assertion)

    def test_an_agent_fact_keeps_the_other_assertions(self):
        rec = FactRecord(**_fact_kwargs(assertion="suggested", metadata={"category": "other", "source": "agent"}))

        assert rec.assertion == "suggested"

    def test_they_survive_a_round_trip(self):
        original = FactRecord(**_fact_kwargs(assertion="stated", subject="Priya", attribute="home city"))

        restored = MemoryRecord.from_doc(original.to_doc())

        assert isinstance(restored, FactRecord)
        assert (restored.assertion, restored.subject, restored.attribute) == ("stated", "Priya", "home city")


# ---------------------------------------------------------------------------
# EpisodicRecord
# ---------------------------------------------------------------------------


class TestEpisodicRecord:
    def test_minimal_valid(self):
        rec = EpisodicRecord(**_episodic_kwargs())
        assert rec.memory_type == "episodic"
        assert rec.title == "Paris trip planning"
        assert rec.started_at == "2026-01-01T09:00:00+00:00"
        assert rec.ended_at == "2026-01-02T10:00:00+00:00"
        assert rec.participants == ["user", "agent"]
        assert rec.events[0] == EpisodeEvent(
            sequence=1,
            description="User asked for Paris trip planning help.",
            occurred_at="2026-01-01T09:00:00+00:00",
            source_turn_ids=["turn-1"],
        )
        assert rec.events[1].description == "Agent recommended booking early."
        assert rec.outcome == EpisodeOutcome(status="successful", description="The trip plan was completed.")
        assert rec.lessons == ["Plan early."]
        assert rec.source_turn_ids == ["turn-1", "turn-2"]

    def test_outcome_is_optional(self):
        rec = EpisodicRecord(**_episodic_kwargs(outcome=None))
        restored = MemoryRecordBase.from_doc(rec.to_doc())
        assert rec.outcome is None
        assert isinstance(restored, EpisodicRecord)
        assert restored.outcome is None

    def test_outcome_status_literal(self):
        with pytest.raises(pydantic.ValidationError, match="outcome"):
            EpisodicRecord(**_episodic_kwargs(outcome={"status": "bogus", "description": "Nope."}))

    def test_ended_at_must_not_be_before_started_at(self):
        with pytest.raises(pydantic.ValidationError, match="ended_at must not be before started_at"):
            EpisodicRecord(
                **_episodic_kwargs(
                    started_at="2026-01-02T10:00:00+00:00",
                    ended_at="2026-01-01T09:00:00+00:00",
                )
            )

    def test_mixed_naive_and_aware_times_compare_without_typeerror(self):
        # A naive date + a tz-aware datetime must not raise TypeError; the naive
        # value is normalized to UTC and the (valid) pair is accepted.
        rec = EpisodicRecord(
            **_episodic_kwargs(
                started_at="2026-03-09",
                ended_at="2026-03-10T09:08:00+00:00",
            )
        )
        assert rec.started_at == "2026-03-09"
        assert rec.ended_at == "2026-03-10T09:08:00+00:00"

    def test_mixed_tz_reversed_order_still_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="ended_at must not be before started_at"):
            EpisodicRecord(
                **_episodic_kwargs(
                    started_at="2026-03-11T00:00:00+00:00",
                    ended_at="2026-03-10",
                )
            )

    def test_id_must_start_with_ep_prefix(self):
        with pytest.raises(pydantic.ValidationError, match="id must start with 'ep_'"):
            EpisodicRecord(**_episodic_kwargs(id="bad-id"))


# ---------------------------------------------------------------------------
# ThreadSummaryRecord
# ---------------------------------------------------------------------------


class TestThreadSummaryRecord:
    def test_minimal_valid(self):
        rec = ThreadSummaryRecord(**_thread_summary_kwargs())
        assert rec.memory_type == "thread_summary"
        assert rec.salience == 1.0

    def test_id_must_start_with_sum_prefix(self):
        with pytest.raises(pydantic.ValidationError, match="id must start with 'summary_'"):
            ThreadSummaryRecord(**_thread_summary_kwargs(id="bad"))

    def test_requires_prompt_id(self):
        kwargs = _thread_summary_kwargs()
        kwargs.pop("prompt_id")
        with pytest.raises(pydantic.ValidationError, match="prompt_id"):
            ThreadSummaryRecord(**kwargs)


# ---------------------------------------------------------------------------
# UserSummaryRecord
# ---------------------------------------------------------------------------


class TestUserSummaryRecord:
    def test_minimal_valid(self):
        rec = UserSummaryRecord(**_user_summary_kwargs())
        assert rec.memory_type == "user_summary"
        assert rec.metadata["thread_ids"] == ["t1", "t2"]

    def test_requires_thread_ids_in_metadata(self):
        with pytest.raises(pydantic.ValidationError, match="thread_ids"):
            UserSummaryRecord(**_user_summary_kwargs(metadata={}))

    def test_id_must_start_with_usum_prefix(self):
        with pytest.raises(pydantic.ValidationError, match="id must start with 'user_summary_'"):
            UserSummaryRecord(**_user_summary_kwargs(id="bad"))


# ---------------------------------------------------------------------------
# ProceduralRecord
# ---------------------------------------------------------------------------


class TestProceduralRecord:
    def test_minimal_valid(self):
        rec = ProceduralRecord(**_procedural_kwargs())
        assert rec.memory_type == "procedural"
        assert rec.procedure_kind == "instruction"
        assert rec.status == "candidate"
        assert rec.applies == "when"
        assert rec.evidence == []
        assert rec.user_support_count == 0
        assert rec.utility_score == 0.5
        assert rec.name == "Be concise"
        assert rec.summary == "Prefer concise responses."
        assert rec.retrieval_text == "Respond concisely when answering the user."
        assert rec.version == 1

    def test_accepts_no_source_ids(self):
        rec = ProceduralRecord(**_procedural_kwargs(source_fact_ids=[], source_episodic_ids=[]))
        assert rec.source_fact_ids == []
        assert rec.source_episodic_ids == []

    def test_skill_needs_two_steps(self):
        one = [ProcedureStep(sequence=1, instruction="Check current context.")]
        with pytest.raises(pydantic.ValidationError, match="at least two steps"):
            ProceduralRecord(**_procedural_kwargs(procedure_kind=ProcedureKind.skill))
        with pytest.raises(pydantic.ValidationError, match="at least two steps"):
            ProceduralRecord(**_procedural_kwargs(procedure_kind=ProcedureKind.skill, steps=one))

    def test_skill_with_two_steps_is_valid(self):
        rec = ProceduralRecord(
            **_procedural_kwargs(
                procedure_kind=ProcedureKind.skill,
                steps=[
                    ProcedureStep(sequence=1, instruction="Check current context.", expected_result="Understood."),
                    ProcedureStep(sequence=2, instruction="Apply the fix."),
                ],
            )
        )
        assert rec.procedure_kind == "skill"
        assert rec.steps[0].instruction == "Check current context."

    @pytest.mark.parametrize(
        ("value", "expected"), [("2030-05-01", "2030-05-01"), ("2030-05-01T09:00:00Z", "2030-05-01")]
    )
    def test_valid_until_keeps_an_iso_date(self, value, expected):
        assert ProceduralRecord(**_procedural_kwargs(valid_until=value)).valid_until == expected

    @pytest.mark.parametrize("value", [None, "", "soon", "2030-13-45", 20300501])
    def test_valid_until_that_is_not_a_date_means_no_known_end(self, value):
        assert ProceduralRecord(**_procedural_kwargs(valid_until=value)).valid_until is None

    @pytest.mark.parametrize(("input_score", "expected"), [(5.0, 1.0), (-1, 0.0), ("not a number", 0.5)])
    def test_utility_score_is_clamped(self, input_score, expected):
        rec = ProceduralRecord(**_procedural_kwargs(utility_score=input_score))
        assert rec.utility_score == expected

    def test_id_must_start_with_proc_prefix(self):
        with pytest.raises(pydantic.ValidationError, match="id must start with 'proc_'"):
            ProceduralRecord(**_procedural_kwargs(id="bad"))

    def test_version_must_be_positive(self):
        with pytest.raises(pydantic.ValidationError):
            ProceduralRecord(**_procedural_kwargs(version=0))

    @pytest.mark.parametrize(
        "bad",
        [{"procedure_kind": "workflow"}, {"evidence": ["guessed"]}, {"applies": "sometimes"}],
        ids=["kind", "evidence", "applies"],
    )
    def test_rejects_values_outside_the_vocabulary(self, bad):
        with pytest.raises(pydantic.ValidationError):
            ProceduralRecord(**_procedural_kwargs(**bad))

    def test_round_trip_preserves_new_fields(self):
        rec = ProceduralRecord(
            **_procedural_kwargs(
                procedure_kind=ProcedureKind.habit,
                applies=ProcedureApplies.always,
                valid_until="2030-05-01",
                status=ProcedureStatus.active,
                evidence=[ProcedureEvidence.corrected, ProcedureEvidence.repeated],
                support_dates=["2025-01-01", "2025-02-01"],
                user_support_count=2,
                practiced_dates=["2025-02-01"],
                source_kind=ProcedureSourceKind.observed_user_preference,
                source_turn_ids=["turn-1"],
            )
        )

        restored = MemoryRecordBase.from_doc(rec.to_doc())

        assert isinstance(restored, ProceduralRecord)
        assert restored.name == "Be concise"
        assert restored.procedure_kind == "habit"
        assert restored.applies == "always"
        assert restored.valid_until == "2030-05-01"
        assert restored.status == "active"
        assert restored.evidence == ["corrected", "repeated"]
        assert restored.support_dates == ["2025-01-01", "2025-02-01"]
        assert restored.user_support_count == 2
        assert restored.practiced_dates == ["2025-02-01"]
        assert restored.source_kind == "observed_user_preference"
        assert restored.source_turn_ids == ["turn-1"]


# ---------------------------------------------------------------------------
# Shared field validators on the base
# ---------------------------------------------------------------------------


class TestTagValidator:
    def test_default_empty(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi")
        assert rec.tags == []

    def test_sorted_and_deduped(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi", tags=["topic:b", "topic:a", "topic:a"])
        assert rec.tags == ["topic:a", "topic:b"]

    def test_lowercased(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi", tags=["Topic:Travel"])
        assert rec.tags == ["topic:travel"]

    def test_invalid_pattern(self):
        with pytest.raises(pydantic.ValidationError, match="Invalid tag"):
            TurnRecord(user_id="u1", role="user", content="hi", tags=["invalid tag!"])

    def test_too_long(self):
        with pytest.raises(pydantic.ValidationError, match="Invalid tag"):
            TurnRecord(user_id="u1", role="user", content="hi", tags=["a" * 101])

    def test_none_becomes_empty(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi", tags=None)
        assert rec.tags == []

    def test_rejects_sys_prefix_from_user_code(self):
        with pytest.raises(pydantic.ValidationError, match="sys:"):
            TurnRecord(user_id="u1", role="user", content="hi", tags=["sys:fact"])

    def test_rejects_more_than_50_tags(self):
        too_many = [f"topic:t{i}" for i in range(51)]
        with pytest.raises(pydantic.ValidationError, match="50"):
            TurnRecord(user_id="u1", role="user", content="hi", tags=too_many)

    def test_internal_context_allows_sys_prefix(self):
        rec = construct_internal(
            FactRecord,
            _fact_kwargs(tags=["sys:fact", "sys:auto-extracted", "topic:ui"]),
        )
        assert "sys:fact" in rec.tags
        assert "sys:auto-extracted" in rec.tags


class TestSalienceConfidence:
    def test_salience_valid(self):
        rec = FactRecord(**_fact_kwargs(salience=0.85))
        assert rec.salience == 0.85

    def test_salience_out_of_range_high(self):
        with pytest.raises(pydantic.ValidationError, match="salience"):
            FactRecord(**_fact_kwargs(salience=1.5))

    def test_salience_out_of_range_low(self):
        with pytest.raises(pydantic.ValidationError, match="salience"):
            FactRecord(**_fact_kwargs(salience=-0.1))

    def test_salience_boundary_zero(self):
        rec = FactRecord(**_fact_kwargs(salience=0.0))
        assert rec.salience == 0.0

    def test_salience_boundary_one(self):
        rec = FactRecord(**_fact_kwargs(salience=1.0))
        assert rec.salience == 1.0

    def test_confidence_valid(self):
        rec = FactRecord(**_fact_kwargs(confidence=0.92))
        assert rec.confidence == 0.92

    def test_confidence_out_of_range_high(self):
        with pytest.raises(pydantic.ValidationError, match="confidence"):
            FactRecord(**_fact_kwargs(confidence=1.5))


class TestContentHashFormat:
    def test_valid_32_hex(self):
        rec = FactRecord(**_fact_kwargs(content_hash=_HEX32))
        assert rec.content_hash == _HEX32

    def test_rejects_non_hex(self):
        with pytest.raises(pydantic.ValidationError, match="content_hash"):
            FactRecord(**_fact_kwargs(content_hash="abc"))

    def test_rejects_uppercase(self):
        with pytest.raises(pydantic.ValidationError, match="content_hash"):
            FactRecord(**_fact_kwargs(content_hash="A" * 32))


class TestPromptVersionFormat:
    def test_valid(self):
        rec = FactRecord(**_fact_kwargs(prompt_version="v1.2.3"))
        assert rec.prompt_version == "v1.2.3"

    def test_rejects_whitespace(self):
        with pytest.raises(pydantic.ValidationError, match="prompt_version"):
            FactRecord(**_fact_kwargs(prompt_version="v 1"))


class TestUseCount:
    def test_default_zero(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi")
        assert rec.use_count == 0

    def test_rejects_negative(self):
        with pytest.raises(pydantic.ValidationError):
            TurnRecord(user_id="u1", role="user", content="hi", use_count=-1)


# ---------------------------------------------------------------------------
# Supersession fields
# ---------------------------------------------------------------------------


class TestSupersession:
    def test_supersede_reason_literal(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi", supersede_reason="duplicate")
        assert rec.supersede_reason == "duplicate"

    def test_supersede_reason_rejects_other(self):
        with pytest.raises(pydantic.ValidationError):
            TurnRecord(user_id="u1", role="user", content="hi", supersede_reason="anything")

    def test_supersedes_ids_and_source_ids(self):
        rec = FactRecord(
            **_fact_kwargs(
                supersedes_ids=["old1"],
                source_memory_ids=["src1"],
            )
        )
        assert rec.supersedes_ids == ["old1"]
        assert rec.source_memory_ids == ["src1"]


# ---------------------------------------------------------------------------
# to_doc / from_doc round-trip & system-field handling
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_fact_round_trip(self, sample_embedding):
        original = FactRecord(**_fact_kwargs(embedding=sample_embedding))
        doc = original.to_doc()
        restored = MemoryRecordBase.from_doc(doc)
        assert isinstance(restored, FactRecord)
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.content_hash == original.content_hash
        assert restored.embedding == sample_embedding

    def test_episodic_round_trip(self):
        original = EpisodicRecord(**_episodic_kwargs())
        doc = original.to_doc()
        assert doc["events"][0] == {
            "sequence": 1,
            "description": "User asked for Paris trip planning help.",
            "occurred_at": "2026-01-01T09:00:00+00:00",
            "source_turn_ids": ["turn-1"],
        }
        assert doc["outcome"] == {"status": "successful", "description": "The trip plan was completed."}
        restored = MemoryRecordBase.from_doc(doc)
        assert isinstance(restored, EpisodicRecord)
        assert restored.title == original.title
        assert restored.participants == ["user", "agent"]
        assert restored.events[1] == EpisodeEvent(
            sequence=2,
            description="Agent recommended booking early.",
            source_turn_ids=["turn-2"],
        )
        assert restored.outcome == EpisodeOutcome(status="successful", description="The trip plan was completed.")
        assert restored.lessons == ["Plan early."]
        assert restored.source_turn_ids == ["turn-1", "turn-2"]

    def test_from_doc_strips_cosmos_system_fields(self, sample_embedding):
        original = FactRecord(**_fact_kwargs(embedding=sample_embedding))
        doc = original.to_doc()
        doc.update({"_rid": "abc", "_ts": 123, "_self": "s", "_attachments": "a"})
        restored = MemoryRecordBase.from_doc(doc)
        assert isinstance(restored, FactRecord)

    def test_from_doc_preserves_etag(self):
        original = FactRecord(**_fact_kwargs())
        doc = original.to_doc()
        doc["_etag"] = "etag-xyz"
        restored = MemoryRecordBase.from_doc(doc)
        assert restored.etag == "etag-xyz"

    def test_from_doc_factory_alias(self):
        original = FactRecord(**_fact_kwargs())
        doc = original.to_doc()
        restored = MemoryRecord.from_doc(doc)
        assert isinstance(restored, FactRecord)
        assert restored.id == original.id

    def test_from_cosmos_dict_alias_back_compat(self):
        original = FactRecord(**_fact_kwargs())
        doc = original.to_cosmos_dict()
        restored = MemoryRecord.from_cosmos_dict(doc)
        assert isinstance(restored, FactRecord)


# ---------------------------------------------------------------------------
# Back-compat dict-style access shim
# ---------------------------------------------------------------------------


class TestDictAccessShim:
    def test_getitem(self):
        rec = FactRecord(**_fact_kwargs())
        assert rec["id"] == "fact_" + _HEX32
        assert rec["content"] == "User prefers tea."
        assert rec["type"] == "fact"

    def test_getitem_raises_keyerror(self):
        rec = FactRecord(**_fact_kwargs())
        with pytest.raises(KeyError):
            _ = rec["bogus"]

    def test_get_with_default(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi")
        assert rec.get("nonexistent", "fallback") == "fallback"

    def test_contains(self):
        rec = FactRecord(**_fact_kwargs())
        assert "id" in rec
        assert "bogus" not in rec


# ---------------------------------------------------------------------------
# to_doc field emission
# ---------------------------------------------------------------------------


class TestToDocFieldEmission:
    def test_tags_always_present(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi")
        assert rec.to_doc()["tags"] == []

    def test_conditional_fields_emitted_when_populated(self, sample_embedding):
        rec = FactRecord(
            **_fact_kwargs(
                embedding=sample_embedding,
                salience=0.8,
                ttl=86400,
                supersedes_ids=["old"],
            )
        )
        d = rec.to_doc()
        assert d["ttl"] == 86400
        assert d["salience"] == 0.8
        assert d["supersedes_ids"] == ["old"]
        assert d["embedding"] == sample_embedding

    def test_omits_unset_optional_fields(self):
        rec = TurnRecord(user_id="u1", role="user", content="hi")
        d = rec.to_doc()
        for k in ("ttl", "embedding", "agent_id", "updated_at"):
            assert k not in d, f"{k} should be omitted"


# ---------------------------------------------------------------------------
# Internal construction helper
# ---------------------------------------------------------------------------


class TestConstructInternal:
    def test_round_trip_via_dict(self):
        data = _fact_kwargs(tags=["sys:fact", "topic:tea"])
        rec = construct_internal(FactRecord, data)
        assert isinstance(rec, FactRecord)
        # sys: tag is preserved through the internal context
        assert "sys:fact" in rec.tags


# ---------------------------------------------------------------------------
# SearchResult / OrchestrationResult
# ---------------------------------------------------------------------------


def test_search_result():
    rec = TurnRecord(user_id="u1", role="user", content="hi")
    sr = SearchResult(record=rec, score=0.95)
    assert sr.record is rec
    assert sr.score == 0.95

    sr_no_score = SearchResult(record=rec)
    assert sr_no_score.score is None


def test_orchestration_result():
    orch = OrchestrationResult(
        runtime_status="Completed",
        output={"result": 42},
        custom_status="done",
        instance_id="inst-1",
    )
    assert orch.runtime_status == "Completed"
    assert orch.output == {"result": 42}
    assert orch.custom_status == "done"
    assert orch.instance_id == "inst-1"


# ---------------------------------------------------------------------------
# MemoryType enum
# ---------------------------------------------------------------------------


def test_memory_type_enum_values():
    expected = {"turn", "thread_summary", "user_summary", "fact", "episodic", "procedural"}
    assert {m.value for m in MemoryType} == expected
