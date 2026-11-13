"""Strict JSON schemas for prompty outputs.

These schemas back ``response_format = {"type": "json_schema", ...}`` calls
on Azure OpenAI. Strict mode (``strict=True``) forces the model to emit
output that exactly matches the schema: no extra keys, no missing keys,
no wrong types. This makes our LLM pipelines behave deterministically
across model families (gpt-4o-mini, gpt-5.x, o-series) and at any
``temperature`` value the model accepts.

OpenAI's strict schema rules require:

* Every property declared under ``properties`` must appear in ``required``.
  There are no truly optional fields; "optional" is expressed with
  ``"type": ["string", "null"]`` so the field is always present but may
  be ``null``.
* ``additionalProperties: false`` on every object - the model cannot
  invent extra keys (e.g. ``reasoning`` or ``confidence`` siblings to
  the real payload that gpt-5.x was leaking into json_object outputs).

Each schema below is keyed by its prompty filename so
``services/pipeline.py`` can look it up and inject it before the LLM call.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# dedup.prompty - reconcile a pool of active facts
# ---------------------------------------------------------------------------
DEDUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradicted_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "winner_id": {"type": "string"},
                    "loser_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["winner_id", "loser_id", "reason"],
                "additionalProperties": False,
            },
        },
        "kept_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["contradicted_pairs", "kept_ids"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# dedup_verify.prompty - one memory's relation to each of its nearest neighbours
#
# One entry per candidate label: the relation, and the details each side states
# that the other lacks. A memory is retired only when it would lose nothing.
# ---------------------------------------------------------------------------
_DEDUP_RELATIONS = [
    "same_claim",
    "seed_contains_candidate",
    "candidate_contains_seed",
    "partial_overlap",
    "update",
    "inconsistent",
    "related",
    "unrelated",
]

DEDUP_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "relation": {"type": "string", "enum": _DEDUP_RELATIONS},
                    "seed_details_missing_from_candidate": {"type": "array", "items": {"type": "string"}},
                    "candidate_details_missing_from_seed": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "label",
                    "relation",
                    "seed_details_missing_from_candidate",
                    "candidate_details_missing_from_seed",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# extract_memories.prompty - extract facts (user- or agent-sourced)
# ---------------------------------------------------------------------------
_FACT_ITEM = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "category": {
            "type": "string",
            "enum": [
                "preference",
                "requirement",
                "biographical",
                "other",
            ],
        },
        "source": {
            "type": "string",
            "enum": ["user", "agent"],
        },
        "confidence": {"type": "number"},
        "salience": {"type": "number"},
        "temporal_context": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "text",
        "category",
        "source",
        "confidence",
        "salience",
        "temporal_context",
        "tags",
    ],
    "additionalProperties": False,
}

EXTRACT_MEMORIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": _FACT_ITEM},
    },
    "required": ["facts"],
    "additionalProperties": False,
}

# extract_memories-v3.prompty - the v2 fact plus how it was asserted, what it is about, the
# names it mentions and the turns it came from. Keys are in the order the model should write
# them: strict mode emits properties in schema order, so each field follows the ones it depends on.
_FACT_ITEM_V3_ORDER = (
    "text",
    "category",
    "source",
    "assertion",
    "subject",
    "attribute",
    "entities",
    "confidence",
    "salience",
    "temporal_context",
    "tags",
    "source_turn_ids",
)
_FACT_ITEM_V3_NEW = {
    "assertion": {
        "type": "string",
        "enum": ["stated", "decided", "suggested", "observed", "directive", "correction"],
    },
    "subject": {"type": "string"},
    "attribute": {"type": ["string", "null"]},
    "entities": {"type": "array", "items": {"type": "string"}},
    "source_turn_ids": {"type": "array", "items": {"type": "string"}},
}
_FACT_ITEM_V3 = {
    "type": "object",
    "properties": {key: {**_FACT_ITEM["properties"], **_FACT_ITEM_V3_NEW}[key] for key in _FACT_ITEM_V3_ORDER},
    "required": list(_FACT_ITEM_V3_ORDER),
    "additionalProperties": False,
}

EXTRACT_MEMORIES_V3_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": _FACT_ITEM_V3}},
    "required": ["facts"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# extract_episode.prompty - extract bounded episodic experience records
# ---------------------------------------------------------------------------
_EPISODE_EVENT = {
    "type": "object",
    "properties": {
        "sequence": {"type": "integer"},
        "description": {"type": "string"},
        "occurred_at": {"type": ["string", "null"]},
        "source_turn_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sequence", "description", "occurred_at", "source_turn_ids"],
    "additionalProperties": False,
}

_EPISODE_OUTCOME = {
    "type": ["object", "null"],
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "successful",
                "partially_successful",
                "failed",
                "abandoned",
                "unknown",
            ],
        },
        "description": {"type": "string"},
    },
    "required": ["status", "description"],
    "additionalProperties": False,
}

_EPISODE_ITEM = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "started_at": {"type": ["string", "null"]},
        "ended_at": {"type": ["string", "null"]},
        "participants": {"type": "array", "items": {"type": "string"}},
        "events": {"type": "array", "items": _EPISODE_EVENT},
        "outcome": _EPISODE_OUTCOME,
        "lessons": {"type": "array", "items": {"type": "string"}},
        "salience": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": [
        "title",
        "summary",
        "started_at",
        "ended_at",
        "participants",
        "events",
        "outcome",
        "lessons",
        "salience",
        "confidence",
    ],
    "additionalProperties": False,
}

EXTRACT_EPISODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "episodes": {"type": "array", "items": _EPISODE_ITEM},
    },
    "required": ["episodes"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# extract_procedure.prompty - learn procedures (instructions, habits, skills) from sources
# ---------------------------------------------------------------------------
_PROCEDURE_STEP = {
    "type": "object",
    "properties": {
        "sequence": {"type": "integer"},
        "instruction": {"type": "string"},
        "expected_result": {"type": ["string", "null"]},
        "on_failure": {"type": ["string", "null"]},
        "tool_name": {"type": ["string", "null"]},
    },
    "required": ["sequence", "instruction", "expected_result", "on_failure", "tool_name"],
    "additionalProperties": False,
}

_PROCEDURE_ITEM = {
    "type": "object",
    "properties": {
        # "add" for a new procedure; "update" revises and "reinforce" re-cites the existing
        # procedure ``updates`` names.
        "action": {"type": "string", "enum": ["add", "update", "reinforce"]},
        "updates": {"type": ["string", "null"]},
        "kind": {"type": "string", "enum": ["instruction", "habit", "skill"]},
        "evidence": {"type": "string", "enum": ["taught", "corrected", "repeated", "practiced", "endorsed"]},
        "name": {"type": "string"},
        "summary": {"type": "string"},
        "applies": {"type": "string", "enum": ["always", "when"]},
        "activation_conditions": {"type": "array", "items": {"type": "string"}},
        "preconditions": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": _PROCEDURE_STEP},
        "success_conditions": {"type": "array", "items": {"type": "string"}},
        "failure_conditions": {"type": "array", "items": {"type": "string"}},
        "safety_constraints": {"type": "array", "items": {"type": "string"}},
        "retrieval_text": {"type": "string"},
        "valid_until": {"type": ["string", "null"]},
        "grounded_in": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": [
        "action",
        "updates",
        "kind",
        "evidence",
        "name",
        "summary",
        "applies",
        "activation_conditions",
        "preconditions",
        "steps",
        "success_conditions",
        "failure_conditions",
        "safety_constraints",
        "retrieval_text",
        "valid_until",
        "grounded_in",
        "confidence",
    ],
    "additionalProperties": False,
}

EXTRACT_PROCEDURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "procedures": {"type": "array", "items": _PROCEDURE_ITEM},
    },
    "required": ["procedures"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# summarize.prompty - first-pass thread summary
#
# Mirrors the 6-field shape the prompty actually instructs the model to emit
# (see ``summarize.prompty`` lines ~69-88). Strict mode would silently drop
# every field outside this list; the consumer in ``services/pipeline.py``
# reads ``parsed["overview"]`` and stores ``parsed`` whole under
# ``metadata.structured_summary``, so the schema must carry the full shape.
# ---------------------------------------------------------------------------
_SUMMARY_ACTION_ITEM = {
    "type": "object",
    "properties": {
        "owner": {"type": ["string", "null"]},
        "task": {"type": "string"},
        "deadline": {"type": ["string", "null"]},
    },
    "required": ["owner", "task", "deadline"],
    "additionalProperties": False,
}

SUMMARIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "open_issues": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": _SUMMARY_ACTION_ITEM},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overview",
        "key_points",
        "decisions",
        "open_issues",
        "action_items",
        "topics",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# summarize_update.prompty - incremental thread summary update
# Same shape as the first-pass schema; both prompties emit the same payload.
# ---------------------------------------------------------------------------
SUMMARIZE_UPDATE_SCHEMA: dict[str, Any] = SUMMARIZE_SCHEMA


# ---------------------------------------------------------------------------
# user_summary.prompty - first-pass user profile
#
# Mirrors the 8 sections the prompty body documents (see
# ``user_summary.prompty`` lines ~35-86 and the JSON example block). Each
# section is a flat string array; the full ``parsed`` dict is persisted under
# ``metadata.structured_summary`` and the ``content`` field is composed from
# ``key_facts`` for vector retrieval.
# ---------------------------------------------------------------------------
USER_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "personal_preferences": {"type": "array", "items": {"type": "string"}},
        "account_environment": {"type": "array", "items": {"type": "string"}},
        "goals_current_work": {"type": "array", "items": {"type": "string"}},
        "behavioral_patterns": {"type": "array", "items": {"type": "string"}},
        "compliance_requirements": {"type": "array", "items": {"type": "string"}},
        "open_items": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "key_facts",
        "personal_preferences",
        "account_environment",
        "goals_current_work",
        "behavioral_patterns",
        "compliance_requirements",
        "open_items",
        "topics",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# user_summary_update.prompty - incremental user profile update
# Same shape as the first-pass schema.
# ---------------------------------------------------------------------------
USER_SUMMARY_UPDATE_SCHEMA: dict[str, Any] = USER_SUMMARY_SCHEMA


# ---------------------------------------------------------------------------
# Prompts that deliberately run with their prompty's plain ``json_object`` format and no fixed seed.
# ``combine_memories`` writes a combined fact that ``dedup_verify`` and a deterministic check then audit
# against its sources, and an unusable answer combines nothing, so a malformed answer costs one retry.
# ---------------------------------------------------------------------------
PLAIN_JSON_PROMPTS: frozenset[str] = frozenset({"combine_memories.prompty"})


# ---------------------------------------------------------------------------
# Registry - maps prompty filename → (schema_name, schema_dict)
# ---------------------------------------------------------------------------
PROMPTY_SCHEMAS: dict[str, tuple[str, dict[str, Any]]] = {
    "dedup.prompty": ("DedupOutput", DEDUP_SCHEMA),
    "dedup_verify.prompty": ("DedupVerifyOutput", DEDUP_VERIFY_SCHEMA),
    "extract_episode.prompty": ("ExtractEpisodesOutput", EXTRACT_EPISODE_SCHEMA),
    "extract_procedure.prompty": ("ExtractProceduresOutput", EXTRACT_PROCEDURE_SCHEMA),
    "extract_memories.prompty": ("ExtractMemoriesOutput", EXTRACT_MEMORIES_SCHEMA),
    "extract_memories-v2.prompty": ("ExtractMemoriesOutput", EXTRACT_MEMORIES_SCHEMA),
    "extract_memories-v3.prompty": ("ExtractMemoriesOutput", EXTRACT_MEMORIES_V3_SCHEMA),
    "summarize.prompty": ("SummarizeOutput", SUMMARIZE_SCHEMA),
    "summarize_update.prompty": ("SummarizeUpdateOutput", SUMMARIZE_UPDATE_SCHEMA),
    "user_summary.prompty": ("UserSummaryOutput", USER_SUMMARY_SCHEMA),
    "user_summary_update.prompty": ("UserSummaryUpdateOutput", USER_SUMMARY_UPDATE_SCHEMA),
}


def response_format_for(filename: str) -> dict[str, Any] | None:
    """Build the ``response_format`` payload for a prompty filename, if known."""
    entry = PROMPTY_SCHEMAS.get(filename)
    if entry is None:
        return None
    name, schema = entry
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }
