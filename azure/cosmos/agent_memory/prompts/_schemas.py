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
* ``additionalProperties: false`` on every object — the model cannot
  invent extra keys (e.g. ``reasoning`` or ``confidence`` siblings to
  the real payload that gpt-5.x was leaking into json_object outputs).

Each schema below is keyed by its prompty filename so
``services/pipeline.py`` can look it up and inject it before the LLM call.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# dedup.prompty — reconcile a pool of active facts
# ---------------------------------------------------------------------------
DEDUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "duplicate_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "merged_content": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": ["number", "null"]},
                    "salience": {"type": ["number", "null"]},
                },
                "required": ["merged_content", "source_ids", "confidence", "salience"],
                "additionalProperties": False,
            },
        },
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
    "required": ["duplicate_groups", "contradicted_pairs", "kept_ids"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# extract_memories.prompty — extract facts + episodic + unclassified
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
                "decision",
                "biographical",
                "temporal",
                "relational",
                "action_item",
            ],
        },
        "subject": {"type": ["string", "null"]},
        "predicate": {"type": ["string", "null"]},
        "object": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "salience": {"type": "number"},
        "temporal_context": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ["ADD", "UPDATE", "CONTRADICT"]},
        "supersedes_id": {"type": ["string", "null"]},
    },
    "required": [
        "text",
        "category",
        "subject",
        "predicate",
        "object",
        "confidence",
        "salience",
        "temporal_context",
        "tags",
        "action",
        "supersedes_id",
    ],
    "additionalProperties": False,
}

_EPISODIC_ITEM = {
    "type": "object",
    "properties": {
        "scope_type": {"type": "string"},
        "scope_value": {"type": "string"},
        "text": {"type": "string"},
        "situation": {"type": ["string", "null"]},
        "action_taken": {"type": ["string", "null"]},
        "outcome": {"type": ["string", "null"]},
        "outcome_valence": {
            "type": ["string", "null"],
            "enum": ["positive", "negative", "mixed", "neutral", None],
        },
        "reasoning": {"type": ["string", "null"]},
        "lesson": {"type": ["string", "null"]},
        "domain": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "salience": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "scope_type",
        "scope_value",
        "text",
        "situation",
        "action_taken",
        "outcome",
        "outcome_valence",
        "reasoning",
        "lesson",
        "domain",
        "confidence",
        "salience",
        "tags",
    ],
    "additionalProperties": False,
}

_UNCLASSIFIED_ITEM = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "confidence": {"type": "number"},
        "salience": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["text", "confidence", "salience", "tags", "reason"],
    "additionalProperties": False,
}

EXTRACT_MEMORIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": _FACT_ITEM},
        "episodic": {"type": "array", "items": _EPISODIC_ITEM},
        "unclassified": {"type": "array", "items": _UNCLASSIFIED_ITEM},
    },
    "required": ["facts", "episodic", "unclassified"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# summarize.prompty — first-pass thread summary
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
# summarize_update.prompty — incremental thread summary update
# Same shape as the first-pass schema; both prompties emit the same payload.
# ---------------------------------------------------------------------------
SUMMARIZE_UPDATE_SCHEMA: dict[str, Any] = SUMMARIZE_SCHEMA


# ---------------------------------------------------------------------------
# user_summary.prompty — first-pass user profile
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
# user_summary_update.prompty — incremental user profile update
# Same shape as the first-pass schema.
# ---------------------------------------------------------------------------
USER_SUMMARY_UPDATE_SCHEMA: dict[str, Any] = USER_SUMMARY_SCHEMA


# ---------------------------------------------------------------------------
# synthesize_procedural.prompty — agent self-improvement / procedural prompt
# ---------------------------------------------------------------------------
SYNTHESIZE_PROCEDURAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "system_prompt": {"type": "string"},
    },
    "required": ["system_prompt"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Registry — maps prompty filename → (schema_name, schema_dict)
# ---------------------------------------------------------------------------
PROMPTY_SCHEMAS: dict[str, tuple[str, dict[str, Any]]] = {
    "dedup.prompty": ("DedupOutput", DEDUP_SCHEMA),
    "extract_memories.prompty": ("ExtractMemoriesOutput", EXTRACT_MEMORIES_SCHEMA),
    "summarize.prompty": ("SummarizeOutput", SUMMARIZE_SCHEMA),
    "summarize_update.prompty": ("SummarizeUpdateOutput", SUMMARIZE_UPDATE_SCHEMA),
    "user_summary.prompty": ("UserSummaryOutput", USER_SUMMARY_SCHEMA),
    "user_summary_update.prompty": ("UserSummaryUpdateOutput", USER_SUMMARY_UPDATE_SCHEMA),
    "synthesize_procedural.prompty": (
        "SynthesizeProceduralOutput",
        SYNTHESIZE_PROCEDURAL_SCHEMA,
    ),
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
