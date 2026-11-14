"""Conformance tests: every strict JSON schema must match its prompty body.

The strict ``response_format`` schemas in ``azure.cosmos.agent_memory.prompts._schemas``
are paired with prompty files that describe the expected output shape in
markdown JSON code-fences. If the two drift apart:

* Required schema keys missing from the prompty → the model never sees them
  described and emits empty values (the root cause of the "raw JSON dump"
  bug for thread/user summaries).
* Prompty keys missing from the schema → strict mode (``additionalProperties:
  false``) rejects perfectly-good model output as invalid.

This module walks ``PROMPTY_SCHEMAS`` and verifies that each prompty contains
at least one JSON code-block whose top-level keys exactly cover the schema's
declared properties. Prompties may include multiple ```json blocks (schema
shape + worked examples + nested fragments); we accept the first block whose
top-level shape lines up with the schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from azure.cosmos.agent_memory.models import FACT_ASSERTIONS, USER_ONLY_ASSERTIONS
from azure.cosmos.agent_memory.prompts._schemas import PLAIN_JSON_PROMPTS, PROMPTY_SCHEMAS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_DIR = _REPO_ROOT / "azure" / "cosmos" / "agent_memory" / "prompts"

_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def _top_level_object_keys(text: str) -> list[set[str]]:
    """Return the top-level keys of every parseable JSON object in ``text``."""
    keys_per_block: list[set[str]] = []
    for data in _parsed_json_blocks(text):
        if isinstance(data, dict):
            keys_per_block.append(set(data.keys()))
    return keys_per_block


def _parsed_json_blocks(text: str) -> list[object]:
    """Return every parseable ```json``` block as a Python object."""
    out: list[object] = []
    for block in _JSON_BLOCK_RE.findall(text):
        block_stripped = block.strip()
        if not block_stripped:
            continue
        try:
            out.append(json.loads(block_stripped))
        except json.JSONDecodeError:
            continue
    return out


def _normalize_type(schema_type: object, data: object) -> object:
    """Pick the best matching type from a union (e.g. ``['object', 'null']``)."""
    if not isinstance(schema_type, list):
        return schema_type
    if isinstance(data, dict) and "object" in schema_type:
        return "object"
    if isinstance(data, list) and "array" in schema_type:
        return "array"
    return None


def _walk_required(data: object, schema: object, path: str) -> list[str]:
    """Recursively check that every ``required`` key in ``schema`` is present.

    Walks into object ``properties`` and array ``items`` (validating only the
    first item, since examples are illustrative not exhaustive). Returns a
    flat list of human-readable error strings - empty list ⇒ conformant.
    """
    if not isinstance(schema, dict):
        return []

    errors: list[str] = []
    schema_type = _normalize_type(schema.get("type"), data)

    if schema_type == "object" and isinstance(data, dict):
        required = schema.get("required", []) or []
        properties = schema.get("properties", {}) or {}
        for req_key in required:
            if req_key not in data:
                errors.append(f"{path}: missing required key '{req_key}'")
        for key, value in data.items():
            if key in properties:
                errors.extend(_walk_required(value, properties[key], f"{path}.{key}"))
    elif schema_type == "array" and isinstance(data, list) and data:
        items_schema = schema.get("items")
        if items_schema:
            errors.extend(_walk_required(data[0], items_schema, f"{path}[0]"))

    return errors


@pytest.mark.parametrize(
    "filename,schema_entry",
    sorted(PROMPTY_SCHEMAS.items()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_prompty_describes_schema_shape(filename: str, schema_entry: tuple[str, dict]) -> None:
    """Each prompty must contain a JSON block matching its strict schema."""
    _name, schema = schema_entry
    properties = set(schema.get("properties", {}).keys())
    required = set(schema.get("required", []))

    prompty_path = _PROMPTS_DIR / filename
    assert prompty_path.exists(), f"Missing prompty file: {prompty_path}"

    text = prompty_path.read_text(encoding="utf-8")
    blocks_keys = _top_level_object_keys(text)

    assert blocks_keys, (
        f"{filename}: no parseable ```json``` blocks found. "
        f"Add an Output Format example matching schema properties "
        f"{sorted(properties)}."
    )

    matching_blocks = [keys for keys in blocks_keys if required <= keys <= properties]

    if not matching_blocks:
        diagnostic_blocks = [sorted(k) for k in blocks_keys]
        pytest.fail(
            f"\n{filename}: no JSON example matches the strict schema.\n"
            f"  schema required:   {sorted(required)}\n"
            f"  schema properties: {sorted(properties)}\n"
            f"  prompty top-level keys per block: {diagnostic_blocks}\n"
            f"\nFix one of:\n"
            f"  - Add or update an example so required keys appear at the top\n"
            f"  - Update the schema in azure/cosmos/agent_memory/prompts/_schemas.py\n"
            f"    if the prompty's intent has changed."
        )


@pytest.mark.parametrize(
    "filename,schema_entry",
    sorted(PROMPTY_SCHEMAS.items()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_prompty_nested_required_fields_present(filename: str, schema_entry: tuple[str, dict]) -> None:
    """Each prompty's JSON example must include all required keys at every nesting level.

    Catches drift inside nested objects (e.g. ``duplicate_groups[].required``)
    that the top-level shape test cannot see. Walks the first JSON block whose
    top-level keys satisfy the schema; if none does, the top-level test owns
    the failure and we skip here.
    """
    _name, schema = schema_entry
    properties = set(schema.get("properties", {}).keys())
    required = set(schema.get("required", []))

    prompty_path = _PROMPTS_DIR / filename
    text = prompty_path.read_text(encoding="utf-8")

    for data in _parsed_json_blocks(text):
        if not isinstance(data, dict):
            continue
        keys = set(data.keys())
        if not (required <= keys <= properties):
            continue
        errors = _walk_required(data, schema, f"${filename}")
        if errors:
            pytest.fail(f"\n{filename}: nested required keys missing in example JSON:\n  " + "\n  ".join(errors))
        return

    pytest.skip(f"{filename}: no top-level matching block - covered by shape test")


def test_every_registered_prompty_file_exists() -> None:
    """Guard against typos / removed prompties leaving dangling schema entries."""
    missing = [filename for filename in PROMPTY_SCHEMAS if not (_PROMPTS_DIR / filename).exists()]
    assert not missing, f"PROMPTY_SCHEMAS references nonexistent files: {missing}"


def test_every_prompty_file_has_a_registered_schema() -> None:
    """Guard against adding a prompty without wiring up a strict schema (unless it is listed as plain JSON)."""
    prompty_files = {path.name for path in _PROMPTS_DIR.glob("*.prompty") if not path.name.startswith("_")}
    unregistered = prompty_files - set(PROMPTY_SCHEMAS) - PLAIN_JSON_PROMPTS
    assert not unregistered, (
        f"These prompty files have no entry in PROMPTY_SCHEMAS - "
        f"strict response_format will not be applied: {sorted(unregistered)}"
    )


# ---------------------------------------------------------------------------
# Fact-extraction prompts: worked examples and the output template
# ---------------------------------------------------------------------------

_FACT_EXTRACTORS = ["extract_memories-v2.prompty", "extract_memories-v3.prompty"]


def _example_sections(filename: str) -> list[tuple[str, str, str]]:
    """``(number, title, section)`` for each worked example in a fact-extraction prompty."""
    text = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    examples_text = text.split("## Few-Shot Examples", 1)[1].split("## Output Format", 1)[0]
    return re.findall(
        r"^### Example (\d+[a-z]?):([^\n]*)\n(.*?)(?=^### Example |\Z)", examples_text, re.MULTILINE | re.DOTALL
    )


def _worked_examples(filename: str) -> dict[str, dict]:
    examples = {}
    for number, _title, section in _example_sections(filename):
        blocks = _JSON_BLOCK_RE.findall(section)
        assert len(blocks) == 1, f"Example {number}: expected one JSON output"
        assert number not in examples, f"Duplicate example number: {number}"
        examples[number] = json.loads(blocks[0])
    assert examples
    return examples


def _fact_schema(filename: str) -> dict:
    _name, schema = PROMPTY_SCHEMAS[filename]
    return schema["properties"]["facts"]["items"]


@pytest.mark.parametrize("filename", _FACT_EXTRACTORS)
def test_all_worked_examples_conform_to_fact_schema(filename: str) -> None:
    fact_schema = _fact_schema(filename)
    for number, output in _worked_examples(filename).items():
        assert set(output) == {"facts"}, number
        assert isinstance(output["facts"], list), number
        for fact in output["facts"]:
            assert isinstance(fact, dict), number
            assert not _walk_required(fact, fact_schema, f"example-{number}")
            assert set(fact) <= set(fact_schema["properties"]), number
            assert isinstance(fact["text"], str) and fact["text"].strip(), number
            for field, field_schema in fact_schema["properties"].items():
                if "enum" in field_schema:
                    assert fact[field] in field_schema["enum"], (number, field)
            for field in ("confidence", "salience"):
                assert isinstance(fact[field], (int, float)) and 0 <= fact[field] <= 1, (number, field)


@pytest.mark.parametrize("filename", _FACT_EXTRACTORS)
def test_the_output_format_asks_for_every_fact_field(filename: str) -> None:
    """The template the model fills lists each field the schema requires; a worked example alone would not show it."""
    fact_schema = _fact_schema(filename)
    output_format = (_PROMPTS_DIR / filename).read_text(encoding="utf-8").split("## Output Format", 1)[1]
    template = json.loads(_JSON_BLOCK_RE.findall(output_format)[0])

    [fact] = template["facts"]
    assert set(fact) == set(fact_schema["properties"]) == set(fact_schema["required"])


def test_v3_fields_are_written_in_schema_order_and_match_the_model_vocabulary() -> None:
    """Strict mode emits properties in schema order, and models imitate the template and examples."""
    fact_schema = _fact_schema("extract_memories-v3.prompty")
    order = list(fact_schema["properties"])
    output_format = (_PROMPTS_DIR / "extract_memories-v3.prompty").read_text(encoding="utf-8").split("## Output Format")
    template = json.loads(_JSON_BLOCK_RE.findall(output_format[1])[0])

    assert list(template["facts"][0]) == order
    for number, output in _worked_examples("extract_memories-v3.prompty").items():
        assert all(list(fact) == order for fact in output["facts"]), number
    assert set(fact_schema["properties"]["assertion"]["enum"]) == FACT_ASSERTIONS
    assert not {"event_date", "valid_until", "durability", "suggested_scope_type", "scope_confidence"} & set(order)


def test_v3_worked_examples_fill_the_structured_fields_as_described() -> None:
    """The demonstrations follow the v3 rules the extractor is asked to apply."""
    examples = _worked_examples("extract_memories-v3.prompty")
    for number, output in examples.items():
        for fact in output["facts"]:
            assert isinstance(fact["subject"], str) and fact["subject"].strip(), number
            assert fact["attribute"] is None or (fact["attribute"] == fact["attribute"].lower()), number
            assert all(isinstance(name, str) and name.strip() for name in fact["entities"]), number
            assert not {"user", "assistant"} & {name.lower() for name in fact["entities"]}, number
            if fact["assertion"] in USER_ONLY_ASSERTIONS:
                assert (fact["source"], fact["subject"]) == ("user", "assistant"), number
            if fact["source"] == "agent":
                # Agent-provided content is never a directive, and is never filed under the assistant's own rules.
                assert fact["assertion"] not in USER_ONLY_ASSERTIONS, number
                assert fact["subject"] != "assistant" and fact["attribute"] is None, number
    assert {fact["assertion"] for output in examples.values() for fact in output["facts"]} == FACT_ASSERTIONS


# A conversation line is ``> Speaker: "..."`` or ``> [timestamp | role]: "..."``.
_CONVERSATION_LINE = re.compile(r"^> (?:\[[^\]|]*\|\s*(?P<role>\w+)\]|(?P<speaker>\w+)):", re.MULTILINE)


def _turn_sources(section: str) -> list[str]:
    """The extraction ``source`` each conversation turn would carry, in ``turn-N`` order."""
    speakers = [(match["role"] or match["speaker"]).lower() for match in _CONVERSATION_LINE.finditer(section)]
    return ["agent" if speaker in {"agent", "assistant"} else "user" for speaker in speakers]


def test_v3_worked_examples_match_what_their_titles_and_cited_turns_imply() -> None:
    """Guard the demonstrations, not live LLM behavior: no numbered example is pinned, so new ones are checked too."""
    filename = "extract_memories-v3.prompty"
    examples = _worked_examples(filename)
    for number, title, section in _example_sections(filename):
        facts = examples[number]["facts"]
        assert (facts == []) is title.strip().endswith("extract nothing"), (number, title)
        turn_sources = _turn_sources(section)
        for fact in facts:
            positions = [int(ref.removeprefix("turn-")) for ref in fact["source_turn_ids"]]
            assert all(1 <= position <= len(turn_sources) for position in positions), (number, fact["text"])
            cited = {turn_sources[position - 1] for position in positions}
            # A fact drawn only from agent turns is an agent fact; one drawn only from user turns is a user fact.
            if len(cited) == 1:
                assert fact["source"] == cited.pop(), (number, fact["text"])


def test_v3_keeps_the_agent_source_rules_of_v2() -> None:
    """Agent-provided answers stay extractable (Example 7b); no field the fact schema lacks is named."""
    text = (_PROMPTS_DIR / "extract_memories-v3.prompty").read_text(encoding="utf-8")

    assert "### Example 7b: Agent-provided information the user will ask about later" in text
    assert "Capture BOTH kinds" in text and "NEVER turn an agent line into a fact *about the user*" in text
    for dropped in ("suggested_scope_type", "event_date", "valid_until", "durability"):
        assert dropped not in text, dropped
    assert "Never write a literal secret into a memory" in text


def test_plain_json_prompts_ask_for_json_without_a_fixed_seed() -> None:
    """The plain-JSON prompts run as validated: the prompty's json_object format, no strict schema, no seed."""
    assert PLAIN_JSON_PROMPTS.isdisjoint(PROMPTY_SCHEMAS)
    for name in sorted(PLAIN_JSON_PROMPTS):
        front_matter = (_PROMPTS_DIR / name).read_text(encoding="utf-8").split("\n---", 1)[0]
        assert re.search(r"response_format:\s*\n\s*type: json_object", front_matter), name
        assert not re.search(r"^\s*seed:", front_matter, re.MULTILINE), name
