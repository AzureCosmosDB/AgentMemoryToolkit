"""Conformance tests: every strict JSON schema must match its prompty body.

The strict ``response_format`` schemas in ``agent_memory_toolkit.prompts._schemas``
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

from agent_memory_toolkit.prompts._schemas import PROMPTY_SCHEMAS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_DIR = _REPO_ROOT / "agent_memory_toolkit" / "prompts"

_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def _top_level_object_keys(text: str) -> list[set[str]]:
    """Return the top-level keys of every parseable JSON object in ``text``.

    Walks every ``` ```json ``` ``` block, attempts ``json.loads`` on each,
    and collects the dict's top-level keys. Unparseable blocks (e.g.,
    fragments with ``...`` placeholders) are skipped — the schema-match
    check only needs one good block.
    """
    keys_per_block: list[set[str]] = []
    for block in _JSON_BLOCK_RE.findall(text):
        block_stripped = block.strip()
        if not block_stripped:
            continue
        try:
            data = json.loads(block_stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            keys_per_block.append(set(data.keys()))
    return keys_per_block


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
            f"  - Update the schema in agent_memory_toolkit/prompts/_schemas.py\n"
            f"    if the prompty's intent has changed."
        )


def test_every_registered_prompty_file_exists() -> None:
    """Guard against typos / removed prompties leaving dangling schema entries."""
    missing = [filename for filename in PROMPTY_SCHEMAS if not (_PROMPTS_DIR / filename).exists()]
    assert not missing, f"PROMPTY_SCHEMAS references nonexistent files: {missing}"


def test_every_prompty_file_has_a_registered_schema() -> None:
    """Guard against adding a prompty without wiring up a strict schema."""
    prompty_files = {path.name for path in _PROMPTS_DIR.glob("*.prompty") if not path.name.startswith("_")}
    unregistered = prompty_files - set(PROMPTY_SCHEMAS)
    assert not unregistered, (
        f"These prompty files have no entry in PROMPTY_SCHEMAS — "
        f"strict response_format will not be applied: {sorted(unregistered)}"
    )
