"""Tenant scope guard: no code path may filter on an owner without the tenant, or build a raw partition key.

A query that names ``c.user_id`` alone reads the same id in every tenant, and a hand-built ``[user_id, ...]`` key
addresses the wrong partition. These checks read the source, so a new query written without the tenant fails here
before it can leak.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(
    path
    for base in (ROOT / "azure" / "cosmos" / "agent_memory", ROOT / "function_app")
    for path in base.rglob("*.py")
    if "prompts" not in path.parts and path.name != "_partitioning.py"
)

# A function that filters on the owner must be tenant-aware: it names the tenant, the shared filters, or the key helper.
TENANT_AWARE = re.compile(r"tenant|OWNER_FILTER|partition_key_for", re.IGNORECASE)
OWNER_FILTER_USE = re.compile(r"""c\.user_id|["']c\.user_id["']""")


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _owner_filters_without_the_tenant() -> list[str]:
    found: list[str] = []
    for path in SOURCES:
        text = path.read_text()
        tree = ast.parse(text)
        for function in _functions(tree):
            source = ast.get_source_segment(text, function) or ""
            if OWNER_FILTER_USE.search(source) and not TENANT_AWARE.search(source):
                found.append(f"{path.relative_to(ROOT)}:{function.lineno} {function.name}")
    return found


def test_every_function_that_filters_on_the_owner_is_tenant_aware() -> None:
    assert _owner_filters_without_the_tenant() == []


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_partition_key_is_built_by_hand(path: Path) -> None:
    text = path.read_text()
    by_hand = re.findall(r"partition_key\s*=\s*\[", text)
    assert by_hand == [], f"{path.relative_to(ROOT)} builds a partition key list by hand; use partition_key_for"


def test_containers_are_created_with_the_tenant_and_owner_key() -> None:
    for name in ("cosmos_memory_client.py", "aio/cosmos_memory_client.py"):
        text = (ROOT / "azure" / "cosmos" / "agent_memory" / name).read_text()
        assert 'PartitionKey(path=list(PARTITION_KEY_PATHS), kind="MultiHash")' in text, name
        assert '"/thread_id"]' not in text.replace('{"path": "/thread_id", "order": "ascending"}', ""), name


def test_an_existing_container_is_checked_for_the_key_when_the_store_is_created() -> None:
    for name in ("cosmos_memory_client.py", "aio/cosmos_memory_client.py"):
        text = (ROOT / "azure" / "cosmos" / "agent_memory" / name).read_text()
        assert text.count("_require_partition_key(") >= 2, name
        assert "partition_key_problem(container_id, properties)" in text, name
