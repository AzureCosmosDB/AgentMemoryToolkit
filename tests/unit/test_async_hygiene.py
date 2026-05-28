"""Async hygiene regression guards for ``agent_memory_toolkit/aio/``.

These tests programmatically inspect the async subpackage to catch
regressions where contributors might inadvertently re-introduce sync I/O on
the event loop. The primary check today is for ``asyncio.to_thread`` call
sites: only two are allowed, and both wrap genuinely synchronous interop
boundaries (a sync counter helper and a sync ``TokenCredential.get_token``).

We match by **enclosing function name** (via AST), not raw line numbers, so
the assertions stay stable across formatting / import reorders inside the
allowed functions. If a new ``to_thread`` call is genuinely needed, update
the ``ALLOWED_CALL_SITES`` set with a justification.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent_memory_toolkit.aio as _aio_pkg

# Each entry is ``(relative_path_under_aio, name_of_an_enclosing_function)``.
# A call site matches if *any* function in its enclosing chain has the
# allowed name — this lets us tolerate nested closures (e.g. the inner
# ``_provider`` inside ``_make_sync_token_provider_for_async``) without
# pinning to the inner name.
ALLOWED_CALL_SITES: set[tuple[str, str]] = {
    # Bridge that lets sync callables be awaited from async auto-trigger code.
    ("auto_trigger.py", "_call_async_compatible"),
    # Sync ``TokenCredential.get_token`` adapter for AsyncAzureOpenAI.
    ("chat.py", "_make_sync_token_provider_for_async"),
}

EXPECTED_TOTAL_CALL_SITES = 2


def _aio_root() -> Path:
    return Path(_aio_pkg.__file__).parent


class _ToThreadFinder(ast.NodeVisitor):
    """Collect (lineno, [enclosing function names]) for every ``asyncio.to_thread`` call."""

    def __init__(self) -> None:
        self._stack: list[str] = []
        self.hits: list[tuple[int, tuple[str, ...]]] = []

    def _visit_func(self, node: ast.AST) -> None:
        self._stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_func(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "to_thread"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
        ):
            self.hits.append((node.lineno, tuple(self._stack)))
        self.generic_visit(node)


def _collect_to_thread_calls() -> list[tuple[str, int, tuple[str, ...]]]:
    """Return ``(relative_path, lineno, enclosing_function_chain)`` for every hit."""
    root = _aio_root()
    all_hits: list[tuple[str, int, tuple[str, ...]]] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "asyncio.to_thread" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        finder = _ToThreadFinder()
        finder.visit(tree)
        rel = path.relative_to(root).as_posix()
        for lineno, chain in finder.hits:
            all_hits.append((rel, lineno, chain))
    return all_hits


def _is_allowed(rel_path: str, chain: tuple[str, ...]) -> bool:
    return any((rel_path, fname) in ALLOWED_CALL_SITES for fname in chain)


def test_asyncio_to_thread_only_in_allowed_call_sites() -> None:
    """Every ``asyncio.to_thread`` call in ``aio/`` must be in the allowed set.

    Failure here means a new sync-to-async bridge was introduced. Either
    rewrite the call to use a native async API, or — if the bridge is
    truly necessary — add ``(file, function)`` to ``ALLOWED_CALL_SITES``
    in this test along with a comment explaining why.
    """
    hits = _collect_to_thread_calls()

    unexpected = [(rel, lineno, chain) for (rel, lineno, chain) in hits if not _is_allowed(rel, chain)]
    assert not unexpected, (
        "Unexpected asyncio.to_thread call site(s) found in "
        "agent_memory_toolkit/aio/. Each new to_thread reintroduces sync "
        "I/O onto the event loop and must be either removed or explicitly "
        "added to ALLOWED_CALL_SITES in this test with justification.\n"
        f"  Unexpected sites: {unexpected}\n"
        f"  Allowed sites:    {sorted(ALLOWED_CALL_SITES)}"
    )

    assert len(hits) == EXPECTED_TOTAL_CALL_SITES, (
        f"Expected exactly {EXPECTED_TOTAL_CALL_SITES} asyncio.to_thread call "
        f"sites in agent_memory_toolkit/aio/; found {len(hits)}.\n"
        f"  All sites: {hits}\n"
        "If an allowed site was intentionally removed, drop the corresponding "
        "entry from ALLOWED_CALL_SITES and update EXPECTED_TOTAL_CALL_SITES."
    )
