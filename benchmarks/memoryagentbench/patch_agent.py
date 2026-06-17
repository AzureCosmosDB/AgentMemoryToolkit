"""Idempotent patch installer for MemoryAgentBench's ``agent.py``.

MemoryAgentBench routes everything through ``AgentWrapper`` in ``agent.py``.
This script injects three small additions:

1. A top-of-file import of :class:`AgentMemoryToolkitBackend`.
2. A new ``_initialize_agent_memory_toolkit_agent`` initializer.
3. A new ``_handle_agent_memory_toolkit_agent`` handler.
4. Dispatch hooks in ``_initialize_agent_by_type`` and ``send_message``.

Each insertion is wrapped in markers so this script can detect and skip
already-installed patches. Run as::

    python patch_agent.py path/to/MemoryAgentBench/agent.py

This is intentionally a small, self-contained edit. For longer-term
integration the recommendation is to maintain a fork of MemoryAgentBench
that already has the dispatch hooks, and import the adapter directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BEGIN_IMPORT = "# >>> agent_memory_toolkit-import (do not edit)"
END_IMPORT = "# <<< agent_memory_toolkit-import"
BEGIN_METHODS = "    # >>> agent_memory_toolkit-methods (do not edit)"
END_METHODS = "    # <<< agent_memory_toolkit-methods"
BEGIN_INIT_DISPATCH = "        # >>> agent_memory_toolkit-init-dispatch"
END_INIT_DISPATCH = "        # <<< agent_memory_toolkit-init-dispatch"
BEGIN_SEND_DISPATCH = "        # >>> agent_memory_toolkit-send-dispatch"
END_SEND_DISPATCH = "        # <<< agent_memory_toolkit-send-dispatch"

IMPORT_BLOCK = f"""\
{BEGIN_IMPORT}
try:
    from benchmarks.memoryagentbench.adapter import AgentMemoryToolkitBackend
except Exception:  # pragma: no cover - import is optional
    AgentMemoryToolkitBackend = None  # type: ignore[assignment]
{END_IMPORT}
"""

METHODS_BLOCK = f"""\
{BEGIN_METHODS}
    def _initialize_agent_memory_toolkit_agent(self, agent_config, dataset_config):
        \"\"\"Initialize the AgentMemoryToolkit-backed agent.\"\"\"
        if AgentMemoryToolkitBackend is None:
            raise ImportError(
                \"AgentMemoryToolkitBackend not importable. Ensure the \"
                \"AgentMemoryToolkit repo is installed (pip install -e ...) \"
                \"and the 'benchmarks' package is on PYTHONPATH.\"
            )
        self.retrieve_num = agent_config.get('retrieve_num', 5)
        self.context = ''
        self._amt_backend = AgentMemoryToolkitBackend(agent_config, dataset_config)
        import time as _time
        self.agent_start_time = _time.time()

    def _handle_agent_memory_toolkit_agent(self, message, memorizing, query_id, context_id):
        \"\"\"Handle messages for the AgentMemoryToolkit agent.\"\"\"
        if memorizing:
            return self._amt_backend.memorize(
                message, query_id=query_id, context_id=context_id
            )
        result = self._amt_backend.query(
            message, query_id=query_id, context_id=context_id
        )
        import time as _time
        self.agent_start_time = _time.time()
        return result
{END_METHODS}
"""

INIT_DISPATCH_BLOCK = f"""\
{BEGIN_INIT_DISPATCH}
        if self._is_agent_type(\"agent_memory_toolkit\"):
            self._initialize_agent_memory_toolkit_agent(agent_config, dataset_config)
            return
{END_INIT_DISPATCH}
"""

SEND_DISPATCH_BLOCK = f"""\
{BEGIN_SEND_DISPATCH}
        if self._is_agent_type(\"agent_memory_toolkit\"):
            return self._handle_agent_memory_toolkit_agent(
                message, memorizing, query_id, context_id
            )
{END_SEND_DISPATCH}
"""


def _has_block(text: str, begin: str) -> bool:
    return begin in text


def _insert_after_line(text: str, needle: str, block: str) -> str:
    idx = text.find(needle)
    if idx == -1:
        raise RuntimeError(f"Could not find anchor: {needle!r}")
    end_of_line = text.find("\n", idx)
    if end_of_line == -1:
        return text + "\n" + block
    return text[: end_of_line + 1] + block + text[end_of_line + 1 :]


def _insert_before_line(text: str, needle: str, block: str) -> str:
    idx = text.find(needle)
    if idx == -1:
        raise RuntimeError(f"Could not find anchor: {needle!r}")
    line_start = text.rfind("\n", 0, idx) + 1
    return text[:line_start] + block + text[line_start:]


def patch(text: str) -> tuple[str, list[str]]:
    """Apply patch to ``agent.py`` text; return (new_text, messages)."""
    messages: list[str] = []

    # 1. Import block at top of file (after the last top-level `import` line).
    if _has_block(text, BEGIN_IMPORT):
        messages.append("import block already present")
    else:
        # Only consider top-level import lines (no leading whitespace), and
        # stop at the first non-import top-level statement so we don't pick
        # up imports buried later in the file. Track parenthesis depth so
        # multi-line ``from x import (...)`` blocks are kept intact.
        lines = text.splitlines(keepends=True)
        last_import = -1
        paren_depth = 0
        in_import = False
        for i, line in enumerate(lines):
            if paren_depth > 0:
                # Continuation of a multi-line import; track parens.
                paren_depth += line.count("(") - line.count(")")
                if paren_depth <= 0:
                    paren_depth = 0
                    if in_import:
                        last_import = i
                        in_import = False
                continue
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line[0].isspace():
                # Indented (inside class/function); ignore.
                continue
            if line.startswith(("import ", "from ")):
                opens = line.count("(") - line.count(")")
                if opens > 0:
                    paren_depth = opens
                    in_import = True
                else:
                    last_import = i
                continue
            # First top-level non-import, non-comment statement: stop.
            if last_import >= 0:
                break
        if last_import < 0:
            new_text = IMPORT_BLOCK + text
        else:
            new_text = (
                "".join(lines[: last_import + 1])
                + IMPORT_BLOCK
                + "".join(lines[last_import + 1 :])
            )
        text = new_text
        messages.append("inserted import block")

    # 2. Methods inside AgentWrapper (before its closing).
    if _has_block(text, BEGIN_METHODS):
        messages.append("methods block already present")
    else:
        # Insert just before the final close of the file as the last method.
        # Anchor: the AgentWrapper class. Insert at the end of the last
        # method definition we can detect. Simplest: append before final
        # "if __name__" or at end-of-file.
        anchor = "if __name__"
        if anchor in text:
            text = _insert_before_line(text, anchor, METHODS_BLOCK + "\n")
        else:
            text = text.rstrip() + "\n\n" + METHODS_BLOCK + "\n"
        messages.append("inserted methods block")

    # 3. Init dispatch at the start of _initialize_agent_by_type body.
    if _has_block(text, BEGIN_INIT_DISPATCH):
        messages.append("init-dispatch already present")
    else:
        anchor = "def _initialize_agent_by_type(self, agent_config, dataset_config):"
        if anchor not in text:
            messages.append("WARN: _initialize_agent_by_type not found; skipped init-dispatch")
        else:
            # Find end of the def line and insert the dispatch right after the
            # docstring (or directly after the def line if no docstring).
            def_idx = text.find(anchor)
            line_end = text.find("\n", def_idx)
            insertion_point = line_end + 1
            # Skip a docstring if present.
            after_def = text[insertion_point:]
            stripped = after_def.lstrip()
            if stripped.startswith(("\"\"\"", "'''")):
                quote = stripped[:3]
                # Skip leading whitespace
                ws = len(after_def) - len(stripped)
                close_pos = after_def.find(quote, ws + 3)
                if close_pos != -1:
                    end_of_doc_line = after_def.find("\n", close_pos)
                    if end_of_doc_line != -1:
                        insertion_point += end_of_doc_line + 1
            text = text[:insertion_point] + INIT_DISPATCH_BLOCK + text[insertion_point:]
            messages.append("inserted init-dispatch")

    # 4. Send dispatch at the start of send_message body.
    if _has_block(text, BEGIN_SEND_DISPATCH):
        messages.append("send-dispatch already present")
    else:
        anchor = "def send_message(self, message, memorizing=False, query_id=None, context_id=None):"
        if anchor not in text:
            messages.append("WARN: send_message not found; skipped send-dispatch")
        else:
            def_idx = text.find(anchor)
            line_end = text.find("\n", def_idx)
            insertion_point = line_end + 1
            after_def = text[insertion_point:]
            stripped = after_def.lstrip()
            if stripped.startswith(("\"\"\"", "'''")):
                quote = stripped[:3]
                ws = len(after_def) - len(stripped)
                close_pos = after_def.find(quote, ws + 3)
                if close_pos != -1:
                    end_of_doc_line = after_def.find("\n", close_pos)
                    if end_of_doc_line != -1:
                        insertion_point += end_of_doc_line + 1
            text = text[:insertion_point] + SEND_DISPATCH_BLOCK + text[insertion_point:]
            messages.append("inserted send-dispatch")

    return text, messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_py", type=Path, help="Path to MemoryAgentBench agent.py")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args(argv)

    if not args.agent_py.is_file():
        print(f"error: {args.agent_py} not found", file=sys.stderr)
        return 2

    original = args.agent_py.read_text(encoding="utf-8")
    patched, messages = patch(original)
    for m in messages:
        print(m)

    if patched == original:
        print("no changes needed")
        return 0

    if args.dry_run:
        print("--- dry-run; not writing ---")
        return 0

    backup = args.agent_py.with_suffix(args.agent_py.suffix + ".bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"backup written: {backup}")
    args.agent_py.write_text(patched, encoding="utf-8")
    print(f"patched: {args.agent_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
