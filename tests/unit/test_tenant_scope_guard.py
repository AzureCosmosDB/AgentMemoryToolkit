"""Tenant scope guard: no code path may read records without the tenant, or build a raw partition key.

A query that names ``c.user_id`` alone reads the same id in every tenant, and a hand-built ``[user_id, ...]`` key
addresses the wrong partition. These checks read the source with ``ast``, so a new query written without the tenant
fails here before it can leak. They are a safety net: what proves that one tenant cannot read another's records are the
behaviour tests of the store and the pipeline.

In every source file the checks enforce that:

* Each query carries the tenant filter in its own expression: ``OWNER_FILTER``, ``TENANT_FILTER``, SQL that reads
  ``c.tenant_id = @...``, ``_QueryBuilder.build_where()``, a ``predicate``, or a variable assigned from one of these.
  A parameter, a docstring, a comment or another query of the same function that mentions the tenant does not count.
  A query is a string that selects ``FROM`` a container, or one that filters on ``user_id``.
* A ``predicate`` (a name or property ending in ``predicate``) is itself built from those filters, so the word cannot
  carry an unscoped condition into a query.
* A ``_QueryBuilder`` is made with ``for_tenant``, which starts every WHERE clause with the tenant.
* A partition key is made with ``partition_key_for``, never written out as a list or a tuple.

The checks find where the tenant filter is, not whether the rest of a WHERE clause keeps it: an ``OR`` that widens
one is for the behaviour tests to catch.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(
    path
    for base in (ROOT / "azure" / "cosmos" / "agent_memory", ROOT / "function_app")
    for path in base.rglob("*.py")
    if "prompts" not in path.parts and path.name != "_partitioning.py"
)

# A query selects from a container (SQL keywords are case-insensitive). A string that filters on the owner counts as one
# too, since it can be the piece of a query that names the owner.
SELECT_FROM = re.compile(r"\bSELECT\b[\s\S]*?\bFROM\s+\w+", re.IGNORECASE)
OWNER_IN_SQL = re.compile(r"\.user_id\s*=\s*@\w+")
# A string can only be a query if one of its lines has this. Reading the source of a string splits the whole file, so
# only these strings are read.
QUERY_LINE = re.compile(r"select|\.user_id", re.IGNORECASE)
# A ``\n`` written inside a literal separates the words of the SQL like a space does.
ESCAPED_SPACE = re.compile(r"\\[nrt]")
# SQL that filters on the tenant, whatever the alias: ``c.tenant_id = @tenant_id``.
TENANT_IN_SQL = re.compile(r"\.tenant_id\s*=\s*@\w+")
TENANT_FILTERS = frozenset({"OWNER_FILTER", "TENANT_FILTER"})

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPES = (*FUNCTIONS, ast.Lambda, ast.Module)

Check = Callable[[str], list[tuple[int, str]]]


def _is_string(node: ast.AST | None) -> bool:
    """A string expression: a literal, an f-string, or a ``+`` chain with one of those in it."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_string(node.left) or _is_string(node.right)
    return isinstance(node, ast.JoinedStr)


def _called(call: ast.Call) -> str:
    """The name a call goes by: ``f`` for ``f()`` and ``g`` for ``a.g()``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    return call.func.attr if isinstance(call.func, ast.Attribute) else ""


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of ``scope`` without those inside the functions defined in it."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, SCOPES):
            pending.extend(ast.iter_child_nodes(node))


def _assignments(node: ast.AST) -> Iterator[tuple[ast.Name | ast.Attribute, ast.expr | None, int | None]]:
    """The names and attributes an assignment binds, each with its value and its place when it is unpacked.

    An augmented assignment (``predicate += " AND ..."``) is left out: it adds to a value assigned elsewhere.
    """
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign):
        targets, value = [node.target], node.value
    else:
        return
    for target in targets:
        places = enumerate(target.elts) if isinstance(target, (ast.Tuple, ast.List)) else [(None, target)]
        for place, element in places:
            if isinstance(element, (ast.Name, ast.Attribute)):
                yield element, value, place


def _name_of(target: ast.Name | ast.Attribute) -> str:
    return target.id if isinstance(target, ast.Name) else target.attr


def _bindings(scope: ast.AST, name: str) -> list[tuple[ast.expr | None, int | None]]:
    """What ``name`` is bound to in ``scope``: the values assigned to it and their places, ``None`` for a parameter."""
    bound: list[tuple[ast.expr | None, int | None]] = []
    if isinstance(scope, (*FUNCTIONS, ast.Lambda)):
        arguments = scope.args
        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
        bound += [(None, None) for parameter in parameters if parameter and parameter.arg == name]
    for node in _own_nodes(scope):
        bound += [
            (value, place)
            for target, value, place in _assignments(node)
            if isinstance(target, ast.Name) and target.id == name
        ]
    return bound


class _Source:
    """A parsed source file, with what the checks look up in it."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tree = ast.parse(text)
        self.parents = {child: parent for parent in ast.walk(self.tree) for child in ast.iter_child_nodes(parent)}
        self.query_lines = {number for number, line in enumerate(text.split("\n"), 1) if QUERY_LINE.search(line)}
        self.functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, FUNCTIONS):
                self.functions.setdefault(node.name, []).append(node)

    def snippet(self, node: ast.AST) -> str:
        return " ".join((ast.get_source_segment(self.text, node) or "").split())[:80]

    def scope_of(self, node: ast.AST) -> ast.AST:
        """The function, lambda or module whose names ``node`` sees."""
        while not isinstance(node, SCOPES):
            node = self.parents[node]
        return node

    def queries(self) -> list[ast.expr]:
        """The outermost string expressions that read records, leaving out docstrings and other bare strings."""
        found: list[ast.expr] = []
        for node in ast.walk(self.tree):
            parent = self.parents.get(node)
            if not _is_string(node) or _is_string(parent) or isinstance(parent, ast.Expr):
                continue
            if self.query_lines.isdisjoint(range(node.lineno, node.end_lineno + 1)):
                continue
            text = ESCAPED_SPACE.sub(" ", ast.get_source_segment(self.text, node) or "")
            if SELECT_FROM.search(text) or OWNER_IN_SQL.search(text):
                found.append(node)
        return sorted(found, key=lambda node: (node.lineno, node.col_offset))

    def carries(
        self, node: ast.expr | None, scope: ast.AST, place: int | None = None, seen: frozenset = frozenset()
    ) -> bool:
        """Whether the SQL that ``node`` evaluates to filters on the tenant, judged from its own expression.

        ``place`` picks one value out of a tuple (``predicate, parameters = ...``). A name is judged by everything
        assigned to it in its function or the module, and a call to a helper of this file by what the helper returns.
        """
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str) and TENANT_IN_SQL.search(node.value) is not None
        if isinstance(node, ast.JoinedStr):
            parts = (part.value if isinstance(part, ast.FormattedValue) else part for part in node.values)
            return any(self.carries(part, scope, None, seen) for part in parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.carries(node.left, scope, None, seen) or self.carries(node.right, scope, None, seen)
        if isinstance(node, ast.Tuple) and place is not None:
            return place < len(node.elts) and self.carries(node.elts[place], scope, None, seen)
        if isinstance(node, ast.Name):
            return node.id in TENANT_FILTERS or self._name_carries(node.id, scope, seen)
        if isinstance(node, ast.Attribute):
            return node.attr in TENANT_FILTERS or node.attr.endswith("predicate")
        if isinstance(node, ast.Call):
            return self._call_carries(node, place, seen)
        return False

    def _name_carries(self, name: str, scope: ast.AST, seen: frozenset) -> bool:
        while True:
            bound = _bindings(scope, name)
            if bound:
                key = (id(scope), name)
                # A value built on its own name adds to what the other assignments carry.
                return key in seen or all(self.carries(value, scope, place, seen | {key}) for value, place in bound)
            if isinstance(scope, ast.Module):
                return False
            scope = self.scope_of(self.parents[scope])

    def _call_carries(self, call: ast.Call, place: int | None, seen: frozenset) -> bool:
        name = _called(call)
        if name == "build_where" or name.endswith("predicate"):
            return True
        helpers = self.functions.get(name, [])
        return bool(helpers) and all(self._returns_carry(helper, place, seen) for helper in helpers)

    def _returns_carry(self, function: ast.AST, place: int | None, seen: frozenset) -> bool:
        if id(function) in seen:
            return False
        returned = [node.value for node in _own_nodes(function) if isinstance(node, ast.Return)]
        return bool(returned) and all(self.carries(value, function, place, seen | {id(function)}) for value in returned)


def unscoped_queries(source: str) -> list[tuple[int, str]]:
    """The queries that do not filter on the tenant in their own expression, as ``(line, snippet)``."""
    src = _Source(source)
    return [
        (query.lineno, src.snippet(query)) for query in src.queries() if not src.carries(query, src.scope_of(query))
    ]


def unscoped_predicates(source: str) -> list[tuple[int, str]]:
    """The predicates that are not built from the tenant filter, as ``(line, snippet)``.

    A predicate is a name or attribute assigned, or a function or property defined, with a name ending in
    ``predicate``. A query takes one in place of the tenant filter, so each must hold that filter itself.
    """
    src = _Source(source)
    found: set[tuple[int, str]] = set()
    for node in ast.walk(src.tree):
        for target, value, place in _assignments(node):
            if _name_of(target).endswith("predicate") and not src.carries(value, src.scope_of(node), place):
                found.add((node.lineno, src.snippet(node)))
        if isinstance(node, FUNCTIONS) and node.name.endswith("predicate"):
            for returned in (child for child in _own_nodes(node) if isinstance(child, ast.Return)):
                if not src.carries(returned.value, node):
                    found.add((returned.lineno, src.snippet(returned)))
    return sorted(found)


def unscoped_query_builders(source: str) -> list[tuple[int, str]]:
    """The ``_QueryBuilder``s made without ``for_tenant``, as ``(line, snippet)``.

    ``build_where()`` counts as the tenant filter because ``for_tenant`` starts every builder with it.
    """
    src = _Source(source)
    calls = [node for node in ast.walk(src.tree) if isinstance(node, ast.Call) and _called(node) == "_QueryBuilder"]
    return sorted((call.lineno, src.snippet(call)) for call in calls)


def _is_partition_key(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        name = target.id
    elif isinstance(target, ast.Attribute):
        name = target.attr
    elif isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
        name = str(target.slice.value)
    else:
        return False
    return "partition_key" in name


def _partition_key_values(node: ast.AST) -> Iterator[ast.expr]:
    """What ``node`` sets as a partition key: ``partition_key=...``, ``partition_key = ...`` or a dict entry."""
    if isinstance(node, ast.keyword):
        if node.arg and "partition_key" in node.arg:
            yield node.value
    elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(_is_partition_key(target) for target in targets):
            yield node.value
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and "partition_key" in str(key.value):
                yield value


def _is_hand_built(value: ast.expr) -> bool:
    """A key written out as a list or a tuple, or made by ``list(...)`` or ``tuple(...)``, also in a conditional."""
    if isinstance(value, ast.IfExp):
        return _is_hand_built(value.body) or _is_hand_built(value.orelse)
    if isinstance(value, ast.BoolOp):
        return any(_is_hand_built(operand) for operand in value.values)
    if isinstance(value, ast.Call):
        return isinstance(value.func, ast.Name) and value.func.id in {"list", "tuple"}
    return isinstance(value, (ast.List, ast.Tuple))


def hand_built_partition_keys(source: str) -> list[tuple[int, str]]:
    """The partition keys written out as a list or a tuple instead of ``partition_key_for``, as ``(line, snippet)``."""
    src = _Source(source)
    values = (value for node in ast.walk(src.tree) for value in _partition_key_values(node))
    return sorted((value.lineno, src.snippet(value)) for value in values if _is_hand_built(value))


CHECKS: tuple[Check, ...] = (
    unscoped_queries,
    unscoped_predicates,
    unscoped_query_builders,
    hand_built_partition_keys,
)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _assert_nothing_flagged(path: Path, check: Check, advice: str) -> None:
    found = check(path.read_text(encoding="utf-8"))
    assert not found, advice + "".join(f"\n  {_relative(path)}:{line}: {snippet}" for line, snippet in found)


@pytest.mark.parametrize("path", SOURCES, ids=_relative)
def test_every_query_filters_on_the_tenant(path: Path) -> None:
    advice = "a query must carry the tenant filter itself: use OWNER_FILTER, TENANT_FILTER or _QueryBuilder.for_tenant"
    _assert_nothing_flagged(path, unscoped_queries, advice)


@pytest.mark.parametrize("path", SOURCES, ids=_relative)
def test_every_predicate_is_built_from_the_tenant_filter(path: Path) -> None:
    _assert_nothing_flagged(path, unscoped_predicates, "a predicate must be built from OWNER_FILTER or TENANT_FILTER")


@pytest.mark.parametrize("path", SOURCES, ids=_relative)
def test_every_query_builder_starts_from_the_tenant(path: Path) -> None:
    _assert_nothing_flagged(path, unscoped_query_builders, "make a _QueryBuilder with _QueryBuilder.for_tenant")


@pytest.mark.parametrize("path", SOURCES, ids=_relative)
def test_no_partition_key_is_built_by_hand(path: Path) -> None:
    _assert_nothing_flagged(path, hand_built_partition_keys, "build a partition key with partition_key_for")


def test_the_checks_see_the_queries_of_the_sources() -> None:
    """Without queries to look at, the checks above would pass by finding nothing."""
    assert sum(len(_Source(path.read_text(encoding="utf-8")).queries()) for path in SOURCES) >= 30


def _marked(source: str) -> list[int]:
    """The lines of a fixture that end in ``# flagged``."""
    return [number for number, line in enumerate(source.splitlines(), start=1) if line.endswith("# flagged")]


# Each source is run through a check, which must flag the lines marked ``# flagged`` and no others.
FLAGGED = [
    pytest.param(
        unscoped_queries,
        '''
def facts(container, tenant_id, user_id):
    """Read the facts of ``user_id`` in the tenant ``tenant_id``."""
    # The tenant is applied by the caller.
    return container.query_items(
        query=f"SELECT * FROM c WHERE c.user_id = @user_id",  # flagged
        parameters=[{"name": "@user_id", "value": user_id}],
    )
''',
        id="query: the tenant is only in a parameter, a docstring and a comment",
    ),
    pytest.param(
        unscoped_queries,
        """
def reads(container, user_id):
    scoped = f"SELECT * FROM c WHERE {OWNER_FILTER} AND c.type = 'turn'"
    unscoped = "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'fact'"  # flagged
    return container.query_items(query=scoped), container.query_items(query=unscoped)
""",
        id="query: a scoped query does not scope another one in the function",
    ),
    pytest.param(
        unscoped_queries,
        """
QUERY = "SELECT * FROM memories m WHERE m.user_id = @user_id"  # flagged
""",
        id="query: another alias",
    ),
    pytest.param(
        unscoped_queries,
        """
def reads(extra):
    first = "SELECT * FROM c WHERE c.user_id = @user_id" + extra  # flagged
    second = "SELECT * FROM c " + "WHERE c.user_id = @user_id"  # flagged
    return first, second
""",
        id="query: concatenation",
    ),
    pytest.param(
        unscoped_queries,
        """
QUERY = "select * from c where c.user_id = @user_id"  # flagged
""",
        id="query: lower case SQL",
    ),
    pytest.param(
        unscoped_queries,
        """
QUERY = "SELECT c.id\\nFROM c WHERE c.type = 'fact'"  # flagged
""",
        id="query: an escaped newline in the literal",
    ),
    pytest.param(
        unscoped_queries,
        """
def owner_clause():
    return "c.user_id = @user_id AND c.type = 'fact'"  # flagged
""",
        id="query: a piece that filters on the owner",
    ),
    pytest.param(
        unscoped_queries,
        """
QUERY = "SELECT * FROM c WHERE {OWNER_FILTER} AND c.type = 'turn'"  # flagged
""",
        id="query: OWNER_FILTER in a string that is not an f-string",
    ),
    pytest.param(
        unscoped_queries,
        """
def reads(predicate):
    where = " WHERE c.type = 'fact'"
    first = f"SELECT * FROM c{where}"  # flagged
    second = f"SELECT * FROM c WHERE {predicate}"  # flagged
    return first, second
""",
        id="query: a variable is judged by its assignments and a parameter is not trusted",
    ),
    pytest.param(
        unscoped_predicates,
        """
def reads(user_id):
    predicate = "c.user_id = @user_id"  # flagged
    owner_predicate = "c.user_id = @user_id AND c.type = 'fact'"  # flagged
    return predicate, owner_predicate
""",
        id="predicate: assigned from a user-only filter",
    ),
    pytest.param(
        unscoped_predicates,
        """
class Scope:
    @property
    def predicate(self):
        return "c.user_id = @user_id"  # flagged
""",
        id="predicate: a property that returns a user-only filter",
    ),
    pytest.param(
        unscoped_predicates,
        """
class Scope:
    predicate: str  # flagged

    def __init__(self, predicate):
        self.predicate = predicate  # flagged
""",
        id="predicate: a field or an attribute set from a parameter",
    ),
    pytest.param(
        unscoped_predicates,
        """
def in_force():
    return "c.type = 'fact'", []


def reads():
    predicate, parameters = in_force()  # flagged
    return predicate, parameters
""",
        id="predicate: unpacked from a helper that returns an unscoped filter",
    ),
    pytest.param(
        unscoped_query_builders,
        """
def reads(user_id):
    qb = _QueryBuilder()  # flagged
    qb.add_filter("c.user_id", "@user_id", user_id)
    return f"SELECT * FROM c{qb.build_where()}"
""",
        id="builder: made without for_tenant",
    ),
    pytest.param(
        hand_built_partition_keys,
        """
def reads(container, tenant_id, user_id, parts):
    container.read_item(item="a", partition_key=[tenant_id, user_id])  # flagged
    container.read_item(item="b", partition_key=(tenant_id, user_id))  # flagged
    container.read_item(item="c", partition_key=list(parts))  # flagged
    container.read_item(item="d", partition_key=parts if user_id else [tenant_id])  # flagged
""",
        id="partition key: a list, a tuple, list() and a conditional",
    ),
    pytest.param(
        hand_built_partition_keys,
        """
def reads(tenant_id, user_id):
    partition_key = [tenant_id, user_id]  # flagged
    options = {"partition_key": [tenant_id, user_id]}  # flagged
    options["partition_key"] = (tenant_id, user_id)  # flagged
    return partition_key, options
""",
        id="partition key: assigned, in a dict or under a key",
    ),
]

SCOPED = [
    pytest.param(
        """
def turns(container):
    return container.query_items(query=f"SELECT * FROM c WHERE {OWNER_FILTER} AND c.type = 'turn'")
""",
        id="OWNER_FILTER",
    ),
    pytest.param(
        """
QUERY = "SELECT * FROM c WHERE c.tenant_id = @tenant_id AND c.user_id = @user_id"
""",
        id="the tenant and owner filters written out",
    ),
    pytest.param(
        """
QUERY = "SELECT * FROM memories m WHERE m.tenant_id = @tenant_id AND m.user_id = @user_id"
""",
        id="another alias",
    ),
    pytest.param(
        """
def reads(qb):
    first = f"SELECT * FROM c{qb.build_where()}"
    second = f"SELECT * FROM c WHERE {TENANT_FILTER}"
    return first, second
""",
        id="build_where() and TENANT_FILTER",
    ),
    pytest.param(
        """
def reads(scope):
    return f"SELECT * FROM c WHERE {scope.predicate} AND c.type = @type"
""",
        id="the predicate of a scope",
    ),
    pytest.param(
        """
def reads():
    first = "SELECT * FROM c WHERE " + OWNER_FILTER + " ORDER BY c._ts"
    second = (
        "SELECT TOP 5 * FROM c "
        f"WHERE {OWNER_FILTER} AND c.type = 'turn' "
        "ORDER BY c.created_at DESC"
    )
    return first, second
""",
        id="concatenation and adjacent literals",
    ),
    pytest.param(
        """
def reads(qb):
    where = qb.build_where()
    return f"SELECT * FROM c{where}"
""",
        id="a WHERE clause held in a variable",
    ),
    pytest.param(
        """
def reads(since):
    query_predicate = OWNER_FILTER
    if since:
        query_predicate += " AND c.created_at > @since"
    return f"SELECT * FROM c WHERE {query_predicate}"
""",
        id="a predicate held in a variable and narrowed",
    ),
    pytest.param(
        """
class Scope:
    @property
    def predicate(self):
        return OWNER_FILTER


def in_force(scope):
    predicate = f"{scope.predicate} AND c.type = @type"
    return predicate, []


def reads(scope):
    predicate, parameters = in_force(scope)
    return f"SELECT * FROM c WHERE {predicate}", parameters
""",
        id="a predicate from a property and a helper",
    ),
    pytest.param(
        '''
def reads(container, tenant_id, user_id, count):
    """Runs SELECT * FROM c WHERE c.user_id = @user_id, but only in this sentence."""
    qb = _QueryBuilder.for_tenant(tenant_id, user_id)
    container.patch_item(
        item="a",
        partition_key=partition_key_for(tenant_id, user_id),
        filter_predicate=f"FROM c WHERE c.n < {count}",
    )
    return qb
''',
        id="a docstring, a patch condition and for_tenant",
    ),
    pytest.param(
        """
def reads(container, tenant_id, user_id, key):
    partition_key = PartitionKey(path=list(PARTITION_KEY_PATHS), kind="MultiHash")
    options = {"partition_key": key}
    options["partition_key"] = key
    container.read_item(item="a", partition_key=partition_key_for(tenant_id, user_id))
    container.read_item(item="b", partition_key=self._key)
    container.query_items(query=q, partition_key=None)
""",
        id="partition keys from the helper, a name, an attribute and None",
    ),
]


@pytest.mark.parametrize(("check", "source"), FLAGGED)
def test_the_checks_flag_exactly_the_marked_lines(check: Check, source: str) -> None:
    assert [line for line, _ in check(source)] == _marked(source)


@pytest.mark.parametrize("source", SCOPED)
def test_scoped_code_is_not_flagged(source: str) -> None:
    assert {check.__name__: check(source) for check in CHECKS} == {check.__name__: [] for check in CHECKS}


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
