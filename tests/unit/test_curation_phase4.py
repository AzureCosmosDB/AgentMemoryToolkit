from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from azure.cosmos.agent_memory.store import MemoryStore


def _fact(**overrides):
    doc = {
        "id": "fact_123",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": "Use the launch checklist.",
        "content_hash": "a" * 32,
        "metadata": {"category": "preference"},
        "created_at": "2026-01-01T00:00:00+00:00",
        "tags": ["sys:fact"],
        "tenant_id": "acme",
        "scope_type": "user",
        "scope_id": "alice",
        "scope_key": "user:alice",
        "acl": {"read": ["user:alice"], "write": ["user:alice"], "annotate": [], "forget": ["user:alice"]},
        "provenance": {"created_by": "user:alice"},
    }
    doc.update(overrides)
    return doc


def _containers(*, memories=None):
    return {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }


def test_promote_allows_target_scope_member_and_stamps_target_lineage():
    memories = MagicMock()
    memories.query_items.return_value = [_fact(provenance={"created_by": "user:alice", "source_ids": ["prior"]})]
    store = MemoryStore(containers=_containers(memories=memories))
    # A target member/writer role authorizes the write into team:eng; the source is the
    # caller's own user scope, so the source read check passes too.
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", roles=["team:eng:member"])

    promoted = store.promote("fact_123", "user:alice", "team:eng", ctx)

    body = memories.upsert_item.call_args.kwargs["body"]
    assert promoted == body
    assert body["id"].startswith("fact_123_promoted_")
    assert body["tenant_id"] == "acme"
    assert body["scope_type"] == "team"
    assert body["scope_id"] == "eng"
    assert body["scope_key"] == "team:eng"
    assert body["acl"]["read"] == ["team:eng"]
    assert body["provenance"]["source_ids"] == ["prior", "fact_123"]
    assert body["metadata"]["promoted_from_memory_id"] == "fact_123"
    assert body["metadata"]["promoted_from_scope"] == "user:alice"
    assert body["metadata"]["curation_status"] == "approved"
    assert body["status"] == "approved"


def test_promote_refuses_without_share_or_assign():
    memories = MagicMock()
    store = MemoryStore(containers=_containers(memories=memories))
    ctx = SecurityContext(tenant_id="acme", principal="user:alice")

    with pytest.raises(ValidationError, match="write permission"):
        store.promote("fact_123", "user:alice", "team:eng", ctx)

    memories.query_items.assert_not_called()
    memories.upsert_item.assert_not_called()


def test_promote_refuses_reading_an_unauthorized_source_scope():
    """The target write is authorized but the source scope is not readable -> refuse.

    Without a source read gate, a caller who can write ``team:public`` could copy a record
    out of ``team:secret`` (a scope it cannot read) into a scope it can read.
    """
    memories = MagicMock()
    store = MemoryStore(containers=_containers(memories=memories))
    # writer on the target team:public, but no read grant on the source team:secret.
    ctx = SecurityContext(tenant_id="acme", principal="user:alice", roles=["team:public:writer"])

    with pytest.raises(ValidationError, match="read permission on source scope"):
        store.promote("fact_123", "team:secret", "team:public", ctx)

    memories.query_items.assert_not_called()
    memories.upsert_item.assert_not_called()
def test_promote_refuses_cross_tenant_source_document():
    """Even with target write + source read authorized, a source doc whose tenant differs
    from the caller must be refused - the real cross-tenant guard, now that the dead
    target-scope tenant check is removed."""
    # Source read is authorized (admin), the query returns a doc stamped with a foreign
    # tenant_id; promote must reject it rather than copy across tenants.
    foreign = _fact(tenant_id="other-tenant")
    memories = MagicMock()
    memories.query_items.return_value = [foreign]
    store = MemoryStore(containers=_containers(memories=memories))
    ctx = SecurityContext(tenant_id="acme", principal="user:root", roles=["tenant:admin"])

    with pytest.raises(ValidationError, match="outside the caller tenant"):
        store.promote("fact_123", "user:alice", "team:eng", ctx)

    memories.upsert_item.assert_not_called()


def test_scope_hint_is_metadata_only_and_does_not_change_placement():
    svc = PipelineService.__new__(PipelineService)
    svc._transcript_metadata_keys = None
    svc._prompt_lineage = lambda _filename: {"prompt_id": "p", "prompt_version": "v1"}  # type: ignore[method-assign]
    svc._run_prompty = lambda *_args, **_kwargs: json.dumps(  # type: ignore[method-assign]
        {
            "facts": [
                {
                    "text": "Use the launch checklist.",
                    "category": "preference",
                    "source": "user",
                    "confidence": 0.8,
                    "salience": 0.7,
                    "temporal_context": None,
                    "tags": [],
                    "suggested_scope_type": "team",
                    "scope_confidence": 0.92,
                }
            ]
        }
    )
    turn = {
        "id": "turn1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "remember the launch checklist",
        "metadata": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "tenant_id": "acme",
        "scope_type": "user",
        "scope_id": "alice",
        "scope_key": "user:alice",
        "acl": {"read": ["user:alice"], "write": ["user:alice"], "annotate": [], "forget": ["user:alice"]},
        "provenance": {"created_by": "user:alice"},
    }

    fact = svc.extract_memories_durable("alice", "t1", turns=[turn])["facts"][0]

    assert fact["scope_key"] == "user:alice"
    assert fact["acl"]["read"] == ["user:alice"]
    assert fact["metadata"]["suggested_scope_type"] == "team"
    assert fact["metadata"]["scope_confidence"] == 0.92
    assert fact["status"] == "candidate"


def test_auto_promote_guard_blocks_low_confidence_no_permission_and_pii():
    store = MemoryStore(containers=_containers())
    store.list_promotion_candidates = MagicMock(  # type: ignore[method-assign]
        return_value=[
            _fact(
                id="low",
                metadata={"category": "preference", "suggested_scope_type": "team", "scope_confidence": 0.2},
            ),
            _fact(
                id="noperm",
                metadata={"category": "preference", "suggested_scope_type": "team", "scope_confidence": 0.99},
            ),
            _fact(
                id="pii",
                metadata={
                    "category": "preference",
                    "suggested_scope_type": "team",
                    "scope_confidence": 0.99,
                    "contains_pii": True,
                },
            ),
        ]
    )
    ctx = SecurityContext(tenant_id="acme", principal="user:alice")

    results = store.auto_promote_candidates(ctx, target_scopes_by_type={"team": "team:eng"}, confidence_threshold=0.9)

    assert [(r["memory_id"], r["reason"]) for r in results] == [
        ("low", "low_confidence"),
        ("noperm", "permission_denied"),
        ("pii", "pii_or_secret"),
    ]


def test_auto_promote_promotes_when_policy_guards_pass():
    candidate = _fact(metadata={"category": "preference", "suggested_scope_type": "team", "scope_confidence": 0.99})
    memories = MagicMock()
    memories.query_items.side_effect = [[candidate], [candidate]]
    store = MemoryStore(containers=_containers(memories=memories))
    ctx = SecurityContext(tenant_id="acme", principal="user:admin", roles=["tenant:admin"])

    results = store.auto_promote_candidates(ctx, target_scopes_by_type={"team": "team:eng"}, confidence_threshold=0.9)

    assert results[0]["status"] == "promoted"
    body = memories.upsert_item.call_args.kwargs["body"]
    assert body["scope_key"] == "team:eng"


def test_normalize_scope_hint_accepts_full_scope_enum():
    from azure.cosmos.agent_memory._curation import normalize_scope_hint

    for scope_type in ["user", "agent", "team", "project", "org", "global"]:
        assert normalize_scope_hint({"suggested_scope_type": scope_type, "scope_confidence": 0.7}) == (scope_type, 0.7)
    # Absent hint is tolerated (returns None) for backward compatibility.
    assert normalize_scope_hint({}) is None
    # Confidence is required whenever a type is present.
    with pytest.raises(ValidationError):
        normalize_scope_hint({"suggested_scope_type": "team"})
    # tenant_id is not a scope and must be rejected as a hint.
    with pytest.raises(ValidationError):
        normalize_scope_hint({"suggested_scope_type": "tenant", "scope_confidence": 0.5})


def test_is_promotable_scope_hint_excludes_user_only():
    from azure.cosmos.agent_memory._curation import PERSONAL_SCOPE_TYPE, is_promotable_scope_hint

    assert PERSONAL_SCOPE_TYPE == "user"
    assert is_promotable_scope_hint("user") is False
    for scope_type in ["agent", "team", "project", "org", "global"]:
        assert is_promotable_scope_hint(scope_type) is True


def test_list_promotion_candidates_query_excludes_user_hint():
    store = MemoryStore.__new__(MemoryStore)
    captured: dict[str, object] = {}

    class _Res:
        allowed_scopes = {"user:alice"}

    store._resolve_scope_action = lambda ctx, scope, action: _Res()  # type: ignore[attr-defined]
    store._memories_container = object()  # type: ignore[attr-defined]

    def _capture(**kwargs):
        captured.update(kwargs)
        return []

    store._query_items = _capture  # type: ignore[attr-defined]
    ctx = SecurityContext(tenant_id="acme", principal="user:alice")

    store.list_promotion_candidates(ctx, from_scope="user:alice")

    query = str(captured["query"])
    params = {p["name"]: p["value"] for p in captured["parameters"]}  # type: ignore[index]
    assert "c.metadata.suggested_scope_type != @personal_scope" in query
    assert params["@personal_scope"] == "user"
