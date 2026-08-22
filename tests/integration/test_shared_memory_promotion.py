"""End-to-end curation / promotion tests (Layer 4 / Phase 4).

Covers manual ``promote`` gated by share/assign, the advisory nature of scope
hints (they never move placement on their own), the manual review queue
(``list_promotion_candidates`` / ``approve_promotion``), and policy
``auto_promote_candidates`` with its confidence threshold and PII exclusion.
"""

from __future__ import annotations

import uuid

import pytest

from azure.cosmos.agent_memory.exceptions import ValidationError
from tests.conftest import INTEGRATION_ENABLED
from tests.integration.conftest import contents

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _has(results: list[dict], needle: str) -> bool:
    return any(needle in c for c in contents(results))


def _hint(scope_type: str, confidence: float, **extra) -> dict:
    return {"suggested_scope_type": scope_type, "scope_confidence": confidence, **extra}


# ---------------------------------------------------------------------------
# Manual promote
# ---------------------------------------------------------------------------


def test_promote_copies_private_record_into_team_and_is_readable(
    make_ctx, write_fact, shared_client, read_record, search_until, foundry_live
):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    marker = f"PROMOTE-{uuid.uuid4().hex[:8]} we standardized on pnpm for all repos"

    memory_id = write_fact(author, "user:alice", marker)
    promoted = shared_client.promote(memory_id, "user:alice", team, author)

    assert promoted["scope_key"] == team
    assert promoted["provenance"]["source"] == "promotion"
    assert memory_id in promoted["provenance"]["source_ids"]
    assert promoted["metadata"]["promoted_from_scope"] == "user:alice"

    teammate = make_ctx("user:carol", groups=[gid])
    got = search_until(teammate, "standardized on pnpm for all repos", lambda r: _has(r, marker))
    assert _has(got, marker), "a teammate must read the promoted (team-scoped) copy"


def test_promote_denied_without_share_permission(make_ctx, write_fact, shared_client, foundry_live):
    author = make_ctx("user:alice")
    memory_id = write_fact(author, "user:alice", f"NOSHARE-{uuid.uuid4().hex[:6]} personal fact")

    foreign_team = f"team:eng-{uuid.uuid4().hex[:8]}"  # alice is NOT a member
    with pytest.raises(ValidationError):
        shared_client.promote(memory_id, "user:alice", foreign_team, author)


# ---------------------------------------------------------------------------
# Advisory hints + review queue
# ---------------------------------------------------------------------------


def test_scope_hint_is_advisory_and_never_moves_placement(make_ctx, shared_client, read_record, foundry_live):
    author = make_ctx("user:alice")
    marker = f"ADVISORY-{uuid.uuid4().hex[:6]} the team uses trunk-based development"
    memory_id = shared_client.session(author, write_scope="user:alice").upsert_memory(
        role="user", content=marker, memory_type="fact", thread_id="t1", metadata=_hint("team", 0.98)
    )
    doc = read_record(tenant_id=author.tenant_id, scope_key="user:alice", thread_id="t1", memory_id=memory_id)
    assert doc["scope_key"] == "user:alice", "a scope hint must NOT relocate the record; placement stays private"
    assert doc["metadata"]["suggested_scope_type"] == "team"


def test_list_promotion_candidates_surfaces_only_hinted_records(make_ctx, shared_client, foundry_live):
    author = make_ctx("user:alice")
    session = shared_client.session(author, write_scope="user:alice")
    hinted_marker = f"CAND-{uuid.uuid4().hex[:8]} deploys go through the release channel"
    session.upsert_memory(
        role="user", content=hinted_marker, memory_type="fact", thread_id="t1", metadata=_hint("team", 0.96)
    )
    session.upsert_memory(
        role="user", content=f"PLAIN-{uuid.uuid4().hex[:8]} unhinted note", memory_type="fact", thread_id="t1"
    )

    candidates = shared_client.list_promotion_candidates(author)
    texts = contents(candidates)
    assert any(hinted_marker in t for t in texts), "hinted record must appear in the review queue"
    assert all("unhinted note" not in t for t in texts), "an unhinted record must not appear"


def test_approve_promotion_promotes_a_queued_candidate(
    make_ctx, shared_client, search_until, foundry_live
):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    marker = f"APPROVE-{uuid.uuid4().hex[:8]} incident retros happen within 48 hours"
    memory_id = shared_client.session(author, write_scope="user:alice").upsert_memory(
        role="user", content=marker, memory_type="fact", thread_id="t1", metadata=_hint("team", 0.97)
    )

    shared_client.approve_promotion(memory_id, from_scope="user:alice", to_scope=team, ctx=author)

    teammate = make_ctx("user:dan", groups=[gid])
    assert _has(search_until(teammate, "incident retros within 48 hours", lambda r: _has(r, marker)), marker)


# ---------------------------------------------------------------------------
# Policy auto-promote
# ---------------------------------------------------------------------------


def test_auto_promote_respects_confidence_threshold(make_ctx, shared_client, foundry_live):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    session = shared_client.session(author, write_scope="user:alice")

    high = f"AUTOHI-{uuid.uuid4().hex[:8]} we cut releases every other tuesday"
    high_id = session.upsert_memory(
        role="user", content=high, memory_type="fact", thread_id="t1", metadata=_hint("team", 0.99)
    )
    low = f"AUTOLO-{uuid.uuid4().hex[:8]} maybe we should try bun someday"
    session.upsert_memory(
        role="user", content=low, memory_type="fact", thread_id="t1", metadata=_hint("team", 0.60)
    )

    results = shared_client.auto_promote_candidates(
        author, target_scopes_by_type={"team": team}, confidence_threshold=0.95
    )
    promoted_ids = {r["memory_id"] for r in results if r["status"] == "promoted"}
    assert high_id in promoted_ids, "a high-confidence hint must be auto-promoted"
    # The low-confidence record is filtered before it is ever a candidate for promotion.
    assert all(r["status"] == "promoted" for r in results), "only eligible (high-confidence) records are returned"


def test_auto_promote_excludes_pii_flagged_records(make_ctx, shared_client, foundry_live):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid])
    marker = f"AUTOPII-{uuid.uuid4().hex[:8]} employee SSN is 111-22-3333"
    pii_id = shared_client.session(author, write_scope="user:alice").upsert_memory(
        role="user",
        content=marker,
        memory_type="fact",
        thread_id="t1",
        metadata=_hint("team", 0.99, pii=True),
    )

    results = shared_client.auto_promote_candidates(
        author, target_scopes_by_type={"team": team}, confidence_threshold=0.95
    )
    by_id = {r["memory_id"]: r for r in results}
    assert by_id.get(pii_id, {}).get("status") == "skipped"
    assert by_id[pii_id]["reason"] == "pii_or_secret"
