"""End-to-end user / thread summary + extraction tests under a real (non-default) tenant.

Thread and user summaries are user-scoped: the pipeline reads the user's turns
and stamps the summary with ``scope_key = user:<id>`` plus provenance. These now
run under an isolated ``tenant_id`` sourced from the ``SecurityContext`` (via the
bound session), proving the pipeline is tenant-aware and no longer pinned to the
default tenant. Requires the live LLM.
"""

from __future__ import annotations

import pytest

from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


def _add_turns(shared_client, ctx, thread_id, turns):
    session = shared_client.session(ctx)
    for role, content in turns:
        session.upsert_memory(role=role, content=content, memory_type="turn", thread_id=thread_id)


def test_thread_summary_is_generated_user_scoped_and_tenant_scoped(shared_client, make_ctx, tenant, foundry_live):
    author = make_ctx("user:alice")  # tenant == the isolated test tenant
    user_id = "alice"
    thread_id = "thread-paris"
    _add_turns(
        shared_client,
        author,
        thread_id,
        [
            ("user", "I'm planning a trip to Paris in the spring."),
            ("agent", "Great - Paris in spring is lovely. Museums or food?"),
            ("user", "Mostly food. I love pastries and want the best bakeries."),
            ("agent", "I'll focus on renowned bakeries and patisseries in Paris."),
        ],
    )

    summary = shared_client.session(author).generate_thread_summary(user_id, thread_id)
    assert summary["type"] == "thread_summary"
    assert summary["content"], "summary content must be non-empty"
    assert summary["scope_key"] == f"user:{user_id}"
    assert summary["tenant_id"] == tenant, "the summary must land in the caller's tenant, not 'default'"
    assert summary["provenance"]["created_by"] == f"user:{user_id}"

    # Read-back through the tenant-aware bound session getter.
    fetched = shared_client.session(author).get_thread_summary(user_id, thread_id)
    assert any(doc["id"] == summary["id"] for doc in fetched), "the persisted summary must be retrievable in-tenant"

    # A different tenant must not see this user's summary.
    other = make_ctx("user:alice", tenant_id="it-other-" + tenant)
    assert not shared_client.session(other).get_thread_summary(user_id, thread_id)


def test_user_summary_spans_threads_and_is_tenant_scoped(shared_client, make_ctx, tenant, foundry_live):
    author = make_ctx("user:alice")
    user_id = "alice"
    _add_turns(
        shared_client,
        author,
        "thread-food",
        [
            ("user", "I am vegetarian and allergic to peanuts."),
            ("agent", "Understood - no meat and no peanuts."),
        ],
    )
    _add_turns(
        shared_client,
        author,
        "thread-travel",
        [
            ("user", "I prefer window seats and travel carry-on only."),
            ("agent", "Noted - window seats and carry-on only."),
        ],
    )
    session = shared_client.session(author)
    session.generate_thread_summary(user_id, "thread-food")
    session.generate_thread_summary(user_id, "thread-travel")

    user_summary = session.generate_user_summary(user_id, ["thread-food", "thread-travel"])
    assert user_summary["type"] == "user_summary"
    assert user_summary["content"], "user summary content must be non-empty"
    assert user_summary["scope_key"] == f"user:{user_id}"
    assert user_summary["tenant_id"] == tenant
    assert user_summary["provenance"]["created_by"] == f"user:{user_id}"


def test_extraction_is_tenant_scoped(shared_client, make_ctx, tenant, foundry_live):
    """Fact extraction reads turns from the caller's tenant and writes facts back there."""
    author = make_ctx("user:alice")
    user_id = "alice"
    thread_id = "thread-prefs"
    _add_turns(
        shared_client,
        author,
        thread_id,
        [
            ("user", "Always deploy to the staging slot before production."),
            ("agent", "Got it - staging before production, every time."),
        ],
    )

    counts = shared_client.session(author).extract_memories(user_id, thread_id)
    assert counts.get("fact_count", 0) >= 1, f"expected at least one extracted fact, got {counts}"

    # The extracted facts must be readable in-tenant.
    in_tenant = shared_client.session(author).search_cosmos("deploy staging before production", top_k=10)
    assert any("staging" in str(r.get("content", "")).lower() for r in in_tenant)
