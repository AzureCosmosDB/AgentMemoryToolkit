"""Slow live-Cosmos TTL lifecycle checks."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from agent_memory_toolkit import CosmosMemoryClient
from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]


def _registered_markers(config: pytest.Config) -> set[str]:
    return {marker.split(":", 1)[0].strip() for marker in config.getini("markers")}


@pytest.fixture(autouse=True)
def _require_slow_selection(pytestconfig: pytest.Config) -> None:
    if "slow" not in _registered_markers(pytestconfig):
        pytest.skip("slow marker is not registered")
    if "slow" not in (pytestconfig.option.markexpr or ""):
        pytest.skip("Run with pytest -m slow to execute TTL lifecycle checks")


@pytest.fixture(scope="module")
def ttl_client(
    cosmos_endpoint,
    cosmos_key,
    cosmos_database,
    cosmos_container,
):
    if not cosmos_endpoint:
        pytest.skip("COSMOS_DB_ENDPOINT not set")
    return CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key or None,
        cosmos_database=cosmos_database,
        cosmos_container=cosmos_container,
        cosmos_turns_container=os.environ.get("COSMOS_TURNS_CONTAINER") or None,
    )


def _find_doc(client: CosmosMemoryClient, user_id: str, thread_id: str, memory_id: str, memory_types: list[str]):
    for doc in client.get_thread(
        thread_id=thread_id,
        user_id=user_id,
        memory_types=memory_types,
        include_superseded=True,
    ):
        if doc.get("id") == memory_id:
            return doc
    return None


def _delete_if_present(client: CosmosMemoryClient, memory_id: str, user_id: str, thread_id: str) -> None:
    try:
        client.delete_cosmos(memory_id=memory_id, user_id=user_id, thread_id=thread_id)
    except Exception:
        pass


def test_turn_ttl_expires_while_episodic_persists(ttl_client: CosmosMemoryClient) -> None:
    user_id = f"ttl-test-user-{uuid.uuid4().hex[:8]}"
    thread_id = f"ttl-test-thread-{uuid.uuid4().hex[:8]}"
    turn_id = ""
    episodic_id = ""
    try:
        turn_id = ttl_client.add_cosmos(
            user_id=user_id,
            role="user",
            content="temporary turn",
            memory_type="turn",
            thread_id=thread_id,
            ttl=60,
        )
        episodic_id = ttl_client.add_cosmos(
            user_id=user_id,
            role="system",
            content="durable episodic memory",
            memory_type="episodic",
            thread_id=thread_id,
            embed=False,
        )

        time.sleep(90)
        assert _find_doc(ttl_client, user_id, thread_id, turn_id, ["turn"]) is None

        time.sleep(30)
        assert _find_doc(ttl_client, user_id, thread_id, episodic_id, ["episodic"]) is not None
    finally:
        if turn_id:
            _delete_if_present(ttl_client, turn_id, user_id, thread_id)
        if episodic_id:
            _delete_if_present(ttl_client, episodic_id, user_id, thread_id)
