"""Integration tests for the change-feed trigger and counter management.

These tests exercise the end-to-end flow: inserting turn documents and
verifying that counters increment inside the counter container as the
change-feed Function processes them.

Enable by setting both::

    AGENT_MEMORY_RUN_INTEGRATION=true
    AGENT_MEMORY_RUN_CHANGEFEED=true

A running Azure Functions host (deployed or local ``func start``) with the
change-feed trigger configured is required, along with the ``turns``,
``counter``, and ``leases`` containers. Post container-split, the change-feed
trigger binds to the ``turns`` container — turn documents must be inserted
there, not into ``memories``.
"""

import os
import time
import uuid

import pytest

from tests.conftest import INTEGRATION_ENABLED

CHANGEFEED_ENABLED = os.environ.get("AGENT_MEMORY_RUN_CHANGEFEED", "").lower() == "true"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (INTEGRATION_ENABLED and CHANGEFEED_ENABLED),
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true and AGENT_MEMORY_RUN_CHANGEFEED=true",
    ),
]


@pytest.fixture(scope="module")
def cosmos_clients():
    """Cosmos DB container clients for turns and counters.

    Uses ``COSMOS_DB_KEY`` when set (relief while control-plane RBAC is in
    private preview); otherwise falls back to ``DefaultAzureCredential``.
    """
    from azure.cosmos import CosmosClient

    endpoint = os.environ.get("COSMOS_DB__accountEndpoint") or os.environ.get("COSMOS_DB_ENDPOINT")
    database_name = os.environ.get("COSMOS_DB_DATABASE", "ai_memory")
    turns_container_name = os.environ.get("COSMOS_DB_TURNS_CONTAINER", "turns")
    counter_container_name = os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counter")
    cosmos_key = os.environ.get("COSMOS_DB_KEY")

    if cosmos_key:
        client = CosmosClient(endpoint, credential=cosmos_key)
    else:
        from azure.identity import DefaultAzureCredential

        client = CosmosClient(endpoint, credential=DefaultAzureCredential())

    db = client.get_database_client(database_name)
    return (
        db.get_container_client(turns_container_name),
        db.get_container_client(counter_container_name),
    )


@pytest.fixture
def unique_ids():
    """Generate unique user_id and thread_id for test isolation."""
    return {
        "user_id": f"test-user-{uuid.uuid4().hex[:8]}",
        "thread_id": f"test-thread-{uuid.uuid4().hex[:8]}",
    }


class TestChangeFeedIntegration:
    """Integration tests for change feed trigger with live Cosmos DB."""

    def _insert_turn(self, turns_container, user_id, thread_id):
        """Insert a single turn document into the turns container."""
        doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "thread_id": thread_id,
            "role": "user",
            "type": "turn",
            "content": f"Test message {uuid.uuid4().hex[:6]}",
            "metadata": {},
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
        turns_container.upsert_item(body=doc)
        return doc

    def _read_counter(self, counter_container, counter_id, user_id, thread_id):
        """Read a counter document, returning None if not found."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return counter_container.read_item(item=counter_id, partition_key=[user_id, thread_id])
        except CosmosResourceNotFoundError:
            return None

    def test_counter_increments_on_turn_insert(self, cosmos_clients, unique_ids):
        """Insert turn documents and verify the thread counter increments.

        Note: This test depends on the change feed trigger running. It inserts
        documents and then polls the counter container for up to 60 seconds
        waiting for the change feed to process them.
        """
        turns, counters = cosmos_clients
        user_id = unique_ids["user_id"]
        thread_id = unique_ids["thread_id"]
        counter_id = f"thread:{user_id}:{thread_id}"

        # Insert 3 turn documents
        for _ in range(3):
            self._insert_turn(turns, user_id, thread_id)

        # Poll for counter to appear (change feed has latency)
        deadline = time.time() + 60
        counter_doc = None
        while time.time() < deadline:
            counter_doc = self._read_counter(counters, counter_id, user_id, thread_id)
            if counter_doc and counter_doc.get("count", 0) >= 3:
                break
            time.sleep(3)

        assert counter_doc is not None, (
            f"Counter {counter_id} was not created within 60s. Is the change feed trigger running?"
        )
        assert counter_doc["count"] >= 3, f"Expected counter >= 3, got {counter_doc['count']}"
