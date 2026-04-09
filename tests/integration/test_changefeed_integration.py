"""Integration tests for the change feed trigger and counter management.

These tests exercise the end-to-end flow: inserting turn documents,
verifying counters increment, and verifying orchestrations are started
at threshold crossings.

Enable by setting::

    AGENT_MEMORY_RUN_INTEGRATION=true

Requires a running Azure Functions host with the change feed trigger
configured and Cosmos DB containers (memories, counters, leases) provisioned.
"""

import os
import sys
import time
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "azure_functions"))

from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]


@pytest.fixture(scope="module")
def cosmos_clients():
    """Create Cosmos DB container clients for memories and counters."""
    from azure.cosmos import CosmosClient
    from azure.identity import DefaultAzureCredential

    endpoint = os.environ["COSMOS_DB_ENDPOINT"]
    database_name = os.environ.get("COSMOS_DB_DATABASE", "ai_memory")
    memories_container_name = os.environ.get("COSMOS_DB_CONTAINER", "memories")
    counters_container_name = os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counters")

    credential = DefaultAzureCredential()
    client = CosmosClient(endpoint, credential=credential)
    db = client.get_database_client(database_name)
    memories = db.get_container_client(memories_container_name)
    counters = db.get_container_client(counters_container_name)
    return memories, counters


@pytest.fixture
def unique_ids():
    """Generate unique user_id and thread_id for test isolation."""
    return {
        "user_id": f"test-user-{uuid.uuid4().hex[:8]}",
        "thread_id": f"test-thread-{uuid.uuid4().hex[:8]}",
    }


class TestChangeFeedIntegration:
    """Integration tests for change feed trigger with live Cosmos DB."""

    def _insert_turn(self, memories_container, user_id, thread_id):
        """Insert a single turn document into the memories container."""
        doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "thread_id": thread_id,
            "role": "user",
            "type": "turn",
            "content": f"Test message {uuid.uuid4().hex[:6]}",
            "metadata": {},
            "embedding": [0.0] * 10,
            "created_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        memories_container.upsert_item(body=doc)
        return doc

    def _read_counter(self, counters_container, counter_id, user_id):
        """Read a counter document, returning None if not found."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return counters_container.read_item(
                item=counter_id, partition_key=user_id
            )
        except CosmosResourceNotFoundError:
            return None

    def test_counter_increments_on_turn_insert(self, cosmos_clients, unique_ids):
        """Insert turn documents and verify the thread counter increments.

        Note: This test depends on the change feed trigger running. It inserts
        documents and then polls the counters container for up to 60 seconds
        waiting for the change feed to process them.
        """
        memories, counters = cosmos_clients
        user_id = unique_ids["user_id"]
        thread_id = unique_ids["thread_id"]
        counter_id = f"thread_counter_{user_id}_{thread_id}"

        # Insert 3 turn documents
        for _ in range(3):
            self._insert_turn(memories, user_id, thread_id)

        # Poll for counter to appear (change feed has latency)
        deadline = time.time() + 60
        counter_doc = None
        while time.time() < deadline:
            counter_doc = self._read_counter(counters, counter_id, user_id)
            if counter_doc and counter_doc.get("count", 0) >= 3:
                break
            time.sleep(3)

        assert counter_doc is not None, (
            f"Counter {counter_id} was not created within 60s. "
            "Is the change feed trigger running?"
        )
        assert counter_doc["count"] >= 3, (
            f"Expected counter >= 3, got {counter_doc['count']}"
        )
