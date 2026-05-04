"""Cosmos DB quickstart for CosmosMemoryClient.

Demonstrates connecting to Azure Cosmos DB and performing CRUD operations:
connect, add, get, update, delete, and push local memories to Cosmos.

Required env vars (or .env file):
    COSMOS_DB_ENDPOINT   – Cosmos DB account endpoint URL
    COSMOS_DB_DATABASE   – database name (default: "ai_memory")
    COSMOS_DB_CONTAINER  – container name (default: "memories")
"""

import os

from dotenv import load_dotenv

from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()


def main() -> None:
    mem = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        ai_foundry_endpoint=os.environ.get("AI_FOUNDRY_ENDPOINT"),
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY"),
        embedding_deployment_name=os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        chat_deployment_name=os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
        use_default_credential=True,
    )

    # Connect to an existing Cosmos DB container
    mem.connect_cosmos()
    print("Connected to Cosmos DB")

    # Add a memory directly to Cosmos. memory_type="turn" skips auto-embedding,
    # which keeps the quickstart runnable without AI Foundry credentials.
    mem.add_cosmos(user_id="u1", role="user", content="Hello from quickstart!", thread_id="t1")
    print("Added memory to Cosmos")

    # Retrieve all memories for the user
    results = mem.get_memories(user_id="u1")
    print(f"Retrieved {len(results)} memories for user 'u1'")

    # Get a specific thread
    thread = mem.get_thread(thread_id="t1")
    print(f"Thread 't1' has {len(thread)} messages")

    # Update the first memory
    memory_id = results[0]["id"]
    mem.update_cosmos(memory_id=memory_id, content="Updated via quickstart")
    print(f"Updated memory {memory_id}")

    # Push a local memory to Cosmos in batch
    mem.add_local(user_id="u1", role="agent", content="Agent response", thread_id="t1")
    mem.push_to_cosmos()
    print("Pushed local memories to Cosmos")

    # Clean up – delete the memories we created
    mem.delete_cosmos(memory_id=memory_id, thread_id="t1", user_id="u1")
    print(f"Deleted memory {memory_id}")

    print("\nQuickstart complete!")


if __name__ == "__main__":
    main()
