"""Async variant of :mod:`scenario_remote_processor`.

Wires :class:`AsyncDurableFunctionProcessor` to the
:class:`AsyncCosmosMemoryClient` so the SDK only writes raw turns and the
sibling Azure Function app handles summarization via the Cosmos change feed.

See ``scenario_remote_processor.py`` for the prerequisites and behavior notes.
"""
from __future__ import annotations

import asyncio
import os

from agent_memory_toolkit.aio import (
    AsyncCosmosMemoryClient,
    AsyncDurableFunctionProcessor,
)


async def main() -> None:
    client = AsyncCosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        # Hand processing off to the sibling Azure Function app.
        processor=AsyncDurableFunctionProcessor(),
    )

    try:
        await client.connect_cosmos()
        user_id = "alice"
        thread_id = "thread-001"

        transcript = [
            ("user", "Hi! I love Cosmos DB."),
            ("agent", "Cosmos DB is fantastic for low-latency global apps."),
            ("user", "Can it do vector search?"),
            ("agent", "Yes — DiskANN indexes power semantic search natively."),
            ("user", "Great. What about hierarchical partition keys?"),
            ("agent", "HPK lets you co-locate related items for efficient queries."),
        ]
        for role, content in transcript:
            await client.add_cosmos(
                user_id=user_id,
                thread_id=thread_id,
                role=role,
                content=content,
                memory_type="turn",
            )
            print(f"  wrote {role:>9}: {content}")

        # No-op locally; the change-feed trigger in the function app does the work.
        await client.process_now(user_id=user_id, thread_id=thread_id)

        # Optional: poll until the function app produces a summary. RU-costly.
        print("\nWaiting for the function app to produce a summary...")
        ok = await client.process_now_and_wait(
            user_id=user_id, thread_id=thread_id, timeout=60.0
        )
        print(f"Summary available: {ok}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
