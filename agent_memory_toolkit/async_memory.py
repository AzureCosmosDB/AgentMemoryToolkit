"""AsyncAgentMemory: Async version of AgentMemory using azure.cosmos.aio and AsyncAzureOpenAI."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from agent_memory_toolkit.memory import VALID_ROLES, VALID_TYPES, _make_memory


class AsyncAgentMemory:
    """Async variant of :class:`AgentMemory`.

    * Cosmos DB operations use ``azure.cosmos.aio``
    * Embeddings use ``openai.AsyncAzureOpenAI``
    * The ``generate_thread_summary`` method uses ``aiohttp`` for non-blocking HTTP
    * Local operations remain synchronous (in-memory list)

    Parameters are identical to :class:`AgentMemory`.
    """

    def __init__(
        self,
        cosmos_endpoint: Optional[str] = None,
        cosmos_credential: Optional["TokenCredential"] = None,
        cosmos_database: Optional[str] = None,
        cosmos_container: Optional[str] = None,
        ai_foundry_endpoint: Optional[str] = None,
        ai_foundry_credential: Optional["TokenCredential"] = None,
        ai_foundry_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        adf_endpoint: Optional[str] = None,
        adf_key: Optional[str] = None,
        use_default_credential: bool = True,
    ) -> None:
        self.local_memory: list[dict[str, Any]] = []

        if use_default_credential and (cosmos_credential is None or ai_foundry_credential is None):
            try:
                from azure.identity.aio import DefaultAzureCredential
                _default = DefaultAzureCredential()
            except ImportError:
                _default = None

            if cosmos_credential is None:
                cosmos_credential = _default
            if ai_foundry_credential is None:
                ai_foundry_credential = _default

        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_credential = cosmos_credential
        self.cosmos_database = cosmos_database
        self.cosmos_container = cosmos_container
        self._cosmos_client = None
        self._cosmos_container_client = None

        self.ai_foundry_endpoint = ai_foundry_endpoint
        self.ai_foundry_credential = ai_foundry_credential
        self.ai_foundry_api_key = ai_foundry_api_key
        self.embedding_model = embedding_model
        self._embeddings_client = None

        self.adf_endpoint = adf_endpoint
        self.adf_key = adf_key

    # ------------------------------------------------------------------
    # Local operations (synchronous – in-memory list)
    # ------------------------------------------------------------------

    def add_local(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        agent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        memory = _make_memory(
            user_id=user_id, role=role, content=content,
            memory_type=memory_type, agent_id=agent_id,
            metadata=metadata, thread_id=thread_id,
        )
        self.local_memory.append(memory)

    def get_local(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        results = self.local_memory
        if memory_id is not None:
            results = [m for m in results if m["id"] == memory_id]
        if user_id is not None:
            results = [m for m in results if m["user_id"] == user_id]
        if role is not None:
            results = [m for m in results if m["role"] == role]
        if memory_type is not None:
            results = [m for m in results if m["type"] == memory_type]
        return results

    def update_local(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        for memory in self.local_memory:
            if memory["id"] == memory_id:
                if content is not None:
                    memory["content"] = content
                if role is not None:
                    if role not in VALID_ROLES:
                        raise ValueError(f"role must be one of {VALID_ROLES}, got '{role}'")
                    memory["role"] = role
                if memory_type is not None:
                    if memory_type not in VALID_TYPES:
                        raise ValueError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")
                    memory["type"] = memory_type
                if metadata is not None:
                    memory["metadata"] = metadata
                memory["updated_at"] = datetime.now(timezone.utc).isoformat()
                return
        raise KeyError(f"No memory found with id '{memory_id}'")

    def delete_local(self, memory_id: str) -> None:
        for i, memory in enumerate(self.local_memory):
            if memory["id"] == memory_id:
                self.local_memory.pop(i)
                return
        raise KeyError(f"No memory found with id '{memory_id}'")

    # ------------------------------------------------------------------
    # Cosmos DB connection (async)
    # ------------------------------------------------------------------

    async def connect_cosmos(
        self,
        endpoint: Optional[str] = None,
        credential: Optional["TokenCredential"] = None,
        database: Optional[str] = None,
        container: Optional[str] = None,
    ) -> None:
        """Establish an async connection to a Cosmos DB container."""
        from azure.cosmos.aio import CosmosClient

        self.cosmos_endpoint = endpoint or self.cosmos_endpoint
        self.cosmos_credential = credential or self.cosmos_credential
        self.cosmos_database = database or self.cosmos_database
        self.cosmos_container = container or self.cosmos_container

        if not self.cosmos_endpoint:
            raise ValueError("cosmos_endpoint is required")
        if not self.cosmos_credential:
            raise ValueError("cosmos_credential is required")
        if not self.cosmos_database:
            raise ValueError("cosmos_database is required")
        if not self.cosmos_container:
            raise ValueError("cosmos_container is required")

        self._cosmos_client = CosmosClient(
            self.cosmos_endpoint, credential=self.cosmos_credential,
        )
        db = self._cosmos_client.get_database_client(self.cosmos_database)
        self._cosmos_container_client = db.get_container_client(self.cosmos_container)

    async def create_memory_store(
        self,
        database: Optional[str] = None,
        container: Optional[str] = None,
        endpoint: Optional[str] = None,
        credential: Optional["TokenCredential"] = None,
        embedding_dimensions: Optional[int] = None,
        embedding_data_type: Optional[str] = None,
        distance_function: Optional[str] = None,
        full_text_language: Optional[str] = None,
    ) -> None:
        """Create the Cosmos DB database and container (async)."""
        import os as _os
        from azure.cosmos.aio import CosmosClient
        from azure.cosmos import PartitionKey, ThroughputProperties

        self.cosmos_endpoint = endpoint or self.cosmos_endpoint
        self.cosmos_credential = credential or self.cosmos_credential
        self.cosmos_database = database or self.cosmos_database
        self.cosmos_container = container or self.cosmos_container

        if not self.cosmos_endpoint:
            raise ValueError("cosmos_endpoint is required")
        if not self.cosmos_credential:
            raise ValueError("cosmos_credential is required")
        if not self.cosmos_database:
            raise ValueError("cosmos_database is required")
        if not self.cosmos_container:
            raise ValueError("cosmos_container is required")

        embedding_dimensions = embedding_dimensions or int(
            _os.environ.get("EMBEDDING_DIMENSIONS", "3072")
        )
        embedding_data_type = (
            embedding_data_type or _os.environ.get("EMBEDDING_DATA_TYPE", "float32")
        )
        distance_function = (
            distance_function or _os.environ.get("EMBEDDING_DISTANCE_FUNCTION", "cosine")
        )
        full_text_language = (
            full_text_language or _os.environ.get("FULL_TEXT_LANGUAGE", "en-US")
        )
        autoscale_max_ru = int(
            _os.environ.get("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")
        )

        self._cosmos_client = CosmosClient(
            self.cosmos_endpoint, credential=self.cosmos_credential,
        )

        db = await self._cosmos_client.create_database_if_not_exists(id=self.cosmos_database)

        partition_key = PartitionKey(path=["/user_id", "/thread_id"], kind="MultiHash")

        vector_embedding_policy = {
            "vectorEmbeddings": [{
                "path": "/embedding",
                "dataType": embedding_data_type,
                "distanceFunction": distance_function,
                "dimensions": embedding_dimensions,
            }]
        }
        indexing_policy = {
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/embedding/*"}],
            "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
            "fullTextIndexes": [{"path": "/content"}],
        }
        full_text_policy = {
            "defaultLanguage": full_text_language,
            "fullTextPaths": [{"path": "/content", "language": full_text_language}],
        }

        container_obj = await db.create_container_if_not_exists(
            id=self.cosmos_container,
            partition_key=partition_key,
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            full_text_policy=full_text_policy,
            offer_throughput=ThroughputProperties(
                auto_scale_max_throughput=autoscale_max_ru,
            ),
        )
        self._cosmos_container_client = container_obj

    def _require_cosmos(self):
        if self._cosmos_container_client is None:
            raise RuntimeError("Cosmos DB is not connected. Call await connect_cosmos() first.")

    # ------------------------------------------------------------------
    # Embeddings helper (async)
    # ------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding using AsyncAzureOpenAI."""
        if self._embeddings_client is None:
            from openai import AsyncAzureOpenAI

            if not self.ai_foundry_endpoint:
                raise ValueError("ai_foundry_endpoint is required for embeddings")

            if self.ai_foundry_api_key:
                self._embeddings_client = AsyncAzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint=self.ai_foundry_endpoint,
                    api_key=self.ai_foundry_api_key,
                )
            else:
                if not self.ai_foundry_credential:
                    raise ValueError(
                        "ai_foundry_credential or ai_foundry_api_key is required for embeddings"
                    )
                from azure.identity.aio import get_bearer_token_provider

                token_provider = get_bearer_token_provider(
                    self.ai_foundry_credential,
                    "https://cognitiveservices.azure.com/.default",
                )
                self._embeddings_client = AsyncAzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint=self.ai_foundry_endpoint,
                    azure_ad_token_provider=token_provider,
                )

        response = await self._embeddings_client.embeddings.create(
            input=[text], model=self.embedding_model,
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Cosmos DB operations (async)
    # ------------------------------------------------------------------

    async def add_cosmos(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        self._require_cosmos()
        memory = _make_memory(
            user_id=user_id, role=role, content=content,
            memory_type=memory_type, metadata=metadata, thread_id=thread_id,
        )
        await self._cosmos_container_client.upsert_item(body=memory)

    async def push_to_cosmos(self, batch_size: int = 25) -> None:
        """Insert all local memories into Cosmos DB in concurrent batches."""
        import asyncio

        self._require_cosmos()

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        for start in range(0, len(self.local_memory), batch_size):
            batch = self.local_memory[start:start + batch_size]
            tasks = [
                asyncio.create_task(
                    self._cosmos_container_client.upsert_item(body=dict(memory))
                )
                for memory in batch
            ]
            if tasks:
                done, pending = await asyncio.wait(tasks)
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()

    async def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        self._require_cosmos()

        conditions: list[str] = []
        parameters: list[dict[str, Any]] = []

        if memory_id is not None:
            conditions.append("c.id = @memory_id")
            parameters.append({"name": "@memory_id", "value": memory_id})
        if user_id is not None:
            conditions.append("c.user_id = @user_id")
            parameters.append({"name": "@user_id", "value": user_id})
        if thread_id is not None:
            conditions.append("c.thread_id = @thread_id")
            parameters.append({"name": "@thread_id", "value": thread_id})
        if role is not None:
            conditions.append("c.role = @role")
            parameters.append({"name": "@role", "value": role})
        if memory_type is not None:
            conditions.append("c.type = @memory_type")
            parameters.append({"name": "@memory_type", "value": memory_type})

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        if recent_k is not None:
            top_clause = "TOP @recent_k "
            parameters.append({"name": "@recent_k", "value": recent_k})
            query = f"SELECT {top_clause}* FROM c{where} ORDER BY c._ts DESC"
        else:
            query = f"SELECT * FROM c{where}"

        items = self._cosmos_container_client.query_items(
            query=query,
            parameters=parameters or None,
        )
        results = [item async for item in items]
        if recent_k is not None:
            results.reverse()
        return results

    async def update_cosmos(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._require_cosmos()

        results = self._cosmos_container_client.query_items(
            query="SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": memory_id}]
        )
        docs = [item async for item in results]
        if not docs:
            raise KeyError(f"No memory found with id '{memory_id}'")

        doc = docs[0]
        if content is not None:
            doc["content"] = content
        if role is not None:
            if role not in VALID_ROLES:
                raise ValueError(f"role must be one of {VALID_ROLES}, got '{role}'")
            doc["role"] = role
        if memory_type is not None:
            if memory_type not in VALID_TYPES:
                raise ValueError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")
            doc["type"] = memory_type
        if metadata is not None:
            doc["metadata"] = metadata
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()

        await self._cosmos_container_client.replace_item(item=doc["id"], body=doc)

    async def delete_cosmos(self, memory_id: str, thread_id: str, user_id: str) -> None:
        self._require_cosmos()

        results = self._cosmos_container_client.query_items(
            query=(
                "SELECT TOP 1 c.id FROM c WHERE c.id = @id "
                "AND c.thread_id = @thread_id AND c.user_id = @user_id"
            ),
            parameters=[
                {"name": "@id", "value": memory_id},
                {"name": "@thread_id", "value": thread_id},
                {"name": "@user_id", "value": user_id},
            ]
        )
        docs = [item async for item in results]
        if not docs:
            raise KeyError(
                f"No memory found with id '{memory_id}' for user_id '{user_id}' and thread_id '{thread_id}'"
            )

        await self._cosmos_container_client.delete_item(
            item=memory_id, partition_key=[user_id, thread_id],
        )

    async def search_cosmos(
        self,
        search_terms: str,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        hybrid_search: bool = False,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        self._require_cosmos()

        query_vector = await self._get_embedding(search_terms)

        conditions: list[str] = []
        parameters: list[dict[str, Any]] = []
        if memory_id is not None:
            conditions.append("c.id = @memory_id")
            parameters.append({"name": "@memory_id", "value": memory_id})
        if user_id is not None:
            conditions.append("c.user_id = @user_id")
            parameters.append({"name": "@user_id", "value": user_id})
        if role is not None:
            conditions.append("c.role = @role")
            parameters.append({"name": "@role", "value": role})
        if memory_type is not None:
            conditions.append("c.type = @memory_type")
            parameters.append({"name": "@memory_type", "value": memory_type})
        if thread_id is not None:
            conditions.append("c.thread_id = @thread_id")
            parameters.append({"name": "@thread_id", "value": thread_id})

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        order_by = "ORDER BY VectorDistance(c.embedding, @embedding)"
        if hybrid_search:
            order_by = (
                "ORDER BY RANK RRF(" 
                "VectorDistance(c.embedding, @embedding), "
                "FullTextScore(c.content, @key_terms)"
                ")"
            )

        query = (
            f"SELECT TOP @top_k c.id, c.user_id, c.role, c.type, c.content, "
            f"c.metadata, c.created_at "
            f"FROM c{where} "
            f"{order_by}"
        )
        parameters.extend([
            {"name": "@top_k", "value": top_k},
            {"name": "@embedding", "value": query_vector},
        ])
        if hybrid_search:
            parameters.append({"name": "@key_terms", "value": search_terms})

        items = self._cosmos_container_client.query_items(
            query=query, parameters=parameters
        )
        return [item async for item in items]

    async def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        self._require_cosmos()

        conditions: list[str] = ["c.thread_id = @thread_id"]
        parameters: list[dict[str, Any]] = [
            {"name": "@thread_id", "value": thread_id},
        ]
        if user_id is not None:
            conditions.append("c.user_id = @user_id")
            parameters.append({"name": "@user_id", "value": user_id})

        if memory_type is not None:
            conditions.append("c.type = @memory_type")
            parameters.append({"name": "@memory_type", "value": memory_type})

        where = " WHERE " + " AND ".join(conditions)
        query = f"SELECT * FROM c{where} ORDER BY c.created_at DESC"

        results = self._cosmos_container_client.query_items(
            query=query, parameters=parameters
        )
        items = [item async for item in results]

        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    # ------------------------------------------------------------------
    # Azure Durable Function – generate_thread_summary (async)
    # ------------------------------------------------------------------

    async def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a thread summary (async)."""
        import asyncio
        import json as _json
        import aiohttp

        if not self.adf_endpoint:
            raise ValueError("adf_endpoint is required to call generate_thread_summary")

        url = f"{self.adf_endpoint.rstrip('/')}/orchestrators/memory_orchestrator"
        if self.adf_key:
            url += f"?code={self.adf_key}"

        body = {
            "user_id": user_id,
            "thread_id": thread_id,
            "thread_summary_only": True,
        }
        if recent_k is not None:
            body["recent_k"] = recent_k

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as resp:
                start_response = await resp.json()

            status_url = start_response.get("statusQueryGetUri")
            if not status_url:
                return start_response

            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(poll_interval)
                async with session.get(status_url) as resp:
                    status = await resp.json()
                runtime_status = status.get("runtimeStatus", "")
                if runtime_status in ("Completed", "Failed", "Terminated"):
                    return status

        raise TimeoutError(
            f"Orchestration did not complete within {timeout}s. "
            f"Check status at: {status_url}"
        )

    # ------------------------------------------------------------------
    # Azure Durable Function – extract_facts (async)
    # ------------------------------------------------------------------

    async def extract_facts(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to extract facts from a thread (async)."""
        import asyncio
        import json as _json
        import aiohttp

        if not self.adf_endpoint:
            raise ValueError("adf_endpoint is required to call extract_facts")

        url = f"{self.adf_endpoint.rstrip('/')}/orchestrators/memory_orchestrator"
        if self.adf_key:
            url += f"?code={self.adf_key}"

        body = {
            "user_id": user_id,
            "thread_id": thread_id,
            "extract_facts_only": True,
        }
        if recent_k is not None:
            body["recent_k"] = recent_k

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as resp:
                start_response = await resp.json()

            status_url = start_response.get("statusQueryGetUri")
            if not status_url:
                return start_response

            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(poll_interval)
                async with session.get(status_url) as resp:
                    status = await resp.json()
                runtime_status = status.get("runtimeStatus", "")
                if runtime_status in ("Completed", "Failed", "Terminated"):
                    return status

        raise TimeoutError(
            f"Orchestration did not complete within {timeout}s. "
            f"Check status at: {status_url}"
        )

    # ------------------------------------------------------------------
    # Azure Durable Function – generate_user_summary (async)
    # ------------------------------------------------------------------

    async def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a cross-thread user summary (async)."""
        import asyncio
        import json as _json
        import aiohttp

        if not self.adf_endpoint:
            raise ValueError("adf_endpoint is required to call generate_user_summary")

        url = f"{self.adf_endpoint.rstrip('/')}/orchestrators/memory_orchestrator"
        if self.adf_key:
            url += f"?code={self.adf_key}"

        body: dict[str, Any] = {
            "user_id": user_id,
            "user_summary_only": True,
        }
        if thread_ids is not None:
            body["thread_ids"] = thread_ids
        if recent_k is not None:
            body["recent_k"] = recent_k

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as resp:
                start_response = await resp.json()

            status_url = start_response.get("statusQueryGetUri")
            if not status_url:
                return start_response

            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(poll_interval)
                async with session.get(status_url) as resp:
                    status = await resp.json()
                runtime_status = status.get("runtimeStatus", "")
                if runtime_status in ("Completed", "Failed", "Terminated"):
                    return status

        raise TimeoutError(
            f"Orchestration did not complete within {timeout}s. "
            f"Check status at: {status_url}"
        )

    # ------------------------------------------------------------------
    # Cosmos DB – get_user_summary (async)
    # ------------------------------------------------------------------

    async def get_user_summary(
        self,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve the user summary document(s) for a user from Cosmos DB (async)."""
        self._require_cosmos()

        query = (
            "SELECT c.id, c.user_id, c.thread_id, c.role, c.type, "
            "c.content, c.metadata, c.created_at "
            "FROM c WHERE c.user_id = @user_id AND c.type = 'user_summary' "
            "ORDER BY c.created_at DESC"
        )
        parameters = [{"name": "@user_id", "value": user_id}]

        results = self._cosmos_container_client.query_items(
            query=query, parameters=parameters
        )
        return [item async for item in results]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the async Cosmos DB client."""
        if self._cosmos_client is not None:
            await self._cosmos_client.close()
            self._cosmos_client = None
            self._cosmos_container_client = None
        if self._embeddings_client is not None:
            await self._embeddings_client.close()
            self._embeddings_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
