"""AgentMemory: A class for managing agent memories locally and (eventually) in Cosmos DB."""

import uuid, os
from datetime import datetime, timezone
from typing import Any, Optional


VALID_ROLES = {"agent", "user", "tool", "system"}
VALID_TYPES = {"turn", "summary", "fact", "user_summary"}


def _make_memory(
    user_id: str,
    role: str,
    content: str,
    memory_type: str = "turn",
    agent_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    memory_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a validated memory dict."""
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got '{role}'")
    if memory_type not in VALID_TYPES:
        raise ValueError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")

    memory = {
        "id": memory_id or str(uuid.uuid4()),
        "user_id": user_id,
        "thread_id": thread_id or str(uuid.uuid4()),
        "role": role,
        "type": memory_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if agent_id is not None:
        memory["agent_id"] = agent_id

    return memory


class AgentMemory:
    """Manages agent memories with local storage and (future) Cosmos DB support.

    Authentication uses ``azure-identity`` by default.  If no explicit
    credential is passed for Cosmos DB or AI Foundry, a
    ``DefaultAzureCredential`` is created automatically.  This supports
    Entra ID (Azure AD) interactive login, managed identity, service
    principal, Azure CLI, and other credential flows out of the box.

    Parameters
    ----------
    cosmos_endpoint : str, optional
        The Cosmos DB account endpoint URL.
    cosmos_credential : TokenCredential, optional
        An Azure credential (e.g. ``DefaultAzureCredential``,
        ``ManagedIdentityCredential``).  Falls back to
        ``DefaultAzureCredential`` when not provided.
    cosmos_database : str, optional
        The Cosmos DB database name.
    cosmos_container : str, optional
        The Cosmos DB container name.
    ai_foundry_endpoint : str, optional
        The Azure OpenAI endpoint URL for generating embeddings
        (e.g. ``https://myaccount.openai.azure.com/``).
    ai_foundry_credential : TokenCredential, optional
        An Azure credential for the AI Foundry endpoint.  Falls back to
        ``DefaultAzureCredential`` when not provided.  Used to obtain
        an Entra ID token for the OpenAI service.
    ai_foundry_api_key : str, optional
        An API key for Azure OpenAI.  When provided this takes
        precedence over *ai_foundry_credential*.
    embedding_model : str, optional
        The embedding model deployment name (default ``text-embedding-3-large``).
    adf_endpoint : str, optional
        Base URL for the Azure Durable Functions API
        (e.g. ``http://localhost:7071/api``).
    adf_key : str, optional
        Function-level key for authenticating to the Azure Function.
        Leave empty when running locally without auth.
    use_default_credential : bool, optional
        When ``True`` (default), automatically creates a
        ``DefaultAzureCredential`` for any credential parameter that is
        not explicitly supplied.  Set to ``False`` to skip this.
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
        embedding_dimensions: Optional[int] = None,
        adf_endpoint: Optional[str] = None,
        adf_key: Optional[str] = None,
        use_default_credential: bool = True,
    ) -> None:
        # Local store
        self.local_memory: list[dict[str, Any]] = []

        # Resolve credentials – fall back to DefaultAzureCredential when
        # an explicit credential is not provided and the caller has not
        # opted out via use_default_credential=False.
        if use_default_credential and (cosmos_credential is None or ai_foundry_credential is None):
            try:
                from azure.identity import DefaultAzureCredential
                _default = DefaultAzureCredential()
            except ImportError:
                _default = None

            if cosmos_credential is None:
                cosmos_credential = _default
            if ai_foundry_credential is None:
                ai_foundry_credential = _default

        # Cosmos DB configuration
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_credential = cosmos_credential
        self.cosmos_database = cosmos_database
        self.cosmos_container = cosmos_container
        self._cosmos_container_client = None

        # Azure OpenAI embedding configuration
        self.ai_foundry_endpoint = ai_foundry_endpoint
        self.ai_foundry_credential = ai_foundry_credential
        self.ai_foundry_api_key = ai_foundry_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions or int(
            os.environ.get("EMBEDDING_DIMENSIONS", "0") or "0"
        ) or None
        self._embeddings_client = None

        # Azure Durable Functions configuration
        self.adf_endpoint = adf_endpoint
        self.adf_key = adf_key

    # ------------------------------------------------------------------
    # Local operations
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
        """Add a new memory to the local store."""
        memory = _make_memory(
            user_id=user_id,
            role=role,
            content=content,
            memory_type=memory_type,
            agent_id=agent_id,
            metadata=metadata,
            thread_id=thread_id,
        )
        self.local_memory.append(memory)

    def get_local(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from the local store.

        All filter parameters are optional. When none are provided every
        memory is returned. Filters are combined with AND logic.
        """
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
        """Update an existing memory in the local store.

        Only the fields that are provided (not ``None``) will be updated.

        Raises ``KeyError`` if no memory with the given id exists.
        """
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
        """Delete a memory from the local store by id.

        Raises ``KeyError`` if no memory with the given id exists.
        """
        for i, memory in enumerate(self.local_memory):
            if memory["id"] == memory_id:
                self.local_memory.pop(i)
                return

        raise KeyError(f"No memory found with id '{memory_id}'")

    # ------------------------------------------------------------------
    # Cosmos DB connection
    # ------------------------------------------------------------------

    def connect_cosmos(
        self,
        endpoint: Optional[str] = None,
        credential: Optional["TokenCredential"] = None,
        database: Optional[str] = None,
        container: Optional[str] = None,
    ) -> None:
        """Establish a connection to a Cosmos DB container.

        Parameters override whatever was set in ``__init__``.  After this
        call the Cosmos CRUD methods are ready to use.

        Raises ``ValueError`` if required connection details are missing.
        """
        from azure.cosmos import CosmosClient

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

        client = CosmosClient(self.cosmos_endpoint, credential=self.cosmos_credential)
        db = client.get_database_client(self.cosmos_database)
        self._cosmos_container_client = db.get_container_client(self.cosmos_container)

    def create_memory_store(
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
        """Create the Cosmos DB database and container for memories.

        Skips creation if the database or container already exists.
        After successful creation the instance is connected and ready
        for CRUD operations.

        The container is configured with:

        * **Hierarchical partition key** – ``/user_id`` then ``/thread_id``
        * **Vector embedding policy** – ``quantizedFlat`` index on the
          ``/embedding`` path
        * **Full-text index & policy** – English analyzer on ``/content``
        * **Autoscale throughput** – max ``1000`` RU/s

        Parameters
        ----------
        database : str, optional
            Database name.  Falls back to ``self.cosmos_database``.
        container : str, optional
            Container name.  Falls back to ``self.cosmos_container``.
        endpoint : str, optional
            Cosmos DB endpoint.  Falls back to ``self.cosmos_endpoint``.
        credential : TokenCredential, optional
            Azure credential.  Falls back to ``self.cosmos_credential``.
        embedding_dimensions : int, optional
            Dimensionality of the embedding vectors.  Falls back to
            env var ``EMBEDDING_DIMENSIONS``, then ``3072``.
        embedding_data_type : str, optional
            Data type for the vector (e.g. ``float32``, ``int8``).
            Falls back to env var ``EMBEDDING_DATA_TYPE``, then
            ``float32``.
        distance_function : str, optional
            Distance function (e.g. ``cosine``, ``euclidean``,
            ``dotproduct``).  Falls back to env var
            ``EMBEDDING_DISTANCE_FUNCTION``, then ``cosine``.
        full_text_language : str, optional
            Language for the full-text index on ``/content``
            (e.g. ``en-US``, ``fr-FR``).  Falls back to env var
            ``FULL_TEXT_LANGUAGE``, then ``en-US``.
        """
        from azure.cosmos import CosmosClient, PartitionKey, ThroughputProperties

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

        import os as _os

        embedding_dimensions = embedding_dimensions or int(
            _os.environ.get("EMBEDDING_DIMENSIONS", "1536")
        )
        embedding_data_type = (
            embedding_data_type
            or _os.environ.get("EMBEDDING_DATA_TYPE", "float32")
        )
        distance_function = (
            distance_function
            or _os.environ.get("EMBEDDING_DISTANCE_FUNCTION", "cosine")
        )
        full_text_language = (
            full_text_language
            or _os.environ.get("FULL_TEXT_LANGUAGE", "en-US")
        )
        autoscale_max_ru = int(
            _os.environ.get("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")
        )

        client = CosmosClient(self.cosmos_endpoint, credential=self.cosmos_credential)

        # ---- Database (create if not exists) ----
        db = client.create_database_if_not_exists(id=self.cosmos_database)

        # ---- Container (create if not exists) ----
        partition_key = PartitionKey(
            path=["/user_id", "/thread_id"],
            kind="MultiHash",
        )

        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": embedding_data_type,
                    "distanceFunction": distance_function,
                    "dimensions": embedding_dimensions,
                }
            ]
        }

        indexing_policy = {
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/embedding/*"}],
            "vectorIndexes": [
                {
                    "path": "/embedding",
                    "type": "quantizedFlat",
                }
            ],
            "fullTextIndexes": [
                {"path": "/content"}
            ],
        }

        full_text_policy = {
            "defaultLanguage": full_text_language,
            "fullTextPaths": [
                {"path": "/content", "language": full_text_language}
            ],
        }

        container_obj = db.create_container_if_not_exists(
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
        """Raise if Cosmos DB is not connected."""
        if self._cosmos_container_client is None:
            raise RuntimeError(
                "Cosmos DB is not connected. Call connect_cosmos() first."
            )

    # ------------------------------------------------------------------
    # Embeddings helper
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding for *text* via Azure OpenAI."""
        if self._embeddings_client is None:
            from openai import AzureOpenAI

            if not self.ai_foundry_endpoint:
                raise ValueError("ai_foundry_endpoint is required for embeddings")

            if self.ai_foundry_api_key:
                self._embeddings_client = AzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint=self.ai_foundry_endpoint,
                    api_key=self.ai_foundry_api_key,
                )
            else:
                if not self.ai_foundry_credential:
                    raise ValueError(
                        "ai_foundry_credential or ai_foundry_api_key is required for embeddings"
                    )
                from azure.identity import get_bearer_token_provider

                token_provider = get_bearer_token_provider(
                    self.ai_foundry_credential,
                    "https://cognitiveservices.azure.com/.default",
                )
                self._embeddings_client = AzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint=self.ai_foundry_endpoint,
                    azure_ad_token_provider=token_provider,
                )

        kwargs: dict[str, Any] = {
            "input": [text],
            "model": self.embedding_model,
        }
        if self.embedding_dimensions:
            kwargs["dimensions"] = self.embedding_dimensions
        response = self._embeddings_client.embeddings.create(**kwargs)
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Cosmos DB operations
    # ------------------------------------------------------------------

    def add_cosmos(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Add a memory to Cosmos DB."""
        self._require_cosmos()
        memory = _make_memory(
            user_id=user_id,
            role=role,
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            thread_id=thread_id,
        )
        self._cosmos_container_client.upsert_item(body=memory)

    def push_to_cosmos(self) -> None:
        """Insert all local memories into Cosmos DB.

        Each local memory is inserted as-is, preserving its existing
        ``id``, ``thread_id``, timestamps, and metadata.
        """
        self._require_cosmos()

        for memory in self.local_memory:
            self._cosmos_container_client.upsert_item(body=dict(memory))

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from Cosmos DB with optional filters.

        Args:
            memory_id: Filter by memory id.
            user_id: Filter by user id.
            thread_id: Filter by thread id.
            role: Filter by role.
            memory_type: Filter by type (raw, summary, fact, etc.).
            recent_k: If specified, return only the *k* most recent documents
                (ordered by ``_ts`` descending, then reversed to chronological).
        """
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
            enable_cross_partition_query=True
        )
        results = list(items)
        if recent_k is not None:
            results.reverse()
        return results

    def update_cosmos(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory in Cosmos DB.

        Raises ``KeyError`` if no memory with the given id exists.
        """
        self._require_cosmos()

        # Fetch current document
        results = list(self._cosmos_container_client.query_items(
            query="SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": memory_id}],
            enable_cross_partition_query=True

        ))
        if not results:
            raise KeyError(f"No memory found with id '{memory_id}'")

        doc = results[0]
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

        self._cosmos_container_client.replace_item(item=doc["id"], body=doc)

    def delete_cosmos(self, memory_id: str, thread_id: str, user_id: str) -> None:
        """Delete a memory from Cosmos DB.

        Raises ``KeyError`` if no memory with the given id exists.
        """
        self._require_cosmos()

        results = list(self._cosmos_container_client.query_items(
            query=(
                "SELECT TOP 1 c.id FROM c WHERE c.id = @id "
                "AND c.thread_id = @thread_id AND c.user_id = @user_id"
            ),
            parameters=[
                {"name": "@id", "value": memory_id},
                {"name": "@thread_id", "value": thread_id},
                {"name": "@user_id", "value": user_id},
            ],
            enable_cross_partition_query=True

        ))
        if not results:
            raise KeyError(
                f"No memory found with id '{memory_id}' for user_id '{user_id}' and thread_id '{thread_id}'"
            )

        self._cosmos_container_client.delete_item(
            item=memory_id,
            partition_key=[user_id, thread_id],
        )

    def search_cosmos(
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
        """Search memories in Cosmos DB.

        1. Embeds ``search_terms`` via the configured AI Foundry model.
        2. Runs a vector similarity query against the Cosmos DB container.
           When ``hybrid_search=True``, combines vector and full-text
           ranking via RRF in the ``ORDER BY`` clause.
          3. Optionally filters by ``memory_id``, ``user_id``, ``role``,
              ``memory_type``, and/or ``thread_id``.
        4. Returns up to ``top_k`` results ordered by similarity.
        """
        self._require_cosmos()

        query_vector = self._get_embedding(search_terms)

        # Build optional WHERE filters
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
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )
        return list(items)

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread from Cosmos DB.

        Parameters
        ----------
        thread_id : str
            The thread to retrieve (required).
        user_id : str, optional
            If provided, only return memories belonging to this user.
        memory_type : str, optional
            If provided, only return memories of this type
            (e.g. ``"turn"``, ``"summary"``, ``"fact"``).
        recent_k : int, optional
            If provided, return only the *k* most recent documents
            (by ``created_at``).  Otherwise all documents are returned.

        Returns
        -------
        list[dict]
            Memories sorted in chronological order (oldest first).
        """
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

        items = list(self._cosmos_container_client.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))

        if recent_k is not None:
            items = items[:recent_k]

        # Return in chronological order (oldest first)
        items.reverse()
        return items

    # ------------------------------------------------------------------
    # Azure Durable Function – generate_thread_summary
    # ------------------------------------------------------------------

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a thread summary.

        Starts the ``memory_orchestrator`` with ``thread_summary_only=True``
        and polls until the orchestration completes or *timeout* seconds
        elapse.

        Parameters
        ----------
        user_id : str
            The user whose memories to summarize.
        thread_id : str
            The conversation thread to summarize.
        recent_k : int, optional
            If provided, only the most recent *k* memories are
            included in the summary.
        poll_interval : float
            Seconds between status polls (default 2).
        timeout : float
            Maximum seconds to wait for completion (default 120).

        Returns
        -------
        dict
            The orchestration result containing the summary.
        """
        import time
        import urllib.request
        import json as _json

        if not self.adf_endpoint:
            raise ValueError("adf_endpoint is required to call generate_thread_summary")

        # Build the starter URL
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

        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            start_response = _json.loads(resp.read().decode("utf-8"))

        status_url = start_response.get("statusQueryGetUri")
        if not status_url:
            return start_response

        # Poll for completion
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            status_req = urllib.request.Request(status_url, method="GET")
            with urllib.request.urlopen(status_req) as resp:
                status = _json.loads(resp.read().decode("utf-8"))
            runtime_status = status.get("runtimeStatus", "")
            if runtime_status in ("Completed", "Failed", "Terminated"):
                return status

        raise TimeoutError(
            f"Orchestration did not complete within {timeout}s. "
            f"Check status at: {status_url}"
        )

    # ------------------------------------------------------------------
    # Azure Durable Function – generate_user_summary
    # ------------------------------------------------------------------

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a cross-thread user summary.

        Aggregates memories across all (or selected) threads for a user
        and produces a structured profile covering preferences, account
        state, compliance details, and behavioural patterns.

        Parameters
        ----------
        user_id : str
            The user to summarize across threads.
        thread_ids : list[str], optional
            If provided, only these threads are included. Otherwise all
            threads for the user are used.
        recent_k : int, optional
            If provided, only the most recent *k* memories per thread
            are included.
        poll_interval : float
            Seconds between status polls (default 2).
        timeout : float
            Maximum seconds to wait for completion (default 120).

        Returns
        -------
        dict
            The orchestration result containing the user summary.
        """
        import time
        import urllib.request
        import json as _json

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

        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            start_response = _json.loads(resp.read().decode("utf-8"))

        status_url = start_response.get("statusQueryGetUri")
        if not status_url:
            return start_response

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            status_req = urllib.request.Request(status_url, method="GET")
            with urllib.request.urlopen(status_req) as resp:
                status = _json.loads(resp.read().decode("utf-8"))
            runtime_status = status.get("runtimeStatus", "")
            if runtime_status in ("Completed", "Failed", "Terminated"):
                return status

        raise TimeoutError(
            f"Orchestration did not complete within {timeout}s. "
            f"Check status at: {status_url}"
        )

    # ------------------------------------------------------------------
    # Cosmos DB – get_user_summary
    # ------------------------------------------------------------------

    def get_user_summary(
        self,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve the user summary document(s) for a user from Cosmos DB.

        Parameters
        ----------
        user_id : str
            The user whose summary to retrieve.

        Returns
        -------
        list[dict]
            User summary documents, newest first.
        """
        self._require_cosmos()

        query = (
            "SELECT c.id, c.user_id, c.thread_id, c.role, c.type, "
            "c.content, c.metadata, c.created_at "
            "FROM c WHERE c.user_id = @user_id AND c.type = 'user_summary' "
            "ORDER BY c.created_at DESC"
        )
        parameters = [{"name": "@user_id", "value": user_id}]

        return list(self._cosmos_container_client.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))

    # ------------------------------------------------------------------
    # Azure Durable Function – extract_facts
    # ------------------------------------------------------------------

    def extract_facts(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to extract facts from a thread.

        Starts the ``memory_orchestrator`` with ``extract_facts_only=True``
        and polls until the orchestration completes or *timeout* seconds
        elapse.

        Parameters
        ----------
        user_id : str
            The user whose memories to extract facts from.
        thread_id : str
            The conversation thread to extract facts from.
        recent_k : int, optional
            If provided, only the most recent *k* memories are
            included in the extraction.
        poll_interval : float
            Seconds between status polls (default 2).
        timeout : float
            Maximum seconds to wait for completion (default 120).

        Returns
        -------
        dict
            The orchestration result containing the extracted facts.
        """
        import time
        import urllib.request
        import json as _json

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

        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            start_response = _json.loads(resp.read().decode("utf-8"))

        status_url = start_response.get("statusQueryGetUri")
        if not status_url:
            return start_response

        # Poll for completion
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            status_req = urllib.request.Request(status_url, method="GET")
            with urllib.request.urlopen(status_req) as resp:
                status = _json.loads(resp.read().decode("utf-8"))
            runtime_status = status.get("runtimeStatus", "")
            if runtime_status in ("Completed", "Failed", "Terminated"):
                return status

        raise TimeoutError(
            f"Orchestration did not complete within {timeout}s. "
            f"Check status at: {status_url}"
        )
