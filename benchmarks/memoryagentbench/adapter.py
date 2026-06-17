"""AgentMemoryToolkit adapter for MemoryAgentBench.

Provides :class:`AgentMemoryToolkitBackend`, a self-contained backend used by a
MemoryAgentBench ``AgentWrapper`` patch (see ``patch_agent.py``).

The backend honours MemoryAgentBench's ``send_message(message, memorizing,
query_id, context_id)`` contract:

* ``memorizing=True``  -> :meth:`memorize` stores a context chunk as a memory
  turn in Cosmos DB.
* ``memorizing=False`` -> :meth:`query` retrieves relevant memories via
  :func:`CosmosMemoryClient.search_cosmos`, asks the configured LLM to answer,
  and returns a standard MemoryAgentBench response dict.

Optional advanced modes invoke the toolkit's processing pipeline
(``extract_facts``, ``generate_thread_summary``, ``generate_user_summary``)
between the ingestion and query phases via :meth:`finalize_context`.

This module deliberately has *no* runtime dependency on MemoryAgentBench code,
so it can be unit-tested in isolation and kept inside this repository.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # Package was renamed; prefer the new path, fall back to the legacy one.
    from azure.cosmos.agent_memory import CosmosMemoryClient
except ImportError:  # pragma: no cover - depends on installed package version
    from agent_memory_toolkit import CosmosMemoryClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Store modes drive what is written during ``memorize`` and what
#: ``finalize_context`` produces. They never affect the search query target,
#: which is independently configured via ``search_memory_types``.
STORE_MODES = ("turns_only", "facts_only", "summary_plus_facts", "user_summary")

#: Search modes select Cosmos search style.
SEARCH_MODES = ("vector", "hybrid")


@dataclass
class AdapterConfig:
    """Resolved configuration for the adapter."""

    agent_name: str = "agent_memory_toolkit"
    model: str = "gpt-4o-mini"
    retrieve_num: int = 5
    max_tokens: int = 512
    temperature: float = 0.0

    # Toolkit-specific
    store_mode: str = "turns_only"
    search_mode: str = "vector"
    search_memory_types: tuple[str, ...] = ("turn",)
    run_id: str = "default"

    # Multi-agent shared memory. When ``writer_agents`` has >1 entry, memorize
    # rotates each write across these agents on the *same* (user_id, thread_id)
    # so they share one memory space. ``query_agents`` optionally restricts
    # retrieval to memories written by a subset of agents (empty = shared read
    # across all agents).
    writer_agents: tuple[str, ...] = ()
    query_agents: tuple[str, ...] = ()

    # Processing
    processing_poll_interval: float = 2.0
    processing_timeout: float = 240.0

    # Misc
    extra: dict[str, Any] = field(default_factory=dict)


def _coerce_types(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("turn",)
    if isinstance(value, str):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ValueError(f"Unsupported memory_types value: {value!r}")


def _coerce_str_list(value: Any) -> tuple[str, ...]:
    """Coerce ``None`` / comma-string / list into a tuple of trimmed strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    raise ValueError(f"Unsupported agent list value: {value!r}")


def build_adapter_config(agent_config: dict[str, Any]) -> AdapterConfig:
    """Build an :class:`AdapterConfig` from a MemoryAgentBench agent_config dict."""
    store_mode = agent_config.get("memory_toolkit_store_mode", "turns_only")
    if store_mode not in STORE_MODES:
        raise ValueError(
            f"memory_toolkit_store_mode must be one of {STORE_MODES}, got {store_mode!r}"
        )
    search_mode = agent_config.get("memory_toolkit_search_mode", "vector")
    if search_mode not in SEARCH_MODES:
        raise ValueError(
            f"memory_toolkit_search_mode must be one of {SEARCH_MODES}, got {search_mode!r}"
        )

    cfg = AdapterConfig(
        agent_name=agent_config.get("agent_name", "agent_memory_toolkit"),
        model=agent_config.get("model", "gpt-4o-mini"),
        retrieve_num=int(agent_config.get("retrieve_num", 5)),
        max_tokens=int(agent_config.get("max_tokens", 512)),
        temperature=float(agent_config.get("temperature", 0.0)),
        store_mode=store_mode,
        search_mode=search_mode,
        search_memory_types=_coerce_types(
            agent_config.get("memory_toolkit_search_memory_types")
        ),
        run_id=str(
            agent_config.get(
                "memory_toolkit_run_id",
                os.environ.get("MAB_RUN_ID", "default"),
            )
        ),
        processing_poll_interval=float(
            agent_config.get("memory_toolkit_processing_poll_interval", 2.0)
        ),
        processing_timeout=float(
            agent_config.get("memory_toolkit_processing_timeout", 240.0)
        ),
        writer_agents=_coerce_str_list(
            agent_config.get("memory_toolkit_writer_agents")
        ),
        query_agents=_coerce_str_list(
            agent_config.get("memory_toolkit_query_agents")
        ),
    )
    return cfg


# ---------------------------------------------------------------------------
# Token counting (best-effort; no hard dependency on tiktoken)
# ---------------------------------------------------------------------------


def _count_tokens(text: str, model: str) -> int:
    try:
        import tiktoken  # type: ignore
    except Exception:
        # Word-count fallback; better than nothing for relative comparisons.
        return max(1, len(text.split()))
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text, disallowed_special=()))


def _accepts_kwarg(func: Any, name: str) -> bool:
    """True if ``func`` accepts keyword ``name`` (directly or via **kwargs)."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    return name in sig.parameters


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class AgentMemoryToolkitBackend:
    """Backend that powers the ``agent_memory_toolkit`` MemoryAgentBench agent.

    Parameters
    ----------
    agent_config:
        MemoryAgentBench's ``agent_config`` dict.
    dataset_config:
        MemoryAgentBench's ``dataset_config`` dict.
    client:
        Optional pre-built :class:`CosmosMemoryClient`. When ``None`` (the
        common case), a client is constructed from environment variables.
    llm_client:
        Optional callable ``(messages, model, max_tokens, temperature) -> str``.
        When ``None``, an OpenAI/Azure OpenAI chat completions client is built
        from environment variables.
    """

    def __init__(
        self,
        agent_config: dict[str, Any],
        dataset_config: dict[str, Any],
        *,
        client: Optional[CosmosMemoryClient] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.cfg = build_adapter_config(agent_config)
        self.sub_dataset = str(dataset_config.get("sub_dataset", "unknown"))
        self.dataset = str(dataset_config.get("dataset", "unknown"))

        self._client = client or self._build_default_client()
        self._llm = llm_client or self._build_default_llm()

        # Track which (user_id, thread_id) pairs have been finalized so
        # finalize_context is idempotent.
        self._finalized: set[tuple[str, str]] = set()
        # Per-context wall-clock for memory_construction_time, mirroring
        # MemoryAgentBench's existing pattern.
        self._context_start: dict[Any, float] = {}
        # Round-robin write counter per context, used in multi-agent mode to
        # attribute each write to one of several agents sharing a thread.
        self._write_seq: dict[Any, int] = {}
        # Detect whether the underlying add_cosmos accepts a ``tags`` argument
        # so multi-agent attribution tags degrade gracefully on older clients.
        self._add_supports_tags = _accepts_kwarg(self._client.add_cosmos, "tags")

        if self.cfg.query_agents and self.cfg.writer_agents:
            unknown = set(self.cfg.query_agents) - set(self.cfg.writer_agents)
            if unknown:
                logger.warning(
                    "query_agents %s are not in writer_agents %s; those reads "
                    "will match nothing.",
                    sorted(unknown),
                    list(self.cfg.writer_agents),
                )

        logger.info(
            "AgentMemoryToolkitBackend ready run_id=%s sub_dataset=%s store=%s "
            "search=%s writers=%s query_agents=%s",
            self.cfg.run_id,
            self.sub_dataset,
            self.cfg.store_mode,
            self.cfg.search_mode,
            list(self.cfg.writer_agents) or "single",
            list(self.cfg.query_agents) or "shared",
        )

    # -- client construction ------------------------------------------------

    @staticmethod
    def _build_default_client() -> CosmosMemoryClient:
        return CosmosMemoryClient(
            cosmos_endpoint=os.environ.get("COSMOS_DB_ENDPOINT"),
            cosmos_database=os.environ.get("COSMOS_DB_DATABASE"),
            cosmos_container=os.environ.get("COSMOS_DB_CONTAINER"),
            cosmos_counter_container=os.environ.get("COSMOS_DB_COUNTERS_CONTAINER"),
            cosmos_lease_container=os.environ.get("COSMOS_DB_LEASE_CONTAINER"),
            cosmos_throughput_mode=os.environ.get("COSMOS_DB_THROUGHPUT_MODE"),
            ai_foundry_endpoint=os.environ.get("AI_FOUNDRY_ENDPOINT"),
            ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY") or None,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
            adf_endpoint=os.environ.get("ADF_ENDPOINT"),
            adf_key=os.environ.get("ADF_KEY"),
        )

    def _build_default_llm(self):
        """Return a callable that performs a chat-completion against an OpenAI-compatible endpoint.

        Uses Azure OpenAI when ``AI_FOUNDRY_ENDPOINT`` is set, otherwise the
        public OpenAI API via ``OPENAI_API_KEY``.
        """
        endpoint = os.environ.get("AI_FOUNDRY_ENDPOINT")
        if endpoint:
            from openai import AzureOpenAI  # type: ignore

            api_key = os.environ.get("AI_FOUNDRY_API_KEY")
            api_version = os.environ.get("AI_FOUNDRY_API_VERSION", "2024-10-21")
            if api_key:
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                )
            else:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                client = AzureOpenAI(
                    azure_ad_token_provider=token_provider,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                )
        else:
            from openai import OpenAI  # type: ignore

            client = OpenAI()

        def _call(messages, model, max_tokens, temperature):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        return _call

    # -- identity helpers ---------------------------------------------------

    def _user_id(self, context_id: Any) -> str:
        # Per-context isolation prevents cross-context retrieval leakage,
        # which matters for accurate retrieval datasets.
        return f"mab::{self.cfg.run_id}::{self.sub_dataset}::ctx{context_id}"

    @staticmethod
    def _thread_id(context_id: Any) -> str:
        return f"context::{context_id}"

    def _writer_for(self, context_id: Any) -> Optional[str]:
        """Pick the writing agent for the next memorize call (round-robin).

        Returns ``None`` when multi-agent mode is disabled, which preserves
        the original single-writer behavior exactly.
        """
        agents = self.cfg.writer_agents
        if not agents:
            return None
        seq = self._write_seq.get(context_id, 0)
        self._write_seq[context_id] = seq + 1
        return agents[seq % len(agents)]

    # -- public API ---------------------------------------------------------

    def memorize(
        self,
        message: str,
        *,
        query_id: Any = None,
        context_id: Any = None,
    ) -> str:
        """Store a benchmark chunk as a ``turn`` memory in Cosmos DB."""
        user_id = self._user_id(context_id)
        thread_id = self._thread_id(context_id)
        self._context_start.setdefault(context_id, time.time())

        metadata: dict[str, Any] = {
            "run_id": self.cfg.run_id,
            "sub_dataset": self.sub_dataset,
            "dataset": self.dataset,
            "context_id": str(context_id) if context_id is not None else None,
            "query_id": str(query_id) if query_id is not None else None,
            "agent_name": self.cfg.agent_name,
        }
        add_kwargs: dict[str, Any] = dict(
            user_id=user_id,
            role="user",
            content=message,
            memory_type="turn",
            thread_id=thread_id,
            metadata=metadata,
        )
        writer = self._writer_for(context_id)
        if writer is not None:
            # Attribute this write to one of several agents sharing the same
            # (user_id, thread_id). Stored in metadata (always filterable
            # post-hoc) and, when supported, as an "agent:<id>" tag.
            metadata["agent_id"] = writer
            if self._add_supports_tags:
                add_kwargs["tags"] = [f"agent:{writer}"]
        self._client.add_cosmos(**add_kwargs)
        return "Memorized"

    def finalize_context(self, context_id: Any) -> None:
        """Run mode-dependent post-processing once per context.

        Called automatically before the first query for a given ``context_id``.
        Safe to invoke repeatedly; subsequent calls are no-ops.
        """
        user_id = self._user_id(context_id)
        thread_id = self._thread_id(context_id)
        key = (user_id, thread_id)
        if key in self._finalized:
            return
        self._finalized.add(key)

        mode = self.cfg.store_mode
        if mode == "turns_only":
            return

        if mode in ("facts_only", "summary_plus_facts"):
            try:
                self._client.extract_facts(
                    user_id=user_id,
                    thread_id=thread_id,
                    poll_interval=self.cfg.processing_poll_interval,
                    timeout=self.cfg.processing_timeout,
                )
            except Exception as exc:
                logger.exception("extract_facts failed for ctx=%s: %s", context_id, exc)
                raise

        if mode == "summary_plus_facts":
            try:
                self._client.generate_thread_summary(
                    user_id=user_id,
                    thread_id=thread_id,
                    poll_interval=self.cfg.processing_poll_interval,
                    timeout=self.cfg.processing_timeout,
                )
            except Exception as exc:
                logger.exception("generate_thread_summary failed ctx=%s: %s", context_id, exc)
                raise

        if mode == "user_summary":
            try:
                self._client.generate_user_summary(
                    user_id=user_id,
                    thread_ids=[thread_id],
                    poll_interval=self.cfg.processing_poll_interval,
                    timeout=self.cfg.processing_timeout,
                )
            except Exception as exc:
                logger.exception("generate_user_summary failed ctx=%s: %s", context_id, exc)
                raise

    def query(
        self,
        message: str,
        *,
        query_id: Any = None,
        context_id: Any = None,
    ) -> dict[str, Any]:
        """Retrieve memories and answer the query via the configured LLM."""
        user_id = self._user_id(context_id)
        thread_id = self._thread_id(context_id)
        ctx_start = self._context_start.get(context_id, time.time())

        # Run any required post-ingestion processing (facts/summaries).
        self.finalize_context(context_id)
        memory_construction_time = time.time() - ctx_start
        query_start = time.time()

        retrieved = self._search(message, user_id=user_id, thread_id=thread_id)
        retrieved_block = self._format_retrieved(retrieved)

        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question using only "
            "the information in the provided memory snippets. If the answer is not "
            "supported by the snippets, say you don't know. Be concise."
        )
        user_prompt = (
            f"Memory snippets:\n{retrieved_block}\n\n"
            f"Question: {message}\n\nAnswer:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            output = self._llm(
                messages,
                self.cfg.model,
                self.cfg.max_tokens,
                self.cfg.temperature,
            )
        except Exception as exc:
            logger.exception("LLM call failed ctx=%s qid=%s: %s", context_id, query_id, exc)
            raise

        query_time_len = time.time() - query_start
        full_input = system_prompt + "\n" + user_prompt
        return {
            "output": output,
            "input_len": _count_tokens(full_input, self.cfg.model),
            "output_len": _count_tokens(output, self.cfg.model),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieved_block,
            "retrieved_count": len(retrieved),
            "store_mode": self.cfg.store_mode,
            "search_mode": self.cfg.search_mode,
            "writer_agents": list(self.cfg.writer_agents),
            "query_agents": list(self.cfg.query_agents),
            "retrieval_scope": "per_agent" if self.cfg.query_agents else "shared",
        }

    # -- helpers ------------------------------------------------------------

    def _search(
        self,
        query_text: str,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[dict[str, Any]]:
        types = self.cfg.search_memory_types
        hybrid = self.cfg.search_mode == "hybrid"
        # When restricting retrieval to a subset of agents, over-fetch so the
        # post-filter still has retrieve_num candidates to return.
        filtering = bool(self.cfg.query_agents)
        fetch_k = self.cfg.retrieve_num * 5 if filtering else self.cfg.retrieve_num

        # When a single memory_type is requested we let Cosmos filter; for
        # multiple types we do per-type queries and merge by similarity rank.
        if len(types) == 1:
            results = self._client.search_cosmos(
                search_terms=query_text,
                user_id=user_id,
                thread_id=thread_id,
                memory_type=types[0],
                hybrid_search=hybrid,
                top_k=fetch_k,
            )
            return self._apply_agent_filter(results)

        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        # Keep results in their per-type rank order, interleaved.
        per_type_results = [
            self._client.search_cosmos(
                search_terms=query_text,
                user_id=user_id,
                thread_id=thread_id,
                memory_type=t,
                hybrid_search=hybrid,
                top_k=fetch_k,
            )
            for t in types
        ]
        for rank in range(fetch_k):
            for results in per_type_results:
                if rank >= len(results):
                    continue
                rec = results[rank]
                rid = rec.get("id")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    merged.append(rec)
        return self._apply_agent_filter(merged)

    def _apply_agent_filter(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Restrict to ``query_agents`` (if set) and cap at ``retrieve_num``."""
        if self.cfg.query_agents:
            allowed = set(self.cfg.query_agents)
            records = [r for r in records if self._record_agent(r) in allowed]
        return records[: self.cfg.retrieve_num]

    @staticmethod
    def _record_agent(rec: dict[str, Any]) -> Optional[str]:
        return rec.get("agent_id") or (rec.get("metadata") or {}).get("agent_id")

    @staticmethod
    def _format_retrieved(records: list[dict[str, Any]]) -> str:
        if not records:
            return "(no memory snippets retrieved)"
        lines = []
        for i, rec in enumerate(records, start=1):
            rec_type = rec.get("type", "turn")
            content = (rec.get("content") or "").strip()
            lines.append(f"[{i}] ({rec_type}) {content}")
        return "\n".join(lines)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


__all__ = [
    "AdapterConfig",
    "AgentMemoryToolkitBackend",
    "STORE_MODES",
    "SEARCH_MODES",
    "build_adapter_config",
]
