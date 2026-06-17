"""Unit tests for the multi-agent shared-memory mode of the adapter."""

from benchmarks.memoryagentbench.adapter import (
    AgentMemoryToolkitBackend,
    build_adapter_config,
)


class FakeClient:
    """Minimal stand-in for CosmosMemoryClient used by the backend."""

    def __init__(self, search_results=None):
        self.adds = []
        self._search_results = search_results or []

    def add_cosmos(self, **kwargs):
        self.adds.append(kwargs)
        return f"id-{len(self.adds)}"

    def search_cosmos(self, **kwargs):
        return list(self._search_results)

    def close(self):
        pass


def _backend(agent_config, search_results=None):
    client = FakeClient(search_results=search_results)
    backend = AgentMemoryToolkitBackend(
        agent_config,
        {"sub_dataset": "s", "dataset": "d"},
        client=client,
        llm_client=lambda *a, **k: "answer",
    )
    return backend, client


def test_build_config_parses_agent_lists():
    cfg = build_adapter_config(
        {
            "memory_toolkit_writer_agents": "researcher, planner",
            "memory_toolkit_query_agents": ["planner"],
        }
    )
    assert cfg.writer_agents == ("researcher", "planner")
    assert cfg.query_agents == ("planner",)


def test_memorize_rotates_writes_across_agents():
    backend, client = _backend({"memory_toolkit_writer_agents": ["a", "b"]})
    for i in range(4):
        backend.memorize(f"msg {i}", context_id="ctx1")
    agents = [c["metadata"]["agent_id"] for c in client.adds]
    assert agents == ["a", "b", "a", "b"]
    assert client.adds[0]["tags"] == ["agent:a"]


def test_memorize_single_writer_is_unchanged():
    backend, client = _backend({})
    backend.memorize("x", context_id="ctx1")
    assert "agent_id" not in client.adds[0]["metadata"]
    assert "tags" not in client.adds[0]


def test_per_agent_filtered_retrieval():
    results = [
        {"id": "1", "content": "from a", "metadata": {"agent_id": "a"}},
        {"id": "2", "content": "from b", "metadata": {"agent_id": "b"}},
        {"id": "3", "content": "from a2", "metadata": {"agent_id": "a"}},
    ]
    backend, _ = _backend(
        {
            "memory_toolkit_writer_agents": ["a", "b"],
            "memory_toolkit_query_agents": ["a"],
            "retrieve_num": 5,
        },
        search_results=results,
    )
    got = backend._search("q", user_id="u", thread_id="t")
    assert [r["id"] for r in got] == ["1", "3"]


def test_shared_retrieval_returns_all_agents():
    results = [
        {"id": "1", "metadata": {"agent_id": "a"}},
        {"id": "2", "metadata": {"agent_id": "b"}},
    ]
    backend, _ = _backend(
        {"memory_toolkit_writer_agents": ["a", "b"], "retrieve_num": 5},
        search_results=results,
    )
    got = backend._search("q", user_id="u", thread_id="t")
    assert {r["id"] for r in got} == {"1", "2"}
