# MemoryAgentBench Integration

This package adds a first-class `agent_memory_toolkit` agent to
[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench), backed
by [`CosmosMemoryClient`](../../agent_memory_toolkit/cosmos_memory_client.py).

It implements the plan in `memoryagentbench_integration_plan.pdf` at the
repo root.

## Layout

| File | Purpose |
|------|---------|
| [adapter.py](adapter.py) | `AgentMemoryToolkitBackend` — memorize / query / finalize |
| [patch_agent.py](patch_agent.py) | Idempotent patch for MemoryAgentBench's `agent.py` |
| [run_benchmark.py](run_benchmark.py) | Orchestrator that patches, validates env, runs `main.py` |
| [configs/AgentMemoryToolkit_gpt-4o-mini.yaml](configs/AgentMemoryToolkit_gpt-4o-mini.yaml) | Sample agent config |
| [configs/agent_memory_toolkit_test.txt](configs/agent_memory_toolkit_test.txt) | Smoke-test config list |

## Setup

1. Clone MemoryAgentBench somewhere outside this repo:
   ```pwsh
   git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench
   ```
2. Create a dedicated Python 3.10 environment for MemoryAgentBench
   (its README pins to 3.10.16). Install MemoryAgentBench requirements,
   then install AgentMemoryToolkit editable from this repo:
   ```pwsh
   pip install -e <path-to-this-repo>/AgentMemoryToolkit
   ```
3. Set the environment variables expected by `CosmosMemoryClient` and the
   chat backend. At minimum:
   - `COSMOS_DB_ENDPOINT`, `COSMOS_DB_DATABASE`, `COSMOS_DB_CONTAINER`
   - `AI_FOUNDRY_ENDPOINT` (Azure OpenAI) **or** `OPENAI_API_KEY`
   - For `facts_only` / `summary_plus_facts` / `user_summary` modes, also
     `ADF_ENDPOINT` (and `ADF_KEY` if used).
4. Copy the sample agent config into MemoryAgentBench:
   ```
   configs/agent_conf/RAG_Agents/gpt-4o-mini/AgentMemoryToolkit_gpt-4o-mini.yaml
   ```

## Run a smoke test

```pwsh
python -m benchmarks.memoryagentbench.run_benchmark `
    --memoryagentbench C:\path\to\MemoryAgentBench `
    --agent-config configs/agent_conf/RAG_Agents/gpt-4o-mini/AgentMemoryToolkit_gpt-4o-mini.yaml `
    --dataset-config configs/data_conf/Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml `
    --max-test-queries-ablation 3
```

The runner:

1. Validates required env vars (extra ones with `--require-processing`).
2. Applies the idempotent patch to MemoryAgentBench's `agent.py` (a `.bak`
   is created the first time).
3. Forces `THREAD_SUMMARY_EVERY_N=FACT_EXTRACTION_EVERY_N=USER_SUMMARY_EVERY_N=0`
   for deterministic timing (override with `--allow-change-feed`).
4. Invokes `python main.py --agent_config ... --dataset_config ...` from
   the MemoryAgentBench checkout, with this repo on `PYTHONPATH` so the
   adapter is importable.

## Identity model

The adapter uses run-scoped IDs to prevent cross-run contamination:

- `user_id  = mab::{run_id}::{sub_dataset}::ctx{context_id}`
- `thread_id = context::{context_id}`

Set `memory_toolkit_run_id` per experiment (or the `MAB_RUN_ID` env var) to
isolate runs. To clean up after a smoke test, query Cosmos by the
`metadata.run_id` field or by `user_id` prefix.

## Modes

| `memory_toolkit_store_mode` | What runs at memorize time | What runs after ingestion (per context) | What you can search |
|-----------------------------|----------------------------|------------------------------------------|---------------------|
| `turns_only`        | `add_cosmos(memory_type='turn')` | nothing | `turn` |
| `facts_only`        | `add_cosmos(memory_type='turn')` | `extract_facts(...)` | `fact` (set `memory_toolkit_search_memory_types: fact`) |
| `summary_plus_facts`| `add_cosmos(memory_type='turn')` | `extract_facts` + `generate_thread_summary` | `fact,summary` |
| `user_summary`      | `add_cosmos(memory_type='turn')` | `generate_user_summary(...)` | `user_summary` |

`memory_toolkit_search_mode` selects `vector` or `hybrid` (RRF) Cosmos search.

## Notes

- The adapter has no runtime dependency on MemoryAgentBench source code; it
  is callable directly from any Python program. The patch script is the
  only piece coupled to MemoryAgentBench.
- For fairness when comparing modes, run `turns_only` first and report it
  as the baseline; report `facts_only` / `summary_plus_facts` as separate
  rows so the LLM-derived memory is not conflated with raw retrieval.
