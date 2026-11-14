# Operations

Runtime knobs for an Agent-Memory-Toolkit deployment. Most ops levers live in `.env` / Function-app App Settings - change them, restart the consumer, and you're done. Deployment-time knobs (Bicep params bound to `azd env set ...`) live in [`infra/README.md`](../infra/README.md).

## Memory lifecycle (TTL)

| Type | Default TTL | Source |
|---|---:|---|
| turn | 30 d | per-doc ttl set at write; the container default (memories_turns) covers documents without one |
| episodic | 90 d | per-doc ttl set at write (memories container) |
| thread_summary | never | container default (memories_summaries, -1) |
| user_summary | never | container default (memories_summaries, -1) |
| fact | never | container default (memories, -1); supersession handles aging |
| procedural | never | container default (memories, -1); supersession handles aging |
| counter | 90 d | container default (counter container; every document in it) |

The counter container's TTL is set when the container is created, by `create_memory_store()` or the Bicep template. It is the container's default, so it applies to every document in it, not only thread counters: user counters and the extraction watermark kept on a thread counter go with them, each 90 days after its last update. A container that already exists keeps its current setting, because `create_container_if_not_exists` does not change it. When a counter expires, its thread or user starts counting again from zero.

Facts and procedures have no default expiry. Deleting one with `delete_memory` also deletes the records folded into it: duplicates, combined-fact sources and facts retired into a procedure (see [`public_api.md`](public_api.md)).

Override per write:

    client.upsert_memory("user-1", "user", text, memory_type="turn", thread_id="thread-1", ttl=60)   # expires in 60 seconds

Override the turns container default at provision time (it applies to documents written without a `ttl`; the SDK sets its own 30-day `ttl` on every turn):

    azd env set MEMORIES_TURNS_DEFAULT_TTL 86400   # 1 day

## Counter-based trigger configuration

Function-app threshold knobs (`THREAD_SUMMARY_EVERY_N`, `FACT_EXTRACTION_EVERY_N`, `DEDUP_EVERY_N`, `USER_SUMMARY_EVERY_N`, `MAX_BATCH_SIZE`, `DEDUP_POOL_SIZE`, `ACTIVITY_TIME_BUDGET_SECONDS`, `MEMORY_PROCESSOR_OWNER`) are Bicep parameters, documented in [`infra/README.md` → Counter-based trigger configuration](../infra/README.md#counter-based-trigger-configuration-function-app-only). Change them with `azd env set ...` then `azd up`. `MAX_BATCH_SIZE` must be at least 1; a lower value falls back to `20`.

## Reconcile, procedure and extraction knobs

These settings are read from the environment: `.env` for the SDK, App Settings for the Function App. On the Function App, `DEDUP_POOL_SIZE` and `ACTIVITY_TIME_BUDGET_SECONDS` are Bicep parameters (`azd env set DEDUP_POOL_SIZE 100`, then `azd up`); the others below are not, so the code defaults apply unless you set them. Set one on the Function App with `az functionapp config appsettings set`, or add it to `infra/modules/functions.bicep` so `azd up` keeps it.

| Setting | Default | Read by | Effect |
|---|---:|---|---|
| `DEDUP_POOL_SIZE` | `50` (max `500`) | SDK, Function App | The `n` of the automatic reconcile: the most facts the dedup pass checks per run, and the size of the contradiction pool. |
| `ACTIVITY_TIME_BUDGET_SECONDS` | `240` (min `30`) | Function App | Seconds a reconcile or procedure activity keeps starting model calls. Keep it below the app's `functionTimeout`. The activity then saves what it finished and the next run continues from its cursor. |
| `PROCEDURAL_SYNTHESIS_AUTO` | `true` | SDK, Function App | Learn procedures after each reconcile. `false` turns it off. |
| `AMT_EXTRACT_MEMORIES_PROMPT` | `extract_memories-v3.prompty` | SDK, Function App | Fact-extraction prompt. `extract_memories-v2.prompty` and `extract_memories.prompty` select the earlier prompts; an unknown value uses the default. |
| `COSMOS_DB_PREFERRED_REGIONS` | unset | SDK clients | Comma-separated Cosmos DB regions, closest first (`West US 3,West US 2`); the `cosmos_preferred_locations` argument overrides it. The Function App's own Cosmos clients do not read it. |

Boolean settings (`PROCEDURAL_SYNTHESIS_AUTO`, `ENABLE_TURN_EMBEDDINGS`) take `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`, `t`/`f` or `y`/`n`, in any case. The SDK and the Function App share one parser, and an invalid value falls back to the default with a warning.

Cost: the dedup pass makes one `dedup_verify` call per new fact that has close neighbours, and each combined group adds a writer call and a verification call, with one retry. Procedure learning reads at most 50 facts and 50 episodes per run, at 10 sources per model call.

The similarity thresholds are fixed in code, not env settings: a fact is folded only into one of its 5 nearest active facts at cosine similarity of at least 0.85; combining looks down to 0.65, in groups of at most 4 facts, into one fact of at most 800 characters.

