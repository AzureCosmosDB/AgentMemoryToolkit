# Infra (`azd` + Bicep)

This folder provisions everything the Agent Memory Toolkit needs in a single Azure subscription via the Azure Developer CLI (`azd`). One command deploys the full stack end-to-end.

## What gets provisioned

`azd up` creates **all** of the following:

- **Cosmos DB for NoSQL** - serverless account (vector and full-text search enabled) with the `ai_memory` database and this container topology:

  | Container | Purpose |
  | --- | --- |
  | `memories` | Facts, episodics, and procedural memories; vector + FTS index |
  | `memories_turns` | Turns; vector + FTS index, 30-day default TTL (`MEMORIES_TURNS_DEFAULT_TTL`, in seconds), change-feed binding target |
  | `memories_summaries` | Thread + user summaries; vector + FTS index (`search_summaries`), composite index for point reads |
  | `leases` | Change-feed checkpoints |
  | `counter` | Atomic counters; 90-day default TTL |
- **AI Foundry** (`Microsoft.CognitiveServices/accounts` with `kind: AIServices`) - with `gpt-4o-mini` and `text-embedding-3-large` deployments
- **User-assigned managed identity (UAMI)** - used by the Function app
- **RBAC role assignments** - Cosmos DB Built-in Data Reader + Contributor, Cognitive Services OpenAI User, Storage Blob/Queue/Table data roles, granted to the UAMI and (except the Storage Queue/Table roles) the deploying user (full table in [Identity & RBAC](#identity--rbac))
- **Function app** - Flex Consumption (Python 3.11), Storage account, App Insights, Log Analytics

The Function app is provisioned by default and is deployed with `MEMORY_PROCESSOR_OWNER=durable`: it processes new turns from the change feed, and SDK clients that load the generated `.env` skip their in-process auto-trigger. To process in your own process instead, run `azd env set MEMORY_PROCESSOR_OWNER inprocess` before `azd up` (the Function app then skips every batch), or use the [SDK-only mode](#sdk-only-mode-skip-the-function-app) below. Flex Consumption is pay-per-execution: with no turns being written the Function app is essentially free (idle cost is the Storage account, ~$0.05/month); otherwise you pay for its executions and, under `durable`, the model calls it makes.

> Advanced escape hatch: set `azd env set DEPLOY_FUNCTION_APP false` and run `azd provision` to skip the Function app + its supporting resources entirely. Not recommended unless you have a strong reason. See [SDK-only mode](#sdk-only-mode-skip-the-function-app) below for the full procedure.

> **Bring-your-own-resources is not supported.** If you already have a Cosmos account or AI Foundry account you want to reuse, point the SDK and Function app at them via the standard `COSMOS_DB_ENDPOINT` / `AI_FOUNDRY_ENDPOINT` environment variables (the Cosmos account needs the `EnableNoSQLVectorSearch` and `EnableNoSQLFullTextSearch` capabilities) and skip `azd up` entirely - you only need the Bicep when you want the toolkit to manage the accounts for you. Wiring BYO accounts into this template proved fragile (cross-RG scoping, account-already-exists races, partial RBAC) for low real-world value.

## Prereqs

- `az` (Azure CLI) and `azd` (Azure Developer CLI) installed
- An Azure subscription with quota for `gpt-4o-mini` and `text-embedding-3-large` in the chosen region (default `eastus2`; allowed: `eastus2`, `swedencentral`, `westus3`, `eastus`)

## Quickstart

```bash
az login
azd auth login

azd env new memorytoolkit-dev
azd env set AZURE_LOCATION eastus2     # required - subscription-scoped Bicep needs it
# Optional: pin a different region
# azd env set AZURE_LOCATION swedencentral

azd up
# ~10 min later: provisioned + function code deployed
```

`azd` writes resource outputs to `.azure/<env-name>/.env`, including:

```
COSMOS_DB_ENDPOINT=...
COSMOS_DB_DATABASE=ai_memory
COSMOS_DB_MEMORIES_CONTAINER=memories
COSMOS_DB_TURNS_CONTAINER=memories_turns
COSMOS_DB_SUMMARIES_CONTAINER=memories_summaries
AI_FOUNDRY_ENDPOINT=...
AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AI_FOUNDRY_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
FUNCTION_APP_NAME=func-...
FUNCTION_APP_URL=https://func-....azurewebsites.net
MEMORY_PROCESSOR_OWNER=durable
```

Source it before running samples or tests:

```bash
set -a && . ./.azure/memorytoolkit-dev/.env && set +a
```

## SDK-only mode (skip the Function app)

If you only ever plan to use `InProcessProcessor` and want to keep the Bicep footprint minimal - no Function app, no Storage account, no App Insights, no Log Analytics, no `leases` or `counter` container:

```bash
azd env new memorytoolkit-sdkonly
azd env set AZURE_LOCATION eastus2
azd env set DEPLOY_FUNCTION_APP false
azd provision   # skips Function app, Storage, App Insights, Log Analytics, storage RBAC, leases + counter containers
```

> Use **`azd provision`** (not `azd up`) for SDK-only. `azd up` always invokes `azd deploy --all`, which looks for a resource tagged `azd-service-name: function_app`. With the Function app turned off there is no such resource and the deploy step fails. `azd provision` runs Bicep without the deploy step.

When `DEPLOY_FUNCTION_APP=false`, the Bicep template automatically sets `MEMORY_PROCESSOR_OWNER=inprocess` in the generated `.env` so the in-process processor's auto-trigger fires. If you previously deployed with `DEPLOY_FUNCTION_APP=true` and are switching to SDK-only mode, also clear the override explicitly:

```bash
azd env set MEMORY_PROCESSOR_OWNER inprocess
```

The in-process auto-trigger counts turns in the `counter` container, and the client's `create_memory_store()` also creates `leases`; this mode creates neither. `CosmosMemoryClient(cosmos_endpoint=...)` calls `create_memory_store()` when it is constructed, which creates any missing container. Cosmos DB allows that with key auth but rejects it for Azure AD data-plane roles, so the constructor then raises: create `counter` and `leases` first (settings as in `infra/modules/cosmos.bicep`), or connect once with key auth.

## Model / deployment names

Two concepts kept separate:

| Concept | What it is | Default |
|---|---|---|
| **Model name** (`*_MODEL_NAME`) | The catalog model published by Azure OpenAI (e.g. `gpt-4o-mini`, `text-embedding-3-large`). | `gpt-4o-mini` / `text-embedding-3-large` |
| **Deployment name** (`*_DEPLOYMENT_NAME`) | The name *you* give the deployment in your AOAI account. Can be anything. | empty → defaults to model name |

Override either before `azd up`:

```bash
# Use a different catalog model with the default deployment name (also set its version, below)
azd env set AI_FOUNDRY_CHAT_MODEL_NAME gpt-4o

# Or pin a custom deployment name (existing or to-be-created)
azd env set AI_FOUNDRY_CHAT_DEPLOYMENT_NAME my-prod-chat
azd env set AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME my-prod-embed
```

The `*_DEPLOYMENT_NAME` value is what the SDK and Function app pass as the `model=` argument to the Azure OpenAI client at runtime.

The model versions are pinned in `infra/modules/ai-foundry.bicep` (`chatModelVersion` is `2024-07-18`, `embeddingModelVersion` is `1`), so a different model also needs its version set there. A newer model can also need a newer API version: `azd env set AZURE_OPENAI_API_VERSION <version>` (default `2024-12-01-preview`).

### Model capacity

Both deployments are created with 30K tokens per minute (TPM). Fact extraction, reconcile (a duplicate check per new fact that has close neighbours, plus combining) and procedure learning all use the chat model, and each extraction call sends a long prompt, so a busy workload can exceed the default. Raise the capacity before `azd up`, within your subscription's quota for the model:

```bash
azd env set AI_FOUNDRY_CHAT_CAPACITY 100        # thousands of TPM
azd env set AI_FOUNDRY_EMBEDDING_CAPACITY 100
```

To cut chat usage instead, see the cost knobs under [Counter-based trigger configuration](#counter-based-trigger-configuration-function-app-only).

## Counter-based trigger configuration (Function app only)

The Function app uses a counter document per `(user_id, thread_id)` to decide when to fire each orchestrator. Every knob is a Bicep parameter bound to an `azd env` variable - `azd up` re-renders them on every deploy, so changing a value is a one-line `azd env set ...` followed by `azd up`.

| `azd env` variable | Bicep param | Default | Effect |
|---|---|---|---|
| `THREAD_SUMMARY_EVERY_N` | `threadSummaryEveryN` | `10` | Run thread-summary orchestration every N turns within a `(user_id, thread_id)`. `0` disables it. |
| `FACT_EXTRACTION_EVERY_N` | `factExtractionEveryN` | `2` | Run fact extraction every N turns within a `(user_id, thread_id)`. Reconcile and procedure learning run on multiples of it (see `DEDUP_EVERY_N`). `0` disables all three. |
| `DEDUP_EVERY_N` | `dedupEveryN` | `5` | Run reconcile (fold duplicate facts, combine facts about one specific thing, resolve contradictions) every Nth fact-extraction, so it fires every `FACT_EXTRACTION_EVERY_N × DEDUP_EVERY_N` turns. Procedure learning is chained after it (unless `PROCEDURAL_SYNTHESIS_AUTO` is `false`). `0` disables both. |
| `USER_SUMMARY_EVERY_N` | `userSummaryEveryN` | `20` | Run user-summary orchestration every N turns from a given `user_id` across all threads. `0` disables it. |
| `EPISODE_EVAL_EVERY_N` | `episodeEvalEveryN` | `4` | Check a thread's open episode for a boundary every N turns within a `(user_id, thread_id)`. `0` disables episodic memory. |
| `EPISODE_IDLE_GAP_SECONDS` | `episodeIdleGapSeconds` | `1800` | A gap longer than this between two consecutive turns closes the open episode. |
| `EPISODE_TOPIC_DRIFT` | `episodeTopicDrift` | `0` | Cosine distance from the open episode's centroid past which a new turn closes it. `0` turns drift detection off, leaving the idle gap and the size cap. |
| `EPISODE_MAX_TURNS` | `episodeMaxTurns` | `40` | An open episode is closed once it reaches this many turns. |
| `EPISODE_MIN_TURNS` | `episodeMinTurns` | `2` | Fewest turns before an idle gap or topic drift may close an episode. |
| `MAX_BATCH_SIZE` | `maxBatchSize` | `20` | Newest items a summary run reads: turns for a thread summary, new memories per thread for a user summary. Extraction is not capped by it; it reads every turn since its last successful run. At least 1; a lower value falls back to `20`. |
| `DEDUP_POOL_SIZE` | `dedupPoolSize` | `50` | The `n` of the automatic reconcile: the most new facts its duplicate pass checks per run (each one with close neighbours costs a model call; facts beyond `n` wait for the next run) and the size of its contradiction pool. `1`-`500`. |
| `ACTIVITY_TIME_BUDGET_SECONDS` | `activityTimeBudgetSeconds` | `240` | Seconds a reconcile or procedure-learning activity keeps starting model calls. It then saves what it finished and the next run continues from its cursor. At least `30`; keep it below the function app timeout (30 minutes by default on Flex Consumption). |
| `MEMORY_PROCESSOR_OWNER` | `memoryProcessorOwner` | `durable` | Backend that owns processing. `durable` = this function-app fleet owns it; SDK clients pointed at the same container will skip auto-triggering. `inprocess` = SDK owns processing instead. |

The Function-app defaults match the SDK in-process defaults so behavior is consistent across backends. The Function-app fleet pays real per-turn LLM cost on every orchestrator invocation; bump `FACT_EXTRACTION_EVERY_N` / `DEDUP_EVERY_N` for cost-sensitive production traffic.

`PROCEDURAL_SYNTHESIS_AUTO` (`true`) and `AMT_EXTRACT_MEMORIES_PROMPT` (`extract_memories-v3.prompty`) are Function-app settings that are not Bicep parameters, so the code defaults apply. Set them with `az functionapp config appsettings set`, or add them to `infra/modules/functions.bicep` so `azd up` keeps them. See [`Docs/operations.md`](../Docs/operations.md#reconcile-procedure-and-extraction-knobs).

Set any `*_EVERY_N` value to `0` to **disable auto-triggering** for that orchestrator. Update at runtime with:

```bash
azd env set THREAD_SUMMARY_EVERY_N 8
azd up   # re-runs provisioning and pushes new App Settings
```

`azd deploy` (code-only) is also supported but will not pick up changes to threshold values because they live in Bicep; always use `azd up` after threshold changes.

## Identity & RBAC

Every data-plane role is granted at the resource (account) scope - never at subscription or RG level. `principalType` is set explicitly on every standard `Microsoft.Authorization/roleAssignments` so first-deploy from a freshly-created service principal succeeds without the usual "PrincipalNotFound" RBAC race.

| Resource | Built-in role | Granted to | Why |
|---|---|---|---|
| Cosmos DB account | `00000000-0000-0000-0000-000000000001` - Cosmos DB Built-in Data Reader | UAMI + deploying user | Explicit read-only scope. Granted alongside Data Contributor so downstream consumers (audit dashboards, analytics jobs) can run as the same identity but be validated by security review as needing only the Reader scope. Cosmos uses its own `sqlRoleAssignments` resource type which does not accept `principalType` (Cosmos enforces it internally via `principalId`). |
| Cosmos DB account | `00000000-0000-0000-0000-000000000002` - Cosmos DB Built-in Data Contributor | UAMI + deploying user | Data-plane reads/writes from Function app + local samples. |
| AI Foundry account | `5e0bd9bd-7b93-4f28-af87-19fc36ad61bd` - Cognitive Services OpenAI User | UAMI + deploying user | Inference (chat + embeddings) from Function app + local samples. |
| Storage account | `b7e6dc6d-f1e8-4753-8033-0f276bb0955b` - Storage Blob Data Owner | UAMI + deploying user | Function-app deployment-from-blob, Durable history blobs, local sample blob inspection. Owner (not Contributor) keeps `azd deploy` symmetric with manual ops scripts that may need lease/ACL operations. |
| Storage account | `974c5e8b-45b9-4653-ba55-5f855dd0fb88` - Storage Queue Data Contributor | UAMI only | Durable Functions task hub (default Azure Storage provider) uses Queues for orchestration messages. Without this, the very first orchestration start returns 403 even though Blob is fine. |
| Storage account | `0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3` - Storage Table Data Contributor | UAMI only | Durable Functions history Tables. Same 403 symptom as queues if missing. |

All Storage roles are skipped when `DEPLOY_FUNCTION_APP=false` (no Storage account = nothing to grant on).

## Cleanup

For a same-name re-deploy, the simplest path is `azd down --purge` followed by `azd up`. `--purge` instructs `azd` to skip the AI Foundry / Log Analytics soft-delete window so the names are free to reuse immediately.

```bash
azd down --purge
```

## CI/CD

Generate a GitHub Actions or Azure Pipelines pipeline for the same flow:

```bash
azd pipeline config
```

If the pipeline signs in as a service principal, also run `azd env set AZURE_PRINCIPAL_TYPE ServicePrincipal`. The default, `User`, makes the AI Foundry and Storage role assignments fail with `PrincipalTypeMismatch`.

## Gotchas

| Gotcha | Mitigation |
| --- | --- |
| First-time provisioning is slow (8–15 min for Cosmos + AI Foundry + Function app) | `azd up` shows progress; just wait |
| `location` property missing - subscription-scoped Bicep requires `AZURE_LOCATION` | Always run `azd env set AZURE_LOCATION eastus2` after `azd env new` |
| AI Foundry region constraints - many regions don't have all features / models | Default `AZURE_LOCATION=eastus2`; supported: `eastus2`, `swedencentral`, `westus3`, `eastus` |
| Model deployment quota - fails if the subscription has zero quota for the model in the chosen region | Request quota or change region; error from Azure points to the right doc |
| Cosmos free-tier limit (one per subscription) | Default is **serverless** - no idle cost, no free-tier conflict |
| AAD propagation - RBAC takes 30–90s; the Function app may briefly 403 on its first invocation after deploy | Retry after a minute |
| Resource naming rules - Storage ≤24 chars lowercase, AI Foundry has its own | Naming uses `take(uniqueString(...), 13)` and `toLower()` to satisfy all rules |

## Architecture choice - AI Foundry

The Bicep uses a single `Microsoft.CognitiveServices/accounts` resource with `kind: AIServices` (named `aif-<token>`) instead of the full AI Foundry hub + project + ML workspace. The AIServices account exposes the same Azure OpenAI endpoint and supports the same `Cognitive Services OpenAI User` RBAC role, which is everything the toolkit needs for embeddings and chat completions. This avoids the extra Storage / Key Vault / App Insights / ML workspace resources a hub-style deployment would create. This is the pattern used by most current `azd`-based Microsoft samples (e.g. `azure-search-openai-demo`, `openai-chat-app-quickstart`).

## File layout

```
infra/
├── main.bicep                        # subscription-scoped entry point
├── main.parameters.json              # binds `azd env` variables to Bicep params
├── abbreviations.json                # standard Azure name prefixes
└── modules/
    ├── identity.bicep                # User-assigned managed identity
    ├── cosmos.bicep                  # Cosmos DB NoSQL serverless account + database + 5 containers
    ├── cosmos-rbac.bicep             # Cosmos data-plane role assignments
    ├── ai-foundry.bicep              # Cognitive Services AIServices account + chat + embedding deployments
    ├── ai-foundry-rbac.bicep         # Cognitive Services OpenAI User role assignments
    ├── functions.bicep               # Flex Consumption Python 3.11 function app + Storage + App Insights + Log Analytics
    └── storage-rbac.bicep            # Blob/Queue/Table role assignments for the function-app UAMI
```

> Runtime ops (TTL tuning, reconcile, procedure and extraction settings) live in [`Docs/operations.md`](../Docs/operations.md). This README owns everything related to `azd up` / Bicep / deployment.
