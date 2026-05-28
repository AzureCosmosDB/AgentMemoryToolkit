# Infra (`azd` + Bicep)

This folder provisions everything the Agent Memory Toolkit needs in a single Azure subscription via the Azure Developer CLI (`azd`). One command deploys the full stack end-to-end.

## What gets provisioned

`azd up` creates **all** of the following:

- **Cosmos DB for NoSQL** — serverless account with the `ai_memory` database and the `memories`, `memories_turns`, `leases`, and `counter` containers
- **AI Foundry** (`Microsoft.CognitiveServices/accounts` with `kind: AIServices`) — with `gpt-4o-mini` and `text-embedding-3-large` deployments
- **User-assigned managed identity (UAMI)** — used by the Function app
- **RBAC role assignments** — Cosmos DB Built-in Data Reader + Contributor, Cognitive Services OpenAI User, Storage Blob/Queue/Table data roles, granted to both the UAMI and the deploying user (full table in [Identity & RBAC](#identity--rbac))
- **Function app** — Flex Consumption (Python 3.11), Storage account, App Insights, Log Analytics

The Function app is **always provisioned**, even if you plan to use `InProcessProcessor` only. Flex Consumption is pay-per-execution — at zero traffic the Function app is essentially free (idle cost is the Storage account, ~$0.05/month). The Function app sits idle and unused for in-process workloads.

> Advanced escape hatch: set `azd env set DEPLOY_FUNCTION_APP false` and run `azd provision` to skip the Function app + its supporting resources entirely. Not recommended unless you have a strong reason. See [SDK-only mode](#sdk-only-mode-skip-the-function-app) below for the full procedure.

> **Bring-your-own-resources is not supported.** If you already have a Cosmos account or AI Foundry account you want to reuse, point the SDK and Function app at them via the standard `COSMOS_DB_ENDPOINT` / `AI_FOUNDRY_ENDPOINT` environment variables and skip `azd up` entirely — you only need the Bicep when you want the toolkit to manage the accounts for you. Wiring BYO accounts into this template proved fragile (cross-RG scoping, account-already-exists races, partial RBAC) for low real-world value.

## Prereqs

- `az` (Azure CLI) and `azd` (Azure Developer CLI) installed
- An Azure subscription with quota for `gpt-4o-mini` and `text-embedding-3-large` in the chosen region (default `eastus2`; allowed: `eastus2`, `swedencentral`, `westus3`, `eastus`)

## Quickstart

```bash
az login
azd auth login

azd env new memorytoolkit-dev
azd env set AZURE_LOCATION eastus2     # required — subscription-scoped Bicep needs it
# Optional: pin a different region
# azd env set AZURE_LOCATION swedencentral

azd up
# ~10 min later: provisioned + function code deployed
```

`azd` writes resource outputs to `.azure/<env-name>/.env`:

```
COSMOS_DB_ENDPOINT=...
COSMOS_DB_DATABASE=ai_memory
COSMOS_DB_CONTAINER=memories
AI_FOUNDRY_ENDPOINT=...
AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AI_FOUNDRY_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
FUNCTION_APP_NAME=func-...
FUNCTION_APP_URL=https://func-....azurewebsites.net
```

Source it before running samples or tests:

```bash
set -a && . ./.azure/memorytoolkit-dev/.env && set +a
```

## SDK-only mode (skip the Function app)

If you only ever plan to use the in-process `MemoryProcessor` and want to keep the Bicep footprint minimal — no Function app, no Storage account, no App Insights, no Log Analytics:

```bash
azd env new memorytoolkit-sdkonly
azd env set AZURE_LOCATION eastus2
azd env set DEPLOY_FUNCTION_APP false
azd provision   # skips Function app, Storage, App Insights, Log Analytics, storage RBAC
```

> Use **`azd provision`** (not `azd up`) for SDK-only. `azd up` always invokes `azd deploy --all`, which looks for a resource tagged `azd-service-name: function_app`. With the Function app turned off there is no such resource and the deploy step fails. `azd provision` runs Bicep without the deploy step.

## Model / deployment names

Two concepts kept separate:

| Concept | What it is | Default |
|---|---|---|
| **Model name** (`*_MODEL_NAME`) | The catalog model published by Azure OpenAI (e.g. `gpt-4o-mini`, `text-embedding-3-large`). | `gpt-4o-mini` / `text-embedding-3-large` |
| **Deployment name** (`*_DEPLOYMENT_NAME`) | The name *you* give the deployment in your AOAI account. Can be anything. | empty → defaults to model name |

Override either before `azd up`:

```bash
# Use a different catalog model with the default deployment name
azd env set AI_FOUNDRY_CHAT_MODEL_NAME gpt-4o

# Or pin a custom deployment name (existing or to-be-created)
azd env set AI_FOUNDRY_CHAT_DEPLOYMENT_NAME my-prod-chat
azd env set AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME my-prod-embed
```

The `*_DEPLOYMENT_NAME` value is what the SDK and Function app pass as the `model=` argument to the Azure OpenAI client at runtime.

## Counter-based trigger configuration (Function app only)

The Function app uses a counter document per `(user_id, thread_id)` to decide when to fire each orchestrator. Every knob is a Bicep parameter bound to an `azd env` variable — `azd up` re-renders them on every deploy, so changing a value is a one-line `azd env set ...` followed by `azd up`.

| `azd env` variable | Bicep param | Default | Effect |
|---|---|---|---|
| `THREAD_SUMMARY_EVERY_N` | `threadSummaryEveryN` | `10` | Run thread-summary orchestration every N turns within a `(user_id, thread_id)`. `0` disables it. |
| `FACT_EXTRACTION_EVERY_N` | `factExtractionEveryN` | `5` | Run fact / episodic / procedural extraction every N turns within a `(user_id, thread_id)`. `0` disables it. |
| `DEDUP_EVERY_N` | `dedupEveryN` | `5` | Run fact dedup every Nth fact-extraction (so dedup actually fires every `FACT_EXTRACTION_EVERY_N × DEDUP_EVERY_N` turns). |
| `USER_SUMMARY_EVERY_N` | `userSummaryEveryN` | `20` | Run user-summary orchestration every N turns from a given `user_id` across all threads. `0` disables it. |
| `MAX_BATCH_SIZE` | `maxBatchSize` | `20` | Maximum number of change-feed items processed per orchestration batch. |
| `MEMORY_PROCESSOR_OWNER` | `memoryProcessorOwner` | `durable` | Backend that owns processing. `durable` = this function-app fleet owns it; SDK clients pointed at the same container will skip auto-triggering. `inprocess` = SDK owns processing instead. |

The defaults are **deliberately higher than the SDK in-process defaults** (the in-process backend uses `FACT_EXTRACTION_EVERY_N=1` for prototype/demo UX). The Function-app fleet pays real per-turn LLM cost on every orchestrator invocation; amortizing keeps it affordable for production traffic. Override either layer to match your workload.

Set any value to `0` to **disable auto-triggering** for that orchestrator. Update at runtime with:

```bash
azd env set THREAD_SUMMARY_EVERY_N 8
azd up   # re-runs provisioning and pushes new App Settings
```

`azd deploy` (code-only) is also supported but will not pick up changes to threshold values because they live in Bicep; always use `azd up` after threshold changes.

## Identity & RBAC

Every data-plane role is granted at the resource (account) scope — never at subscription or RG level. `principalType` is set explicitly on every standard `Microsoft.Authorization/roleAssignments` so first-deploy from a freshly-created service principal succeeds without the usual "PrincipalNotFound" RBAC race.

| Resource | Built-in role | Granted to | Why |
|---|---|---|---|
| Cosmos DB account | `00000000-0000-0000-0000-000000000001` — Cosmos DB Built-in Data Reader | UAMI + deploying user | Explicit read-only scope. Granted alongside Data Contributor so downstream consumers (audit dashboards, analytics jobs) can run as the same identity but be validated by security review as needing only the Reader scope. Cosmos uses its own `sqlRoleAssignments` resource type which does not accept `principalType` (Cosmos enforces it internally via `principalId`). |
| Cosmos DB account | `00000000-0000-0000-0000-000000000002` — Cosmos DB Built-in Data Contributor | UAMI + deploying user | Data-plane reads/writes from Function app + local samples. |
| AI Foundry account | `5e0bd9bd-7b93-4f28-af87-19fc36ad61bd` — Cognitive Services OpenAI User | UAMI + deploying user | Inference (chat + embeddings) from Function app + local samples. |
| Storage account | `b7e6dc6d-f1e8-4753-8033-0f276bb0955b` — Storage Blob Data Owner | UAMI + deploying user | Function-app deployment-from-blob, Durable history blobs, local sample blob inspection. Owner (not Contributor) keeps `azd deploy` symmetric with manual ops scripts that may need lease/ACL operations. |
| Storage account | `974c5e8b-45b9-4653-ba55-5f855dd0fb88` — Storage Queue Data Contributor | UAMI only | Durable Functions task hub (default Azure Storage provider) uses Queues for orchestration messages. Without this, the very first orchestration start returns 403 even though Blob is fine. |
| Storage account | `0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3` — Storage Table Data Contributor | UAMI only | Durable Functions history Tables. Same 403 symptom as queues if missing. |

All Storage roles are skipped when `DEPLOY_FUNCTION_APP=false` (no Storage account = nothing to grant on).

## Cleanup

For a same-name re-deploy, the simplest path is `azd down --purge` followed by `azd up`. `--purge` instructs `azd` to skip the Cosmos / AI Foundry soft-delete window so the names are free to reuse immediately.

```bash
azd down --purge
```

## CI/CD

Generate a GitHub Actions or Azure Pipelines pipeline for the same flow:

```bash
azd pipeline config
```

## Gotchas

| Gotcha | Mitigation |
| --- | --- |
| First-time provisioning is slow (8–15 min for Cosmos + AI Foundry + Function app) | `azd up` shows progress; just wait |
| `location` property missing — subscription-scoped Bicep requires `AZURE_LOCATION` | Always run `azd env set AZURE_LOCATION eastus2` after `azd env new` |
| AI Foundry region constraints — many regions don't have all features / models | Default `AZURE_LOCATION=eastus2`; supported: `eastus2`, `swedencentral`, `westus3`, `eastus` |
| Model deployment quota — fails if the subscription has zero quota for the model in the chosen region | Request quota or change region; error from Azure points to the right doc |
| Cosmos free-tier limit (one per subscription) | Default is **serverless** — no idle cost, no free-tier conflict |
| AAD propagation — RBAC takes 30–90s; the Function app may briefly 403 on its first invocation after deploy | Retry after a minute. `dependsOn` chains in Bicep ensure roles exist before the Function app starts |
| Resource naming rules — Storage ≤24 chars lowercase, AI Foundry has its own | Naming uses `take(uniqueString(...), 13)` and `toLower()` to satisfy all rules |

## Architecture choice — AI Foundry

The Bicep uses a single `Microsoft.CognitiveServices/accounts` resource with `kind: AIServices` (named `aif-<token>`) instead of the full AI Foundry hub + project + ML workspace. The AIServices account exposes the same Azure OpenAI endpoint and supports the same `Cognitive Services OpenAI User` RBAC role, which is everything the toolkit needs for embeddings and chat completions. This avoids the extra Storage / Key Vault / App Insights / ML workspace resources a hub-style deployment would create. This is the pattern used by most current `azd`-based Microsoft samples (e.g. `azure-search-openai-demo`, `openai-chat-app-quickstart`).

## File layout

```
infra/
├── main.bicep                        # subscription-scoped entry point
├── main.parameters.json              # binds ${AZURE_*} env vars to Bicep params
├── abbreviations.json                # standard Azure name prefixes
└── modules/
    ├── identity.bicep                # User-assigned managed identity
    ├── cosmos.bicep                  # Cosmos DB NoSQL serverless account + database + 4 containers
    ├── cosmos-rbac.bicep             # Cosmos data-plane role assignments
    ├── ai-foundry.bicep              # Cognitive Services AIServices account + chat + embedding deployments
    ├── ai-foundry-rbac.bicep         # Cognitive Services OpenAI User role assignments
    ├── functions.bicep               # Flex Consumption Python 3.11 function app + Storage + App Insights + Log Analytics
    └── storage-rbac.bicep            # Blob/Queue/Table role assignments for the function-app UAMI
```

> Runtime ops (TTL tuning, monitoring, scaling beyond defaults) live in [`Docs/operations.md`](../Docs/operations.md). This README owns everything related to `azd up` / Bicep / deployment.
