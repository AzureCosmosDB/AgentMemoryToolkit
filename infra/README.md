# Infra (`azd` + Bicep)

This folder provisions everything the Agent Memory Toolkit needs in a single Azure subscription via the Azure Developer CLI (`azd`). One command deploys the full stack end-to-end.

## What gets provisioned

`azd up` creates **all** of the following:

- **Cosmos DB for NoSQL** — serverless account with the `ai_memory` database and `memories`, `leases`, and `counter` containers
- **AI Foundry** (`Microsoft.CognitiveServices/accounts` with `kind: AIServices`) — with `gpt-4o-mini` and `text-embedding-3-large` deployments
- **User-assigned managed identity (UAMI)** — used by the Function app
- **RBAC role assignments** — Cosmos DB Built-in Data Contributor + Cognitive Services OpenAI User + Storage Blob Data Owner, granted to both the UAMI and the deploying user
- **Function app** — Flex Consumption (Python 3.11), Storage account, App Insights, Log Analytics

The Function app is **always provisioned**, even if you plan to use `InProcessProcessor` only. Flex Consumption is pay-per-execution — at zero traffic the Function app is essentially free (idle cost is the Storage account, ~$0.05/month). The Function app sits idle and unused for in-process workloads.

> Advanced escape hatch: set `azd env set DEPLOY_FUNCTION_APP false` before `azd up` to skip the Function app + its supporting resources entirely. Not recommended unless you have a strong reason.

## Prereqs

- `az` (Azure CLI) and `azd` (Azure Developer CLI) installed
- An Azure subscription with quota for `gpt-4o-mini` and `text-embedding-3-large` in the chosen region (default `eastus2`; allowed: `eastus2`, `swedencentral`, `westus3`, `eastus`)

## Quickstart

```bash
az login
azd auth login

azd env new memorytoolkit-dev
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

## Bring-your-own-resources (BYOR)

Reuse an existing Cosmos account or AI Foundry account by setting the corresponding `azd` env vars **before** `azd up`:

```bash
azd env set USE_EXISTING_COSMOS true
azd env set EXISTING_COSMOS_ACCOUNT my-cosmos
azd env set EXISTING_COSMOS_RESOURCE_GROUP my-rg

azd env set USE_EXISTING_AI_FOUNDRY true
azd env set EXISTING_AI_FOUNDRY_NAME my-aif
azd env set EXISTING_AI_FOUNDRY_RESOURCE_GROUP my-rg

azd up
```

Bicep references these via the `existing` keyword — RBAC and (where the existing resource is in the same RG) container/deployment creation still run.

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

The Function app uses a counter document per `(user_id, thread_id)` to decide when to fire each orchestrator. Tune via app settings (set automatically by `azd up`):

| App setting | Default | Effect |
|---|---|---|
| `THREAD_SUMMARY_EVERY_N` | `10` | Run thread-summary orchestration every N turns within a `(user_id, thread_id)`. |
| `FACT_EXTRACTION_EVERY_N` | `1` | Run fact / episodic / procedural extraction every N turns within a `(user_id, thread_id)`. |
| `USER_SUMMARY_EVERY_N` | `20` | Run user-summary orchestration every N turns from a given `user_id` across all threads. |

Set any value to `0` to **disable auto-triggering** for that orchestrator. Update at runtime with:

```bash
azd env set THREAD_SUMMARY_EVERY_N 8
azd deploy
```

## Cleanup

```bash
azd down --purge
```

`--purge` skips the Cosmos / AI Foundry soft-delete window so the names are free to reuse immediately.

## CI/CD

Generate a GitHub Actions or Azure Pipelines pipeline for the same flow:

```bash
azd pipeline config
```

## Gotchas

| Gotcha | Mitigation |
| --- | --- |
| First-time provisioning is slow (8–15 min for Cosmos + AI Foundry + Function app) | `azd up` shows progress; just wait |
| AI Foundry region constraints — many regions don't have all features / models | Default `AZURE_LOCATION=eastus2`; supported: `eastus2`, `swedencentral`, `westus3` |
| Model deployment quota — fails if the subscription has zero quota for the model in the chosen region | Request quota or change region; error from Azure points to the right doc |
| Cosmos free-tier limit (one per subscription) | Default is **serverless** — no idle cost, no free-tier conflict |
| AAD propagation — RBAC takes 30–90s; the Function app may briefly 403 on its first invocation after deploy | Retry after a minute. `dependsOn` chains in Bicep ensure roles exist before the Function app starts |
| Resource naming rules — Storage ≤24 chars lowercase, AI Foundry has its own | Naming uses `take(uniqueString(...), 13)` and `toLower()` to satisfy all rules |

## Architecture choice — AI Foundry

The Bicep uses a single `Microsoft.CognitiveServices/accounts` resource with `kind: AIServices` (named `aif-<token>`) instead of the full AI Foundry hub + project + ML workspace. The AIServices account exposes the same Azure OpenAI endpoint and supports the same `Cognitive Services OpenAI User` RBAC role, which is everything the toolkit needs for embeddings and chat completions. This avoids the extra Storage / Key Vault / App Insights / ML workspace resources a hub-style deployment would create. This is the pattern used by most current `azd`-based Microsoft samples (e.g. `azure-search-openai-demo`, `openai-chat-app-quickstart`).

## File layout

```
infra/
├── main.bicep                  # subscription-scoped entry point
├── main.parameters.json        # binds ${AZURE_*} env vars
├── abbreviations.json          # standard Azure name prefixes
└── modules/
    ├── cosmos.bicep            # NoSQL serverless account + DB + 3 containers
    ├── ai-foundry.bicep        # AIServices account + 2 model deployments
    ├── functions.bicep         # Flex Consumption Python 3.11 function app
    ├── identity.bicep          # User-assigned managed identity
    └── rbac.bicep              # Cosmos / AI Foundry / Storage role assignments
```
