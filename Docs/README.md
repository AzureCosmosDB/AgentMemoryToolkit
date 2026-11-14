# Docs

This folder contains the main project documentation for Agent Memory Toolkit.

## Table of Contents

| Document | Purpose |
|----------|---------|
| [concepts.md](concepts.md) | Explains the core memory model, including memory types (turn, summary, fact, episodic, procedural, user summary), threads, roles, embeddings and search (result slots), the processing pipeline and fact extraction, secret redaction, reconciliation, automatic change feed processing, and shared Cosmos throughput configuration. |
| [local_testing.md](local_testing.md) | Covers local setup, environment configuration, RBAC, Cosmos provisioning, running the toolkit and Azure Functions locally, and testing change feed auto-processing with serverless or autoscale container provisioning. |
| [azure_testing.md](azure_testing.md) | Covers Azure deployment, cloud configuration, required services, change feed settings, throughput mode configuration, and validation steps for running the toolkit in Azure. |
| [design_patterns.md](design_patterns.md) | Shows when and how to call CRUD operations, summarization, fact extraction, memory retrieval and search slots, procedures and standing instructions, reconciliation, and secret handling and deletion in chat and multi-agent applications, including automatic processing via the change feed. |
| [public_api.md](public_api.md) | Documents the `CosmosMemoryClient` and `AsyncCosmosMemoryClient` methods (connection, CRUD, retrieval, processing, tagging), fact and procedure fields, secret redaction, topology validation, and extension points. |
| [operations.md](operations.md) | Covers runtime settings for a deployment: memory TTLs, counter-based trigger thresholds, and reconcile, procedure, extraction prompt and Cosmos DB region settings. |
| [architecture/orchestrators.md](architecture/orchestrators.md) | Describes the Function App's Durable orchestrators and their activity chains, the time-budgeted reconcile and procedure activities, and how failed activities are reported without exposing message text. |
| [troubleshooting.md](troubleshooting.md) | Helps diagnose common setup, authentication, Cosmos DB, embeddings, vector search (including hidden records and result slots), Durable Functions (including failed activities and logs), change feed, and async client issues. |
| [RELEASING.md](RELEASING.md) | Maintainer runbook for versioning, the pre-flight checklist, cutting a release, the namespace package check, and yanking a bad release. |

## Recommended Reading Order

1. Start with [concepts.md](concepts.md) to understand the data model and memory lifecycle.
2. Use [local_testing.md](local_testing.md) to get the toolkit running and validated on your machine.
3. Use [azure_testing.md](azure_testing.md) when you are ready to deploy or validate the full stack in Azure.
4. See [design_patterns.md](design_patterns.md) for integration patterns in real applications.
5. Keep [public_api.md](public_api.md) at hand for client methods and fields, and use [operations.md](operations.md) to tune TTLs and runtime settings.
6. Read [architecture/orchestrators.md](architecture/orchestrators.md) to see how the Function App chains its processing activities.
7. Use [troubleshooting.md](troubleshooting.md) when setup, processing, search, or automatic change feed behavior does not work as expected.