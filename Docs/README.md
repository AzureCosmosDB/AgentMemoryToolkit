# Docs

This folder contains the main project documentation for Agent Memory Toolkit.

## Table of Contents

| Document | Purpose |
|----------|---------|
| [concepts.md](concepts.md) | Explains the core memory model, including memory types (turn, summary, fact, user summary), threads, roles, the processing pipeline, automatic change feed processing, and shared Cosmos throughput configuration. |
| [local_testing.md](local_testing.md) | Covers local setup, environment configuration, RBAC, Cosmos provisioning, running the toolkit and Azure Functions locally, and testing change feed auto-processing with serverless or autoscale container provisioning. |
| [azure_testing.md](azure_testing.md) | Covers Azure deployment, cloud configuration, required services, change feed settings, throughput mode configuration, and validation steps for running the toolkit in Azure. |
| [design_patterns.md](design_patterns.md) | Shows when and how to call CRUD operations, summarization, fact extraction, and memory retrieval in chat and multi-agent applications, including automatic processing via the change feed. |
| [troubleshooting.md](troubleshooting.md) | Helps diagnose common setup, authentication, Cosmos DB, embeddings, Durable Functions, vector search, and change feed issues. |

## Recommended Reading Order

1. Start with [concepts.md](concepts.md) to understand the data model and memory lifecycle.
2. Use [local_testing.md](local_testing.md) to get the toolkit running and validated on your machine.
3. Use [azure_testing.md](azure_testing.md) when you are ready to deploy or validate the full stack in Azure.
4. See [design_patterns.md](design_patterns.md) for integration patterns in real applications.
5. Use [troubleshooting.md](troubleshooting.md) when setup, processing, search, or automatic change feed behavior does not work as expected.