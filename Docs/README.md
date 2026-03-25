# Docs

This folder contains the main project documentation for Agent Memory Toolkit.

## Table of Contents

| Document | Purpose |
|----------|---------|
| [concepts.md](concepts.md) | Explains the core memory model, including memory types (turn, summary, fact, user summary), threads, roles, and the processing pipeline. |
| [local_testing.md](local_testing.md) | Covers local setup, environment configuration, RBAC, Cosmos provisioning, and running the toolkit and Azure Functions locally. |
| [azure_testing.md](azure_testing.md) | Covers Azure deployment, cloud configuration, required services, and validation steps for running the toolkit in Azure. |
| [design_patterns.md](design_patterns.md) | Shows when and how to call CRUD operations, summarization, fact extraction, and memory retrieval in chat and multi-agent applications. |

## Recommended Reading Order

1. Start with [concepts.md](concepts.md) to understand the data model and memory lifecycle.
2. Use [local_testing.md](local_testing.md) to get the toolkit running and validated on your machine.
3. Use [azure_testing.md](azure_testing.md) when you are ready to deploy or validate the full stack in Azure.
4. See [design_patterns.md](design_patterns.md) for integration patterns in real applications.