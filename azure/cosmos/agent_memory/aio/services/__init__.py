"""Async service re-exports.

Mirrors :mod:`azure.cosmos.agent_memory.services` but only contains the async
variants. Sync code should import from ``azure.cosmos.agent_memory.services``;
async code should import from ``azure.cosmos.agent_memory.aio.services``.
"""

from __future__ import annotations

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService

__all__ = ["AsyncPipelineService"]
