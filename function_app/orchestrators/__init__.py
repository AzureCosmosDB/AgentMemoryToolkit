"""Durable orchestrators + activity functions.

Each orchestrator is a thin chain of activities that delegate to
:class:`azure.cosmos.agent_memory.services.pipeline.PipelineService`. The pipeline owns
all prompts and business logic; activities are deliberately small.
"""
