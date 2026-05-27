"""Durable orchestrators + activity functions.

Each orchestrator is a thin chain of activities that delegate to
:class:`agent_memory_toolkit.services.pipeline.PipelineService`. The pipeline owns
all prompts and business logic; activities are deliberately small.
"""
