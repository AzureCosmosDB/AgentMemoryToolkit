"""Durable orchestrators + activity functions.

Each orchestrator is a thin chain of activities that delegate to
:class:`azure.cosmos.agent_memory.services.pipeline.PipelineService`. The pipeline owns
all prompts and business logic; activities are deliberately small.

Every activity is wrapped by :func:`shared.error_redaction.redact_errors`, so a failure reaches the
Durable history only as ``<code> (<ExcType>)``, never as exception text; activity calls retry per
:func:`orchestrators._retry.default_retry_options`.
"""
