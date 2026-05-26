"""Top-level Durable Functions app entry point.

Registers the change-feed trigger and orchestrator/activity blueprints into
a single :class:`df.DFApp`. Azure Functions discovers ``app`` here.
"""

from __future__ import annotations

import azure.durable_functions as df
import azure.functions as func
from orchestrators import extract_memories as extract_memories_bp
from orchestrators import synthesize_procedural as synthesize_procedural_bp
from orchestrators import thread_summary as thread_summary_bp
from orchestrators import user_summary as user_summary_bp
from triggers import change_feed as change_feed_bp

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

app.register_functions(change_feed_bp.bp)
app.register_functions(thread_summary_bp.bp)
app.register_functions(extract_memories_bp.bp)
app.register_functions(synthesize_procedural_bp.bp)
app.register_functions(user_summary_bp.bp)
