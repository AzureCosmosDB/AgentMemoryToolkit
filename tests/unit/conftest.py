"""Unit-test conftest: install lightweight stubs for Azure Functions packages.

``azure-functions`` and ``azure-durable-functions`` are Azure Functions
runtime packages that are not installed in the library's dev dependencies —
they're only needed when *running* the function app, not when testing the
helper logic inside it.  This conftest inserts minimal stubs into
``sys.modules`` before any test module is imported so that
``tests/unit/test_changefeed.py`` (and any other test that imports from
``azure_functions/``) can be collected without the runtime packages present.
"""

import sys
import types


def _install_azure_functions_stubs() -> None:
    # Inject stub sub-modules directly into sys.modules.  Do NOT create or
    # overwrite the top-level ``azure`` entry — that is a real namespace
    # package provided by azure-cosmos (a core dependency), and replacing it
    # with a plain module breaks its sub-package imports.

    # ----- azure.functions -----
    if "azure.functions" not in sys.modules:
        af = types.ModuleType("azure.functions")

        class _AuthLevel:
            FUNCTION = "function"

        af.AuthLevel = _AuthLevel
        af.DocumentList = list
        af.HttpRequest = object
        af.HttpResponse = object
        sys.modules["azure.functions"] = af

    # ----- azure.durable_functions -----
    if "azure.durable_functions" not in sys.modules:
        adf = types.ModuleType("azure.durable_functions")

        class _Noop:
            """A do-nothing stand-in for DFApp / Blueprint decorators."""

            def __init__(self, *args, **kwargs):
                pass

            def register_functions(self, *args, **kwargs):
                pass

            def route(self, *args, **kwargs):
                return lambda fn: fn

            def durable_client_input(self, *args, **kwargs):
                return lambda fn: fn

            def orchestration_trigger(self, *args, **kwargs):
                return lambda fn: fn

            def cosmos_db_trigger(self, *args, **kwargs):
                return lambda fn: fn

            def activity_trigger(self, *args, **kwargs):
                return lambda fn: fn

        adf.Blueprint = _Noop
        adf.DFApp = _Noop
        adf.DurableOrchestrationContext = object
        sys.modules["azure.durable_functions"] = adf


_install_azure_functions_stubs()
