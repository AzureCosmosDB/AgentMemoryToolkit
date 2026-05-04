"""Path bootstrap for function-app unit tests.

The function-app source lives in ``function_app/`` (a sibling of the SDK
``agent_memory_toolkit/``) and is *not* a package — Azure Functions discovers
modules by file name.  We add the directory to ``sys.path`` here so tests
can ``import shared.counters``, ``import triggers.change_feed`` etc. without
each test file having to repeat the ``sys.path`` dance.
"""

import os
import sys

_FUNCTION_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "function_app"))
if _FUNCTION_APP_DIR not in sys.path:
    sys.path.insert(0, _FUNCTION_APP_DIR)
