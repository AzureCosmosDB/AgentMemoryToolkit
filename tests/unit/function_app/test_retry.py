"""Unit tests for ``function_app/orchestrators/_retry.py``.

The helper builds a single ``df.RetryOptions`` instance shared by all
orchestrators. We patch ``df.RetryOptions`` to capture the call args without
depending on the SDK's actual constructor signature (which differs across
``azure-functions-durable`` versions).

The installed ``azure.durable_functions`` SDK exposes only
``first_retry_interval_in_milliseconds`` and ``max_number_of_attempts`` on
:class:`df.RetryOptions`; backoff coefficient is not configurable via this
constructor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from orchestrators import _retry
from orchestrators._retry import default_retry_options


def _patch_retry_ctor(**kwargs):
    """Patch ``df.RetryOptions`` regardless of whether it's a real class or
    a stubbed-out attribute (the unit-test conftest installs a fake
    ``azure.durable_functions`` module that omits ``RetryOptions``)."""
    return patch.object(_retry.df, "RetryOptions", create=True, **kwargs)


def test_default_retry_options_returns_retry_options_instance():
    sentinel = MagicMock(name="RetryOptions")
    with _patch_retry_ctor(return_value=sentinel) as ctor:
        result = default_retry_options()
    assert result is sentinel
    assert ctor.call_count == 1


def test_default_retry_options_uses_three_attempts():
    with _patch_retry_ctor() as ctor:
        default_retry_options()
    kwargs = ctor.call_args.kwargs
    assert kwargs["max_number_of_attempts"] == 3


def test_default_retry_options_uses_two_second_first_interval():
    with _patch_retry_ctor() as ctor:
        default_retry_options()
    kwargs = ctor.call_args.kwargs
    assert kwargs["first_retry_interval_in_milliseconds"] == 2000


def test_default_retry_options_only_passes_supported_kwargs():
    """The real SDK constructor accepts only two kwargs - pinning that here
    so a future change cannot silently re-introduce the original bug."""
    with _patch_retry_ctor() as ctor:
        default_retry_options()
    kwargs = ctor.call_args.kwargs
    assert set(kwargs.keys()) == {
        "first_retry_interval_in_milliseconds",
        "max_number_of_attempts",
    }


def test_default_retry_options_is_called_freshly_each_time():
    """No caching - each call constructs a new RetryOptions."""
    with _patch_retry_ctor(side_effect=lambda **kw: MagicMock()) as ctor:
        a = default_retry_options()
        b = default_retry_options()
    assert ctor.call_count == 2
    assert a is not b
