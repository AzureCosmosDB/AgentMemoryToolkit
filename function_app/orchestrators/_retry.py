"""Shared retry options for activity calls (spec §8.6)."""

from __future__ import annotations

import azure.durable_functions as df


def default_retry_options():
    """``max_attempts=3, first_retry_interval=2s``.

    Note: the installed ``azure.durable_functions`` SDK exposes only
    ``first_retry_interval_in_milliseconds`` and ``max_number_of_attempts``
    on :class:`df.RetryOptions`. Backoff coefficient is not configurable
    via this constructor (defaults are applied internally).
    """
    return df.RetryOptions(
        first_retry_interval_in_milliseconds=2000,
        max_number_of_attempts=3,
    )

