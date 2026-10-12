from __future__ import annotations

import email.utils
from datetime import UTC, datetime

import httpx

from job_hunt.errors import ErrorKind, ProviderError


def retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(float(value), 0)
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value)
        return max((parsed - datetime.now(UTC)).total_seconds(), 0)


def raise_provider_error(provider: str, response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    body = response.text[:1000]
    if response.status_code in {401, 403}:
        kind, retryable = ErrorKind.AUTHENTICATION, False
    elif response.status_code == 429:
        kind, retryable = ErrorKind.RATE_LIMIT, True
    elif response.status_code >= 500:
        kind, retryable = ErrorKind.TRANSIENT_PROVIDER, True
    else:
        kind, retryable = ErrorKind.BUSINESS, False
    raise ProviderError(
        f"{provider} returned HTTP {response.status_code}: {body}",
        kind,
        retryable=retryable,
        provider=provider,
        status_code=response.status_code,
        retry_after=retry_after_seconds(response),
    )
