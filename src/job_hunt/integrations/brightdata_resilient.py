from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.brightdata import BrightDataProvider
from job_hunt.logging import logger

_RATE_LIMIT_CODES = {"rate_limit", "rate_limited", "too_many_requests"}
_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "too many requests", "http 429", "status 429")


def _text(value: object) -> str:
    return "" if value in (None, "") else str(value).strip()


def _has_usable_job_payload(record: Mapping[str, Any]) -> bool:
    title = _text(record.get("job_title") or record.get("title"))
    company = _text(record.get("company_name") or record.get("company"))
    description = _text(
        record.get("description_text")
        or record.get("job_summary")
        or record.get("job_description")
        or record.get("job_description_formatted")
        or record.get("description")
    )
    return bool(title and company and description)


def _rate_limit_record(record: Mapping[str, Any]) -> bool:
    code = _text(record.get("error_code")).casefold()
    message = _text(record.get("error") or record.get("error_message") or record.get("message")).casefold()
    return code in _RATE_LIMIT_CODES or any(marker in message for marker in _RATE_LIMIT_MARKERS)


class ResilientBrightDataProvider(BrightDataProvider):
    """Add record-level retry semantics on top of Bright Data HTTP handling.

    Bright Data can return HTTP 200 while individual scraper records contain an
    error such as ``error_code=rate_limit``. Those records describe crawler or
    target-site throttling, not API-key capacity. They must therefore surface as
    retryable provider errors without cooling down or rotating the API-key pool.
    """

    def _request_with_keys(
        self,
        *,
        operation: str,
        input_count: int,
        request: Callable[[str], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        rows = super()._request_with_keys(
            operation=operation,
            input_count=input_count,
            request=request,
        )

        rate_limited = [row for row in rows if _rate_limit_record(row)]
        usable = [row for row in rows if _has_usable_job_payload(row)]
        if rate_limited and not usable:
            codes = Counter(_text(row.get("error_code")) or "unknown" for row in rate_limited)
            messages = [
                _text(row.get("error") or row.get("error_message") or row.get("message")) for row in rate_limited
            ]
            first_message = next((message for message in messages if message), "crawler rate limit")
            logger().warning(
                "brightdata_record_rate_limited",
                operation=operation,
                input_count=input_count,
                raw_count=len(rows),
                rate_limited_count=len(rate_limited),
                error_codes=dict(codes),
                error_message=first_message[:300],
            )
            raise ProviderError(
                f"Bright Data crawler rate limited {operation}: {first_message}",
                ErrorKind.RATE_LIMIT,
                retryable=True,
                provider="brightdata",
            )

        return rows
