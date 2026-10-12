from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from job_hunt.logging import logger
from job_hunt.retry import retry_transient


class BaserowRows(Protocol):
    def list_fields(self, table_id: int) -> list[dict[str, Any]]: ...

    def iter_rows(self, table_id: int, **params: object) -> Iterable[Mapping[str, Any]]: ...


def _created_on_field(fields: list[dict[str, Any]]) -> str | None:
    for field in fields:
        field_type = str(field.get("type") or "").strip().casefold()
        name = str(field.get("name") or "").strip()
        if field_type == "created_on" and name:
            return name
    return None


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, Mapping):
        value = value.get("value") or value.get("date")
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _as_date(value: object) -> date | None:
    parsed = _as_datetime(value)
    if parsed is not None:
        return parsed.date()
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _linkedin_job_id(row: Mapping[str, object]) -> str | None:
    value = row.get("Job ID")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        candidate = value.strip()
        return candidate if candidate.isdigit() else None
    return None


def _lookback_seconds(*, lookback_hours: int | float | None, lookback_seconds: int | None) -> int:
    if lookback_hours is not None and lookback_seconds is not None:
        raise ValueError("Specify either lookback_hours or lookback_seconds, not both")
    if lookback_seconds is not None:
        if lookback_seconds <= 0:
            raise ValueError("lookback_seconds must be positive")
        return lookback_seconds
    if lookback_hours is None or lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")
    return max(1, int(lookback_hours * 3600))


def recent_baserow_rows(
    client: BaserowRows,
    table_id: int,
    *,
    lookback_hours: int | float | None = None,
    lookback_seconds: int | None = None,
    now: datetime | None = None,
) -> list[Mapping[str, Any]]:
    """Return rows stored in Baserow during the requested recent window.

    A Baserow Created on field gives exact storage-time semantics. Existing Jobs tables
    without that optional field fall back conservatively to the posting Date field so
    deployment does not require a schema migration.
    """
    resolved_lookback_seconds = _lookback_seconds(
        lookback_hours=lookback_hours,
        lookback_seconds=lookback_seconds,
    )
    reference = now or datetime.now(UTC)
    reference = reference.replace(tzinfo=UTC) if reference.tzinfo is None else reference.astimezone(UTC)
    cutoff = reference - timedelta(seconds=resolved_lookback_seconds)

    created_field = _created_on_field(retry_transient(client.list_fields, table_id))
    order_field = created_field or "Date"
    result: list[Mapping[str, Any]] = []

    if created_field is None:
        logger().warning(
            "baserow_recent_job_ids_using_posted_date_fallback",
            table_id=table_id,
            lookback_seconds=resolved_lookback_seconds,
        )

    # Sorting newest-first bounds the scan by the requested lookback instead of the
    # lifetime size of the Jobs table. Lazy pagination stops requesting older pages.
    for raw in client.iter_rows(table_id, order_by=f"-{order_field}"):
        if created_field is not None:
            created_at = _as_datetime(raw.get(created_field))
            if created_at is None:
                continue
            if created_at < cutoff:
                break
        else:
            posted_date = _as_date(raw.get("Date"))
            if posted_date is None:
                continue
            if posted_date < cutoff.date():
                break
        result.append(raw)

    return result


def recent_baserow_linkedin_job_ids(
    client: BaserowRows,
    table_id: int,
    *,
    lookback_hours: int | float | None = None,
    lookback_seconds: int | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Return recent LinkedIn job IDs for provider-side deduplication."""
    result: list[str] = []
    seen: set[str] = set()
    for row in recent_baserow_rows(
        client,
        table_id,
        lookback_hours=lookback_hours,
        lookback_seconds=lookback_seconds,
        now=now,
    ):
        job_id = _linkedin_job_id(row)
        if job_id is None or job_id in seen:
            continue
        seen.add(job_id)
        result.append(job_id)
    return result
