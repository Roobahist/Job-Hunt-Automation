from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PurgeRowsSummary:
    scanned: int
    matched: int
    rechecked: int
    skipped_changed: int
    deleted: int
    executed: bool


def status_option_id(row: Mapping[str, object], field: str = "Status") -> int | None:
    current = row.get(field)
    if isinstance(current, Mapping):
        value = current.get("id")
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(current, int) and not isinstance(current, bool):
        return current
    return None


def matching_status_row_ids(
    rows: Iterable[Mapping[str, object]],
    *,
    target_status_id: int,
    status_field: str = "Status",
) -> tuple[list[int], int]:
    row_ids: list[int] = []
    scanned = 0
    for row in rows:
        scanned += 1
        if status_option_id(row, status_field) != target_status_id:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, int) or isinstance(row_id, bool):
            raise ValueError("A matching row is missing a valid integer id")
        row_ids.append(row_id)
    return row_ids, scanned


def purge_rows_by_status(
    *,
    rows: Iterable[Mapping[str, object]],
    target_status_id: int,
    get_row: Callable[[int], Mapping[str, object]],
    delete_row: Callable[[int], None],
    execute: bool,
    status_field: str = "Status",
) -> PurgeRowsSummary:
    candidate_ids, scanned = matching_status_row_ids(
        rows,
        target_status_id=target_status_id,
        status_field=status_field,
    )

    if not execute:
        return PurgeRowsSummary(
            scanned=scanned,
            matched=len(candidate_ids),
            rechecked=0,
            skipped_changed=0,
            deleted=0,
            executed=False,
        )

    rechecked = 0
    skipped_changed = 0
    deleted = 0
    for row_id in candidate_ids:
        # Baserow's public REST API does not expose an atomic conditional delete.
        # Re-read immediately before each individual DELETE so rows changed by another
        # service after the initial scan are skipped instead of being removed in bulk.
        current = get_row(row_id)
        rechecked += 1
        if status_option_id(current, status_field) != target_status_id:
            skipped_changed += 1
            continue
        delete_row(row_id)
        deleted += 1

    return PurgeRowsSummary(
        scanned=scanned,
        matched=len(candidate_ids),
        rechecked=rechecked,
        skipped_changed=skipped_changed,
        deleted=deleted,
        executed=True,
    )
