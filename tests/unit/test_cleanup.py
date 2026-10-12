from __future__ import annotations

import pytest

from job_hunt.application.cleanup import purge_rows_by_status


def test_purge_rows_by_status_dry_run_does_not_recheck_or_delete() -> None:
    rows = [
        {"id": 1, "Status": {"id": 7, "value": "Dropped"}},
        {"id": 2, "Status": 7},
        {"id": 3, "Status": {"id": 4, "value": "New"}},
    ]
    events: list[tuple[str, int]] = []

    dry_run = purge_rows_by_status(
        rows=rows,
        target_status_id=7,
        get_row=lambda row_id: events.append(("get", row_id)) or rows[row_id - 1],
        delete_row=lambda row_id: events.append(("delete", row_id)),
        execute=False,
    )

    assert dry_run.scanned == 3
    assert dry_run.matched == 2
    assert dry_run.rechecked == 0
    assert dry_run.skipped_changed == 0
    assert dry_run.deleted == 0
    assert dry_run.executed is False
    assert events == []


def test_purge_rows_by_status_rechecks_each_candidate_immediately_before_delete() -> None:
    rows = [
        {"id": 1, "Status": {"id": 7, "value": "Dropped"}},
        {"id": 2, "Status": {"id": 7, "value": "Dropped"}},
        {"id": 3, "Status": {"id": 4, "value": "New"}},
    ]
    current_rows = {
        1: {"id": 1, "Status": {"id": 7, "value": "Dropped"}},
        # Another service changed row 2 after the initial scan.
        2: {"id": 2, "Status": {"id": 4, "value": "New"}},
    }
    events: list[tuple[str, int]] = []

    def get_row(row_id: int) -> dict[str, object]:
        events.append(("get", row_id))
        return current_rows[row_id]

    def delete_row(row_id: int) -> None:
        events.append(("delete", row_id))

    executed = purge_rows_by_status(
        rows=rows,
        target_status_id=7,
        get_row=get_row,
        delete_row=delete_row,
        execute=True,
    )

    assert executed.scanned == 3
    assert executed.matched == 2
    assert executed.rechecked == 2
    assert executed.skipped_changed == 1
    assert executed.deleted == 1
    assert executed.executed is True
    assert events == [("get", 1), ("delete", 1), ("get", 2)]


def test_purge_rows_by_status_rejects_matching_row_without_id() -> None:
    with pytest.raises(ValueError, match="valid integer id"):
        purge_rows_by_status(
            rows=[{"Status": {"id": 7}}],
            target_status_id=7,
            get_row=lambda row_id: {},
            delete_row=lambda row_id: None,
            execute=True,
        )
