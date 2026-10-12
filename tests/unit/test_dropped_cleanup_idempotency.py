from __future__ import annotations

from typing import Any

import httpx

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.baserow import BaserowClient, BaserowJobRepository


def _missing_row_error(row_id: int) -> ProviderError:
    detail = f"The rows ['{row_id}'] do not exist."
    return ProviderError(
        f'baserow returned HTTP 404: {{"error":"ERROR_ROW_DOES_NOT_EXIST","detail":"{detail}"}}',
        ErrorKind.BUSINESS,
        retryable=False,
        provider="baserow",
        status_code=404,
    )


def _transient_delete_error() -> ProviderError:
    return ProviderError(
        "Baserow network request failed: response lost after DELETE",
        ErrorKind.TRANSIENT_PROVIDER,
        retryable=True,
        provider="baserow",
        retry_after=0,
    )


class CleanupClient:
    def __init__(self, snapshot_rows: list[dict[str, Any]], live_rows: dict[int, dict[str, Any]]) -> None:
        self.snapshot_rows = snapshot_rows
        self.live_rows = live_rows
        self.delete_calls: list[int] = []
        self.ambiguous_delete_once: set[int] = set()

    def iter_rows(self, _: int) -> object:
        return iter(self.snapshot_rows)

    def get_row(self, _: int, row_id: int) -> dict[str, Any]:
        row = self.live_rows.get(row_id)
        if row is None:
            raise _missing_row_error(row_id)
        return row

    def delete_row(self, _: int, row_id: int) -> None:
        self.delete_calls.append(row_id)
        if row_id in self.ambiguous_delete_once:
            self.ambiguous_delete_once.remove(row_id)
            self.live_rows.pop(row_id, None)
            raise _transient_delete_error()
        if row_id not in self.live_rows:
            raise _missing_row_error(row_id)
        self.live_rows.pop(row_id)


def _repo(client: CleanupClient) -> BaserowJobRepository:
    return BaserowJobRepository(
        client,
        1,
        {"new": 10, "dropped": 11, "toApply": 12},
    )  # type: ignore[arg-type]


def test_baserow_delete_treats_missing_row_as_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(
            404,
            json={"error": "ERROR_ROW_DOES_NOT_EXIST", "detail": "The rows ['41'] do not exist."},
        )

    http_client = httpx.Client(base_url="https://api.baserow.io", transport=httpx.MockTransport(handler))
    client = BaserowClient("token", client=http_client)

    client.delete_row(1, 41)


def test_cleanup_accepts_404_after_ambiguous_delete() -> None:
    client = CleanupClient(
        snapshot_rows=[{"id": 41, "Status": {"id": 11, "value": "Dropped"}}],
        live_rows={41: {"id": 41, "Status": {"id": 11, "value": "Dropped"}}},
    )
    client.ambiguous_delete_once.add(41)

    assert _repo(client).delete_status("dropped") == 1
    assert client.delete_calls == [41, 41]
    assert 41 not in client.live_rows


def test_cleanup_skips_row_missing_after_initial_scan() -> None:
    client = CleanupClient(
        snapshot_rows=[{"id": 52, "Status": {"id": 11, "value": "Dropped"}}],
        live_rows={},
    )

    assert _repo(client).delete_status("dropped") == 0
    assert client.delete_calls == []


def test_cleanup_rechecks_status_before_delete() -> None:
    client = CleanupClient(
        snapshot_rows=[{"id": 63, "Status": {"id": 11, "value": "Dropped"}}],
        live_rows={63: {"id": 63, "Status": {"id": 12, "value": "To Apply"}}},
    )

    assert _repo(client).delete_status("dropped") == 0
    assert client.delete_calls == []


def test_cleanup_deduplicates_ids_from_paginated_snapshot() -> None:
    client = CleanupClient(
        snapshot_rows=[
            {"id": 74, "Status": {"id": 11, "value": "Dropped"}},
            {"id": 74, "Status": {"id": 11, "value": "Dropped"}},
        ],
        live_rows={74: {"id": 74, "Status": {"id": 11, "value": "Dropped"}}},
    )

    assert _repo(client).delete_status("dropped") == 1
    assert client.delete_calls == [74]
