from __future__ import annotations

import httpx
import respx

from job_hunt.integrations.baserow import BaserowClient


@respx.mock
def test_baserow_auth_filter_pagination_and_writes() -> None:
    rows_route = respx.get("https://api.baserow.io/api/database/rows/table/1/").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "results": [{"id": 1}],
                    "next": "https://api.baserow.io/api/database/rows/table/1/?page=2",
                },
            ),
            httpx.Response(200, json={"results": [{"id": 2}], "next": None}),
        ]
    )
    client = BaserowClient("token")
    rows = list(client.iter_rows(1, **{"filter__Link__equal": "https://x"}))
    assert rows == [{"id": 1}, {"id": 2}]
    request = rows_route.calls[0].request
    assert request.headers["Authorization"] == "Token token"
    assert "user_field_names=true" in str(request.url)
    assert "filter__Link__equal=https%3A%2F%2Fx" in str(request.url)


@respx.mock
def test_baserow_create_update_and_upload_via_url() -> None:
    create = respx.post("https://api.baserow.io/api/database/rows/table/1/").mock(
        return_value=httpx.Response(200, json={"id": 9})
    )
    update = respx.patch("https://api.baserow.io/api/database/rows/table/1/9/").mock(
        return_value=httpx.Response(200, json={"id": 9, "Score": 80})
    )
    upload = respx.post("https://api.baserow.io/api/user-files/upload-via-url/").mock(
        return_value=httpx.Response(200, json={"name": "x.pdf"})
    )
    client = BaserowClient("token")
    assert client.create_row(1, {"Title": "T"})["id"] == 9
    assert client.update_row(1, 9, {"Score": 80})["Score"] == 80
    assert client.upload_via_url("https://cdn/x.pdf")["name"] == "x.pdf"
    assert create.calls[0].request.url.params["user_field_names"] == "true"
    assert update.called and upload.called


@respx.mock
def test_baserow_delete_row_uses_single_row_endpoint() -> None:
    delete = respx.delete("https://api.baserow.io/api/database/rows/table/1/41/").mock(return_value=httpx.Response(204))
    client = BaserowClient("token")

    client.delete_row(1, 41)

    assert delete.call_count == 1
