from __future__ import annotations

import json

import httpx
import respx

from job_hunt.integrations.telegram import TelegramNotifier


def _request_json(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content.decode("utf-8"))


@respx.mock
def test_terminal_failure_updates_processing_message_instead_of_deleting_it() -> None:
    delete_route = respx.post("https://api.telegram.org/botsecret/deleteMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": True})
    )
    edit_route = respx.post("https://api.telegram.org/botsecret/editMessageCaption").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )

    result = TelegramNotifier("secret").edit_processing_message(
        "42",
        "31",
        caption="Status: ❌ Failed\nError: retries exhausted",
        job_url="https://example.com/job",
        row_id=77,
    )

    assert result == "31"
    assert len(delete_route.calls) == 0
    assert len(edit_route.calls) == 1
    payload = _request_json(edit_route.calls[0].request)
    assert payload["caption"] == "Status: ❌ Failed\nError: retries exhausted"


@respx.mock
def test_terminal_failure_before_notification_init_still_creates_debug_message() -> None:
    send_route = respx.post("https://api.telegram.org/botsecret/sendDocument").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )

    result = TelegramNotifier("secret").send_processing_message(
        "42",
        caption="Status: ❌ Failed",
        job_url="https://example.com/job",
        row_id=77,
    )

    assert result == "31"
    assert len(send_route.calls) == 1
