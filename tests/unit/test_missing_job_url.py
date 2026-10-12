from __future__ import annotations

from typing import Any

from job_hunt.domain.models import Job
from job_hunt.integrations.baserow import BaserowJobRepository
from job_hunt.integrations.telegram import TelegramNotifier
from job_hunt.url_utils import valid_http_url


class StubBaserowClient:
    def __init__(self) -> None:
        self.find_calls: list[tuple[int, str, object]] = []

    def find_equal(self, table_id: int, field: str, value: object) -> dict[str, Any] | None:
        self.find_calls.append((table_id, field, value))
        return None


def _job(url: str) -> Job:
    return Job(source="fillout", url=url, company_name="Example", title="Role", description="Description")


def test_valid_http_url_rejects_placeholders_and_relative_paths() -> None:
    assert valid_http_url(None) is None
    assert valid_http_url("") is None
    assert valid_http_url("/") is None
    assert valid_http_url("example.com/job") is None
    assert valid_http_url("https://example.com/job") == "https://example.com/job"


def test_baserow_persists_invalid_job_url_as_blank() -> None:
    client = StubBaserowClient()
    repository = BaserowJobRepository(client, 1, {"new": 10})  # type: ignore[arg-type]

    assert repository._job_fields(_job("/"))["Link"] is None
    assert repository.find(_job("/")) is None
    assert client.find_calls == []


def test_telegram_omits_open_job_button_when_url_is_invalid() -> None:
    processing = TelegramNotifier._processing_actions("/", 77)
    processing_rows = processing["inline_keyboard"]
    assert isinstance(processing_rows, list)
    assert processing_rows == [[{"text": "Skip Processing", "callback_data": "status:dropped:77"}]]

    final = TelegramNotifier._actions("/", 77, "run-1")
    final_rows = final["inline_keyboard"]
    assert isinstance(final_rows, list)
    assert all(button.get("text") != "Open Job" for row in final_rows for button in row)
