from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.http import raise_provider_error, retry_after_seconds
from job_hunt.integrations.telegram import TelegramNotifier


def _request_json(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content.decode("utf-8"))


def test_http_error_classification_and_retry_after() -> None:
    response = httpx.Response(429, headers={"Retry-After": "7"}, text="slow")
    assert retry_after_seconds(response) == 7
    with pytest.raises(ProviderError) as caught:
        raise_provider_error("test", response)
    assert caught.value.kind is ErrorKind.RATE_LIMIT
    assert caught.value.retryable
    with pytest.raises(ProviderError) as auth:
        raise_provider_error("test", httpx.Response(401, text="no"))
    assert auth.value.kind is ErrorKind.AUTHENTICATION
    with pytest.raises(ProviderError) as upstream:
        raise_provider_error("test", httpx.Response(503, text="down"))
    assert upstream.value.kind is ErrorKind.TRANSIENT_PROVIDER
    assert upstream.value.retryable
    with pytest.raises(ProviderError) as business:
        raise_provider_error("test", httpx.Response(400, text="invalid"))
    assert business.value.kind is ErrorKind.BUSINESS
    assert not business.value.retryable


@respx.mock
def test_telegram_uses_one_multipart_media_group(tmp_path: Path) -> None:
    files = []
    for index in range(3):
        path = tmp_path / f"{index}.pdf"
        path.write_bytes(b"pdf")
        files.append(path)
    route = respx.post("https://api.telegram.org/botsecret/sendMediaGroup").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": [{"message_id": 12}]})
    )
    result = TelegramNotifier("secret").send_documents("42", files, "caption")
    assert result == "12"
    body = route.calls[0].request.content
    assert body.count(b'Content-Disposition: form-data; name="file_') == 3


@respx.mock
def test_processing_message_starts_as_placeholder_and_can_be_skipped() -> None:
    route = respx.post("https://api.telegram.org/botsecret/sendDocument").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    message_id = TelegramNotifier("secret").send_processing_message(
        "42",
        caption="Status: processing",
        job_url="https://example.com/job",
        row_id=77,
    )
    assert message_id == "31"
    body = route.calls[0].request.content
    assert b"processing.txt" in body
    assert b"Skip Processing" in body
    assert b"status%3Adropped%3A77" in body or b"status:dropped:77" in body


@respx.mock
def test_dropped_job_does_not_create_late_processing_placeholder() -> None:
    route = respx.post("https://api.telegram.org/botsecret/sendDocument").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    message_id = TelegramNotifier("secret").send_processing_message(
        "42",
        caption="Status: ⛔ Dropped",
        job_url="https://example.com/job",
        row_id=77,
    )
    assert message_id == ""
    assert len(route.calls) == 0


@respx.mock
def test_processing_message_updates_caption_in_place() -> None:
    route = respx.post("https://api.telegram.org/botsecret/editMessageCaption").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    result = TelegramNotifier("secret").edit_processing_message(
        "42",
        "31",
        caption="Status: qualification",
        job_url="https://example.com/job",
        row_id=77,
    )
    assert result == "31"
    payload = _request_json(route.calls[0].request)
    assert payload["message_id"] == 31
    assert payload["caption"] == "Status: qualification"
    assert payload["reply_markup"]["inline_keyboard"][1][0]["text"] == "Skip Processing"


@respx.mock
def test_failed_processing_message_exposes_regenerate_action() -> None:
    route = respx.post("https://api.telegram.org/botsecret/editMessageCaption").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    result = TelegramNotifier("secret").edit_processing_message(
        "42",
        "31",
        caption="Status: ❌ Failed\nError: repair failed",
        job_url="https://example.com/job",
        row_id=77,
        run_id="run-123",
    )
    assert result == "31"
    payload = _request_json(route.calls[0].request)
    keyboard = payload["reply_markup"]["inline_keyboard"]
    assert keyboard[-1][0] == {"text": "Regenerate", "callback_data": "retry:run-123"}


@respx.mock
def test_dropped_processing_message_is_deleted_instead_of_edited() -> None:
    delete_route = respx.post("https://api.telegram.org/botsecret/deleteMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": True})
    )
    edit_route = respx.post("https://api.telegram.org/botsecret/editMessageCaption").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    result = TelegramNotifier("secret").edit_processing_message(
        "42",
        "31",
        caption="Status: ⛔ Dropped manually",
        job_url="https://example.com/job",
        row_id=None,
    )
    assert result == "31"
    assert len(delete_route.calls) == 1
    assert len(edit_route.calls) == 0
    assert _request_json(delete_route.calls[0].request) == {"chat_id": "42", "message_id": 31}


@respx.mock
def test_processing_message_is_replaced_by_zip_with_final_actions(tmp_path: Path) -> None:
    archive = tmp_path / "application.zip"
    archive.write_bytes(b"zip")
    route = respx.post("https://api.telegram.org/botsecret/editMessageMedia").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    result = TelegramNotifier("secret").finalize_processing_message(
        "42",
        "31",
        archive,
        caption="Status: complete",
        job_url="https://example.com/job",
        row_id=77,
        run_id="run-1",
    )
    assert result == "31"
    body = route.calls[0].request.content
    assert b"application.zip" in body
    assert b"status%3Aapplied%3A77" in body or b"status:applied:77" in body
    assert b"status%3Adropped%3A77" in body or b"status:dropped:77" in body
    assert b"retry%3Arun-1" in body or b"retry:run-1" in body


@respx.mock
def test_late_complete_progress_refresh_preserves_final_keyboard() -> None:
    route = respx.post("https://api.telegram.org/botsecret/editMessageCaption").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 31}})
    )
    TelegramNotifier("secret").edit_processing_message(
        "42",
        "31",
        caption="Status: ✅ Complete",
        job_url="https://example.com/job",
        row_id=77,
    )
    payload = _request_json(route.calls[0].request)
    assert "reply_markup" not in payload


@respx.mock
def test_telegram_archive_document_carries_workflow_controls(tmp_path: Path) -> None:
    archive = tmp_path / "application.zip"
    archive.write_bytes(b"zip")
    route = respx.post("https://api.telegram.org/botsecret/sendDocument").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 14}})
    )
    notifier = TelegramNotifier("secret")
    assert (
        notifier.send_document_with_actions(
            "42",
            archive,
            caption="Designer at Example",
            job_url="https://example.com/job",
            row_id=77,
            run_id="run-1",
        )
        == "14"
    )
    payload = route.calls[0].request.content
    assert b"application.zip" in payload
    assert b"Designer at Example" in payload
    assert b"status%3Aapplied%3A77" in payload or b"status:applied:77" in payload
    assert b"status%3Adropped%3A77" in payload or b"status:dropped:77" in payload
    assert b"retry%3Arun-1" in payload or b"retry:run-1" in payload


@respx.mock
def test_telegram_action_message_has_workflow_controls() -> None:
    route = respx.post("https://api.telegram.org/botsecret/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 13}})
    )
    notifier = TelegramNotifier("secret")
    assert (
        notifier.send_application_actions(
            "42",
            caption="Designer at Example",
            job_url="https://example.com/job",
            row_id=77,
            run_id="run-1",
        )
        == "13"
    )
    payload = route.calls[0].request.read().decode()
    assert "status:applied:77" in payload
    assert "status:dropped:77" in payload
    assert "retry:run-1" in payload


@respx.mock
def test_application_bundle_sends_only_zip_with_caption_and_actions(tmp_path: Path) -> None:
    artifacts = []
    for name in ("application.zip", "Person_CV.pdf", "Person_CL.pdf"):
        path = tmp_path / name
        path.write_bytes(b"data")
        artifacts.append(path)

    document_route = respx.post("https://api.telegram.org/botsecret/sendDocument").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 21}})
    )

    result = TelegramNotifier("secret").send_application_bundle(
        "42",
        artifacts,
        caption="Designer at Example",
        job_url="https://example.com/job",
        row_id=77,
        run_id="run-1",
    )

    assert result == "21"
    assert len(document_route.calls) == 1
    body = document_route.calls[0].request.content
    assert b"application.zip" in body
    assert b"Person_CV.pdf" not in body
    assert b"Person_CL.pdf" not in body
    assert b"Designer at Example" in body
    assert b"status%3Aapplied%3A77" in body or b"status:applied:77" in body
    assert b"status%3Adropped%3A77" in body or b"status:dropped:77" in body
    assert b"retry%3Arun-1" in body or b"retry:run-1" in body


def test_application_bundle_requires_exactly_one_zip(tmp_path: Path) -> None:
    pdf = tmp_path / "Person_CV.pdf"
    pdf.write_bytes(b"pdf")
    with pytest.raises(ValueError, match="exactly one ZIP"):
        TelegramNotifier("x").send_application_bundle(
            "1",
            [pdf],
            caption="x",
            job_url="https://example.com/job",
            row_id=1,
            run_id="run",
        )


def test_telegram_enforces_api_album_size(tmp_path: Path) -> None:
    one = tmp_path / "one"
    one.write_text("x")
    with pytest.raises(ValueError, match="2 to 10"):
        TelegramNotifier("x").send_documents("1", [one], "x")
