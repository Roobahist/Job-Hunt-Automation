from __future__ import annotations

from uuid import UUID

from job_hunt import _failure_caption, _notification_context


def test_failure_caption_replaces_processing_status_and_includes_actual_error() -> None:
    caption = "\n".join(
        [
            "Job URL received",
            "🏢 Pending normalization",
            "",
            "Status: ⚙️ Fetching job page and extracting details with AI",
        ]
    )

    result = _failure_caption(
        caption,
        ValueError("job_page_content_extraction did not return company, title, and description"),
    )

    assert "Status: ❌ Failed" in result
    assert "Error: ValueError: job_page_content_extraction did not return company, title, and description" in result
    assert "Status: ⚙️" not in result


def test_failure_caption_respects_telegram_caption_limit() -> None:
    result = _failure_caption(
        "Job content received\n\nStatus: ⚙️ Preparing content",
        RuntimeError("x" * 5000),
    )

    assert len(result) <= 1024
    assert result.startswith("Job content received")
    assert "Status: ❌ Failed" in result
    assert "Error: RuntimeError:" in result


def test_notification_context_extracts_process_submission_run() -> None:
    run_id = UUID("11111111-1111-1111-1111-111111111111")

    context = _notification_context(
        "job_hunt.process_submission",
        ("nova", {"entry_type": "url"}, str(run_id)),
        None,
    )

    assert context == ("nova", run_id)


def test_notification_context_ignores_unrelated_tasks() -> None:
    assert _notification_context("job_hunt.discover_tenant", ("nova", "run"), None) is None
