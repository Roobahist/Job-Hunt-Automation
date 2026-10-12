from __future__ import annotations

from uuid import UUID

from job_hunt.api.app import _fillout_intake_caption

_RUN_ID = UUID("11111111-1111-1111-1111-111111111111")


def test_url_intake_caption_describes_fetch_and_ai_extraction() -> None:
    url = "https://example.com/jobs/123?source=fillout"
    caption, job_url = _fillout_intake_caption(
        {"entry_type": "url", "job_url": url},
        _RUN_ID,
    )

    assert job_url == url
    assert f"🔗 URL: {url}" in caption
    assert "Status: ⚙️ Fetching job page and extracting details with AI" in caption


def test_ai_content_intake_caption_describes_content_extraction() -> None:
    caption, job_url = _fillout_intake_caption(
        {"entry_type": "ai_content", "source_url": None},
        _RUN_ID,
    )

    assert job_url is None
    assert "Status: ⚙️ Preparing content and extracting job details with AI" in caption
