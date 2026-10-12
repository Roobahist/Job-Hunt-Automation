from __future__ import annotations

from typing import Any

from job_hunt.application import normalization
from job_hunt.application.normalization import SubmissionNormalizer
from job_hunt.application.page_content import prepare_page_content
from job_hunt.domain.models import Job, UrlSubmission


class _Discovery:
    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> dict[str, Any]:
        raise AssertionError("LinkedIn discovery should not run for a normal URL submission")


class _Extractor:
    def __init__(self, returned_url: str) -> None:
        self.returned_url = returned_url
        self.received_content = ""
        self.received_source_url = ""

    def extract_job(self, content: str, source_url: str) -> Job:
        self.received_content = content
        self.received_source_url = source_url
        return Job(
            source="web",
            url=self.returned_url,
            company_name="Example",
            title="Data Analyst",
            description="Analyze data",
        )


def test_prepare_page_content_removes_nonvisible_markup_without_clipping_visible_text() -> None:
    visible = "Useful job text " + ("x" * 150_000) + " tail marker"
    html = f"<html><style>ignore-style</style><script>ignore-script</script><body>{visible}</body></html>"

    prepared = prepare_page_content(html)

    assert prepared == visible
    assert "ignore-style" not in prepared
    assert "ignore-script" not in prepared
    assert "tail marker" in prepared


def test_url_submission_preserves_submitted_url_and_sanitizes_page(monkeypatch: Any) -> None:
    submitted_url = "https://bba.csod.com/ux/ats/careersite/4/home/requisition/3080?c=bba"
    extractor = _Extractor("/")
    monkeypatch.setattr(
        normalization,
        "fetch_public_text",
        lambda url: "<html><script>tracking()</script><body>Junior Advisor job description</body></html>",
    )
    normalizer = SubmissionNormalizer(_Discovery(), extractor, "https://www.linkedin.com/jobs/view/{jobId}")

    job = normalizer.normalize(
        UrlSubmission(entry_type="url", job_url=submitted_url),
        country="ca",
        max_concurrency=1,
    )

    assert job.url == submitted_url
    assert job.identity == "https://bba.csod.com/ux/ats/careersite/4/home/requisition/3080"
    assert extractor.received_source_url == submitted_url
    assert extractor.received_content == "Junior Advisor job description"


def test_url_submission_keeps_submitted_url_even_if_extractor_returns_another_url(monkeypatch: Any) -> None:
    submitted_url = "https://example.com/jobs/original"
    extractor = _Extractor("https://example.com/jobs/canonical")
    monkeypatch.setattr(normalization, "fetch_public_text", lambda url: "Job text")
    normalizer = SubmissionNormalizer(_Discovery(), extractor, "https://www.linkedin.com/jobs/view/{jobId}")

    job = normalizer.normalize(
        UrlSubmission(entry_type="url", job_url=submitted_url),
        country="ca",
        max_concurrency=1,
    )

    assert job.url == submitted_url
