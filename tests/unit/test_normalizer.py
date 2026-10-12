from __future__ import annotations

from datetime import UTC, datetime

from job_hunt.application.normalization import (
    SubmissionNormalizer,
    fillout_submission,
    job_from_provider,
)
from job_hunt.domain.models import (
    AiContentSubmission,
    EntryType,
    ExternalSubmission,
    IndeedSubmission,
    Job,
    LinkedInSubmission,
    UrlSubmission,
)


class Discovery:
    def fetch_linkedin(self, job_id: int, **_: object) -> dict[str, object]:
        return {
            "id": job_id,
            "companyName": "C",
            "title": "T",
            "description": "D",
            "datePosted": "2026-08-19",
        }

    def fetch_indeed(self, job_id: str) -> dict[str, object]:
        return {
            "jobId": job_id,
            "jobUrl": f"https://www.indeed.com/viewjob?jk={job_id}",
            "companyName": "Indeed Company",
            "title": "Indeed Role",
            "descriptionText": "Indeed description",
            "location": "Calgary, AB",
            "employmentType": "Full-time",
            "postedAt": "2026-09-05T15:30:00Z",
        }


class Extractor:
    def extract_job(self, _: str, url: str) -> Job:
        return Job(source="web", url=url, company_name="C", title="T", description="D")


def test_normalizer_linkedin_ai_and_url(monkeypatch: object) -> None:
    normalizer = SubmissionNormalizer(Discovery(), Extractor(), "https://linkedin/{jobId}")  # type: ignore[arg-type]
    linkedin = normalizer.normalize(
        LinkedInSubmission(entry_type=EntryType.LINKEDIN, linkedin_job_id=5),
        country="ca",
        max_concurrency=1,
    )
    assert linkedin.external_id == "5"
    assert linkedin.url == "https://linkedin/5"
    assert linkedin.published_at == datetime(2026, 8, 19, tzinfo=UTC)
    ai = normalizer.normalize(
        AiContentSubmission(entry_type=EntryType.AI_CONTENT, page_content="posting", source_url="https://x/job"),
        country="ca",
        max_concurrency=1,
    )
    assert ai.identity == "https://x/job"


def test_normalizer_indeed_uses_provider_and_indeed_identity() -> None:
    normalizer = SubmissionNormalizer(Discovery(), Extractor(), "https://linkedin/{jobId}")  # type: ignore[arg-type]
    job = normalizer.normalize(
        IndeedSubmission(entry_type=EntryType.INDEED, indeed_job_id="a4d97fa27223cc84"),
        country="ca",
        max_concurrency=1,
    )

    assert job.source == "indeed"
    assert job.external_id == "a4d97fa27223cc84"
    assert job.identity == "indeed:a4d97fa27223cc84"
    assert job.url == "https://www.indeed.com/viewjob?jk=a4d97fa27223cc84"
    assert job.company_name == "Indeed Company"
    assert job.contract_type == "Full-time"
    assert job.published_at == datetime(2026, 9, 5, 15, 30, tzinfo=UTC)


def test_url_normalizer_fetches_public_text(monkeypatch: object) -> None:
    monkeypatch.setattr("job_hunt.application.normalization.fetch_public_text", lambda _: "posting")  # type: ignore[attr-defined]
    normalizer = SubmissionNormalizer(Discovery(), Extractor(), "https://linkedin/{jobId}")  # type: ignore[arg-type]
    job = normalizer.normalize(
        UrlSubmission(entry_type=EntryType.URL, job_url="https://x/job"),
        country="ca",
        max_concurrency=1,
    )
    assert job.company_name == "C"


def test_linkedin_url_uses_provider_instead_of_page_extraction(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "job_hunt.application.normalization.fetch_public_text",
        lambda _: (_ for _ in ()).throw(AssertionError("page fetch should not run")),
    )  # type: ignore[attr-defined]
    normalizer = SubmissionNormalizer(Discovery(), Extractor(), "https://linkedin/{jobId}")  # type: ignore[arg-type]
    job = normalizer.normalize(
        UrlSubmission(
            entry_type=EntryType.URL,
            job_url="https://ca.linkedin.com/jobs/view/machine-learning-specialist-at-example-4452378707",
        ),
        country="ca",
        max_concurrency=1,
    )
    assert job.external_id == "4452378707"
    assert job.published_at == datetime(2026, 8, 19, tzinfo=UTC)


def test_provider_timestamp_date_is_supported() -> None:
    job = job_from_provider(
        {
            "jobId": 123456,
            "url": "https://www.linkedin.com/jobs/view/123456",
            "companyName": "C",
            "title": "T",
            "descriptionText": "Build data systems",
            "location": "Canada",
            "employmentType": "Full-time",
            "postedAt": "2026-06-12T00:00:00.000Z",
        }
    )
    assert job.external_id == "123456"
    assert job.company_name == "C"
    assert job.description == "Build data systems"
    assert job.contract_type == "Full-time"
    assert job.published_at == datetime(2026, 6, 12, tzinfo=UTC)


def test_fillout_actual_entry_values_match_exported_form() -> None:
    ai = fillout_submission({"entryType": "AI", "pageContent": "text", "jobUrl": ""}, {})
    assert isinstance(ai, AiContentSubmission)
    assert ai.page_content == "text"
    assert ai.source_url is None

    linkedin = fillout_submission({"entryType": "LinkedIn", "linkedinJobId": "123"}, {})
    assert isinstance(linkedin, LinkedInSubmission)
    assert linkedin.linkedin_job_id == 123

    indeed = fillout_submission({"entryType": "Indeed", "indeedJobId": "a4d97fa27223cc84"}, {})
    assert isinstance(indeed, IndeedSubmission)
    assert indeed.indeed_job_id == "a4d97fa27223cc84"

    external = fillout_submission(
        {
            "entryType": "External",
            "companyName": "Example",
            "jobTitle": "Data Scientist",
            "jobDescription": "Analyze data",
            "jobUrl": "https://example.com/job",
            "location": "",
            "contractType": "",
            "publishedAt": "",
            "externalJobId": "",
        },
        {},
    )
    assert isinstance(external, ExternalSubmission)
    assert external.location is None
    assert external.contract_type is None
    assert external.published_at is None
    assert external.external_job_id is None

    url = fillout_submission({"entryType": "URL", "jobUrl": "https://x/job"}, {})
    assert isinstance(url, UrlSubmission)


def test_fillout_ai_entry_alias_remains_supported() -> None:
    ai = fillout_submission({"entryType": "AI Entry", "pageContent": "text"}, {})
    assert isinstance(ai, AiContentSubmission)


def test_fillout_indeed_question_id_mapping() -> None:
    submission = fillout_submission(
        {
            "submission": {
                "questions": [
                    {"id": "entry-widget", "value": "Indeed"},
                    {"id": "indeed-widget", "value": "e1b2a54bc6e64c84"},
                ]
            }
        },
        {
            "entryType": "entry-widget",
            "indeedJobId": "indeed-widget",
        },
    )
    assert isinstance(submission, IndeedSubmission)
    assert submission.indeed_job_id == "e1b2a54bc6e64c84"


def test_fillout_semantic_keys_work_with_different_configured_field_ids() -> None:
    submission = fillout_submission(
        {"entryType": "LinkedIn", "linkedinJobId": "123"},
        {
            "entryType": "tenant-specific-entry-widget",
            "linkedinJobId": "tenant-specific-linkedin-widget",
        },
    )
    assert isinstance(submission, LinkedInSubmission)
    assert submission.linkedin_job_id == 123


def test_fillout_configured_ids_still_take_precedence() -> None:
    submission = fillout_submission(
        {
            "entryType": "URL",
            "jobUrl": "https://semantic.example/job",
            "opaque-entry-id": "LinkedIn",
            "opaque-linkedin-id": "456",
        },
        {
            "entryType": "opaque-entry-id",
            "linkedinJobId": "opaque-linkedin-id",
        },
    )
    assert isinstance(submission, LinkedInSubmission)
    assert submission.linkedin_job_id == 456
