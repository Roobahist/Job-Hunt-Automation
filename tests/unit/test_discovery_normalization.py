from __future__ import annotations

from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from job_hunt.application.discovery import build_search_url, normalize_discovery, search_criteria_active
from job_hunt.application.normalization import SubmissionNormalizer, fillout_submission
from job_hunt.domain.models import (
    AiContentSubmission,
    EntryType,
    ExternalSubmission,
    Job,
    LinkedInSubmission,
    UrlSubmission,
)


def provider_row(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": 1,
        "url": "https://example.com/jobs/1",
        "companyName": "Good Company",
        "title": "Data Analyst",
        "description": "Analyze data",
    }
    values.update(updates)
    return values


class DiscoveryStub:
    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> dict[str, object]:
        raise AssertionError("LinkedIn discovery should not be used in these tests")


class ExtractorStub:
    def __init__(self, job: Job | None = None, error: ValueError | None = None) -> None:
        self.job = job
        self.error = error

    def extract_job(self, content: str, source_url: str) -> Job:
        if self.error is not None:
            raise self.error
        assert self.job is not None
        return self.job


def normalizer(extractor: ExtractorStub) -> SubmissionNormalizer:
    return SubmissionNormalizer(DiscoveryStub(), extractor, "https://linkedin.com/jobs/view/{jobId}")


def test_search_url_uses_config_interval_and_ignores_criteria_date_filter() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {
            "Keywords": "data scientist",
            "Location": "alberta",
            "Date Filter": "r86400",
        },
        1,
    )
    assert url == (
        "https://www.linkedin.com/jobs/search/?keywords=data+scientist&location=alberta&f_TPR=r3600&sortBy=R"
    )


def test_search_url_rebuilds_from_current_columns_and_ignores_stale_generated_url() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {
            "Generated URL": (
                "https://www.linkedin.com/jobs/search/?keywords=stale&location=old&f_E=6&f_WT=1&f_TPR=r86400&sortBy=DD"
            ),
            "Keywords": "software engineer",
            "Location": "canada",
            "Experience Level": [
                {"id": 1, "value": "Entry Level"},
                {"id": 2, "value": "Internship"},
                {"id": 3, "value": "Associate"},
            ],
            "Job Type": [{"value": "Full-time"}],
            "Workplace Type": [{"value": "Remote"}],
            "Date Posted": 99,
        },
        2,
    )
    params = parse_qs(urlsplit(url).query)

    assert params["keywords"] == ["software engineer"]
    assert params["location"] == ["canada"]
    assert params["f_E"] == ["2,1,3"]
    assert params["f_JT"] == ["F"]
    assert params["f_WT"] == ["2"]
    assert params["f_TPR"] == ["r7200"]
    assert params["sortBy"] == ["R"]
    assert "stale" not in url
    assert "old" not in url


def test_search_url_accepts_baserow_select_labels_and_preencoded_codes() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {
            "Keywords": "machine learning",
            "Experience Level": "2,3",
            "Job Type": "C",
            "Workplace Type": "3",
        },
        3,
    )
    params = parse_qs(urlsplit(url).query)
    assert params["f_E"] == ["2,3"]
    assert params["f_JT"] == ["C"]
    assert params["f_WT"] == ["3"]
    assert params["f_TPR"] == ["r10800"]


def test_search_url_rejects_non_positive_schedule_interval() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        build_search_url("https://x/search/", {"Keywords": "data science"}, 0)


def test_search_criteria_requires_active_flag() -> None:
    assert search_criteria_active({"Active": True})
    assert search_criteria_active({"Active": "yes"})
    assert search_criteria_active({"active": 1})
    assert not search_criteria_active({"Active": False})
    assert not search_criteria_active({"Active": "off"})
    assert not search_criteria_active({})


def test_discovery_normalization_does_not_post_filter_or_deduplicate() -> None:
    rows = [
        provider_row(),
        provider_row(),
        provider_row(id=2, url="https://x/2", title="Senior Lead"),
    ]
    jobs = normalize_discovery(rows, ["good company"], ["senior", "lead"])
    assert len(jobs) == 3
    assert [job.title for job in jobs] == ["Data Analyst", "Data Analyst", "Senior Lead"]


def test_fillout_normalizes_linkedin_and_external() -> None:
    linkedin = fillout_submission({"entryType": "linkedin", "linkedinJobId": "42"}, {})
    assert isinstance(linkedin, LinkedInSubmission)
    external = fillout_submission(
        {
            "entryType": "external",
            "companyName": "C",
            "jobTitle": "T",
            "jobDescription": "D",
            "jobUrl": "https://example.com/j",
        },
        {},
    )
    assert isinstance(external, ExternalSubmission)


def test_fillout_flattens_official_questions_shape() -> None:
    payload = {
        "submission": {
            "questions": [
                {"id": "entry-id", "value": "linkedin"},
                {"questionId": "job-id", "value": "42"},
            ]
        }
    }
    submission = fillout_submission(
        payload,
        {"entryType": "entry-id", "linkedinJobId": "job-id"},
    )
    assert isinstance(submission, LinkedInSubmission)
    assert submission.linkedin_job_id == 42


def test_url_mode_rejects_incomplete_extraction_and_suggests_ai_mode() -> None:
    incomplete = Job(
        source="web",
        url="https://example.com/job",
        company_name="",
        title="AI Engineer",
        description="Build AI systems.",
    )
    submission = UrlSubmission(entry_type=EntryType.URL, job_url="https://example.com/job")

    with (
        patch("job_hunt.application.normalization.fetch_public_text", return_value="job page"),
        pytest.raises(ValueError) as exc_info,
    ):
        normalizer(ExtractorStub(job=incomplete)).normalize(
            submission,
            country="ca",
            max_concurrency=1,
        )

    message = str(exc_info.value)
    assert "company_name" in message
    assert "Use AI mode and paste the full job description instead" in message


def test_url_mode_translates_extractor_failure_into_ai_mode_guidance() -> None:
    submission = UrlSubmission(entry_type=EntryType.URL, job_url="https://example.com/job")

    with (
        patch("job_hunt.application.normalization.fetch_public_text", return_value="job page"),
        pytest.raises(ValueError) as exc_info,
    ):
        normalizer(ExtractorStub(error=ValueError("description missing"))).normalize(
            submission,
            country="ca",
            max_concurrency=1,
        )

    assert "Use AI mode and paste the full job description instead" in str(exc_info.value)


def test_ai_mode_rejects_incomplete_extraction_and_suggests_manual_mode() -> None:
    incomplete = Job(
        source="web",
        url="",
        company_name="",
        title="Finance Expert",
        description="Evaluate finance model outputs.",
    )
    submission = AiContentSubmission(
        entry_type=EntryType.AI_CONTENT,
        page_content="Finance Expert job description",
    )

    with pytest.raises(ValueError) as exc_info:
        normalizer(ExtractorStub(job=incomplete)).normalize(
            submission,
            country="ca",
            max_concurrency=1,
        )

    message = str(exc_info.value)
    assert "company_name" in message
    assert "Use Manual mode and provide the required job details" in message


def test_ai_mode_accepts_only_complete_required_job_information() -> None:
    complete = Job(
        source="web",
        url="",
        company_name="Example Co",
        title="Finance Expert",
        description="Evaluate finance model outputs.",
    )
    submission = AiContentSubmission(
        entry_type=EntryType.AI_CONTENT,
        page_content="Complete posting",
    )

    job = normalizer(ExtractorStub(job=complete)).normalize(
        submission,
        country="ca",
        max_concurrency=1,
    )

    assert job.company_name == "Example Co"
    assert job.title == "Finance Expert"
    assert job.description == "Evaluate finance model outputs."
