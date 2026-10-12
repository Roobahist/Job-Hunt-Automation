from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from job_hunt.application.normalization import job_from_provider
from job_hunt.integrations.baserow import BaserowJobRepository
from job_hunt.integrations.brightdata import _canonical_job
from job_hunt.integrations.brightdata_indeed import _canonical_indeed_job
from job_hunt.integrations.discovery_skip import recent_baserow_linkedin_job_ids


def _repository() -> BaserowJobRepository:
    return BaserowJobRepository(
        MagicMock(),
        table_id=1,
        status_options={"new": 1, "dropped": 2, "toApply": 3},
        contract_type_options={"fullTime": 17},
    )


def test_coregent_actor_output_maps_to_existing_baserow_job_fields() -> None:
    raw = {
        "jobId": "4450774820",
        "jobUrl": "https://www.linkedin.com/jobs/view/4450774820/",
        "title": "Software Developer (2026-035)",
        "companyName": "Circle Cardiovascular Imaging",
        "location": "Calgary, Alberta, Canada",
        "employmentType": "Full-time",
        "postedAt": "2026-09-02T00:00:00.000Z",
        "descriptionText": "Design, develop, test, maintain, and improve medical imaging software.",
    }

    values = _repository()._job_fields(job_from_provider(raw), include_status=True)

    assert values == {
        "Job ID": "4450774820",
        "Company Name": "Circle Cardiovascular Imaging",
        "Title": "Software Developer (2026-035)",
        "Location": "Calgary, Alberta, Canada",
        "Job Description": "Design, develop, test, maintain, and improve medical imaging software.",
        "Contract Type": 17,
        "Link": "https://www.linkedin.com/jobs/view/4450774820",
        "Date": "2026-09-02",
        "Status": 1,
    }


def test_brightdata_output_maps_to_same_existing_baserow_job_fields() -> None:
    raw = {
        "url": "https://www.linkedin.com/jobs/view/software-developer-at-circle-4450774820?_l=en",
        "job_posting_id": "4450774820",
        "job_title": "Software Developer (2026-035)",
        "company_name": "Circle Cardiovascular Imaging",
        "job_location": "Calgary, Alberta, Canada",
        "job_summary": "Design, develop, test, maintain, and improve medical imaging software.",
        "job_employment_type": "Full-time",
        "job_posted_date": "2026-09-02T00:00:00.000Z",
    }
    canonical = _canonical_job(raw)
    assert canonical is not None

    values = _repository()._job_fields(job_from_provider(canonical), include_status=True)

    assert values == {
        "Job ID": "4450774820",
        "Company Name": "Circle Cardiovascular Imaging",
        "Title": "Software Developer (2026-035)",
        "Location": "Calgary, Alberta, Canada",
        "Job Description": "Design, develop, test, maintain, and improve medical imaging software.",
        "Contract Type": 17,
        "Link": "https://www.linkedin.com/jobs/view/software-developer-at-circle-4450774820",
        "Date": "2026-09-02",
        "Status": 1,
    }


def test_brightdata_indeed_output_maps_to_existing_baserow_job_fields() -> None:
    job_id = "a4d97fa27223cc84"
    job_url = f"https://www.indeed.com/viewjob?jk={job_id}"
    canonical = _canonical_indeed_job(
        {
            "jobid": job_id,
            "company_name": "Example Analytics",
            "date_posted_parsed": "2026-09-05T15:30:00.000Z",
            "job_title": "Data Analyst",
            "description_text": "Build dashboards and analyze operational data.",
            "location": "Calgary, AB",
            "job_type": ["Full-time"],
        },
        expected_job_id=job_id,
        authoritative_url=job_url,
    )
    assert canonical is not None
    job = job_from_provider(canonical, source="indeed").model_copy(update={"url": job_url})

    values = _repository()._job_fields(job, include_status=True)

    assert values == {
        "Job ID": job_id,
        "Company Name": "Example Analytics",
        "Title": "Data Analyst",
        "Location": "Calgary, AB",
        "Job Description": "Build dashboards and analyze operational data.",
        "Contract Type": 17,
        "Link": job_url,
        "Date": "2026-09-05",
        "Status": 1,
    }


def test_recent_skip_ids_use_exact_created_on_storage_time_when_available() -> None:
    client = MagicMock()
    client.list_fields.return_value = [
        {"name": "Date", "type": "date"},
        {"name": "Created On", "type": "created_on"},
    ]
    client.iter_rows.return_value = [
        {"Job ID": 4450774820, "Created On": "2026-09-04T14:30:00Z", "Date": "2026-09-02"},
        {"Job ID": 4450774821, "Created On": "2026-09-04T12:29:59Z", "Date": "2026-09-04"},
        {"Job ID": "external-non-linkedin", "Created On": "2026-09-04T14:45:00Z", "Date": "2026-09-04"},
    ]

    ids = recent_baserow_linkedin_job_ids(
        client,
        99,
        lookback_hours=2,
        now=datetime(2026, 9, 4, 14, 30, tzinfo=UTC),
    )

    assert ids == ["4450774820"]


def test_recent_skip_ids_support_second_resolution_for_verification_windows() -> None:
    client = MagicMock()
    client.list_fields.return_value = [{"name": "Created On", "type": "created_on"}]
    client.iter_rows.return_value = [
        {"Job ID": 4450774820, "Created On": "2026-09-04T14:30:00Z"},
        {"Job ID": 4450774821, "Created On": "2026-09-04T13:15:45Z"},
        {"Job ID": 4450774822, "Created On": "2026-09-04T13:15:44Z"},
    ]

    ids = recent_baserow_linkedin_job_ids(
        client,
        99,
        lookback_seconds=4455,
        now=datetime(2026, 9, 4, 14, 30, tzinfo=UTC),
    )

    assert ids == ["4450774820", "4450774821"]


def test_recent_skip_ids_fall_back_to_posted_date_without_schema_change() -> None:
    client = MagicMock()
    client.list_fields.return_value = [{"name": "Date", "type": "date"}]
    client.iter_rows.return_value = [
        {"Job ID": 4450774820, "Date": "2026-09-04"},
        {"Job ID": 4450774821, "Date": "2026-09-03"},
        {"Job ID": 4450774822, "Date": "2026-09-02"},
    ]

    ids = recent_baserow_linkedin_job_ids(
        client,
        99,
        lookback_hours=2,
        now=datetime(2026, 9, 4, 14, 30, tzinfo=UTC),
    )

    assert ids == ["4450774820"]
