from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from job_hunt.errors import ProviderError
from job_hunt.integrations.brightdata_indeed import BrightDataIndeedProvider

DATASET_ID = "gd_l4dx9j9sscpvs7no2"
JOB_ID = "a4d97fa27223cc84"
JOB_URL = f"https://www.indeed.com/viewjob?jk={JOB_ID}"


def _factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[str], httpx.Client]:
    def build(api_key: str) -> httpx.Client:
        return httpx.Client(
            base_url="https://api.brightdata.com",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    return build


def _job(job_id: str = JOB_ID) -> dict[str, object]:
    return {
        "jobid": job_id,
        "company_name": "Example Analytics",
        "date_posted_parsed": "2026-09-05T15:30:00.000Z",
        "job_title": "Data Analyst",
        "description_text": "Build dashboards and analyze operational data.",
        "location": "Calgary, AB",
        "job_type": ["Full-time"],
        "salary": "$70,000-$85,000 a year",
    }


def test_indeed_collect_uses_expected_request_and_canonical_output() -> None:
    BrightDataIndeedProvider._unavailable_until.clear()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=[_job()], request=request)

    provider = BrightDataIndeedProvider(
        ["key-one"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
    )

    row = provider.fetch_indeed(JOB_ID)

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/datasets/v3/scrape"
    assert request.url.params["dataset_id"] == DATASET_ID
    assert request.url.params["notify"] == "false"
    assert request.url.params["include_errors"] == "true"
    assert request.url.params["format"] == "json"
    assert json.loads(request.content) == {
        "input": [{"url": JOB_URL}],
        "limit_per_input": None,
    }

    assert row["jobId"] == JOB_ID
    assert row["jobUrl"] == JOB_URL
    assert row["title"] == "Data Analyst"
    assert row["companyName"] == "Example Analytics"
    assert row["descriptionText"] == "Build dashboards and analyze operational data."
    assert row["location"] == "Calgary, AB"
    assert row["employmentType"] == "Full-time"
    assert row["postedAt"] == "2026-09-05T15:30:00.000Z"
    assert row["_jobFetchProvider"] == "brightdata_indeed"


def test_indeed_collect_rotates_keys_after_capacity_failure() -> None:
    BrightDataIndeedProvider._unavailable_until.clear()
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer exhausted-key":
            return httpx.Response(402, json={"error": "insufficient balance"}, request=request)
        return httpx.Response(200, json=[_job()], request=request)

    provider = BrightDataIndeedProvider(
        ["exhausted-key", "working-key"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
    )

    row = provider.fetch_indeed(JOB_ID)

    assert attempted == ["Bearer exhausted-key", "Bearer working-key"]
    assert row["jobId"] == JOB_ID


def test_indeed_collect_ignores_error_and_mismatched_rows() -> None:
    BrightDataIndeedProvider._unavailable_until.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"error": "empty page", "error_code": "crawl_error"},
                _job("different-job-id"),
                _job(),
            ],
            request=request,
        )

    provider = BrightDataIndeedProvider(
        ["key-one"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
    )

    assert provider.fetch_indeed(JOB_ID)["jobId"] == JOB_ID


def test_indeed_collect_rejects_response_without_complete_job() -> None:
    BrightDataIndeedProvider._unavailable_until.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"jobid": JOB_ID, "company_name": "Example", "job_title": "Data Analyst"}],
            request=request,
        )

    provider = BrightDataIndeedProvider(
        ["key-one"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
    )

    with pytest.raises(ProviderError, match="was not found"):
        provider.fetch_indeed(JOB_ID)
