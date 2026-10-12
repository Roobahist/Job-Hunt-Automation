from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.brightdata_indeed import BrightDataIndeedProvider
from job_hunt.integrations.brightdata_resilient import ResilientBrightDataProvider

LINKEDIN_DATASET_ID = "gd_lpfll7v5hcqtkxl6l"
INDEED_DATASET_ID = "gd_l4dx9j9sscpvs7no2"
LINKEDIN_SEARCH_URL = "https://www.linkedin.com/jobs/search/?keywords=data%20scientist&f_TPR=r3600"
LINKEDIN_JOB_ID = 4461425040
INDEED_JOB_ID = "48ddfac3828876b4"


def _factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[str], httpx.Client]:
    def build(api_key: str) -> httpx.Client:
        return httpx.Client(
            base_url="https://api.brightdata.com",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    return build


def _rate_limit_record(url: str) -> dict[str, object]:
    return {
        "timestamp": "2026-09-06T01:05:23.239Z",
        "input": {"url": url},
        "error": f"Crawler error: Navigation failed for {url}: too many requests",
        "error_code": "rate_limit",
    }


def _linkedin_job() -> dict[str, object]:
    return {
        "url": f"https://www.linkedin.com/jobs/view/{LINKEDIN_JOB_ID}",
        "job_posting_id": str(LINKEDIN_JOB_ID),
        "job_title": "Data Scientist",
        "company_name": "Example Analytics",
        "job_summary": "Build and evaluate machine learning systems.",
    }


def test_discovery_record_rate_limit_is_retryable_when_no_job_was_returned() -> None:
    ResilientBrightDataProvider._unavailable_until.clear()
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(request.headers["Authorization"])
        return httpx.Response(200, json=[_rate_limit_record(LINKEDIN_SEARCH_URL)], request=request)

    provider = ResilientBrightDataProvider(
        ["key-one", "key-two"],
        dataset_id=LINKEDIN_DATASET_ID,
        client_factory=_factory(handler),
    )

    with pytest.raises(ProviderError) as caught:
        provider.discover([LINKEDIN_SEARCH_URL], max_items=20)

    assert caught.value.kind is ErrorKind.RATE_LIMIT
    assert caught.value.retryable is True
    assert caught.value.retry_after is None
    assert attempted == ["Bearer key-one"]


def test_discovery_keeps_partial_jobs_when_another_input_is_rate_limited() -> None:
    ResilientBrightDataProvider._unavailable_until.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[_rate_limit_record(LINKEDIN_SEARCH_URL), _linkedin_job()],
            request=request,
        )

    provider = ResilientBrightDataProvider(
        ["key-one"],
        dataset_id=LINKEDIN_DATASET_ID,
        client_factory=_factory(handler),
    )

    rows = provider.discover([LINKEDIN_SEARCH_URL], max_items=20)

    assert [row["jobId"] for row in rows] == [str(LINKEDIN_JOB_ID)]


def test_linkedin_single_job_record_rate_limit_is_retryable() -> None:
    ResilientBrightDataProvider._unavailable_until.clear()
    job_url = f"https://www.linkedin.com/jobs/view/{LINKEDIN_JOB_ID}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_rate_limit_record(job_url)], request=request)

    provider = ResilientBrightDataProvider(
        ["key-one"],
        dataset_id=LINKEDIN_DATASET_ID,
        client_factory=_factory(handler),
    )

    with pytest.raises(ProviderError) as caught:
        provider.fetch_linkedin(LINKEDIN_JOB_ID)

    assert caught.value.kind is ErrorKind.RATE_LIMIT
    assert caught.value.retryable is True


def test_indeed_single_job_record_rate_limit_is_retryable() -> None:
    BrightDataIndeedProvider._unavailable_until.clear()
    job_url = f"https://www.indeed.com/viewjob?jk={INDEED_JOB_ID}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_rate_limit_record(job_url)], request=request)

    provider = BrightDataIndeedProvider(
        ["key-one"],
        dataset_id=INDEED_DATASET_ID,
        client_factory=_factory(handler),
    )

    with pytest.raises(ProviderError) as caught:
        provider.fetch_indeed(INDEED_JOB_ID)

    assert caught.value.kind is ErrorKind.RATE_LIMIT
    assert caught.value.retryable is True
    assert "too many requests" in str(caught.value).casefold()
