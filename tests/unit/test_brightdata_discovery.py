from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from job_hunt.integrations.brightdata import BrightDataProvider

DATASET_ID = "gd_lpfll7v5hcqtkxl6l"
SEARCH_URL = (
    "https://www.linkedin.com/jobs/search/?f_TPR=r3600&f_WT=2&geoId=103644278&keywords=machine%20learning"
)


def _factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[str], httpx.Client]:
    def build(api_key: str) -> httpx.Client:
        return httpx.Client(
            base_url="https://api.brightdata.com",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    return build


def _job(*, job_id: str = "4461425040", title: str = "AI/ML Engineer", company: str = "NexDew") -> dict[str, object]:
    return {
        "url": f"https://www.linkedin.com/jobs/view/ai-ml-engineer-at-nexdew-{job_id}?_l=en",
        "job_posting_id": job_id,
        "job_title": title,
        "company_name": company,
        "company_id": "143093019",
        "job_location": "United States",
        "job_summary": "Build and deploy machine learning systems.",
        "job_seniority_level": "Mid-Senior level",
        "job_function": "Engineering and Information Technology",
        "job_employment_type": "Full-time",
        "job_industries": "IT Services and IT Consulting",
        "company_url": "https://www.linkedin.com/company/nexdew-technologies-llc",
        "job_posted_time": "59 minutes ago",
        "job_num_applicants": 0,
        "job_posted_date": "2026-09-05T00:10:10.138Z",
        "job_description_formatted": "<strong>Build ML systems</strong>",
        "is_easy_apply": False,
        "timestamp": "2026-09-05T01:09:10.226Z",
    }


def test_brightdata_uses_generated_urls_and_normalizes_output() -> None:
    BrightDataProvider._unavailable_until.clear()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=[_job()], request=request)

    provider = BrightDataProvider(
        ["key-one"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
        company_exclusions=["Randstad"],
        title_exclusions=["Senior"],
    )

    rows = provider.discover([SEARCH_URL], max_items=20)

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/datasets/v3/scrape"
    assert request.url.params["dataset_id"] == DATASET_ID
    assert request.url.params["type"] == "discover_new"
    assert request.url.params["discover_by"] == "url"
    assert request.url.params["notify"] == "false"
    assert request.url.params["include_errors"] == "true"
    assert request.url.params["format"] == "json"
    assert json.loads(request.content) == {"input": [{"url": SEARCH_URL}]}

    assert rows[0]["jobId"] == "4461425040"
    assert rows[0]["jobUrl"] == "https://www.linkedin.com/jobs/view/ai-ml-engineer-at-nexdew-4461425040?_l=en"
    assert rows[0]["title"] == "AI/ML Engineer"
    assert rows[0]["companyName"] == "NexDew"
    assert rows[0]["location"] == "United States"
    assert rows[0]["employmentType"] == "Full-time"
    assert rows[0]["postedAt"] == "2026-09-05T00:10:10.138000Z"
    assert rows[0]["descriptionText"] == "Build and deploy machine learning systems."
    assert rows[0]["jobFunctions"] == ["Engineering and Information Technology"]
    assert rows[0]["industries"] == ["IT Services and IT Consulting"]
    assert rows[0]["_discoveryProvider"] == "brightdata"


def test_brightdata_skips_error_rows_exclusions_and_duplicates_after_response() -> None:
    BrightDataProvider._unavailable_until.clear()
    response_rows = [
        {
            "timestamp": "2026-09-05T01:10:44.649Z",
            "input": {"url": SEARCH_URL},
            "error": "Crawler error: Returned empty page. No results",
            "error_code": "crawl_error",
        },
        _job(),
        _job(),
        _job(job_id="4461425041", title="Senior ML Engineer"),
        _job(job_id="4461425042", company="Randstad Canada"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_rows, request=request)

    provider = BrightDataProvider(
        ["key-one"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
        company_exclusions=["Randstad"],
        title_exclusions=["senior"],
    )

    rows = provider.discover([SEARCH_URL], max_items=20)

    assert [row["jobId"] for row in rows] == ["4461425040"]


def test_brightdata_uses_relative_posted_time_when_absolute_date_is_missing() -> None:
    BrightDataProvider._unavailable_until.clear()
    row = _job()
    row.pop("job_posted_date")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[row], request=request)

    provider = BrightDataProvider(["key-one"], dataset_id=DATASET_ID, client_factory=_factory(handler))

    result = provider.discover([SEARCH_URL], max_items=20)

    assert result[0]["postedAt"] == "2026-09-05T00:10:10.226000Z"


def test_brightdata_rotates_api_keys_after_quota_failure() -> None:
    BrightDataProvider._unavailable_until.clear()
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["Authorization"]
        attempted.append(auth)
        if auth == "Bearer exhausted-key":
            return httpx.Response(402, json={"error": "insufficient balance"}, request=request)
        return httpx.Response(200, json=[_job()], request=request)

    provider = BrightDataProvider(
        ["exhausted-key", "working-key"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
        quota_cooldown_seconds=3600,
    )

    rows = provider.discover([SEARCH_URL], max_items=20)

    assert attempted == ["Bearer exhausted-key", "Bearer working-key"]
    assert rows[0]["jobId"] == "4461425040"


def test_brightdata_follows_snapshot_after_sync_timeout() -> None:
    BrightDataProvider._unavailable_until.clear()
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(202, json={"snapshot_id": "s_test"}, request=request)
        if request.url.path == "/datasets/v3/progress/s_test":
            return httpx.Response(200, json={"snapshot_id": "s_test", "status": "ready"}, request=request)
        if request.url.path == "/datasets/v3/snapshot/s_test":
            return httpx.Response(200, json=[_job()], request=request)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    provider = BrightDataProvider(
        ["key-one"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
        snapshot_poll_seconds=1,
        sleep=lambda _: None,
    )

    rows = provider.discover([SEARCH_URL], max_items=20)

    assert paths == [
        "/datasets/v3/scrape",
        "/datasets/v3/progress/s_test",
        "/datasets/v3/snapshot/s_test",
    ]
    assert rows[0]["jobId"] == "4461425040"


def test_brightdata_collect_by_url_builds_job_url_and_normalizes_output() -> None:
    BrightDataProvider._unavailable_until.clear()
    captured: list[httpx.Request] = []
    row = _job()
    row.pop("url")
    row.pop("job_posting_id")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=[row], request=request)

    provider = BrightDataProvider(["key-one"], dataset_id=DATASET_ID, client_factory=_factory(handler))

    result = provider.fetch_linkedin(4461425040)

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/datasets/v3/scrape"
    assert dict(request.url.params) == {"dataset_id": DATASET_ID, "format": "json"}
    assert json.loads(request.content) == [{"url": "https://www.linkedin.com/jobs/view/4461425040"}]
    assert result["jobId"] == "4461425040"
    assert result["jobUrl"] == "https://www.linkedin.com/jobs/view/4461425040"
    assert result["title"] == "AI/ML Engineer"
    assert result["companyName"] == "NexDew"
    assert result["descriptionText"] == "Build and deploy machine learning systems."
    assert result["employmentType"] == "Full-time"
    assert result["_linkedinFetchProvider"] == "brightdata"


def test_brightdata_collect_by_url_rotates_keys_after_quota_failure() -> None:
    BrightDataProvider._unavailable_until.clear()
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["Authorization"]
        attempted.append(auth)
        if auth == "Bearer exhausted-key":
            return httpx.Response(402, json={"error": "insufficient balance"}, request=request)
        return httpx.Response(200, json=[_job()], request=request)

    provider = BrightDataProvider(
        ["exhausted-key", "working-key"],
        dataset_id=DATASET_ID,
        client_factory=_factory(handler),
    )

    result = provider.fetch_linkedin(4461425040)

    assert attempted == ["Bearer exhausted-key", "Bearer working-key"]
    assert result["jobId"] == "4461425040"


def test_brightdata_collect_by_url_rejects_mismatched_job_identity() -> None:
    BrightDataProvider._unavailable_until.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_job(job_id="4461425041")], request=request)

    provider = BrightDataProvider(["key-one"], dataset_id=DATASET_ID, client_factory=_factory(handler))

    try:
        provider.fetch_linkedin(4461425040)
    except Exception as exc:
        assert "4461425040" in str(exc)
    else:
        raise AssertionError("Expected mismatched Bright Data job identity to be rejected")
