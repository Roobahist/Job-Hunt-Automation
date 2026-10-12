# ruff: noqa: I001
from __future__ import annotations

from typing import Any

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.indeed_router import IndeedFetchRouter


JOB_ID = "48ddfac3828876b4"


class FakeApifyIndeed:
    def __init__(self, result: dict[str, Any] | ProviderError) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def fetch_indeed(self, job_id: str, *, country: str) -> dict[str, Any]:
        self.calls.append((job_id, country))
        if isinstance(self.result, ProviderError):
            raise self.result
        return self.result


class FakeBrightDataIndeed:
    def __init__(self, result: dict[str, Any] | ProviderError) -> None:
        self.result = result
        self.calls: list[str] = []

    def fetch_indeed(self, job_id: str) -> dict[str, Any]:
        self.calls.append(job_id)
        if isinstance(self.result, ProviderError):
            raise self.result
        return self.result


def _job(provider: str) -> dict[str, Any]:
    return {
        "jobId": JOB_ID,
        "jobUrl": f"https://www.indeed.com/viewjob?jk={JOB_ID}",
        "title": "Data Scientist",
        "companyName": "Example Inc.",
        "descriptionText": "Description",
        "_jobFetchProvider": provider,
    }


def test_apify_is_primary_and_brightdata_is_not_called_on_success() -> None:
    apify = FakeApifyIndeed(_job("apify_indeed"))
    brightdata = FakeBrightDataIndeed(_job("brightdata_indeed"))
    router = IndeedFetchRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        country="ca",
    )

    result = router.fetch_indeed(JOB_ID)

    assert result["_jobFetchProvider"] == "apify_indeed"
    assert apify.calls == [(JOB_ID, "CA")]
    assert brightdata.calls == []


def test_brightdata_can_be_used_only_when_explicitly_enabled_after_apify() -> None:
    apify = FakeApifyIndeed(
        ProviderError(
            "Apify actor unavailable",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="apify",
        )
    )
    brightdata = FakeBrightDataIndeed(_job("brightdata_indeed"))
    router = IndeedFetchRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        country="ca",
        order=("apify", "brightdata"),
    )

    result = router.fetch_indeed(JOB_ID)

    assert result["_jobFetchProvider"] == "brightdata_indeed"
    assert apify.calls == [(JOB_ID, "CA")]
    assert brightdata.calls == [JOB_ID]


def test_apify_failure_is_preserved_when_brightdata_is_not_configured() -> None:
    expected = ProviderError(
        "Apify actor unavailable",
        ErrorKind.RATE_LIMIT,
        retryable=True,
        provider="apify",
    )
    apify = FakeApifyIndeed(expected)
    router = IndeedFetchRouter(apify, brightdata=None, country="ca")  # type: ignore[arg-type]

    try:
        router.fetch_indeed(JOB_ID)
    except ProviderError as exc:
        assert exc is expected
    else:
        raise AssertionError("Expected the Apify provider error")
