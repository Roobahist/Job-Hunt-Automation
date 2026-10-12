from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.discovery_router import DiscoveryRouter

SEARCH_URL = "https://www.linkedin.com/jobs/search/?keywords=data&f_TPR=r3600&f_WT=2"


class FakeBrightData:
    def __init__(
        self,
        result: list[dict[str, Any]] | ProviderError,
        *,
        fetch_result: Mapping[str, Any] | ProviderError | None = None,
    ) -> None:
        self.result = result
        self.fetch_result = fetch_result
        self.calls: list[tuple[list[str], int]] = []
        self.fetch_calls: list[int] = []

    def discover(self, urls: Sequence[str], *, max_items: int) -> list[dict[str, Any]]:
        self.calls.append((list(urls), max_items))
        if isinstance(self.result, ProviderError):
            raise self.result
        return list(self.result)

    def fetch_linkedin(self, job_id: int) -> Mapping[str, Any]:
        self.fetch_calls.append(job_id)
        if isinstance(self.fetch_result, ProviderError):
            raise self.fetch_result
        if self.fetch_result is not None:
            return self.fetch_result
        return {"jobId": str(job_id), "_linkedinFetchProvider": "brightdata"}


class FakeIndeed:
    def __init__(self, result: Mapping[str, Any] | ProviderError) -> None:
        self.result = result
        self.calls: list[str] = []

    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]:
        self.calls.append(job_id)
        if isinstance(self.result, ProviderError):
            raise self.result
        return self.result


class FakeApify:
    def __init__(
        self,
        results: Mapping[str, list[dict[str, Any]] | ProviderError],
        *,
        fetch_result: Mapping[str, Any] | ProviderError | None = None,
    ) -> None:
        self.results = dict(results)
        self.fetch_result = fetch_result
        self.calls: list[dict[str, Any]] = []
        self.fetch_calls: list[tuple[int, str, int]] = []

    def discover_strategy(
        self,
        strategy: str,
        urls: Sequence[str],
        *,
        max_items: int,
        skip_job_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "strategy": strategy,
                "urls": list(urls),
                "max_items": max_items,
                "skip_job_ids": list(skip_job_ids) if skip_job_ids is not None else None,
            }
        )
        result = self.results[strategy]
        if isinstance(result, ProviderError):
            raise result
        return list(result)

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        self.fetch_calls.append((job_id, country, max_concurrency))
        if isinstance(self.fetch_result, ProviderError):
            raise self.fetch_result
        if self.fetch_result is not None:
            return self.fetch_result
        return {"jobId": str(job_id), "country": country, "maxConcurrency": max_concurrency}


def _row(job_id: str, *, provider: str) -> dict[str, Any]:
    return {
        "jobId": job_id,
        "jobUrl": f"https://www.linkedin.com/jobs/view/{job_id}",
        "title": f"Job {job_id}",
        "companyName": "Example",
        "descriptionText": "Description",
        "_discoveryProvider": provider,
    }


def _router(
    apify: FakeApify,
    brightdata: FakeBrightData | None,
    *,
    indeed: FakeIndeed | None = None,
    linkedin_fetch_order: Sequence[str] = ("apify", "brightdata"),
) -> DiscoveryRouter:
    return DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=indeed,  # type: ignore[arg-type]
        discovery_order=("brightdata", "coregent", "legacy"),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
        linkedin_fetch_order=linkedin_fetch_order,
    )


def test_brightdata_success_always_runs_coregent_verification_with_double_window() -> None:
    brightdata = FakeBrightData([_row("4461425040", provider="brightdata")])
    apify = FakeApify(
        {
            "coregent": [
                _row("4461425040", provider="coregent"),
                _row("4461425041", provider="coregent"),
            ],
            "legacy": [],
        }
    )
    lookbacks: list[int] = []

    def recent_ids(seconds: int) -> list[str]:
        lookbacks.append(seconds)
        return ["4460000000"]

    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("brightdata", "coregent", "legacy"),
        recent_job_ids_provider=recent_ids,
        default_window_seconds=3600,
    )

    rows = router.discover([SEARCH_URL], max_items=20)

    assert [row["jobId"] for row in rows] == ["4461425040", "4461425041"]
    assert brightdata.calls == [([SEARCH_URL], 20)]
    assert lookbacks == [7200]
    assert apify.calls == [
        {
            "strategy": "coregent",
            "urls": ["https://www.linkedin.com/jobs/search/?keywords=data&f_TPR=r7200&f_WT=2"],
            "max_items": 20,
            "skip_job_ids": ["4460000000", "4461425040"],
        }
    ]


def test_verification_only_jobs_are_not_dropped_when_brightdata_hits_primary_cap() -> None:
    brightdata = FakeBrightData([_row("4461425040", provider="brightdata")])
    apify = FakeApify({"coregent": [_row("4461425041", provider="coregent")], "legacy": []})
    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("brightdata", "coregent", "legacy"),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
    )

    rows = router.discover([SEARCH_URL], max_items=1)

    assert [row["jobId"] for row in rows] == ["4461425040", "4461425041"]


def test_brightdata_quota_exhaustion_falls_back_to_next_strategy() -> None:
    brightdata = FakeBrightData(
        ProviderError(
            "all Bright Data keys exhausted",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="brightdata",
        )
    )
    apify = FakeApify({"coregent": [_row("4461425041", provider="coregent")], "legacy": []})

    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("brightdata", "coregent", "legacy"),
        recent_job_ids_provider=lambda seconds: ["4460000000"] if seconds == 7200 else [],
        default_window_seconds=3600,
    )

    rows = router.discover([SEARCH_URL], max_items=20)

    assert [row["jobId"] for row in rows] == ["4461425041"]
    assert apify.calls[0]["strategy"] == "coregent"
    assert apify.calls[0]["skip_job_ids"] == ["4460000000"]


def test_priority_can_bypass_brightdata_when_apify_strategy_is_first() -> None:
    brightdata = FakeBrightData([_row("4461425040", provider="brightdata")])
    apify = FakeApify({"legacy": [_row("4461425042", provider="legacy")], "coregent": []})

    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("legacy", "brightdata", "coregent"),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
    )

    rows = router.discover([SEARCH_URL], max_items=20)

    assert [row["jobId"] for row in rows] == ["4461425042"]
    assert brightdata.calls == []
    assert [call["strategy"] for call in apify.calls] == ["legacy"]


def test_coregent_verification_failure_does_not_discard_brightdata_results() -> None:
    primary = _row("4461425040", provider="brightdata")
    brightdata = FakeBrightData([primary])
    apify = FakeApify(
        {
            "coregent": ProviderError(
                "Coregent unavailable",
                ErrorKind.RATE_LIMIT,
                retryable=True,
                provider="apify",
            ),
            "legacy": [],
        }
    )

    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("brightdata", "coregent", "legacy"),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
    )

    assert router.discover([SEARCH_URL], max_items=20) == [primary]


def test_missing_f_tpr_uses_configured_interval_for_verification() -> None:
    url = "https://www.linkedin.com/jobs/search/?keywords=data&f_WT=2"
    brightdata = FakeBrightData([_row("4461425040", provider="brightdata")])
    apify = FakeApify({"coregent": [], "legacy": []})
    lookbacks: list[int] = []

    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("brightdata",),
        recent_job_ids_provider=lambda seconds: lookbacks.append(seconds) or [],
        default_window_seconds=5400,
    )

    router.discover([url], max_items=20)

    assert lookbacks == [10800]
    assert apify.calls[0]["urls"] == ["https://www.linkedin.com/jobs/search/?keywords=data&f_WT=2&f_TPR=r10800"]


def test_linkedin_job_fetch_uses_apify_first_by_default() -> None:
    apify_result = _row("4461425040", provider="apify")
    apify = FakeApify({"coregent": [], "legacy": []}, fetch_result=apify_result)
    brightdata = FakeBrightData([], fetch_result=_row("4461425040", provider="brightdata"))
    router = _router(apify, brightdata)

    result = router.fetch_linkedin(4461425040, country="ca", max_concurrency=2)

    assert result == apify_result
    assert apify.fetch_calls == [(4461425040, "ca", 2)]
    assert brightdata.fetch_calls == []


def test_linkedin_job_fetch_falls_back_to_brightdata_when_apify_is_unavailable() -> None:
    apify = FakeApify(
        {"coregent": [], "legacy": []},
        fetch_result=ProviderError(
            "all Apify accounts exhausted",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="apify",
        ),
    )
    brightdata_result = _row("4461425040", provider="brightdata")
    brightdata = FakeBrightData([], fetch_result=brightdata_result)
    router = _router(apify, brightdata)

    result = router.fetch_linkedin(4461425040, country="ca", max_concurrency=2)

    assert result == brightdata_result
    assert apify.fetch_calls == [(4461425040, "ca", 2)]
    assert brightdata.fetch_calls == [4461425040]


def test_linkedin_job_fetch_priority_can_make_brightdata_primary() -> None:
    apify = FakeApify({"coregent": [], "legacy": []}, fetch_result=_row("4461425040", provider="apify"))
    brightdata_result = _row("4461425040", provider="brightdata")
    brightdata = FakeBrightData([], fetch_result=brightdata_result)
    router = _router(apify, brightdata, linkedin_fetch_order=("brightdata", "apify"))

    result = router.fetch_linkedin(4461425040, country="ca", max_concurrency=2)

    assert result == brightdata_result
    assert brightdata.fetch_calls == [4461425040]
    assert apify.fetch_calls == []


def test_linkedin_job_fetch_preserves_apify_error_when_brightdata_is_not_configured() -> None:
    apify_error = ProviderError(
        "Apify unavailable",
        ErrorKind.RATE_LIMIT,
        retryable=True,
        provider="apify",
    )
    apify = FakeApify({"coregent": [], "legacy": []}, fetch_result=apify_error)
    router = _router(apify, None)

    try:
        router.fetch_linkedin(4461425040, country="ca", max_concurrency=2)
    except ProviderError as exc:
        assert exc is apify_error
    else:
        raise AssertionError("Expected the original Apify failure")


def test_indeed_job_fetch_delegates_to_brightdata_provider() -> None:
    apify = FakeApify({"coregent": [], "legacy": []})
    expected = {
        "jobId": "a4d97fa27223cc84",
        "jobUrl": "https://www.indeed.com/viewjob?jk=a4d97fa27223cc84",
    }
    indeed = FakeIndeed(expected)
    router = _router(apify, None, indeed=indeed)

    assert router.fetch_indeed("a4d97fa27223cc84") == expected
    assert indeed.calls == ["a4d97fa27223cc84"]


def test_indeed_job_fetch_fails_clearly_when_provider_is_not_configured() -> None:
    apify = FakeApify({"coregent": [], "legacy": []})
    router = _router(apify, None)

    try:
        router.fetch_indeed("a4d97fa27223cc84")
    except ProviderError as exc:
        assert exc.kind is ErrorKind.CONFIGURATION
    else:
        raise AssertionError("Expected Indeed provider configuration failure")
