from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.discovery_router import DiscoveryRouter

SEARCH_URL = "https://www.linkedin.com/jobs/search/?keywords=data&f_TPR=r3600"


class FakeRoobahist:
    def __init__(
        self,
        discovery_result: list[dict[str, Any]] | ProviderError,
        *,
        fetch_result: Mapping[str, Any] | ProviderError | None = None,
    ) -> None:
        self.discovery_result = discovery_result
        self.fetch_result = fetch_result
        self.discovery_calls: list[tuple[list[str], int, list[str]]] = []
        self.fetch_calls: list[tuple[int, str, int]] = []

    def discover(
        self,
        urls: Sequence[str],
        *,
        max_items: int,
        skip_job_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        self.discovery_calls.append((list(urls), max_items, list(skip_job_ids)))
        if isinstance(self.discovery_result, ProviderError):
            raise self.discovery_result
        return list(self.discovery_result)

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        self.fetch_calls.append((job_id, country, max_concurrency))
        if isinstance(self.fetch_result, ProviderError):
            raise self.fetch_result
        if self.fetch_result is None:
            return {"id": str(job_id)}
        return self.fetch_result


class FakeApify:
    def __init__(self, *, fetch_result: Mapping[str, Any] | ProviderError | None = None) -> None:
        self.fetch_result = fetch_result
        self.discovery_calls: list[str] = []
        self.fetch_calls: list[int] = []

    def discover_strategy(
        self,
        strategy: str,
        urls: Sequence[str],
        *,
        max_items: int,
        skip_job_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        del urls, max_items, skip_job_ids
        self.discovery_calls.append(strategy)
        return [{"id": "fallback"}]

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        del country, max_concurrency
        self.fetch_calls.append(job_id)
        if isinstance(self.fetch_result, ProviderError):
            raise self.fetch_result
        if self.fetch_result is None:
            return {"id": str(job_id), "provider": "apify"}
        return self.fetch_result


class FakeBrightData:
    def __init__(self, fetch_result: Mapping[str, Any]) -> None:
        self.fetch_result = fetch_result
        self.discovery_calls = 0
        self.fetch_calls: list[int] = []

    def discover(self, urls: Sequence[str], *, max_items: int) -> list[dict[str, Any]]:
        del urls, max_items
        self.discovery_calls += 1
        return []

    def fetch_linkedin(self, job_id: int) -> Mapping[str, Any]:
        self.fetch_calls.append(job_id)
        return self.fetch_result


def test_roobahist_is_used_first_for_discovery_and_receives_recent_skip_ids() -> None:
    primary = FakeRoobahist([{"id": "4450774820"}])
    apify = FakeApify()
    lookbacks: list[int] = []

    def recent_ids(seconds: int) -> list[str]:
        lookbacks.append(seconds)
        return ["4440000001", "4440000002"]

    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        roobahist=primary,
        brightdata=None,
        indeed=None,
        discovery_order=("roobahist", "coregent", "legacy"),
        recent_job_ids_provider=recent_ids,
        default_window_seconds=3600,
    )

    assert router.discover([SEARCH_URL], max_items=20) == [{"id": "4450774820"}]
    assert lookbacks == [7200]
    assert primary.discovery_calls == [([SEARCH_URL], 20, ["4440000001", "4440000002"])]
    assert apify.discovery_calls == []


def test_roobahist_discovery_failure_falls_back_to_existing_provider() -> None:
    primary = FakeRoobahist(
        ProviderError(
            "primary unavailable",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="apify",
        )
    )
    apify = FakeApify()
    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        roobahist=primary,
        brightdata=None,
        indeed=None,
        discovery_order=("roobahist", "legacy"),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
    )

    assert router.discover([SEARCH_URL], max_items=20) == [{"id": "fallback"}]
    assert apify.discovery_calls == ["legacy"]


def test_roobahist_is_used_first_for_direct_linkedin_job_fetch() -> None:
    expected = {"id": "4450774820", "provider": "roobahist"}
    primary = FakeRoobahist([], fetch_result=expected)
    apify = FakeApify(fetch_result={"id": "4450774820", "provider": "apify"})
    brightdata = FakeBrightData({"id": "4450774820", "provider": "brightdata"})
    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        roobahist=primary,
        brightdata=brightdata,  # type: ignore[arg-type]
        indeed=None,
        discovery_order=("roobahist",),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
        linkedin_fetch_order=("roobahist", "apify", "brightdata"),
    )

    assert router.fetch_linkedin(4450774820, country="ca", max_concurrency=3) == expected
    assert primary.fetch_calls == [(4450774820, "ca", 3)]
    assert apify.fetch_calls == []
    assert brightdata.fetch_calls == []


def test_direct_fetch_falls_back_when_roobahist_actor_fails() -> None:
    primary = FakeRoobahist(
        [],
        fetch_result=ProviderError(
            "primary unavailable",
            ErrorKind.TRANSIENT_PROVIDER,
            retryable=True,
            provider="apify",
        ),
    )
    fallback = {"id": "4450774820", "provider": "apify"}
    apify = FakeApify(fetch_result=fallback)
    router = DiscoveryRouter(
        apify,  # type: ignore[arg-type]
        roobahist=primary,
        brightdata=None,
        indeed=None,
        discovery_order=("roobahist",),
        recent_job_ids_provider=lambda _: [],
        default_window_seconds=3600,
        linkedin_fetch_order=("roobahist", "apify", "brightdata"),
    )

    assert router.fetch_linkedin(4450774820, country="ca", max_concurrency=1) == fallback
    assert primary.fetch_calls == [(4450774820, "ca", 1)]
    assert apify.fetch_calls == [4450774820]
