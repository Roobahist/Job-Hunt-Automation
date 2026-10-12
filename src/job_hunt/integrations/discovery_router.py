from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.apify import ApifyProvider
from job_hunt.integrations.brightdata import BrightDataProvider
from job_hunt.integrations.brightdata_indeed import BrightDataIndeedProvider
from job_hunt.logging import logger

_SUPPORTED_DISCOVERY_STRATEGIES = {"roobahist", "brightdata", "coregent", "legacy"}
_SUPPORTED_LINKEDIN_FETCH_STRATEGIES = {"roobahist", "apify", "brightdata"}
_LINKEDIN_JOB_ID = re.compile(r"(?:^|[-/])(\d{6,})(?:/)?$")
RecentJobIdsProvider = Callable[[int], Sequence[str]]


class LinkedInPrimaryProvider(Protocol):
    def discover(
        self,
        urls: Sequence[str],
        *,
        max_items: int,
        skip_job_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]: ...

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]: ...


def _job_id(row: Mapping[str, Any]) -> str | None:
    for key in ("jobId", "job_posting_id", "id", "job_id"):
        value = row.get(key)
        if isinstance(value, bool) or value in (None, ""):
            continue
        candidate = str(value).strip()
        if candidate.isdigit():
            return candidate

    raw_url = row.get("jobUrl") or row.get("url") or row.get("link")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    path = urlsplit(raw_url.strip()).path.rstrip("/")
    match = _LINKEDIN_JOB_ID.search(path)
    return match.group(1) if match else None


def _identity(row: Mapping[str, Any]) -> str | None:
    job_id = _job_id(row)
    if job_id is not None:
        return f"id:{job_id}"
    raw_url = row.get("jobUrl") or row.get("url") or row.get("link")
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    parsed = urlsplit(raw_url.strip())
    canonical = urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), "", ""))
    return f"url:{canonical}"


def _window_seconds(url: str, default_seconds: int) -> int:
    values = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    raw = values.get("f_TPR", "")
    if raw.startswith("r") and raw[1:].isdigit() and int(raw[1:]) > 0:
        return int(raw[1:])
    return default_seconds


def _with_window_seconds(url: str, seconds: int) -> str:
    if seconds <= 0:
        raise ValueError("LinkedIn discovery window must be positive")
    parsed = urlsplit(url)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    rewritten: list[tuple[str, str]] = []
    replaced = False
    for key, value in pairs:
        if key == "f_TPR":
            if not replaced:
                rewritten.append((key, f"r{seconds}"))
                replaced = True
            continue
        rewritten.append((key, value))
    if not replaced:
        rewritten.append(("f_TPR", f"r{seconds}"))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(rewritten), parsed.fragment))


def _merge_unique(*groups: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    merged: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            identity = _identity(row)
            if identity is not None and identity in seen:
                continue
            if identity is not None:
                seen.add(identity)
            merged.append(row)
    return merged


class DiscoveryRouter:
    """Route search discovery and provider-backed job lookups without changing downstream contracts."""

    def __init__(
        self,
        apify: ApifyProvider,
        *,
        roobahist: LinkedInPrimaryProvider | None = None,
        brightdata: BrightDataProvider | None,
        indeed: BrightDataIndeedProvider | None,
        discovery_order: Sequence[str],
        recent_job_ids_provider: RecentJobIdsProvider,
        default_window_seconds: int,
        linkedin_fetch_order: Sequence[str] = ("roobahist", "apify", "brightdata"),
    ) -> None:
        self.apify = apify
        self.roobahist = roobahist
        self.brightdata = brightdata
        self.indeed = indeed
        self.discovery_order = tuple(str(value).strip() for value in discovery_order if str(value).strip())
        self.linkedin_fetch_order = tuple(str(value).strip() for value in linkedin_fetch_order if str(value).strip())
        self.recent_job_ids_provider = recent_job_ids_provider
        self.default_window_seconds = default_window_seconds

        if not self.discovery_order:
            raise ValueError("At least one discovery strategy is required")
        unknown = set(self.discovery_order) - _SUPPORTED_DISCOVERY_STRATEGIES
        if unknown:
            raise ValueError(f"Unknown discovery strategies: {sorted(unknown)}")
        if len(set(self.discovery_order)) != len(self.discovery_order):
            raise ValueError("Discovery strategies must not contain duplicates")

        if not self.linkedin_fetch_order:
            raise ValueError("At least one LinkedIn job fetch strategy is required")
        unknown_fetch = set(self.linkedin_fetch_order) - _SUPPORTED_LINKEDIN_FETCH_STRATEGIES
        if unknown_fetch:
            raise ValueError(f"Unknown LinkedIn job fetch strategies: {sorted(unknown_fetch)}")
        if len(set(self.linkedin_fetch_order)) != len(self.linkedin_fetch_order):
            raise ValueError("LinkedIn job fetch strategies must not contain duplicates")

        if self.default_window_seconds <= 0:
            raise ValueError("default_window_seconds must be positive")

    def _recent_skip_ids(self, urls: Sequence[str], *, window_multiplier: int = 2) -> list[str]:
        lookback_seconds = max(_window_seconds(url, self.default_window_seconds) * window_multiplier for url in urls)
        return [str(value).strip() for value in self.recent_job_ids_provider(lookback_seconds) if str(value).strip()]

    def _coregent_verification(
        self,
        urls: Sequence[str],
        brightdata_rows: Sequence[Mapping[str, Any]],
        *,
        max_items: int,
    ) -> list[dict[str, Any]]:
        doubled_urls = [
            _with_window_seconds(url, _window_seconds(url, self.default_window_seconds) * 2) for url in urls
        ]
        verification_window_seconds = max(_window_seconds(url, self.default_window_seconds) for url in doubled_urls)
        stored_ids = [
            str(value).strip()
            for value in self.recent_job_ids_provider(verification_window_seconds)
            if str(value).strip()
        ]
        brightdata_ids = [job_id for row in brightdata_rows if (job_id := _job_id(row)) is not None]
        skip_ids = list(dict.fromkeys([*stored_ids, *brightdata_ids]))

        logger().info(
            "brightdata_coregent_verification_started",
            input_count=len(doubled_urls),
            brightdata_count=len(brightdata_rows),
            verification_window_seconds=verification_window_seconds,
            stored_skip_count=len(stored_ids),
            brightdata_skip_count=len(brightdata_ids),
            skip_count=len(skip_ids),
        )
        try:
            rows = list(
                self.apify.discover_strategy(
                    "coregent",
                    doubled_urls,
                    max_items=max_items,
                    skip_job_ids=skip_ids,
                )
            )
        except ProviderError as exc:
            logger().warning(
                "brightdata_coregent_verification_failed",
                error_kind=exc.kind,
                retryable=exc.retryable,
                error_message=str(exc),
            )
            return []

        logger().info(
            "brightdata_coregent_verification_succeeded",
            result_count=len(rows),
            verification_window_seconds=verification_window_seconds,
        )
        return rows

    def _run_strategy(
        self,
        strategy: str,
        urls: Sequence[str],
        *,
        max_items: int,
    ) -> list[Mapping[str, Any]]:
        if strategy == "roobahist":
            if self.roobahist is None:
                raise ProviderError(
                    "Roobahist LinkedIn discovery is not configured",
                    ErrorKind.CONFIGURATION,
                    provider="apify",
                )
            return list(
                self.roobahist.discover(
                    urls,
                    max_items=max_items,
                    skip_job_ids=self._recent_skip_ids(urls),
                )
            )

        if strategy == "brightdata":
            if self.brightdata is None:
                raise ProviderError(
                    "Bright Data discovery is not configured",
                    ErrorKind.CONFIGURATION,
                    provider="brightdata",
                )
            primary = list(self.brightdata.discover(urls, max_items=max_items))
            verification = self._coregent_verification(urls, primary, max_items=max_items)
            return _merge_unique(primary, verification)

        if strategy == "coregent":
            return list(
                self.apify.discover_strategy(
                    "coregent",
                    urls,
                    max_items=max_items,
                    skip_job_ids=self._recent_skip_ids(urls),
                )
            )

        if strategy == "legacy":
            return list(self.apify.discover_strategy("legacy", urls, max_items=max_items))

        raise ValueError(f"Unsupported discovery strategy: {strategy}")

    def discover(self, urls: Sequence[str], *, max_items: int) -> list[Mapping[str, Any]]:
        if max_items <= 0:
            raise ValueError("max_items must be positive")

        last_error: ProviderError | None = None
        for strategy in self.discovery_order:
            if (strategy == "roobahist" and self.roobahist is None) or (
                strategy == "brightdata" and self.brightdata is None
            ):
                logger().warning(
                    "discovery_strategy_unconfigured",
                    strategy=strategy,
                    configured_order=list(self.discovery_order),
                )
                continue

            logger().info(
                "discovery_strategy_started",
                strategy=strategy,
                configured_order=list(self.discovery_order),
                input_count=len(urls),
                max_items=max_items,
            )
            try:
                rows = self._run_strategy(strategy, urls, max_items=max_items)
            except ProviderError as exc:
                last_error = exc
                logger().warning(
                    "discovery_strategy_failed",
                    strategy=strategy,
                    error_kind=exc.kind,
                    retryable=exc.retryable,
                    error_message=str(exc),
                )
                continue

            logger().info(
                "discovery_strategy_succeeded",
                strategy=strategy,
                result_count=len(rows),
            )
            return rows

        if last_error is not None:
            raise last_error
        raise ProviderError(
            f"No configured discovery strategy is available for {list(self.discovery_order)}",
            ErrorKind.CONFIGURATION,
            provider="discovery",
        )

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        last_error: ProviderError | None = None
        for strategy in self.linkedin_fetch_order:
            if (strategy == "roobahist" and self.roobahist is None) or (
                strategy == "brightdata" and self.brightdata is None
            ):
                logger().warning(
                    "linkedin_job_fetch_strategy_unconfigured",
                    strategy=strategy,
                    job_id=str(job_id),
                    configured_order=list(self.linkedin_fetch_order),
                )
                continue

            logger().info(
                "linkedin_job_fetch_strategy_started",
                strategy=strategy,
                job_id=str(job_id),
                configured_order=list(self.linkedin_fetch_order),
            )
            try:
                if strategy == "roobahist":
                    if self.roobahist is None:
                        continue
                    row = self.roobahist.fetch_linkedin(
                        job_id,
                        country=country,
                        max_concurrency=max_concurrency,
                    )
                elif strategy == "apify":
                    row = self.apify.fetch_linkedin(
                        job_id,
                        country=country,
                        max_concurrency=max_concurrency,
                    )
                elif strategy == "brightdata":
                    if self.brightdata is None:
                        continue
                    row = self.brightdata.fetch_linkedin(job_id)
                else:
                    raise ValueError(f"Unsupported LinkedIn job fetch strategy: {strategy}")
            except ProviderError as exc:
                last_error = exc
                logger().warning(
                    "linkedin_job_fetch_strategy_failed",
                    strategy=strategy,
                    job_id=str(job_id),
                    error_kind=exc.kind,
                    retryable=exc.retryable,
                    error_message=str(exc),
                )
                continue

            logger().info(
                "linkedin_job_fetch_strategy_succeeded",
                strategy=strategy,
                job_id=str(job_id),
            )
            return row

        if last_error is not None:
            raise last_error
        raise ProviderError(
            f"No configured LinkedIn job fetch strategy is available for {list(self.linkedin_fetch_order)}",
            ErrorKind.CONFIGURATION,
            provider="linkedin_job_fetch",
        )

    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]:
        if self.indeed is None:
            raise ProviderError(
                "Bright Data Indeed job collection is not configured",
                ErrorKind.CONFIGURATION,
                provider="brightdata",
            )
        return self.indeed.fetch_indeed(job_id)
