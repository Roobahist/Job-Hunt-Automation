from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from threading import Lock
from typing import Any, ClassVar
from urllib.parse import urlparse

from apify_client import ApifyClient

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.logging import logger
from job_hunt.state import RedisState

_RETRY_SECONDS = re.compile(r"(?:retry|try again)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_ACTOR_CAPACITY_MARKERS = (
    "maximum charged results",
    "minimum is 150",
    "maximum charge",
    "charge limit reached",
    "cannot fund",
)
_MINIMUM_CHARGE_ERROR = "Apify actor stopped because the account cannot fund the minimum charged results"
_MINIMUM_CHARGE_LOOPS = 2
_SUPPORTED_DISCOVERY_STRATEGIES = {"coregent", "legacy"}
_ROOBAHIST_ACTOR_ID = "munificent_geastrum/apify-linkedin"
_ROOBAHIST_DEFAULT_TIMEOUT_SECONDS = 900


class ApifyProvider:
    _lock: ClassVar[Lock] = Lock()
    _unavailable_until: ClassVar[dict[str, float]] = {}

    def __init__(
        self,
        tokens: Sequence[str],
        search_actor_id: str,
        single_actor_id: str,
        *,
        search_actor_ids: Mapping[str, str] | None = None,
        discovery_order: Sequence[str] = ("legacy",),
        skip_job_ids_provider: Callable[[], Sequence[str]] | None = None,
        company_exclusions: Sequence[str] = (),
        title_exclusions: Sequence[str] = (),
        quota_cooldown_seconds: int = 3600,
        state: RedisState | None = None,
    ) -> None:
        self.tokens = [token for token in tokens if token]
        if not self.tokens:
            raise ValueError("At least one Apify token is required")
        self.search_actor_id = search_actor_id
        self.single_actor_id = single_actor_id
        configured_actor_ids = {"legacy": str(search_actor_id).strip()}
        if search_actor_ids is not None:
            configured_actor_ids.update(
                {
                    str(strategy).strip(): str(actor_id).strip()
                    for strategy, actor_id in search_actor_ids.items()
                    if str(strategy).strip() and str(actor_id).strip()
                }
            )
        self.search_actor_ids = configured_actor_ids
        self.discovery_order = tuple(str(strategy).strip() for strategy in discovery_order if str(strategy).strip())
        if not self.discovery_order:
            raise ValueError("At least one Apify discovery strategy is required")
        unknown = set(self.discovery_order) - _SUPPORTED_DISCOVERY_STRATEGIES
        if unknown:
            raise ValueError(f"Unknown Apify discovery strategies: {sorted(unknown)}")
        self.skip_job_ids_provider = skip_job_ids_provider
        self.company_exclusions = [str(value).strip() for value in company_exclusions if str(value).strip()]
        self.title_exclusions = [str(value).strip() for value in title_exclusions if str(value).strip()]
        self.quota_cooldown_seconds = quota_cooldown_seconds
        self.state = state

    @staticmethod
    def _token_id(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()[:12]

    @staticmethod
    def _is_capacity_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status in {401, 402, 403, 429}:
            return True
        text = str(exc).lower()
        markers = (
            "quota",
            "rate limit",
            "rate_limit",
            "too many requests",
            "usage limit",
            "limit exceeded",
            "insufficient",
            "payment required",
            "not enough",
            "monthly usage",
            "account limit",
            *_ACTOR_CAPACITY_MARKERS,
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_minimum_charge_error(exc: Exception) -> bool:
        return str(exc) == _MINIMUM_CHARGE_ERROR

    @staticmethod
    def _actor_log_capacity_error(log_text: str | None) -> bool:
        if not log_text:
            return False
        lowered = log_text.lower()
        return any(marker in lowered for marker in _ACTOR_CAPACITY_MARKERS)

    def _cooldown_seconds(self, exc: Exception) -> int:
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return max(1, int(retry_after))
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            raw = headers.get("retry-after")
            try:
                if raw is not None:
                    return max(1, int(float(raw)))
            except (TypeError, ValueError):
                pass
        match = _RETRY_SECONDS.search(str(exc))
        if match:
            return max(1, int(float(match.group(1))))
        return self.quota_cooldown_seconds

    def _available_tokens(self) -> list[str]:
        state = self.state
        if state is not None:
            available = [token for token in self.tokens if state.available("apify", self._token_id(token))]
            if available:
                return available
            retry_after = min(max(1, state.remaining_cooldown("apify", self._token_id(token))) for token in self.tokens)
            raise ProviderError(
                "Apify capacity is cooling down for all configured accounts",
                ErrorKind.RATE_LIMIT,
                retryable=True,
                provider="apify",
                retry_after=retry_after,
            )

        now = time.monotonic()
        with self._lock:
            available = [token for token in self.tokens if self._unavailable_until.get(self._token_id(token), 0) <= now]
            if available:
                return available
            retry_after = min(
                max(1, int(self._unavailable_until.get(self._token_id(token), now) - now) + 1) for token in self.tokens
            )
        raise ProviderError(
            "Apify capacity is cooling down for all configured accounts",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="apify",
            retry_after=retry_after,
        )

    def _cooldown(self, token: str, seconds: int) -> None:
        token_id = self._token_id(token)
        if self.state is not None:
            self.state.cooldown("apify", token_id, seconds=seconds)
            return
        with self._lock:
            self._unavailable_until[token_id] = time.monotonic() + seconds

    @classmethod
    def _run_with_client(
        cls,
        client: ApifyClient,
        actor_id: str,
        run_input: Mapping[str, Any],
        max_items: int | None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> list[dict[str, Any]]:
        actor = client.actor(actor_id)
        if actor_id == _ROOBAHIST_ACTOR_ID:
            resolved_timeout = _ROOBAHIST_DEFAULT_TIMEOUT_SECONDS if timeout_secs is None else timeout_secs
            run = actor.call(
                run_input=dict(run_input),
                max_items=max_items,
                timeout_secs=resolved_timeout,
                max_total_charge_usd=(
                    Decimal(str(max_total_charge_usd)) if max_total_charge_usd is not None else None
                ),
            )
        else:
            run = actor.call(run_input=dict(run_input), max_items=max_items)
        if not run or not run.get("defaultDatasetId"):
            raise ProviderError(
                "Apify run returned no dataset",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="apify",
            )
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        if not items and run.get("id"):
            log_text = client.log(str(run["id"])).get()
            if cls._actor_log_capacity_error(log_text):
                raise RuntimeError(_MINIMUM_CHARGE_ERROR)
        return items

    def _run(
        self,
        actor_id: str,
        run_input: Mapping[str, Any],
        max_items: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> Iterable[dict[str, Any]]:
        tokens = self._available_tokens()
        queue = deque(tokens)
        failures: list[str] = []
        cooldowns: list[int] = []
        minimum_charge_attempts = 0
        minimum_charge_limit = len(tokens) * _MINIMUM_CHARGE_LOOPS

        while queue:
            token = queue.popleft()
            try:
                yield from self._run_with_client(
                    ApifyClient(token),
                    actor_id,
                    run_input,
                    max_items,
                    timeout_secs=timeout_secs,
                    max_total_charge_usd=max_total_charge_usd,
                )
                return
            except ProviderError:
                raise
            except Exception as exc:
                failures.append(f"{self._token_id(token)}: {exc}")
                if self._is_minimum_charge_error(exc):
                    minimum_charge_attempts += 1
                    if minimum_charge_attempts >= minimum_charge_limit:
                        raise ProviderError(
                            "Apify is out of access: every configured account failed the minimum-charge coverage check "
                            f"for {_MINIMUM_CHARGE_LOOPS} full loops",
                            ErrorKind.RATE_LIMIT,
                            retryable=False,
                            provider="apify",
                        ) from exc
                    queue.append(token)
                    continue
                if self._is_capacity_error(exc):
                    cooldown = self._cooldown_seconds(exc)
                    cooldowns.append(cooldown)
                    self._cooldown(token, cooldown)
                    continue
                raise ProviderError(
                    f"Apify request failed: {exc}",
                    ErrorKind.TRANSIENT_PROVIDER,
                    retryable=True,
                    provider="apify",
                ) from exc

        raise ProviderError(
            "Apify capacity exhausted for all configured accounts: " + "; ".join(failures),
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="apify",
            retry_after=min(cooldowns) if cooldowns else None,
        )

    @staticmethod
    def _valid_http_url(value: str) -> bool:
        parsed = urlparse(value.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _legacy_discovery_input(self, urls: Sequence[str], max_items: int) -> dict[str, Any]:
        return {
            "startUrls": [{"url": url} for url in urls],
            "maxItems": max_items,
            "saveOnlyUniqueItems": True,
            "companyExclude": self.company_exclusions,
            "jobTitleExclude": self.title_exclusions,
        }

    def _coregent_discovery_input(
        self,
        urls: Sequence[str],
        max_items: int,
        *,
        skip_job_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        resolved_skip_ids = (
            [str(value).strip() for value in skip_job_ids if str(value).strip()]
            if skip_job_ids is not None
            else (
                [str(value).strip() for value in self.skip_job_ids_provider() if str(value).strip()]
                if self.skip_job_ids_provider is not None
                else []
            )
        )
        return {
            "easyApplyOnly": False,
            "excludeJobsWithoutSalary": False,
            "excludeRecruitmentAgencies": False,
            "expandLargeSearches": False,
            "extractPublicEmails": False,
            "includeCompanyDetails": False,
            "includeRecruiterDetails": False,
            "keepUnconfirmedWorkplace": False,
            "maxResults": max_items,
            "proxyConfiguration": {"useApifyProxy": True},
            "removeDuplicates": True,
            "skipJobIds": list(dict.fromkeys(resolved_skip_ids)),
            "searchUrls": list(urls),
            "underTenApplicantsOnly": False,
            "excludeCompanies": self.company_exclusions,
            "excludeJobTitleKeywords": self.title_exclusions,
        }

    def _discovery_input(
        self,
        strategy: str,
        urls: Sequence[str],
        max_items: int,
        *,
        skip_job_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if strategy == "coregent":
            return self._coregent_discovery_input(urls, max_items, skip_job_ids=skip_job_ids)
        if strategy == "legacy":
            return self._legacy_discovery_input(urls, max_items)
        raise ValueError(f"Unsupported Apify discovery strategy: {strategy}")

    def discover_strategy(
        self,
        strategy: str,
        urls: Sequence[str],
        *,
        max_items: int,
        skip_job_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if strategy not in _SUPPORTED_DISCOVERY_STRATEGIES:
            raise ValueError(f"Unsupported Apify discovery strategy: {strategy}")
        search_urls = [url.strip() for url in urls if isinstance(url, str) and self._valid_http_url(url)]
        if not search_urls:
            raise ProviderError(
                "No valid LinkedIn search URLs were generated",
                ErrorKind.BUSINESS,
                provider="apify",
            )

        actor_id = self.search_actor_ids.get(strategy, "").strip()
        if not actor_id:
            raise ProviderError(
                f"Apify discovery strategy {strategy!r} is not configured",
                ErrorKind.CONFIGURATION,
                provider="apify",
            )

        logger().info(
            "apify_discovery_strategy_started",
            strategy=strategy,
            actor_id=actor_id,
            input_count=len(search_urls),
            max_items=max_items,
        )
        rows = list(
            self._run(
                actor_id,
                self._discovery_input(
                    strategy,
                    search_urls,
                    max_items,
                    skip_job_ids=skip_job_ids,
                ),
                max_items=max_items,
            )
        )
        logger().info(
            "apify_discovery_strategy_succeeded",
            strategy=strategy,
            actor_id=actor_id,
            result_count=len(rows),
        )
        return rows

    def discover(self, urls: Sequence[str], *, max_items: int) -> Iterable[Mapping[str, Any]]:
        last_error: ProviderError | None = None
        for strategy in self.discovery_order:
            actor_id = self.search_actor_ids.get(strategy, "").strip()
            if not actor_id:
                logger().warning(
                    "apify_discovery_strategy_unconfigured",
                    strategy=strategy,
                    configured_order=list(self.discovery_order),
                )
                continue
            try:
                return self.discover_strategy(strategy, urls, max_items=max_items)
            except ProviderError as exc:
                last_error = exc
                logger().warning(
                    "apify_discovery_strategy_failed",
                    strategy=strategy,
                    actor_id=actor_id,
                    error_kind=exc.kind,
                    retryable=exc.retryable,
                    error_message=str(exc),
                )

        if last_error is not None:
            raise last_error
        raise ProviderError(
            f"No configured Apify discovery actor is available for strategies {list(self.discovery_order)}",
            ErrorKind.BUSINESS,
            provider="apify",
        )

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        job_url = f"https://www.linkedin.com/jobs/view/{job_id}"
        items = list(
            self._run(
                self.single_actor_id,
                {
                    "jobUrls": [job_url],
                    "proxy": {
                        "useApifyProxy": True,
                        "apifyProxyCountry": country.upper(),
                    },
                    "maxConcurrency": max_concurrency,
                },
                max_items=1,
            )
        )
        if not items:
            raise ProviderError(
                f"LinkedIn job {job_id} was not found in the Apify response",
                ErrorKind.BUSINESS,
                provider="apify",
            )
        return items[0]
