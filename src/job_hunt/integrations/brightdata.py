from __future__ import annotations

import hashlib
import html
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

import httpx

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.logging import logger
from job_hunt.state import RedisState

_MAX_SYNC_INPUTS = 20
_CAPACITY_MARKERS = (
    "insufficient",
    "balance",
    "quota",
    "credit",
    "rate limit",
    "too many requests",
    "maximum limit",
    "concurrency limit",
    "account is suspended",
)
_RETRY_SECONDS = re.compile(r"(?:retry|try again)[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_LINKEDIN_JOB_ID = re.compile(r"(?:^|[-/])(\d{6,})(?:/)?$")
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_RELATIVE_POSTED = re.compile(
    r"(?P<count>\d+)\+?\s*(?P<unit>minute|hour|day|week|month)s?\s+ago",
    re.IGNORECASE,
)

BrightDataClientFactory = Callable[[str], httpx.Client]


def _default_client_factory(api_key: str, timeout_seconds: float) -> httpx.Client:
    return httpx.Client(
        base_url="https://api.brightdata.com",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout_seconds,
    )


def _strip_html(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return _WHITESPACE.sub(" ", html.unescape(_HTML_TAG.sub(" ", value))).strip()


def _as_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _linkedin_job_id_from_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").casefold()
    if host != "linkedin.com" and not host.endswith(".linkedin.com"):
        return None
    if "/jobs/view/" not in parsed.path:
        return None
    match = _LINKEDIN_JOB_ID.search(parsed.path.rstrip("/"))
    return match.group(1) if match else None


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _relative_posted_at(value: object, *, anchor: datetime | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().casefold()
    if text in {"just now", "now"}:
        return anchor or datetime.now(UTC)
    match = _RELATIVE_POSTED.search(text)
    if match is None:
        return None
    count = int(match.group("count"))
    unit = match.group("unit").casefold()
    delta = {
        "minute": timedelta(minutes=count),
        "hour": timedelta(hours=count),
        "day": timedelta(days=count),
        "week": timedelta(weeks=count),
        "month": timedelta(days=30 * count),
    }[unit]
    return (anchor or datetime.now(UTC)) - delta


def _posted_at(record: Mapping[str, Any]) -> str | None:
    absolute = _parse_datetime(record.get("job_posted_date"))
    if absolute is None:
        anchor = _parse_datetime(record.get("timestamp"))
        absolute = _relative_posted_at(record.get("job_posted_time"), anchor=anchor)
    return absolute.isoformat().replace("+00:00", "Z") if absolute is not None else None


def _is_error_record(record: Mapping[str, Any]) -> bool:
    if not record.get("error") and not record.get("error_code"):
        return False
    return not any(
        record.get(key)
        for key in (
            "job_posting_id",
            "job_title",
            "company_name",
            "job_summary",
            "job_description_formatted",
        )
    )


def _canonical_job(record: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_url = str(record.get("url") or "").strip()
    job_id = str(record.get("job_posting_id") or "").strip() or _linkedin_job_id_from_url(raw_url)
    if not raw_url and job_id:
        raw_url = f"https://www.linkedin.com/jobs/view/{job_id}"

    title = str(record.get("job_title") or "").strip()
    company = str(record.get("company_name") or "").strip()
    description = str(record.get("job_summary") or "").strip() or _strip_html(record.get("job_description_formatted"))
    if not raw_url or not title or not company or not description:
        return None

    canonical = dict(record)
    canonical.update(
        {
            "jobId": job_id or None,
            "jobUrl": raw_url,
            "title": title,
            "companyName": company,
            "companyLinkedInUrl": record.get("company_url"),
            "companyId": record.get("company_id"),
            "location": record.get("job_location"),
            "employmentType": record.get("job_employment_type"),
            "seniorityLevel": record.get("job_seniority_level"),
            "postedAt": _posted_at(record),
            "applicantCount": record.get("job_num_applicants"),
            "easyApply": bool(record.get("is_easy_apply", False)),
            "descriptionText": description,
            "descriptionHtml": record.get("job_description_formatted"),
            "applyUrl": record.get("apply_link") or raw_url,
            "jobFunctions": _as_list(record.get("job_function")),
            "industries": _as_list(record.get("job_industries")),
            "salary": record.get("base_salary") or record.get("job_base_pay_range"),
            "_discoveryProvider": "brightdata",
        }
    )
    return canonical


def _record_identity(record: Mapping[str, Any]) -> str | None:
    job_id = record.get("jobId") or record.get("job_posting_id")
    if job_id not in (None, ""):
        return f"id:{str(job_id).strip()}"
    url = record.get("jobUrl") or record.get("url")
    if isinstance(url, str) and url.strip():
        return f"url:{_canonical_url(url)}"
    return None


class BrightDataProvider:
    _lock: ClassVar[Lock] = Lock()
    _unavailable_until: ClassVar[dict[str, float]] = {}

    def __init__(
        self,
        api_keys: Sequence[str],
        *,
        dataset_id: str,
        company_exclusions: Sequence[str] = (),
        title_exclusions: Sequence[str] = (),
        request_timeout_seconds: float = 75,
        snapshot_timeout_seconds: float = 300,
        snapshot_poll_seconds: float = 5,
        quota_cooldown_seconds: int = 3600,
        state: RedisState | None = None,
        client_factory: BrightDataClientFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_keys = [str(key).strip() for key in api_keys if str(key).strip()]
        if not self.api_keys:
            raise ValueError("At least one Bright Data API key is required")
        self.dataset_id = str(dataset_id).strip()
        if not self.dataset_id:
            raise ValueError("Bright Data dataset ID is required")
        self.company_exclusions = [str(value).strip() for value in company_exclusions if str(value).strip()]
        self.title_exclusions = [str(value).strip() for value in title_exclusions if str(value).strip()]
        self.request_timeout_seconds = request_timeout_seconds
        self.snapshot_timeout_seconds = snapshot_timeout_seconds
        self.snapshot_poll_seconds = snapshot_poll_seconds
        self.quota_cooldown_seconds = quota_cooldown_seconds
        self.state = state
        self.client_factory = client_factory or (
            lambda api_key: _default_client_factory(api_key, self.request_timeout_seconds)
        )
        self.sleep = sleep
        self.monotonic = monotonic

    @staticmethod
    def _key_id(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()[:12]

    @staticmethod
    def _valid_search_url(value: str) -> bool:
        parsed = urlsplit(value.strip())
        host = (parsed.hostname or "").casefold()
        return (
            parsed.scheme in {"http", "https"}
            and (host == "linkedin.com" or host.endswith(".linkedin.com"))
            and parsed.path.startswith("/jobs/")
        )

    def _available_keys(self) -> list[str]:
        if self.state is not None:
            available = [key for key in self.api_keys if self.state.available("brightdata", self._key_id(key))]
            if available:
                return available
            retry_after = min(
                max(1, self.state.remaining_cooldown("brightdata", self._key_id(key))) for key in self.api_keys
            )
            raise ProviderError(
                "Bright Data capacity is cooling down for all configured accounts",
                ErrorKind.RATE_LIMIT,
                retryable=True,
                provider="brightdata",
                retry_after=retry_after,
            )

        now = self.monotonic()
        with self._lock:
            available = [key for key in self.api_keys if self._unavailable_until.get(self._key_id(key), 0) <= now]
            if available:
                return available
            retry_after = min(
                max(1, int(self._unavailable_until.get(self._key_id(key), now) - now) + 1) for key in self.api_keys
            )
        raise ProviderError(
            "Bright Data capacity is cooling down for all configured accounts",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="brightdata",
            retry_after=retry_after,
        )

    def _cooldown(self, api_key: str, seconds: int) -> None:
        key_id = self._key_id(api_key)
        if self.state is not None:
            self.state.cooldown("brightdata", key_id, seconds=seconds)
            return
        with self._lock:
            self._unavailable_until[key_id] = self.monotonic() + seconds

    def _cooldown_seconds(self, response: httpx.Response | None, message: str) -> int:
        if response is not None:
            raw = response.headers.get("retry-after")
            try:
                if raw is not None:
                    return max(1, int(float(raw)))
            except ValueError:
                pass
        match = _RETRY_SECONDS.search(message)
        if match:
            return max(1, int(float(match.group(1))))
        return self.quota_cooldown_seconds

    @staticmethod
    def _response_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()
        if isinstance(payload, Mapping):
            for key in ("error", "error_message", "message"):
                value = payload.get(key)
                if value:
                    return str(value)
        return str(payload)

    def _http_error(self, response: httpx.Response) -> ProviderError:
        message = self._response_message(response) or f"HTTP {response.status_code}"
        status = response.status_code
        if status in {402, 429}:
            return ProviderError(
                f"Bright Data capacity error: {message}",
                ErrorKind.RATE_LIMIT,
                retryable=True,
                provider="brightdata",
                status_code=status,
                retry_after=self._cooldown_seconds(response, message),
            )
        if status in {401, 403}:
            return ProviderError(
                f"Bright Data credential is unavailable: {message}",
                ErrorKind.AUTHENTICATION,
                retryable=False,
                provider="brightdata",
                status_code=status,
                retry_after=self._cooldown_seconds(response, message),
            )
        if status in {400, 404, 409, 422}:
            return ProviderError(
                f"Bright Data rejected the request: {message}",
                ErrorKind.BUSINESS,
                provider="brightdata",
                status_code=status,
            )
        if status >= 500:
            return ProviderError(
                f"Bright Data request failed: {message}",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="brightdata",
                status_code=status,
            )
        return ProviderError(
            f"Bright Data returned HTTP {status}: {message}",
            ErrorKind.TRANSIENT_PROVIDER,
            retryable=True,
            provider="brightdata",
            status_code=status,
        )

    @staticmethod
    def _records(payload: object) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("data", "results", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, Mapping)]
        raise ProviderError(
            "Bright Data returned an unexpected response shape",
            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
            provider="brightdata",
        )

    def _wait_for_snapshot(self, client: httpx.Client, snapshot_id: str) -> list[dict[str, Any]]:
        deadline = self.monotonic() + self.snapshot_timeout_seconds
        while self.monotonic() < deadline:
            response = client.get(f"/datasets/v3/progress/{snapshot_id}")
            if response.status_code != 200:
                raise self._http_error(response)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError(
                    "Bright Data snapshot progress response was not valid JSON",
                    ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                    provider="brightdata",
                ) from exc
            if not isinstance(payload, Mapping):
                raise ProviderError(
                    "Bright Data snapshot progress response was not an object",
                    ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                    provider="brightdata",
                )

            status = str(payload.get("status") or "").casefold()
            if status == "ready":
                break
            if status == "failed":
                message = str(payload.get("error_message") or payload.get("message") or "snapshot failed")
                lowered = message.casefold()
                kind = (
                    ErrorKind.RATE_LIMIT
                    if any(marker in lowered for marker in _CAPACITY_MARKERS)
                    else ErrorKind.TRANSIENT_PROVIDER
                )
                raise ProviderError(
                    f"Bright Data snapshot failed: {message}",
                    kind,
                    retryable=True,
                    provider="brightdata",
                    retry_after=self.quota_cooldown_seconds if kind is ErrorKind.RATE_LIMIT else None,
                )

            logger().info(
                "brightdata_snapshot_waiting",
                snapshot_id=snapshot_id,
                status=status or "unknown",
            )
            self.sleep(self.snapshot_poll_seconds)
        else:
            raise ProviderError(
                f"Bright Data snapshot {snapshot_id} did not finish within {self.snapshot_timeout_seconds:g}s",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="brightdata",
            )

        while self.monotonic() < deadline:
            response = client.get(f"/datasets/v3/snapshot/{snapshot_id}", params={"format": "json"})
            if response.status_code == 202:
                self.sleep(self.snapshot_poll_seconds)
                continue
            if response.status_code != 200:
                raise self._http_error(response)
            try:
                return self._records(response.json())
            except ValueError as exc:
                raise ProviderError(
                    "Bright Data snapshot response was not valid JSON",
                    ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                    provider="brightdata",
                ) from exc

        raise ProviderError(
            f"Bright Data snapshot {snapshot_id} became ready but could not be downloaded before timeout",
            ErrorKind.TRANSIENT_PROVIDER,
            retryable=True,
            provider="brightdata",
        )

    def _rows_from_sync_response(
        self,
        response: httpx.Response,
        client: httpx.Client,
        *,
        input_count: int,
        operation: str,
    ) -> list[dict[str, Any]]:
        if response.status_code == 202:
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError(
                    "Bright Data returned HTTP 202 without a JSON snapshot response",
                    ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                    provider="brightdata",
                ) from exc
            snapshot_id = str(payload.get("snapshot_id") or "").strip() if isinstance(payload, Mapping) else ""
            if not snapshot_id:
                raise ProviderError(
                    "Bright Data returned HTTP 202 without a snapshot_id",
                    ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                    provider="brightdata",
                )
            logger().info(
                "brightdata_snapshot_started",
                snapshot_id=snapshot_id,
                input_count=input_count,
                operation=operation,
            )
            return self._wait_for_snapshot(client, snapshot_id)
        if response.status_code != 200:
            raise self._http_error(response)
        try:
            return self._records(response.json())
        except ValueError as exc:
            raise ProviderError(
                f"Bright Data {operation} response was not valid JSON",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="brightdata",
            ) from exc

    def _scrape_with_key(self, api_key: str, urls: Sequence[str]) -> list[dict[str, Any]]:
        with self.client_factory(api_key) as client:
            response = client.post(
                "/datasets/v3/scrape",
                params={
                    "dataset_id": self.dataset_id,
                    "type": "discover_new",
                    "discover_by": "url",
                    "notify": "false",
                    "include_errors": "true",
                    "format": "json",
                },
                json={"input": [{"url": url} for url in urls]},
            )
            return self._rows_from_sync_response(
                response,
                client,
                input_count=len(urls),
                operation="discovery",
            )

    def _collect_job_with_key(self, api_key: str, job_url: str) -> list[dict[str, Any]]:
        with self.client_factory(api_key) as client:
            response = client.post(
                "/datasets/v3/scrape",
                params={"dataset_id": self.dataset_id, "format": "json"},
                json=[{"url": job_url}],
            )
            return self._rows_from_sync_response(
                response,
                client,
                input_count=1,
                operation="linkedin_job_collect",
            )

    def _request_with_keys(
        self,
        *,
        operation: str,
        input_count: int,
        request: Callable[[str], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        keys = self._available_keys()
        queue = deque(keys)
        failures: list[str] = []
        cooldowns: list[int] = []

        while queue:
            api_key = queue.popleft()
            key_id = self._key_id(api_key)
            logger().info(
                "brightdata_key_attempt",
                key_id=key_id,
                input_count=input_count,
                operation=operation,
            )
            try:
                rows = request(api_key)
                logger().info(
                    "brightdata_key_succeeded",
                    key_id=key_id,
                    raw_count=len(rows),
                    operation=operation,
                )
                return rows
            except ProviderError as exc:
                if exc.kind in {ErrorKind.RATE_LIMIT, ErrorKind.AUTHENTICATION}:
                    cooldown = max(1, int(exc.retry_after or self.quota_cooldown_seconds))
                    cooldowns.append(cooldown)
                    failures.append(f"{key_id}: {exc}")
                    self._cooldown(api_key, cooldown)
                    logger().warning(
                        "brightdata_key_unavailable",
                        key_id=key_id,
                        operation=operation,
                        error_kind=exc.kind,
                        status_code=exc.status_code,
                        cooldown_seconds=cooldown,
                        error_message=str(exc),
                    )
                    continue
                raise
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Bright Data network request failed: {exc}",
                    ErrorKind.TRANSIENT_PROVIDER,
                    retryable=True,
                    provider="brightdata",
                ) from exc

        raise ProviderError(
            "Bright Data capacity exhausted for all configured accounts: " + "; ".join(failures),
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="brightdata",
            retry_after=min(cooldowns) if cooldowns else None,
        )

    def _request_batch(self, urls: Sequence[str]) -> list[dict[str, Any]]:
        return self._request_with_keys(
            operation="discovery",
            input_count=len(urls),
            request=lambda api_key: self._scrape_with_key(api_key, urls),
        )

    def _postprocess(self, rows: Sequence[Mapping[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
        error_codes: Counter[str] = Counter()
        malformed = 0
        duplicates = 0
        excluded = 0
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()

        for raw in rows:
            if _is_error_record(raw):
                code = str(raw.get("error_code") or "unknown")
                error_codes[code] += 1
                continue

            job = _canonical_job(raw)
            if job is None:
                malformed += 1
                continue

            company = str(job.get("companyName") or "").casefold()
            title = str(job.get("title") or "").casefold()
            if any(term.casefold() in company for term in self.company_exclusions) or any(
                term.casefold() in title for term in self.title_exclusions
            ):
                excluded += 1
                continue

            identity = _record_identity(job)
            if identity is not None and identity in seen:
                duplicates += 1
                continue
            if identity is not None:
                seen.add(identity)
            jobs.append(job)

        capped = jobs[:max_items]
        logger().info(
            "brightdata_discovery_postprocessed",
            raw_count=len(rows),
            accepted_count=len(capped),
            uncapped_accepted_count=len(jobs),
            error_count=sum(error_codes.values()),
            error_codes=dict(error_codes),
            malformed_count=malformed,
            duplicate_count=duplicates,
            excluded_count=excluded,
            capped_count=max(0, len(jobs) - len(capped)),
        )
        return capped

    def discover(self, urls: Sequence[str], *, max_items: int) -> list[dict[str, Any]]:
        search_urls = [url.strip() for url in urls if isinstance(url, str) and self._valid_search_url(url)]
        if not search_urls:
            raise ProviderError(
                "No valid LinkedIn search URLs were generated",
                ErrorKind.BUSINESS,
                provider="brightdata",
            )
        if max_items <= 0:
            raise ValueError("max_items must be positive")

        raw_rows: list[dict[str, Any]] = []
        for index in range(0, len(search_urls), _MAX_SYNC_INPUTS):
            batch = search_urls[index : index + _MAX_SYNC_INPUTS]
            logger().info(
                "brightdata_discovery_batch_started",
                batch_number=(index // _MAX_SYNC_INPUTS) + 1,
                input_count=len(batch),
                dataset_id=self.dataset_id,
            )
            raw_rows.extend(self._request_batch(batch))
        return self._postprocess(raw_rows, max_items=max_items)

    def fetch_linkedin(self, job_id: int) -> Mapping[str, Any]:
        if job_id <= 0:
            raise ValueError("LinkedIn job ID must be positive")

        expected_id = str(job_id)
        job_url = f"https://www.linkedin.com/jobs/view/{job_id}"
        logger().info(
            "brightdata_linkedin_job_collect_started",
            job_id=expected_id,
            dataset_id=self.dataset_id,
        )
        rows = self._request_with_keys(
            operation="linkedin_job_collect",
            input_count=1,
            request=lambda api_key: self._collect_job_with_key(api_key, job_url),
        )

        error_codes: Counter[str] = Counter()
        malformed = 0
        mismatched = 0
        for raw in rows:
            if _is_error_record(raw):
                error_codes[str(raw.get("error_code") or "unknown")] += 1
                continue

            returned_id = str(raw.get("job_posting_id") or "").strip()
            returned_url_id = _linkedin_job_id_from_url(raw.get("url"))
            if returned_id and returned_id != expected_id:
                mismatched += 1
                continue
            if returned_url_id and returned_url_id != expected_id:
                mismatched += 1
                continue

            enriched = dict(raw)
            enriched["job_posting_id"] = expected_id
            enriched["url"] = job_url
            job = _canonical_job(enriched)
            if job is None:
                malformed += 1
                continue
            job["_linkedinFetchProvider"] = "brightdata"
            logger().info(
                "brightdata_linkedin_job_collect_succeeded",
                job_id=expected_id,
                raw_count=len(rows),
                error_count=sum(error_codes.values()),
                malformed_count=malformed,
                mismatched_count=mismatched,
            )
            return job

        logger().warning(
            "brightdata_linkedin_job_collect_no_usable_result",
            job_id=expected_id,
            raw_count=len(rows),
            error_count=sum(error_codes.values()),
            error_codes=dict(error_codes),
            malformed_count=malformed,
            mismatched_count=mismatched,
        )
        raise ProviderError(
            f"LinkedIn job {job_id} was not found in the Bright Data response",
            ErrorKind.BUSINESS,
            provider="brightdata",
        )
