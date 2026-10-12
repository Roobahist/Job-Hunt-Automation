from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qs, urlsplit

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.brightdata_resilient import ResilientBrightDataProvider
from job_hunt.logging import logger


def _text(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            candidate = str(item).strip()
            if candidate:
                return candidate
        return ""
    return str(value).strip()


def _indeed_job_id_from_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").casefold()
    if host != "indeed.com" and not host.endswith(".indeed.com"):
        return None
    if parsed.path.rstrip("/") != "/viewjob":
        return None
    values = parse_qs(parsed.query).get("jk", [])
    return str(values[0]).strip() if values and str(values[0]).strip() else None


def _is_error_record(record: Mapping[str, Any]) -> bool:
    if not record.get("error") and not record.get("error_code"):
        return False
    return not any(
        record.get(key)
        for key in ("jobid", "job_title", "company_name", "description_text")
    )


def _canonical_indeed_job(
    record: Mapping[str, Any],
    *,
    expected_job_id: str,
    authoritative_url: str,
) -> dict[str, Any] | None:
    returned_id = _text(record.get("jobid") or record.get("job_id") or record.get("id"))
    returned_url_id = _indeed_job_id_from_url(
        record.get("url") or record.get("job_url") or record.get("input_url")
    )
    if returned_id and returned_id != expected_job_id:
        return None
    if returned_url_id and returned_url_id != expected_job_id:
        return None

    title = _text(record.get("job_title") or record.get("title"))
    company = _text(record.get("company_name") or record.get("company"))
    description = _text(
        record.get("description_text")
        or record.get("job_description")
        or record.get("description")
    )
    if not title or not company or not description:
        return None

    canonical = dict(record)
    canonical.update(
        {
            "jobId": expected_job_id,
            "jobUrl": authoritative_url,
            "title": title,
            "companyName": company,
            "descriptionText": description,
            "location": _text(
                record.get("location")
                or record.get("job_location")
                or record.get("formatted_location")
            )
            or None,
            "employmentType": _text(
                record.get("job_type")
                or record.get("employment_type")
                or record.get("job_employment_type")
            )
            or None,
            "postedAt": record.get("date_posted_parsed") or record.get("date_posted"),
            "salary": record.get("salary") or record.get("salary_info") or record.get("salary_text"),
            "applyUrl": authoritative_url,
            "_jobFetchProvider": "brightdata_indeed",
        }
    )
    return canonical


class BrightDataIndeedProvider(ResilientBrightDataProvider):
    """Collect one Indeed posting while reusing Bright Data credential and retry behavior."""

    def _collect_indeed_with_key(self, api_key: str, job_url: str) -> list[dict[str, Any]]:
        with self.client_factory(api_key) as client:
            response = client.post(
                "/datasets/v3/scrape",
                params={
                    "dataset_id": self.dataset_id,
                    "notify": "false",
                    "include_errors": "true",
                    "format": "json",
                },
                json={
                    "input": [{"url": job_url}],
                    "limit_per_input": None,
                },
            )
            return self._rows_from_sync_response(
                response,
                client,
                input_count=1,
                operation="indeed_job_collect",
            )

    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]:
        normalized_id = str(job_id).strip()
        if not normalized_id:
            raise ValueError("Indeed job ID is required")

        job_url = f"https://www.indeed.com/viewjob?jk={normalized_id}"
        logger().info(
            "brightdata_indeed_job_collect_started",
            job_id=normalized_id,
            dataset_id=self.dataset_id,
        )
        rows = self._request_with_keys(
            operation="indeed_job_collect",
            input_count=1,
            request=lambda api_key: self._collect_indeed_with_key(api_key, job_url),
        )

        error_codes: Counter[str] = Counter()
        malformed = 0
        mismatched = 0
        for raw in rows:
            if _is_error_record(raw):
                error_codes[str(raw.get("error_code") or "unknown")] += 1
                continue

            returned_id = _text(raw.get("jobid") or raw.get("job_id") or raw.get("id"))
            returned_url_id = _indeed_job_id_from_url(
                raw.get("url") or raw.get("job_url") or raw.get("input_url")
            )
            if returned_id and returned_id != normalized_id:
                mismatched += 1
                continue
            if returned_url_id and returned_url_id != normalized_id:
                mismatched += 1
                continue

            job = _canonical_indeed_job(
                raw,
                expected_job_id=normalized_id,
                authoritative_url=job_url,
            )
            if job is None:
                malformed += 1
                continue

            logger().info(
                "brightdata_indeed_job_collect_succeeded",
                job_id=normalized_id,
                raw_count=len(rows),
                error_count=sum(error_codes.values()),
                malformed_count=malformed,
                mismatched_count=mismatched,
            )
            return job

        logger().warning(
            "brightdata_indeed_job_collect_no_usable_result",
            job_id=normalized_id,
            raw_count=len(rows),
            error_count=sum(error_codes.values()),
            error_codes=dict(error_codes),
            malformed_count=malformed,
            mismatched_count=mismatched,
        )
        raise ProviderError(
            f"Indeed job {normalized_id} was not found in the Bright Data response",
            ErrorKind.BUSINESS,
            provider="brightdata",
        )
