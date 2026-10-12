from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.apify import ApifyProvider
from job_hunt.integrations.apify_actors import INDEED_SINGLE_JOB_ACTOR
from job_hunt.logging import logger
from job_hunt.state import RedisState


def _text(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_indeed_job(
    record: Mapping[str, Any],
    *,
    expected_job_id: str,
    authoritative_url: str,
) -> dict[str, Any] | None:
    returned_id = _text(record.get("id") or record.get("jobKey") or record.get("job_id"))
    if returned_id and returned_id != expected_job_id:
        return None

    title_data = _mapping(record.get("title"))
    company_data = _mapping(record.get("company"))
    description_data = _mapping(record.get("description"))
    urls = _mapping(record.get("urls"))
    location_data = _mapping(record.get("location"))
    classification = _mapping(record.get("classification"))
    dates = _mapping(record.get("dates"))
    apply_data = _mapping(record.get("apply"))
    salary_data = _mapping(record.get("salary"))

    title = _text(title_data.get("text") or title_data.get("normalized") or record.get("job_title"))
    company = _text(company_data.get("name") or record.get("company_name"))
    description = _text(
        description_data.get("text")
        or record.get("description_text")
        or record.get("job_description")
        or record.get("snippet")
    )
    if not title or not company or not description:
        return None

    location = _text(
        location_data.get("formattedShort")
        or location_data.get("formatted")
        or location_data.get("fullAddress")
        or location_data.get("city")
    )
    employment_type = _text(classification.get("jobType"))
    apply_url = _text(
        apply_data.get("url")
        or urls.get("apply")
        or urls.get("external")
        or urls.get("indeed")
        or authoritative_url
    )
    salary = salary_data.get("text") or record.get("salary")

    canonical = dict(record)
    canonical.update(
        {
            "jobId": expected_job_id,
            "jobUrl": authoritative_url,
            "title": title,
            "companyName": company,
            "descriptionText": description,
            "descriptionHtml": description_data.get("html"),
            "location": location or None,
            "employmentType": employment_type or None,
            "postedAt": dates.get("posted") or dates.get("onIndeed"),
            "salary": salary,
            "applyUrl": apply_url or authoritative_url,
            "_jobFetchProvider": "apify_indeed",
        }
    )
    return canonical


class ApifyIndeedProvider(ApifyProvider):
    """Fetch one known Indeed posting with the application-owned Indeed Actor."""

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        actor_id: str = INDEED_SINGLE_JOB_ACTOR,
        quota_cooldown_seconds: int = 3600,
        state: RedisState | None = None,
    ) -> None:
        resolved_actor_id = str(actor_id).strip()
        if not resolved_actor_id:
            raise ValueError("Indeed Apify actor ID is required")
        super().__init__(
            tokens,
            resolved_actor_id,
            resolved_actor_id,
            quota_cooldown_seconds=quota_cooldown_seconds,
            state=state,
        )
        self.indeed_actor_id = resolved_actor_id

    def fetch_indeed(self, job_id: str, *, country: str) -> Mapping[str, Any]:
        normalized_id = str(job_id).strip()
        if not normalized_id:
            raise ValueError("Indeed job ID is required")

        normalized_country = str(country).strip().upper() or "US"
        job_url = f"https://www.indeed.com/viewjob?jk={normalized_id}"
        logger().info(
            "apify_indeed_job_collect_started",
            actor_id=self.indeed_actor_id,
            job_id=normalized_id,
            country=normalized_country,
        )
        rows = list(
            self._run(
                self.indeed_actor_id,
                {
                    "jobKeys": [normalized_id],
                    "country": normalized_country,
                    "searchMode": "basic",
                },
                max_items=1,
            )
        )

        malformed = 0
        mismatched = 0
        for raw in rows:
            returned_id = _text(raw.get("id") or raw.get("jobKey") or raw.get("job_id"))
            if returned_id and returned_id != normalized_id:
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
                "apify_indeed_job_collect_succeeded",
                actor_id=self.indeed_actor_id,
                job_id=normalized_id,
                raw_count=len(rows),
                malformed_count=malformed,
                mismatched_count=mismatched,
            )
            return job

        logger().warning(
            "apify_indeed_job_collect_no_usable_result",
            actor_id=self.indeed_actor_id,
            job_id=normalized_id,
            raw_count=len(rows),
            malformed_count=malformed,
            mismatched_count=mismatched,
        )
        raise ProviderError(
            f"Indeed job {normalized_id} was not found in the Apify response",
            ErrorKind.BUSINESS,
            provider="apify",
        )
