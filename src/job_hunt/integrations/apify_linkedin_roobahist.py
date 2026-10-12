from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.logging import logger


class ApifyActorRunner(Protocol):
    def _run(
        self,
        actor_id: str,
        run_input: Mapping[str, Any],
        max_items: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> Iterable[dict[str, Any]]: ...


_SEARCH_FIELDS: dict[str, bool] = {
    "jobId": True,
    "jobUrn": False,
    "jobUrl": True,
    "title": True,
    "companyName": True,
    "companyUrl": False,
    "location": True,
    "postedAt": True,
    "postedText": False,
    "searchUrls": False,
    "companyLogoUrl": False,
    "companyId": False,
    "companySlug": False,
    "jobSlug": False,
    "scrapedAt": False,
    "searchPosition": False,
    "searchSalaryText": False,
}

_JOB_FIELDS: dict[str, bool] = {
    "jobDescription": True,
    "jobDescriptionHtml": False,
    "jobSeniorityLevel": False,
    "jobEmploymentType": True,
    "jobFunctions": False,
    "jobIndustries": False,
    "jobApplicantCountText": False,
    "jobApplicantCount": False,
    "jobApplicantCountType": False,
    "jobCriteria": False,
    "jobSalaryText": False,
    "jobSalaryMin": False,
    "jobSalaryMax": False,
    "jobSalaryCurrency": False,
    "jobSalaryPeriod": False,
    "jobApplyMethod": False,
    "jobApplyUrl": False,
    "jobPosterProfileUrl": False,
    "jobPosterName": False,
    "jobPostedAtTimestamp": False,
    "jobIsReposted": False,
    "jobWorkplaceType": False,
    "jobDetailStatus": False,
    "jobBenefits": False,
    "jobClosedNotice": False,
    "jobIsVerified": False,
}

_DEFAULT_JOB_DETAIL_CONCURRENCY = 3
_DEFAULT_RUN_TIMEOUT_SECONDS = 900


def _text(value: object) -> str:
    return str(value or "").strip()


def _optional_text(value: object) -> str | None:
    text = _text(value)
    return text or None


class RoobahistLinkedInProvider:
    """Adapter for roobahist/apify-linkedin with a stable provider-neutral row contract."""

    def __init__(
        self,
        runner: ApifyActorRunner,
        actor_id: str,
        *,
        company_inclusions: Sequence[str] = (),
        company_exclusions: Sequence[str] = (),
        title_inclusions: Sequence[str] = (),
        title_exclusions: Sequence[str] = (),
        description_inclusions: Sequence[str] = (),
        description_exclusions: Sequence[str] = (),
        run_timeout_seconds: int = _DEFAULT_RUN_TIMEOUT_SECONDS,
        run_max_total_charge_usd: float | None = None,
        job_detail_concurrency: int = _DEFAULT_JOB_DETAIL_CONCURRENCY,
    ) -> None:
        self.runner = runner
        self.actor_id = actor_id.strip()
        if not self.actor_id:
            raise ValueError("Roobahist LinkedIn Actor ID is required")
        if run_timeout_seconds < 0:
            raise ValueError("run_timeout_seconds must be non-negative")
        if run_max_total_charge_usd is not None and run_max_total_charge_usd <= 0:
            raise ValueError("run_max_total_charge_usd must be positive when provided")
        if not 1 <= job_detail_concurrency <= 5:
            raise ValueError("job_detail_concurrency must be between 1 and 5")
        self.run_timeout_seconds = run_timeout_seconds
        self.run_max_total_charge_usd = run_max_total_charge_usd
        self.job_detail_concurrency = job_detail_concurrency
        self.company_inclusions = [str(value).strip() for value in company_inclusions if str(value).strip()]
        self.company_exclusions = [str(value).strip() for value in company_exclusions if str(value).strip()]
        self.title_inclusions = [str(value).strip() for value in title_inclusions if str(value).strip()]
        self.title_exclusions = [str(value).strip() for value in title_exclusions if str(value).strip()]
        self.description_inclusions = [str(value).strip() for value in description_inclusions if str(value).strip()]
        self.description_exclusions = [str(value).strip() for value in description_exclusions if str(value).strip()]

    def _base_input(self, max_results: int) -> dict[str, Any]:
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        return {
            "searchFields": dict(_SEARCH_FIELDS),
            "enrichJobInfo": True,
            "jobFields": dict(_JOB_FIELDS),
            "enrichCompanyInfo": False,
            "jobDetailConcurrency": self.job_detail_concurrency,
            "maxResults": max_results,
            "removeDuplicates": True,
        }

    def _run_actor(self, run_input: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
        if self.run_max_total_charge_usd is None:
            return self.runner._run(
                self.actor_id,
                run_input,
                max_items=None,
                timeout_secs=self.run_timeout_seconds,
            )
        return self.runner._run(
            self.actor_id,
            run_input,
            max_items=None,
            timeout_secs=self.run_timeout_seconds,
            max_total_charge_usd=self.run_max_total_charge_usd,
        )

    @staticmethod
    def _canonical_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": _text(row.get("jobId")),
            "url": _text(row.get("jobUrl")),
            "companyName": _text(row.get("companyName")),
            "title": _text(row.get("title")),
            "description": _text(row.get("jobDescription")),
            "location": _optional_text(row.get("location")),
            "employmentType": _optional_text(row.get("jobEmploymentType")),
            "postedAt": row.get("postedAt"),
        }

    @staticmethod
    def _complete(row: Mapping[str, Any]) -> bool:
        return all(_text(row.get(field)) for field in ("id", "url", "companyName", "title", "description"))

    def discover(
        self,
        urls: Sequence[str],
        *,
        max_items: int,
        skip_job_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        search_urls = [str(url).strip() for url in urls if str(url).strip()]
        if not search_urls:
            raise ProviderError(
                "No LinkedIn search URLs were supplied to the Roobahist Actor",
                ErrorKind.BUSINESS,
                provider="apify",
            )
        resolved_skip_ids = list(dict.fromkeys(str(value).strip() for value in skip_job_ids if str(value).strip()))
        run_input = self._base_input(max_items) | {
            "searchUrls": [{"url": url, "inferGeoId": True} for url in search_urls],
            "skipJobIds": resolved_skip_ids,
            "includeCompanyNames": self.company_inclusions,
            "excludeCompanyNames": self.company_exclusions,
            "includeJobTitleKeywords": self.title_inclusions,
            "excludeJobTitleKeywords": self.title_exclusions,
            "includeJobDescriptionKeywords": self.description_inclusions,
            "excludeJobDescriptionKeywords": self.description_exclusions,
        }
        logger().info(
            "roobahist_linkedin_discovery_started",
            actor_id=self.actor_id,
            input_count=len(search_urls),
            max_items=max_items,
            skip_job_ids=len(resolved_skip_ids),
            job_detail_concurrency=self.job_detail_concurrency,
            run_timeout_seconds=self.run_timeout_seconds,
            include_companies=len(self.company_inclusions),
            exclude_companies=len(self.company_exclusions),
            include_title_keywords=len(self.title_inclusions),
            exclude_title_keywords=len(self.title_exclusions),
            include_description_keywords=len(self.description_inclusions),
            exclude_description_keywords=len(self.description_exclusions),
        )
        raw_rows = list(self._run_actor(run_input))
        rows: list[dict[str, Any]] = []
        incomplete = 0
        for raw in raw_rows:
            row = self._canonical_row(raw)
            if not self._complete(row):
                incomplete += 1
                continue
            rows.append(row)
        if incomplete:
            logger().warning(
                "roobahist_linkedin_incomplete_rows_skipped",
                actor_id=self.actor_id,
                incomplete_count=incomplete,
                result_count=len(rows),
            )
        logger().info(
            "roobahist_linkedin_discovery_succeeded",
            actor_id=self.actor_id,
            raw_result_count=len(raw_rows),
            result_count=len(rows),
        )
        return rows

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        del country, max_concurrency
        if job_id <= 0:
            raise ValueError("LinkedIn job ID must be positive")
        run_input = self._base_input(1) | {
            "jobIds": [str(job_id)],
            "skipJobIds": [],
            "includeCompanyNames": [],
            "excludeCompanyNames": [],
            "includeJobTitleKeywords": [],
            "excludeJobTitleKeywords": [],
            "includeJobDescriptionKeywords": [],
            "excludeJobDescriptionKeywords": [],
        }
        logger().info(
            "roobahist_linkedin_direct_fetch_started",
            actor_id=self.actor_id,
            job_id=str(job_id),
            job_detail_concurrency=self.job_detail_concurrency,
            run_timeout_seconds=self.run_timeout_seconds,
        )
        raw_rows = list(self._run_actor(run_input))
        for raw in raw_rows:
            row = self._canonical_row(raw)
            if _text(row.get("id")) != str(job_id):
                continue
            if not self._complete(row):
                raise ProviderError(
                    f"Roobahist LinkedIn Actor returned incomplete enriched data for job {job_id}",
                    ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                    retryable=True,
                    provider="apify",
                )
            logger().info(
                "roobahist_linkedin_direct_fetch_succeeded",
                actor_id=self.actor_id,
                job_id=str(job_id),
            )
            return row
        raise ProviderError(
            f"LinkedIn job {job_id} was not found in the Roobahist Actor response",
            ErrorKind.BUSINESS,
            provider="apify",
        )
