from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from job_hunt.application.page_content import prepare_page_content
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import (
    AiContentSubmission,
    EntryType,
    ExternalSubmission,
    IndeedSubmission,
    Job,
    JobSubmission,
    LinkedInSubmission,
    SubmissionBase,
    UrlSubmission,
)
from job_hunt.errors import ProviderError, is_context_length_error
from job_hunt.ports import DiscoveryProvider, JobExtractor
from job_hunt.security import fetch_public_text
from job_hunt.url_utils import valid_http_url

_LINKEDIN_JOB_ID = re.compile(r"(?:^|[-/])(\d{6,})(?:/)?$")
_ENTRY_TYPE_ALIASES = {
    "linkedin": EntryType.LINKEDIN,
    "linkedin_entry": EntryType.LINKEDIN,
    "indeed": EntryType.INDEED,
    "indeed_entry": EntryType.INDEED,
    "external": EntryType.EXTERNAL,
    "external_entry": EntryType.EXTERNAL,
    "ai": EntryType.AI_CONTENT,
    "ai_entry": EntryType.AI_CONTENT,
    "ai_content": EntryType.AI_CONTENT,
    "url": EntryType.URL,
    "url_entry": EntryType.URL,
}


def linkedin_job_id_from_url(url: str) -> int | None:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host != "linkedin.com" and not host.endswith(".linkedin.com"):
        return None
    if "/jobs/view/" not in parts.path:
        return None
    match = _LINKEDIN_JOB_ID.search(parts.path.rstrip("/"))
    return int(match.group(1)) if match else None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_datetime(int(text))
    try:
        return _as_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return None


def job_from_provider(data: Mapping[str, Any], *, source: str = "linkedin") -> Job:
    external_id = data.get("id") or data.get("jobId") or data.get("job_id")
    url = data.get("url") or data.get("link") or data.get("jobUrl") or data.get("inputUrl")
    company = data.get("companyName") or data.get("company") or data.get("company_name")
    title = data.get("title") or data.get("jobTitle") or data.get("job_title")
    description = (
        data.get("description")
        or data.get("descriptionText")
        or data.get("jobDescription")
        or data.get("job_description")
    )
    published = (
        data.get("publishedAt")
        or data.get("published_at")
        or data.get("postedAt")
        or data.get("posted_at")
        or data.get("datePosted")
        or data.get("date_posted")
        or data.get("postedDate")
        or data.get("publishedDate")
    )
    contract_type = (
        data.get("contractType")
        or data.get("contract_type")
        or data.get("employmentType")
        or data.get("employment_type")
    )
    return assign_identity(
        Job(
            source=source,
            external_id=str(external_id) if external_id else None,
            url=str(url or ""),
            company_name=str(company or ""),
            title=str(title or ""),
            description=str(description or ""),
            location=str(data.get("location") or "") or None,
            contract_type=str(contract_type or "") or None,
            published_at=_parse_datetime(published),
            provider_data=dict(data),
        )
    )


def _with_submission_controls(job: Job, submission: SubmissionBase) -> Job:
    if not submission.reuse_existing_score and not submission.reuse_existing_data:
        return job
    provider_data = dict(job.provider_data)
    provider_data["reuse_existing_score"] = submission.reuse_existing_score
    provider_data["reuse_existing_data"] = submission.reuse_existing_data
    return job.model_copy(update={"provider_data": provider_data})


def _assign_preserving_source_url(job: Job, source_url: str) -> Job:
    authoritative_url = valid_http_url(source_url)
    prepared = job.model_copy(update={"url": authoritative_url}) if authoritative_url else job
    identified = assign_identity(prepared)
    if authoritative_url is None:
        return identified
    return identified.model_copy(update={"url": authoritative_url})


def _missing_required_fields(job: Job) -> list[str]:
    values = {
        "company_name": job.company_name,
        "job_title": job.title,
        "job_description": job.description,
    }
    return [name for name, value in values.items() if not str(value or "").strip()]


def _require_complete_job(job: Job, *, guidance: str) -> Job:
    missing = _missing_required_fields(job)
    if missing:
        raise ValueError(f"Required job information is missing: {', '.join(missing)}. {guidance}")
    return job


class SubmissionNormalizer:
    def __init__(self, discovery: DiscoveryProvider, extractor: JobExtractor, linkedin_template: str) -> None:
        self.discovery = discovery
        self.extractor = extractor
        self.linkedin_template = linkedin_template

    def _linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Job:
        raw = self.discovery.fetch_linkedin(job_id, country=country, max_concurrency=max_concurrency)
        enriched = dict(raw) | {
            "id": raw.get("id") or raw.get("jobId") or raw.get("job_id") or job_id,
            "url": (
                raw.get("url")
                or raw.get("link")
                or raw.get("jobUrl")
                or raw.get("inputUrl")
                or self.linkedin_template.format(jobId=job_id)
            ),
        }
        job = job_from_provider(enriched)
        return _require_complete_job(
            job,
            guidance="The LinkedIn source did not provide enough information to process this job.",
        )

    def _indeed(self, job_id: str) -> Job:
        raw = self.discovery.fetch_indeed(job_id)
        job_url = f"https://www.indeed.com/viewjob?jk={job_id}"
        enriched = dict(raw) | {
            "id": raw.get("id") or raw.get("jobId") or raw.get("job_id") or job_id,
            "url": raw.get("url") or raw.get("link") or raw.get("jobUrl") or raw.get("inputUrl") or job_url,
        }
        # Identity uses the Indeed job ID, while the stored URL keeps the jk query
        # parameter because it is part of the canonical Indeed posting address.
        job = job_from_provider(enriched, source="indeed").model_copy(update={"url": job_url})
        return _require_complete_job(
            job,
            guidance="The Indeed source did not provide enough information to process this job.",
        )

    def _extract_with_guidance(self, content: str, source_url: str, *, guidance: str) -> Job:
        try:
            extracted = self.extractor.extract_job(content, source_url)
        except ValueError as exc:
            raise ValueError(f"Could not extract all required job information. {guidance}") from exc
        except ProviderError as exc:
            if is_context_length_error(exc):
                raise ValueError(
                    "The job content is too large for all configured LLM context windows. " + guidance
                ) from exc
            raise
        return _require_complete_job(extracted, guidance=guidance)

    def normalize(self, submission: JobSubmission, *, country: str, max_concurrency: int) -> Job:
        if isinstance(submission, ExternalSubmission):
            job = assign_identity(
                Job(
                    source=submission.source,
                    external_id=submission.external_job_id,
                    url=str(submission.job_url),
                    company_name=submission.company_name,
                    title=submission.job_title,
                    description=submission.job_description,
                    location=submission.location,
                    contract_type=submission.contract_type,
                    published_at=submission.published_at,
                )
            )
            completed = _require_complete_job(
                job,
                guidance="Use Manual mode and provide the missing required fields.",
            )
            return _with_submission_controls(completed, submission)
        if isinstance(submission, LinkedInSubmission):
            job = self._linkedin(
                submission.linkedin_job_id,
                country=country,
                max_concurrency=max_concurrency,
            )
            return _with_submission_controls(job, submission)
        if isinstance(submission, IndeedSubmission):
            return _with_submission_controls(self._indeed(submission.indeed_job_id), submission)
        if isinstance(submission, UrlSubmission):
            source_url = str(submission.job_url)
            linkedin_job_id = linkedin_job_id_from_url(source_url)
            if linkedin_job_id is not None:
                job = self._linkedin(
                    linkedin_job_id,
                    country=country,
                    max_concurrency=max_concurrency,
                )
                return _with_submission_controls(job, submission)
            content = prepare_page_content(fetch_public_text(source_url))
            guidance = "Use AI mode and paste the full job description instead."
            extracted = self._extract_with_guidance(content, source_url, guidance=guidance)
            job = _assign_preserving_source_url(extracted, source_url)
            return _with_submission_controls(job, submission)
        if isinstance(submission, AiContentSubmission):
            source_url = str(submission.source_url or "")
            content = prepare_page_content(submission.page_content)
            guidance = (
                "The pasted posting does not contain enough information. "
                "Use Manual mode and provide the required job details."
            )
            extracted = self._extract_with_guidance(content, source_url, guidance=guidance)
            job = _assign_preserving_source_url(extracted, source_url)
            return _with_submission_controls(job, submission)
        raise TypeError(f"Unsupported submission: {submission.entry_type}")


def _entry_type(value: object) -> EntryType | str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return _ENTRY_TYPE_ALIASES.get(normalized, normalized)


def _optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def fillout_submission(payload: Mapping[str, Any], field_ids: Mapping[str, str]) -> JobSubmission:
    source = payload.get("submission", payload)
    if not isinstance(source, Mapping):
        source = payload
    flattened: dict[str, Any] = dict(source)
    questions = source.get("questions", [])
    if isinstance(questions, list):
        for question in questions:
            if not isinstance(question, Mapping):
                continue
            question_id = question.get("id") or question.get("questionId")
            if question_id:
                flattened[str(question_id)] = question.get("value")

    def value(name: str) -> Any:
        configured_id = field_ids.get(name)
        if configured_id and configured_id in flattened:
            return flattened[configured_id]
        return flattened.get(name)

    entry = _entry_type(value("entryType"))
    if entry == EntryType.LINKEDIN:
        return LinkedInSubmission(entry_type=EntryType.LINKEDIN, linkedin_job_id=int(value("linkedinJobId")))
    if entry == EntryType.INDEED:
        return IndeedSubmission(entry_type=EntryType.INDEED, indeed_job_id=str(value("indeedJobId") or "").strip())
    if entry == EntryType.EXTERNAL:
        return ExternalSubmission(
            entry_type=EntryType.EXTERNAL,
            company_name=value("companyName"),
            job_title=value("jobTitle"),
            job_description=value("jobDescription"),
            job_url=value("jobUrl"),
            location=_optional(value("location")),
            contract_type=_optional(value("contractType")),
            published_at=_optional(value("publishedAt")),
            external_job_id=_optional(value("externalJobId")),
        )
    if entry == EntryType.AI_CONTENT:
        return AiContentSubmission(
            entry_type=EntryType.AI_CONTENT,
            page_content=value("pageContent"),
            source_url=_optional(value("jobUrl")),
        )
    return UrlSubmission(entry_type=EntryType.URL, job_url=value("jobUrl"))
