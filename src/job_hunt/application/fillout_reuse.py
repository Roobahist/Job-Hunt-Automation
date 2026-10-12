from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from job_hunt.application.normalization import linkedin_job_id_from_url
from job_hunt.application.triage import job_from_baserow_row
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import (
    AiContentSubmission,
    ExternalSubmission,
    IndeedSubmission,
    Job,
    JobSubmission,
    LinkedInSubmission,
    UrlSubmission,
)
from job_hunt.retry import retry_transient
from job_hunt.url_utils import valid_http_url


class ExistingJobRepository(Protocol):
    def find_reference(
        self,
        *,
        external_id: str | None = None,
        url: str | None = None,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class ExistingFilloutResolution:
    row: Mapping[str, Any] | None
    job: Job | None


def _lookup_reference(
    submission: JobSubmission,
    linkedin_template: str,
) -> tuple[str | None, str | None, str]:
    if isinstance(submission, LinkedInSubmission):
        job_id = str(submission.linkedin_job_id)
        return job_id, linkedin_template.format(jobId=job_id), "linkedin"
    if isinstance(submission, IndeedSubmission):
        job_id = submission.indeed_job_id
        return job_id, f"https://www.indeed.com/viewjob?jk={job_id}", "indeed"
    if isinstance(submission, ExternalSubmission):
        return submission.external_job_id, str(submission.job_url), submission.source
    if isinstance(submission, UrlSubmission):
        url = str(submission.job_url)
        linkedin_job_id = linkedin_job_id_from_url(url)
        if linkedin_job_id is not None:
            return str(linkedin_job_id), url, "linkedin"
        return None, url, "url"
    if isinstance(submission, AiContentSubmission):
        return None, str(submission.source_url) if submission.source_url else None, "ai_content"
    raise TypeError(f"Unsupported submission: {submission.entry_type}")


def resolve_existing_fillout(
    submission: JobSubmission,
    repository: ExistingJobRepository,
    linkedin_template: str,
) -> ExistingFilloutResolution:
    external_id, url, source = _lookup_reference(submission, linkedin_template)
    if not external_id and not url:
        return ExistingFilloutResolution(row=None, job=None)

    row = retry_transient(
        repository.find_reference,
        external_id=external_id,
        url=url,
    )
    if row is None:
        return ExistingFilloutResolution(row=None, job=None)

    for field in ("Company Name", "Title", "Job Description"):
        if not str(row.get(field) or "").strip():
            return ExistingFilloutResolution(row=row, job=None)

    stored = job_from_baserow_row(row)
    stored_url = valid_http_url(stored.url)
    reference_url = valid_http_url(url or "")
    updates: dict[str, object] = {
        "source": source,
        "url": stored_url or reference_url or "",
    }
    if not stored.external_id and external_id:
        updates["external_id"] = external_id
    return ExistingFilloutResolution(row=row, job=assign_identity(stored.model_copy(update=updates)))
