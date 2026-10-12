from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from job_hunt.application.job_equivalence import best_processed_equivalent, jobs_equivalent
from job_hunt.domain.models import Job


def _job(*, location: str, title: str | None = None, description: str | None = None) -> Job:
    return Job(
        source="linkedin",
        external_id="1",
        url="https://www.linkedin.com/jobs/view/1",
        company_name="Example AI",
        title=title or f"Machine Learning Engineer - {location}",
        description=description
        or (
            f"Join our {location} team to build production machine learning systems, "
            "design model-serving APIs, improve inference performance, and collaborate "
            "with product and data teams on reliable AI features."
        ),
        location=location,
        contract_type="Full-time",
    )


def _row(row_id: int, job: Job, **extra: Any) -> dict[str, Any]:
    return {
        "id": row_id,
        "Job ID": str(row_id),
        "Company Name": job.company_name,
        "Title": job.title,
        "Job Description": job.description,
        "Location": job.location,
        "Contract Type": {"value": job.contract_type},
        "Link": job.url,
        "Status": {"id": 10, "value": "New"},
        **extra,
    }


def test_equivalence_ignores_each_postings_location() -> None:
    calgary = _job(location="Calgary, Alberta, Canada")
    toronto = _job(location="Toronto, Ontario, Canada")

    assert jobs_equivalent(calgary, toronto)


def test_equivalence_rejects_meaningfully_different_description() -> None:
    original = _job(location="Calgary, Alberta, Canada")
    different = _job(
        location="Toronto, Ontario, Canada",
        description=(
            "Lead enterprise sales across Canada, own strategic accounts, negotiate commercial agreements, "
            "and build executive relationships with customers."
        ),
    )

    assert not jobs_equivalent(original, different)


def test_equivalence_requires_same_company() -> None:
    left = _job(location="Calgary, Alberta, Canada")
    right = _job(location="Toronto, Ontario, Canada").model_copy(update={"company_name": "Different Corp"})

    assert not jobs_equivalent(left, right)


def test_best_processed_equivalent_prefers_completed_artifacts() -> None:
    target = _job(location="Vancouver, British Columbia, Canada")
    dropped_job = _job(location="Calgary, Alberta, Canada")
    completed_job = _job(location="Toronto, Ontario, Canada")
    rows = [
        _row(1, dropped_job, Status={"id": 11, "value": "Dropped"}),
        _row(
            2,
            completed_job,
            Status={"id": 12, "value": "To Apply"},
            Score=91,
            CV=[{"name": "cv.pdf"}],
            **{
                "Cover Letter": [{"name": "cover.pdf"}],
                "Application ZIP": [{"name": "application.zip"}],
            },
        ),
    ]

    selected = best_processed_equivalent(target, rows, new_status_id=10)

    assert selected is not None
    assert selected["id"] == 2


def test_best_processed_equivalent_prefers_scored_row_over_status_only_row() -> None:
    target = _job(location="Vancouver, British Columbia, Canada")
    status_only = _job(location="Calgary, Alberta, Canada")
    scored = _job(location="Toronto, Ontario, Canada")
    rows = [
        _row(1, status_only, Status={"id": 11, "value": "Dropped"}),
        _row(2, scored, Score=76),
    ]

    selected = best_processed_equivalent(target, rows, new_status_id=10)

    assert selected is not None
    assert selected["id"] == 2


def test_same_role_can_match_across_current_and_previous_storage_window() -> None:
    # The window itself is enforced by recent_baserow_rows; this fixture verifies the
    # matching data remains independent of the posting date/location differences.
    current = _job(location="Edmonton, Alberta, Canada").model_copy(
        update={"published_at": datetime(2026, 9, 18, 8, tzinfo=UTC)}
    )
    previous = _job(location="Seattle, Washington, USA").model_copy(
        update={"published_at": datetime(2026, 9, 18, 7, tzinfo=UTC)}
    )

    assert jobs_equivalent(current, previous)
