from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from job_hunt.application.normalization import job_from_provider
from job_hunt.domain.models import Job

_EXPERIENCE_LEVEL_CODES = {
    "internship": "1",
    "entry level": "2",
    "entry": "2",
    "associate": "3",
    "mid-senior": "4",
    "mid-senior level": "4",
    "mid senior level": "4",
    "director": "5",
    "executive": "6",
}
_JOB_TYPE_CODES = {
    "full-time": "F",
    "full time": "F",
    "part-time": "P",
    "part time": "P",
    "contract": "C",
    "temporary": "T",
    "volunteer": "V",
    "internship": "I",
    "other": "O",
}
_WORKPLACE_TYPE_CODES = {
    "on-site": "1",
    "onsite": "1",
    "remote": "2",
    "hybrid": "3",
}
_SORT_BY_CODES = {
    "relevance": "R",
    "date": "DD",
    "r": "R",
    "dd": "DD",
}


def search_criteria_active(criteria: Mapping[str, Any]) -> bool:
    """Return whether a Baserow search-criteria row is enabled for discovery."""
    value = criteria.get("Active")
    if value is None:
        value = criteria.get("active")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "active", "enabled"}
    return False


def _linkedin_date_filter(interval_hours: int | float, *, window_seconds: int | None = None) -> str:
    if window_seconds is not None:
        if window_seconds <= 0:
            raise ValueError("LinkedIn discovery window must be greater than zero seconds")
        return f"r{window_seconds}"
    if interval_hours <= 0:
        raise ValueError("LinkedIn schedule interval must be greater than zero hours")
    return f"r{int(interval_hours * 3600)}"


def _criteria_value(criteria: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = criteria.get(name)
        if value not in (None, "", []):
            return value
    return None


def _baserow_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value") or value.get("name") or value.get("id")
    return value


def _selected_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        selected = _baserow_scalar(value)
        return [str(selected).strip()] if selected not in (None, "") else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            values.extend(_selected_values(item))
        return values
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _linkedin_filter_codes(value: Any, mapping: Mapping[str, str]) -> str | None:
    codes: list[str] = []
    for selected in _selected_values(value):
        normalized = selected.casefold()
        code = mapping.get(normalized)
        if code is None and selected in mapping.values():
            code = selected
        if code is not None and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else None


def _linkedin_sort_by(value: Any) -> str:
    selected = _baserow_scalar(value)
    if selected in (None, ""):
        return "R"
    return _SORT_BY_CODES.get(str(selected).strip().casefold(), "R")


def build_search_url(
    base_url: str,
    criteria: Mapping[str, Any],
    interval_hours: int | float,
    *,
    window_seconds: int | None = None,
) -> str:
    """Rebuild a LinkedIn URL from the current criteria row for every discovery run."""
    params = {
        "keywords": _criteria_value(criteria, "Keywords", "keywords"),
        "location": _criteria_value(criteria, "Location", "location"),
        "f_E": _linkedin_filter_codes(
            _criteria_value(criteria, "Experience Level", "experienceLevel"),
            _EXPERIENCE_LEVEL_CODES,
        ),
        "f_JT": _linkedin_filter_codes(
            _criteria_value(criteria, "Job Type", "jobType"),
            _JOB_TYPE_CODES,
        ),
        "f_WT": _linkedin_filter_codes(
            _criteria_value(criteria, "Workplace Type", "workplaceType"),
            _WORKPLACE_TYPE_CODES,
        ),
        "f_TPR": _linkedin_date_filter(interval_hours, window_seconds=window_seconds),
        "sortBy": _linkedin_sort_by(_criteria_value(criteria, "Sort By", "sortBy")),
        "distance": _criteria_value(criteria, "Distance", "distance"),
        "f_C": _criteria_value(criteria, "Company ID", "companyId"),
        "f_AL": "true" if _criteria_value(criteria, "Easy Apply", "easyApply") is True else None,
    }
    encoded = urlencode({key: value for key, value in params.items() if value not in (None, "")})
    return base_url.rstrip("?") + (f"?{encoded}" if encoded else "")


def excluded(job: Job, company_terms: Sequence[str], title_terms: Sequence[str]) -> bool:
    company = job.company_name.casefold()
    title = job.title.casefold()
    return any(term.casefold() in company for term in company_terms) or any(
        term.casefold() in title for term in title_terms
    )


def normalize_discovery(
    rows: Iterable[Mapping[str, Any]], company_terms: Sequence[str], title_terms: Sequence[str]
) -> list[Job]:
    # Exclusions and same-run duplicate removal are intentionally handled by the
    # discovery provider before results are fetched. This layer translates valid
    # provider rows and marks them so repository-level cross-location grouping only
    # applies to scheduled discovery, not Fillout/manual submissions.
    del company_terms, title_terms
    jobs: list[Job] = []
    for row in rows:
        try:
            job = job_from_provider(row)
        except ValueError:
            continue
        provider_data = dict(job.provider_data)
        provider_data["_scheduled_discovery"] = True
        jobs.append(job.model_copy(update={"provider_data": provider_data}))
    return jobs
