from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any

from job_hunt.domain.models import Job

_TITLE_SIMILARITY = 0.98
_DESCRIPTION_SIMILARITY = 0.99
_ARTIFACT_FIELDS = ("CV", "Cover Letter", "Application ZIP")
_WORDS = re.compile(r"[a-z0-9]+")


def _display(value: object) -> str:
    if isinstance(value, Mapping):
        raw = value.get("value") or value.get("name")
        return str(raw or "").strip()
    return str(value or "").strip()


def _normalized(value: object) -> str:
    return " ".join(_WORDS.findall(_display(value).casefold()))


def _location_phrases(location: object) -> tuple[str, ...]:
    raw = _display(location)
    if not raw:
        return ()
    phrases = {_normalized(raw)}
    for part in re.split(r"[,|;/]", raw):
        phrase = _normalized(part)
        # Two-letter province/state abbreviations are too generic to remove safely.
        if len(phrase) >= 4:
            phrases.add(phrase)
    return tuple(sorted((phrase for phrase in phrases if phrase), key=len, reverse=True))


def normalized_without_location(value: object, location: object) -> str:
    text = _normalized(value)
    for phrase in _location_phrases(location):
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
    return " ".join(text.split())


def _similar(left: str, right: str, threshold: float) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    longest = max(len(left), len(right))
    if longest == 0 or min(len(left), len(right)) / longest < threshold - 0.01:
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold


def jobs_equivalent(left: Job, right: Job) -> bool:
    if _normalized(left.company_name) != _normalized(right.company_name):
        return False

    left_title = normalized_without_location(left.title, left.location)
    right_title = normalized_without_location(right.title, right.location)
    if not _similar(left_title, right_title, _TITLE_SIMILARITY):
        return False

    if (
        left.contract_type
        and right.contract_type
        and _normalized(left.contract_type) != _normalized(right.contract_type)
    ):
        return False

    left_description = normalized_without_location(left.description, left.location)
    right_description = normalized_without_location(right.description, right.location)
    return _similar(left_description, right_description, _DESCRIPTION_SIMILARITY)


def job_from_baserow_row(row: Mapping[str, Any]) -> Job | None:
    company = _display(row.get("Company Name"))
    title = _display(row.get("Title"))
    description = _display(row.get("Job Description"))
    if not company or not title or not description:
        return None
    row_id = row.get("id")
    internal_id = row_id if isinstance(row_id, int) and not isinstance(row_id, bool) else 0
    return Job(
        source="baserow",
        external_id=_display(row.get("Job ID")) or None,
        url=_display(row.get("Link")),
        company_name=company,
        title=title,
        description=description,
        location=_display(row.get("Location")) or None,
        contract_type=_display(row.get("Contract Type")) or None,
        internal_id=internal_id,
    )


def equivalent_rows(job: Job, rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    for row in rows:
        other = job_from_baserow_row(row)
        if other is not None and jobs_equivalent(job, other):
            matches.append(row)
    return matches


def select_option_id(value: object) -> int | None:
    if isinstance(value, Mapping):
        candidate = value.get("id")
        return candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def stored_score(row: Mapping[str, Any]) -> int | None:
    value = row.get("Score")
    if value is None or isinstance(value, bool):
        return None
    try:
        score = int(float(str(value).strip()))
    except ValueError:
        return None
    return score if 0 <= score <= 100 else None


def has_artifacts(row: Mapping[str, Any]) -> bool:
    return all(bool(row.get(field)) for field in _ARTIFACT_FIELDS)


def processing_result_rank(row: Mapping[str, Any], *, new_status_id: int) -> int:
    status_id = select_option_id(row.get("Status"))
    score = stored_score(row)
    if has_artifacts(row):
        return 4
    if score is not None:
        return 3
    if status_id is not None and status_id != new_status_id:
        return 2
    return 0


def best_processed_equivalent(
    job: Job,
    rows: Iterable[Mapping[str, Any]],
    *,
    new_status_id: int,
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_rank = 0
    for row in equivalent_rows(job, rows):
        rank = processing_result_rank(row, new_status_id=new_status_id)
        if rank > best_rank:
            best = row
            best_rank = rank
    return best
