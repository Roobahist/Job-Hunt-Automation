from __future__ import annotations

from typing import Any

from job_hunt.application.triage import triage_new_jobs
from job_hunt.domain.models import Job, Qualification


class Compatibility:
    def __init__(self, results: dict[int, bool]) -> None:
        self.results = results
        self.calls: list[int] = []

    def compatible(self, job: Job, _prompts: dict[str, Any]) -> bool:
        self.calls.append(job.internal_id)
        return self.results[job.internal_id]


class Qualifier:
    def __init__(self, scores: dict[int, int]) -> None:
        self.scores = scores
        self.calls: list[int] = []

    def qualify(self, job: Job, _master_cv: dict[str, Any], _prompts: dict[str, Any]) -> Qualification:
        self.calls.append(job.internal_id)
        score = self.scores[job.internal_id]
        return Qualification(score=score, should_apply=score >= 50, reasoning="test")


class Repository:
    def __init__(self) -> None:
        self.status_changes: list[tuple[int, str]] = []
        self.saved_qualifications: list[tuple[int, Qualification]] = []

    def set_status(self, row_id: int, status_key: str) -> None:
        self.status_changes.append((row_id, status_key))

    def save_qualification(self, row_id: int, result: Qualification) -> None:
        self.saved_qualifications.append((row_id, result))


def row(row_id: int, *, status_id: int = 1, score: object = None) -> dict[str, Any]:
    return {
        "id": row_id,
        "Status": {"id": status_id, "value": "New" if status_id == 1 else "Dropped"},
        "Job ID": 1000 + row_id,
        "Company Name": f"Company {row_id}",
        "Title": f"Role {row_id}",
        "Job Description": f"Description {row_id}",
        "Link": f"https://example.com/jobs/{row_id}",
        "Location": "Calgary",
        "Contract Type": {"id": 10, "value": "Full-time"},
        "Score": score,
    }


def test_triage_filters_new_rows_and_reuses_existing_scores() -> None:
    repository = Repository()
    compatibility = Compatibility({1: False, 2: True, 3: True})
    qualifier = Qualifier({3: 60})

    summary = triage_new_jobs(
        rows=[row(99, status_id=2), row(1), row(2, score=85), row(3)],
        new_status_id=1,
        threshold=70,
        compatibility_filter=compatibility,
        qualifier=qualifier,
        repository=repository,  # type: ignore[arg-type]
        prompts={},
        master_cv={},
    )

    assert compatibility.calls == [1, 2, 3]
    assert qualifier.calls == [3]
    assert repository.status_changes == [(1, "dropped"), (3, "dropped")]
    assert len(repository.saved_qualifications) == 1
    assert repository.saved_qualifications[0][0] == 3
    assert summary.processed == 3
    assert summary.kept == 1
    assert summary.dropped == 2
    assert summary.compatibility_dropped == 1
    assert summary.qualification_dropped == 1
    assert summary.scored == 1
    assert summary.reused_scores == 1
    assert summary.errors == 0
    assert summary.next_after_id == 3
    assert summary.decisions[1].action == "keep_new"
    assert summary.decisions[1].score == 85


def test_triage_dry_run_scores_but_never_writes() -> None:
    repository = Repository()
    compatibility = Compatibility({1: True})
    qualifier = Qualifier({1: 20})

    summary = triage_new_jobs(
        rows=[row(1)],
        new_status_id=1,
        threshold=70,
        compatibility_filter=compatibility,
        qualifier=qualifier,
        repository=repository,  # type: ignore[arg-type]
        prompts={},
        master_cv={},
        dry_run=True,
    )

    assert qualifier.calls == [1]
    assert repository.status_changes == []
    assert repository.saved_qualifications == []
    assert summary.dropped == 1
    assert summary.next_after_id == 1
    assert summary.decisions[0].action == "would_drop"
    assert summary.decisions[0].score == 20


def test_triage_stops_at_invalid_row_without_advancing_cursor_past_it() -> None:
    repository = Repository()
    compatibility = Compatibility({1: True})
    qualifier = Qualifier({1: 90})
    invalid = row(2)
    invalid["Job Description"] = ""

    summary = triage_new_jobs(
        rows=[row(3), invalid, row(1)],
        new_status_id=1,
        threshold=70,
        compatibility_filter=compatibility,
        qualifier=qualifier,
        repository=repository,  # type: ignore[arg-type]
        prompts={},
        master_cv={},
    )

    assert compatibility.calls == [1]
    assert qualifier.calls == [1]
    assert summary.processed == 2
    assert summary.errors == 1
    assert summary.kept == 1
    assert summary.next_after_id == 1
    assert summary.decisions[0].row_id == 1
    assert summary.decisions[1].row_id == 2
    assert summary.decisions[1].action == "error"


def test_triage_after_id_and_limit_produce_non_overlapping_sorted_batches() -> None:
    repository = Repository()
    compatibility = Compatibility({2: True, 3: True, 4: True, 5: True})
    qualifier = Qualifier({2: 90, 3: 90, 4: 90, 5: 90})
    rows = [row(5), row(2), row(4), row(1), row(3)]

    first = triage_new_jobs(
        rows=rows,
        new_status_id=1,
        threshold=70,
        compatibility_filter=compatibility,
        qualifier=qualifier,
        repository=repository,  # type: ignore[arg-type]
        prompts={},
        master_cv={},
        limit=2,
        after_id=1,
    )

    assert [decision.row_id for decision in first.decisions] == [2, 3]
    assert first.next_after_id == 3

    second = triage_new_jobs(
        rows=rows,
        new_status_id=1,
        threshold=70,
        compatibility_filter=compatibility,
        qualifier=qualifier,
        repository=repository,  # type: ignore[arg-type]
        prompts={},
        master_cv={},
        limit=2,
        after_id=first.next_after_id,
    )

    assert [decision.row_id for decision in second.decisions] == [4, 5]
    assert second.next_after_id == 5
