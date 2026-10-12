from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from job_hunt.application.workflow import ApplicationWorkflow
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, Qualification, TailoredContent


class Repository:
    def __init__(self, score: object) -> None:
        self.row: dict[str, object] = {"id": 7, "Score": score}
        self.calls: list[tuple[str, object]] = []

    def find(self, job: Job) -> dict[str, object]:
        self.calls.append(("find", job.identity))
        return self.row

    def create(self, job: Job) -> dict[str, object]:
        self.calls.append(("create", job.identity))
        return {"id": 8}

    def reset(self, row_id: int, job: Job) -> dict[str, object]:
        self.calls.append(("reset", row_id))
        return {"id": row_id}

    def clear_qualification(self, row_id: int) -> None:
        self.calls.append(("clear_qualification", row_id))

    def save_qualification(self, row_id: int, result: Qualification) -> None:
        self.calls.append(("qualification", result.score))
        self.row["Score"] = result.score

    def save_artifacts(self, row_id: int, uploaded_files: object) -> None:
        self.calls.append(("artifacts", row_id))

    def set_status(self, row_id: int, status: str) -> None:
        self.calls.append(("status", status))

    def has_status(self, row_id: int, status: str) -> bool:
        self.calls.append(("has_status", status))
        return False


class AI:
    def __init__(self, score: int = 91) -> None:
        self.score = score
        self.qualify_calls = 0

    def qualify(self, *_: object) -> Qualification:
        self.qualify_calls += 1
        return Qualification(score=self.score, reasoning="new score")

    def tailor(self, *_: object) -> TailoredContent:
        return TailoredContent(cv={}, cover_letter={})


class Renderer:
    def render(self, *_: object, **__: object) -> Any:
        raise AssertionError("render should not run in qualification tests")


class Publisher:
    def publish(self, *_: object) -> dict[str, object]:
        raise AssertionError("publish should not run in qualification tests")


def job(*, reuse_data: bool, reuse_score: bool) -> Job:
    base = assign_identity(
        Job(
            source="linkedin",
            external_id="4452378707",
            url="https://www.linkedin.com/jobs/view/4452378707",
            company_name="C",
            title="T",
            description="D",
        )
    )
    return base.model_copy(
        update={
            "provider_data": {
                "reuse_existing_data": reuse_data,
                "reuse_existing_score": reuse_score,
            }
        }
    )


def workflow(repository: Repository, ai: AI) -> ApplicationWorkflow:
    return ApplicationWorkflow(repository, ai, ai, Renderer(), Publisher(), Path("runs"))


def qualify(repository: Repository, ai: AI, candidate: Job) -> object:
    return workflow(repository, ai).persist_and_qualify(
        candidate,
        run_id=uuid4(),
        master_cv={},
        prompts={},
        threshold=60,
        force=True,
    )


def test_fillout_existing_complete_row_reuses_data_and_score() -> None:
    repository = Repository(score=82)
    ai = AI()

    result = qualify(repository, ai, job(reuse_data=True, reuse_score=True))

    assert result.passed  # type: ignore[attr-defined]
    assert result.score == 82  # type: ignore[attr-defined]
    assert ai.qualify_calls == 0
    assert ("reset", 7) not in repository.calls
    assert ("clear_qualification", 7) not in repository.calls
    assert ("status", "new") in repository.calls


def test_fillout_existing_complete_row_scores_when_stored_score_is_empty() -> None:
    repository = Repository(score=None)
    ai = AI(score=91)

    result = qualify(repository, ai, job(reuse_data=True, reuse_score=True))

    assert result.passed  # type: ignore[attr-defined]
    assert result.score == 91  # type: ignore[attr-defined]
    assert ai.qualify_calls == 1
    assert ("reset", 7) not in repository.calls
    assert ("clear_qualification", 7) not in repository.calls
    assert ("qualification", 91) in repository.calls


def test_fillout_incomplete_row_refreshes_data_but_reuses_existing_score() -> None:
    repository = Repository(score="74")
    ai = AI()

    result = qualify(repository, ai, job(reuse_data=False, reuse_score=True))

    assert result.passed  # type: ignore[attr-defined]
    assert result.score == 74  # type: ignore[attr-defined]
    assert ai.qualify_calls == 0
    assert ("reset", 7) in repository.calls
    assert ("clear_qualification", 7) not in repository.calls
