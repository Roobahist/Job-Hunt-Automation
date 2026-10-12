from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from job_hunt.application.workflow import ApplicationWorkflow
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, Qualification, TailoredContent
from job_hunt.errors import ErrorKind, ProviderError, WorkflowError


class Repository:
    def __init__(self, existing: bool = False, dropped: bool = False, existing_score: int | None = None) -> None:
        self.existing = {"id": 7, "Score": existing_score} if existing else None
        self.dropped = dropped
        self.calls: list[tuple[str, object]] = []

    def find(self, job: Job) -> dict[str, object] | None:
        self.calls.append(("find", job.identity))
        return self.existing

    def create(self, job: Job) -> dict[str, int]:
        self.calls.append(("create", job.internal_id))
        return {"id": 8}

    def reset(self, row_id: int, job: Job) -> dict[str, int]:
        self.calls.append(("reset", row_id))
        return {"id": row_id}

    def clear_qualification(self, row_id: int) -> None:
        self.calls.append(("clear_qualification", row_id))

    def save_qualification(self, row_id: int, result: Qualification) -> None:
        self.calls.append(("qualification", result.score))

    def save_artifacts(self, row_id: int, uploaded_files: object) -> None:
        self.calls.append(("artifacts", row_id))

    def set_status(self, row_id: int, status_key: str) -> None:
        self.calls.append(("status", status_key))
        self.dropped = status_key == "dropped"

    def has_status(self, row_id: int, status_key: str) -> bool:
        self.calls.append(("has_status", status_key))
        return self.dropped if status_key == "dropped" else False


class AI:
    def __init__(self, qualification: Qualification) -> None:
        self.qualification = qualification
        self.qualify_calls = 0
        self.tailor_calls = 0

    def qualify(self, *_: object) -> Qualification:
        self.qualify_calls += 1
        return self.qualification

    def tailor(self, *_: object) -> TailoredContent:
        self.tailor_calls += 1
        return TailoredContent(cv={}, cover_letter={})


class Renderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, *_: object, **__: object) -> Any:
        self.calls += 1
        return SimpleNamespace(notification_paths=lambda: [Path("/tmp/application.zip")])


class Publisher:
    def __init__(self) -> None:
        self.calls = 0

    def publish(self, *_: object) -> dict[str, object]:
        self.calls += 1
        return {"CV": []}


def make_workflow(repo: Repository, result: Qualification) -> tuple[ApplicationWorkflow, Publisher, AI, Renderer]:
    publisher = Publisher()
    ai = AI(result)
    renderer = Renderer()
    return ApplicationWorkflow(repo, ai, ai, renderer, publisher, Path("runs")), publisher, ai, renderer


def application_job() -> Job:
    return assign_identity(
        Job(
            source="x",
            external_id="1",
            url="https://x/jobs/1",
            company_name="C",
            title="T",
            description="D",
        )
    )


def qualify(workflow: ApplicationWorkflow, *, force: bool = False, resume_row_id: int | None = None) -> object:
    return workflow.persist_and_qualify(
        application_job(),
        run_id=uuid4(),
        master_cv={},
        prompts={},
        threshold=33,
        force=force,
        resume_row_id=resume_row_id,
    )


def generate(workflow: ApplicationWorkflow, *, row_id: int, score: int) -> object:
    return workflow.generate_documents(
        application_job(),
        run_id=uuid4(),
        row_id=row_id,
        score=score,
        master_cv={},
        prompts={},
        applicant_filename="Person",
    )


def test_below_threshold_marks_row_dropped_and_stops_at_qualification_boundary() -> None:
    repo = Repository()
    workflow, publisher, _, _ = make_workflow(repo, Qualification(score=32, should_apply=True, reasoning="low"))
    result = qualify(workflow)
    assert not result.passed  # type: ignore[attr-defined]
    assert result.score == 32  # type: ignore[attr-defined]
    assert publisher.calls == 0
    assert ("qualification", 32) in repo.calls
    assert ("status", "dropped") in repo.calls
    assert all(call[0] != "artifacts" for call in repo.calls)


def test_existing_non_dropped_row_is_not_requalified_automatically() -> None:
    repo = Repository(existing=True, dropped=False, existing_score=75)
    workflow, _, ai, _ = make_workflow(repo, Qualification(score=90, should_apply=True, reasoning="unused"))
    result = qualify(workflow)
    assert not result.passed  # type: ignore[attr-defined]
    assert result.score == 75  # type: ignore[attr-defined]
    assert ai.qualify_calls == 0
    assert ("reset", 7) not in repo.calls
    assert ("clear_qualification", 7) not in repo.calls
    assert ("status", "new") not in repo.calls
    assert ("qualification", 90) not in repo.calls


def test_existing_dropped_row_is_not_requalified_automatically() -> None:
    repo = Repository(existing=True, dropped=True, existing_score=75)
    workflow, _, ai, _ = make_workflow(repo, Qualification(score=99, should_apply=True, reasoning="unused"))
    result = qualify(workflow)
    assert not result.passed  # type: ignore[attr-defined]
    assert result.score == 75  # type: ignore[attr-defined]
    assert ai.qualify_calls == 0
    assert ("reset", 7) not in repo.calls
    assert ("clear_qualification", 7) not in repo.calls
    assert ("status", "new") not in repo.calls


def test_retry_resumes_matching_row_instead_of_treating_it_as_duplicate() -> None:
    repo = Repository(existing=True, dropped=False, existing_score=None)
    workflow, _, ai, _ = make_workflow(repo, Qualification(score=90, should_apply=True, reasoning="good"))

    result = qualify(workflow, resume_row_id=7)

    assert result.passed  # type: ignore[attr-defined]
    assert result.score == 90  # type: ignore[attr-defined]
    assert ai.qualify_calls == 1
    assert ("reset", 7) not in repo.calls
    assert ("clear_qualification", 7) not in repo.calls
    assert ("qualification", 90) in repo.calls


def test_retry_refuses_to_claim_row_owned_by_another_run() -> None:
    repo = Repository(existing=True, dropped=False)
    workflow, _, ai, _ = make_workflow(repo, Qualification(score=90, should_apply=True, reasoning="unused"))

    with pytest.raises(WorkflowError) as caught:
        qualify(workflow, resume_row_id=8)

    assert caught.value.kind == ErrorKind.BUSINESS
    assert ai.qualify_calls == 0
    assert ("reset", 7) not in repo.calls
    assert all(call[0] != "qualification" for call in repo.calls)


def test_forced_retry_resumes_without_resetting_or_clearing_row_again() -> None:
    repo = Repository(existing=True, dropped=False, existing_score=None)
    workflow, _, ai, _ = make_workflow(repo, Qualification(score=1, should_apply=False, reasoning="forced"))

    result = qualify(workflow, force=True, resume_row_id=7)

    assert result.passed  # type: ignore[attr-defined]
    assert ai.qualify_calls == 1
    assert ("reset", 7) not in repo.calls
    assert ("clear_qualification", 7) not in repo.calls
    assert ("status", "new") not in repo.calls
    assert ("qualification", 1) in repo.calls


def test_manual_drop_after_persistence_stops_before_qualification() -> None:
    repo = Repository()
    workflow, _, ai, _ = make_workflow(repo, Qualification(score=90, should_apply=True, reasoning="unused"))

    def persisted(_row_id: int) -> None:
        repo.dropped = True

    result = workflow.persist_and_qualify(
        application_job(),
        run_id=uuid4(),
        master_cv={},
        prompts={},
        threshold=33,
        force=False,
        persisted=persisted,
    )
    assert not result.passed
    assert result.score is None
    assert ai.qualify_calls == 0
    assert all(call[0] != "qualification" for call in repo.calls)


def test_manual_drop_during_qualification_does_not_save_new_score() -> None:
    repo = Repository()

    class DroppingQualifier(AI):
        def qualify(self, *_: object) -> Qualification:
            result = super().qualify()
            repo.dropped = True
            return result

    ai = DroppingQualifier(Qualification(score=90, should_apply=True, reasoning="unused"))
    workflow = ApplicationWorkflow(repo, ai, ai, Renderer(), Publisher(), Path("runs"))
    result = qualify(workflow)
    assert not result.passed  # type: ignore[attr-defined]
    assert result.score == 90  # type: ignore[attr-defined]
    assert ai.qualify_calls == 1
    assert ("qualification", 90) not in repo.calls


def test_document_generation_is_explicit_after_qualification() -> None:
    repo = Repository()
    workflow, publisher, _, _ = make_workflow(
        repo,
        Qualification(score=90, should_apply=False, reasoning="metadata only"),
    )
    qualification = qualify(workflow)
    assert qualification.score is not None  # type: ignore[attr-defined]
    result = generate(workflow, row_id=qualification.row_id, score=qualification.score)  # type: ignore[attr-defined]
    assert result.passed  # type: ignore[attr-defined]
    assert publisher.calls == 1
    assert result.notification_paths == ("/tmp/application.zip",)  # type: ignore[attr-defined]


def test_progress_callback_reports_major_pipeline_boundaries() -> None:
    repo = Repository()
    workflow, _, _, _ = make_workflow(repo, Qualification(score=90, should_apply=True, reasoning="good"))
    events: list[tuple[str, str]] = []
    qualification = workflow.persist_and_qualify(
        application_job(),
        run_id=uuid4(),
        master_cv={},
        prompts={},
        threshold=33,
        force=False,
        progress=lambda stage, event: events.append((stage, event)),
    )
    assert qualification.score is not None
    workflow.generate_documents(
        application_job(),
        run_id=uuid4(),
        row_id=qualification.row_id,
        score=qualification.score,
        master_cv={},
        prompts={},
        applicant_filename="Person",
        progress=lambda stage, event: events.append((stage, event)),
    )
    assert events == [
        ("persistence", "start"),
        ("persistence", "finish"),
        ("qualification", "start"),
        ("qualification", "finish"),
        ("tailoring", "start"),
        ("tailoring", "finish"),
        ("rendering", "start"),
        ("rendering", "finish"),
        ("artifact_upload", "start"),
        ("artifact_upload", "finish"),
    ]


def test_manual_drop_before_documents_skips_tailoring_rendering_and_publish() -> None:
    repo = Repository(dropped=True)
    workflow, publisher, ai, renderer = make_workflow(
        repo,
        Qualification(score=90, should_apply=True, reasoning="good"),
    )
    events: list[tuple[str, str]] = []
    result = workflow.generate_documents(
        application_job(),
        run_id=uuid4(),
        row_id=8,
        score=90,
        master_cv={},
        prompts={},
        applicant_filename="Person",
        progress=lambda stage, event: events.append((stage, event)),
    )
    assert not result.passed
    assert not result.artifacts_published
    assert result.notification_paths == ()
    assert ai.tailor_calls == 0
    assert renderer.calls == 0
    assert publisher.calls == 0
    assert events == []


def test_manual_drop_after_tailoring_stops_before_rendering() -> None:
    repo = Repository()

    class DroppingAI(AI):
        def tailor(self, *_: object) -> TailoredContent:
            result = super().tailor()
            repo.dropped = True
            return result

    publisher = Publisher()
    ai = DroppingAI(Qualification(score=90, should_apply=True, reasoning="good"))
    renderer = Renderer()
    workflow = ApplicationWorkflow(repo, ai, ai, renderer, publisher, Path("runs"))
    result = generate(workflow, row_id=8, score=90)
    assert not result.passed  # type: ignore[attr-defined]
    assert ai.tailor_calls == 1
    assert renderer.calls == 0
    assert publisher.calls == 0


def test_manual_drop_after_rendering_stops_before_upload() -> None:
    repo = Repository()

    class DroppingRenderer(Renderer):
        def render(self, *_: object, **__: object) -> Any:
            result = super().render()
            repo.dropped = True
            return result

    publisher = Publisher()
    ai = AI(Qualification(score=90, should_apply=True, reasoning="good"))
    renderer = DroppingRenderer()
    workflow = ApplicationWorkflow(repo, ai, ai, renderer, publisher, Path("runs"))
    result = generate(workflow, row_id=8, score=90)
    assert not result.passed  # type: ignore[attr-defined]
    assert ai.tailor_calls == 1
    assert renderer.calls == 1
    assert publisher.calls == 0


def test_force_resets_new_and_overrides_below_threshold_drop_rule() -> None:
    repo = Repository(existing=True, dropped=True)
    workflow, publisher, _, _ = make_workflow(repo, Qualification(score=1, should_apply=False, reasoning="no"))
    result = qualify(workflow, force=True)
    assert result.passed  # type: ignore[attr-defined]
    assert ("clear_qualification", 7) in repo.calls
    assert ("status", "new") in repo.calls
    assert ("status", "dropped") not in repo.calls
    assert ("qualification", 1) in repo.calls
    assert publisher.calls == 0


def test_permanent_error_is_not_retried() -> None:
    class BrokenRepository(Repository):
        def find(self, job: Job) -> None:
            raise WorkflowError("bad", ErrorKind.VALIDATION)

    workflow, _, _, _ = make_workflow(
        BrokenRepository(),
        Qualification(score=1, should_apply=False, reasoning="no"),
    )
    with pytest.raises(WorkflowError):
        qualify(workflow)


def test_qualification_rate_limit_escapes_immediately_for_celery_deferral() -> None:
    class RateLimitedAI(AI):
        def __init__(self) -> None:
            super().__init__(Qualification(score=0, should_apply=False, reasoning="unused"))
            self.calls = 0

        def qualify(self, *_: object) -> Qualification:
            self.calls += 1
            raise ProviderError(
                "local requests-per-minute budget exhausted",
                ErrorKind.RATE_LIMIT,
                retryable=True,
                provider="gemini",
                retry_after=47,
            )

    repo = Repository()
    ai = RateLimitedAI()
    workflow = ApplicationWorkflow(repo, ai, ai, Renderer(), Publisher(), Path("runs"))
    with pytest.raises(ProviderError) as caught:
        qualify(workflow)
    assert caught.value.retry_after == 47
    assert ai.calls == 1
    assert all(call[0] != "qualification" for call in repo.calls)
