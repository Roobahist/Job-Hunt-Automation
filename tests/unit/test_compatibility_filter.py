from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from job_hunt.application.compatibility import GeminiCompatibilityFilter
from job_hunt.application.workflow import ApplicationWorkflow
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, PromptDefinition, Qualification, TailoredContent


class StructuredClient:
    def __init__(self, compatible: bool) -> None:
        self.compatible = compatible
        self.prompts: list[str] = []

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.prompts.append(prompt)
        assert definition.key == "job_compatibility_filter"
        return {"compatible": self.compatible}


class Repository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def find(self, job: Job) -> None:
        return None

    def create(self, job: Job) -> dict[str, int]:
        return {"id": 8}

    def reset(self, row_id: int, job: Job) -> dict[str, int]:
        return {"id": row_id}

    def save_qualification(self, row_id: int, result: Qualification) -> None:
        self.calls.append(("qualification", result.score))

    def save_artifacts(self, row_id: int, uploaded_files: object) -> None:
        self.calls.append(("artifacts", row_id))

    def set_status(self, row_id: int, status_key: str) -> None:
        self.calls.append(("status", status_key))

    def has_status(self, row_id: int, status_key: str) -> bool:
        return False


class AI:
    def __init__(self) -> None:
        self.qualify_calls = 0

    def qualify(self, *_: object) -> Qualification:
        self.qualify_calls += 1
        return Qualification(score=90, should_apply=True, reasoning="good")

    def tailor(self, *_: object) -> TailoredContent:
        return TailoredContent(cv={}, cover_letter={})


class Renderer:
    def render(self, *_: object, **__: object) -> object:
        raise AssertionError("rendering must not run during compatibility filtering")


class Publisher:
    def publish(self, *_: object) -> dict[str, object]:
        raise AssertionError("publishing must not run during compatibility filtering")


def prompt() -> PromptDefinition:
    return PromptDefinition(
        key="job_compatibility_filter",
        version=1,
        template="Evaluate [[scraped_job_json]] and [[job_json]]",
        output_structure={
            "type": "object",
            "properties": {"compatible": {"type": "boolean"}},
            "required": ["compatible"],
            "additionalProperties": False,
        },
        temperature=0,
    )


def job() -> Job:
    return assign_identity(
        Job(
            source="linkedin",
            external_id="123",
            url="https://example.com/jobs/123",
            company_name="Example",
            title="Data Scientist",
            description="Remote role for candidates authorized to work in Canada.",
            location="Toronto, Ontario, Canada",
            provider_data={"workplaceType": "REMOTE", "description": "full scraped text"},
        )
    )


def test_filter_receives_raw_scraped_provider_payload() -> None:
    client = StructuredClient(True)
    compatibility = GeminiCompatibilityFilter(client)  # type: ignore[arg-type]
    assert compatibility.compatible(job(), {"job_compatibility_filter": prompt()})
    assert "workplaceType" in client.prompts[0]
    assert "full scraped text" in client.prompts[0]


def test_false_compatibility_drops_without_qualification() -> None:
    repo = Repository()
    ai = AI()
    compatibility = GeminiCompatibilityFilter(StructuredClient(False))  # type: ignore[arg-type]
    workflow = ApplicationWorkflow(
        repo,
        ai,
        ai,
        Renderer(),
        Publisher(),
        Path("runs"),
        compatibility_filter=compatibility,
    )
    result = workflow.persist_and_qualify(
        job(),
        run_id=uuid4(),
        master_cv={},
        prompts={"job_compatibility_filter": prompt()},
        threshold=33,
        force=False,
    )
    assert not result.passed
    assert result.score is None
    assert ai.qualify_calls == 0
    assert ("status", "dropped") in repo.calls
    assert all(call[0] != "qualification" for call in repo.calls)


def test_true_compatibility_continues_to_qualification() -> None:
    repo = Repository()
    ai = AI()
    compatibility = GeminiCompatibilityFilter(StructuredClient(True))  # type: ignore[arg-type]
    workflow = ApplicationWorkflow(
        repo,
        ai,
        ai,
        Renderer(),
        Publisher(),
        Path("runs"),
        compatibility_filter=compatibility,
    )
    result = workflow.persist_and_qualify(
        job(),
        run_id=uuid4(),
        master_cv={},
        prompts={"job_compatibility_filter": prompt()},
        threshold=33,
        force=False,
    )
    assert result.passed
    assert result.score == 90
    assert ai.qualify_calls == 1
    assert ("qualification", 90) in repo.calls


def test_forced_submission_bypasses_compatibility_and_resets_status_new() -> None:
    repo = Repository()
    ai = AI()
    blocked_client = StructuredClient(False)
    compatibility = GeminiCompatibilityFilter(blocked_client)  # type: ignore[arg-type]
    workflow = ApplicationWorkflow(
        repo,
        ai,
        ai,
        Renderer(),
        Publisher(),
        Path("runs"),
        compatibility_filter=compatibility,
    )
    result = workflow.persist_and_qualify(
        job(),
        run_id=uuid4(),
        master_cv={},
        prompts={"job_compatibility_filter": prompt()},
        threshold=33,
        force=True,
    )
    assert result.passed
    assert ai.qualify_calls == 1
    assert blocked_client.prompts == []
    assert ("status", "new") in repo.calls
    assert ("status", "dropped") not in repo.calls
