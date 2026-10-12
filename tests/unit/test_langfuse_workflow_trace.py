from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from job_hunt.application import workflow as workflow_module
from job_hunt.application.workflow import ApplicationWorkflow, QualificationResult, WorkflowResult
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job


class Observation:
    def __init__(self) -> None:
        self.output: Any = None

    def update(self, **kwargs: Any) -> None:
        self.output = kwargs.get("output")


def _job() -> Job:
    return assign_identity(
        Job(
            source="linkedin",
            external_id="123",
            url="https://example.com/jobs/123",
            company_name="Example",
            title="ML Engineer",
            description="Build useful ML systems.",
        )
    )


def test_qualification_and_documents_join_one_trace_with_meaningful_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = UUID("14471d60-d158-432b-88bb-f4cb85a3f10d")
    trace_context = {"trace_id": "1" * 32}
    trace_seeds: list[str] = []
    spans: list[tuple[dict[str, Any], Observation]] = []

    def fake_trace_context(*, enabled: bool, seed: str) -> dict[str, str] | None:
        assert enabled
        trace_seeds.append(seed)
        return trace_context

    @contextmanager
    def fake_span(**kwargs: Any) -> Iterator[Observation]:
        observation = Observation()
        spans.append((kwargs, observation))
        yield observation

    monkeypatch.setattr(workflow_module, "langfuse_trace_context", fake_trace_context)
    monkeypatch.setattr(workflow_module, "langfuse_span", fake_span)
    monkeypatch.setattr(workflow_module, "job_trace_context", lambda *_args, **_kwargs: nullcontext())

    workflow = ApplicationWorkflow(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path,
        tenant_key="nova",
        langfuse_enabled=True,
    )
    monkeypatch.setattr(
        workflow,
        "_persist_and_qualify_impl",
        lambda *_args, **_kwargs: QualificationResult(row_id=7150, passed=True, score=88, attach_trace=False),
    )
    monkeypatch.setattr(
        workflow,
        "_generate_documents_impl",
        lambda *_args, **_kwargs: WorkflowResult(
            row_id=7150,
            passed=True,
            artifacts_published=True,
            score=88,
            notification_paths=("/tmp/application.zip",),
        ),
    )

    job = _job()
    master_cv = {"name": "Nova", "skills": ["Python", "ML"]}

    qualification = workflow.persist_and_qualify(
        job,
        run_id=run_id,
        master_cv=master_cv,
        prompts={},
        threshold=70,
        force=True,
    )
    documents = workflow.generate_documents(
        job,
        run_id=run_id,
        row_id=qualification.row_id,
        score=qualification.score or 0,
        master_cv=master_cv,
        prompts={},
        applicant_filename="Novacandidate",
    )

    expected_seed = f"job-hunt:nova:{run_id}"
    assert trace_seeds == [expected_seed, expected_seed]
    assert [span[0]["name"] for span in spans] == ["qualification-pipeline", "document-pipeline"]
    assert all(span[0]["trace_name"] == "job-application" for span in spans)
    assert all(span[0]["trace_context"] == trace_context for span in spans)
    assert all(span[0]["session_id"] == "nova:123" for span in spans)

    qualification_input = spans[0][0]["input"]
    assert qualification_input["job"]["description"] == "Build useful ML systems."
    assert qualification_input["master_cv"] == master_cv
    assert qualification_input["qualification_threshold"] == 70
    assert qualification_input["force"] is True
    assert spans[0][1].output == {
        "row_id": 7150,
        "passed": True,
        "score": 88,
        "attach_trace": False,
    }

    document_input = spans[1][0]["input"]
    assert document_input["job"]["title"] == "ML Engineer"
    assert document_input["master_cv"] == master_cv
    assert document_input["qualification_score"] == 88
    assert spans[1][1].output == {
        "row_id": 7150,
        "passed": True,
        "artifacts_published": True,
        "score": 88,
        "notification_paths": ["/tmp/application.zip"],
    }
    assert documents.artifacts_published


def test_langfuse_session_uses_generated_job_id_when_external_id_is_missing(tmp_path: Path) -> None:
    workflow = ApplicationWorkflow(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        tmp_path,
        tenant_key="atlas",
        langfuse_enabled=True,
    )
    job = assign_identity(
        Job(
            source="web",
            url="https://example.com/careers/designer",
            company_name="Example",
            title="Designer",
            description="Design useful products.",
        )
    )

    assert workflow._langfuse_session_id(job) == f"atlas:{job.internal_id}"
