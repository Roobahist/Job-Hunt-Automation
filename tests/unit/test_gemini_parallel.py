from __future__ import annotations

from threading import Barrier, Lock

import pytest

from job_hunt.domain.models import Job
from job_hunt.integrations.gemini_parallel import (
    ParallelGeminiWorkflowAI,
    ParallelAtlasGeminiWorkflowAI,
)


def test_nova_independent_branches_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    ai = ParallelGeminiWorkflowAI(object(), {}, parallelism=3)  # type: ignore[arg-type]
    barrier = Barrier(3, timeout=2)
    lock = Lock()
    started: list[str] = []

    def branch(name: str, result: object) -> object:
        with lock:
            started.append(name)
        barrier.wait()
        return result

    monkeypatch.setattr(
        ai,
        "_project_pipeline",
        lambda *_: branch("projects", ([0], [{"title": "P", "content": ["p"]}])),
    )
    monkeypatch.setattr(
        ai,
        "_work_pipeline",
        lambda *_: branch("work", ([0], [{"title": "W", "content": ["w"]}])),
    )
    monkeypatch.setattr(
        ai,
        "_skills_pipeline",
        lambda *_: branch("skills", [{"label": "Skills", "value": "Python"}]),
    )
    monkeypatch.setattr(ai, "_summary_pipeline", lambda *_: ["summary"])
    monkeypatch.setattr(ai, "_definition", lambda *_: object())
    monkeypatch.setattr(ai, "_run", lambda *_: {"paragraphs": ["1", "2", "3"]})

    job = Job(source="x", url="https://x", company_name="C", title="T", description="D")
    content = ai.tailor(
        job,
        {"summary": ["old"], "projects": [], "work_experience": [], "skills": []},
        {},
    )
    assert sorted(started) == ["projects", "skills", "work"]
    assert content.cv["summary"] == ["summary"]
    assert content.cover_letter["paragraphs"] == ["1", "2", "3"]


def test_atlas_independent_branches_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    ai = ParallelAtlasGeminiWorkflowAI(object(), {}, parallelism=3)  # type: ignore[arg-type]
    barrier = Barrier(3, timeout=2)
    lock = Lock()
    started: list[str] = []

    def branch(name: str, result: object) -> object:
        with lock:
            started.append(name)
        barrier.wait()
        return result

    monkeypatch.setattr(
        ai,
        "_work_pipeline_sections",
        lambda *_: branch("work", [{"title": "W", "content": ["w"]}]),
    )
    monkeypatch.setattr(
        ai,
        "_skills_pipeline_sections",
        lambda *_: branch("skills", [{"label": "Skills", "value": "Python"}]),
    )
    monkeypatch.setattr(ai, "_references_decision", lambda *_: branch("references", False))
    monkeypatch.setattr(ai, "_summary_pipeline_sections", lambda *_: ["summary"])
    monkeypatch.setattr(ai, "_definition", lambda *_: object())
    monkeypatch.setattr(ai, "_run", lambda *_: {"paragraphs": ["1", "2", "3"]})

    source = {
        "sections": [
            {"type": "text", "content": ["old"]},
            {"type": "entries", "entries": []},
            {"type": "education", "items": []},
            {"type": "label_rows", "rows": []},
            {"type": "references", "items": [{"name": "R"}]},
        ]
    }
    job = Job(source="x", url="https://x", company_name="C", title="T", description="D")
    content = ai.tailor(job, source, {})
    assert sorted(started) == ["references", "skills", "work"]
    sections = content.cv["sections"]
    assert isinstance(sections, list)
    assert next(section for section in sections if section["type"] == "references")["items"] == []
