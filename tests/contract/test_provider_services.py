from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from job_hunt.domain.models import ArtifactBundle, Job, PromptDefinition
from job_hunt.errors import ProviderError
from job_hunt.integrations.artifacts import APPLICATION_ZIP_FIELD, BaserowArtifactPublisher
from job_hunt.integrations.gemini import GeminiStructuredClient, GeminiWorkflowAI


def bundle(tmp_path: Path) -> ArtifactBundle:
    paths = [tmp_path / name for name in ("cv.json", "cv.tex", "cv.pdf", "cl.json", "cl.tex", "cl.pdf", "all.zip")]
    for path in paths:
        path.write_text("x")
    return ArtifactBundle(
        run_directory=tmp_path,
        cv_json=paths[0],
        cv_tex=paths[1],
        cv_pdf=paths[2],
        cover_letter_json=paths[3],
        cover_letter_tex=paths[4],
        cover_letter_pdf=paths[5],
        archive=paths[6],
    )


def definition(key: str, schema: dict[str, Any]) -> PromptDefinition:
    return PromptDefinition(
        key=key,
        version=1,
        template=key,
        output_structure=schema,
        temperature=0.3,
    )


def test_baserow_publisher_uploads_final_pdfs_and_zip(tmp_path: Path) -> None:
    uploaded: list[str] = []

    class Base:
        def upload_file(self, path: Path) -> dict[str, str]:
            uploaded.append(path.name)
            return {"name": path.name}

    result = BaserowArtifactPublisher(Base()).publish(bundle(tmp_path))  # type: ignore[arg-type]
    assert uploaded == ["cv.pdf", "cl.pdf", "all.zip"]
    assert result["CV"] == [{"name": "cv.pdf"}]
    assert result["Cover Letter"] == [{"name": "cl.pdf"}]
    assert result[APPLICATION_ZIP_FIELD] == [{"name": "all.zip"}]


def test_gemini_schema_failure_calls_auto_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "string"},
            }
        },
        "required": ["paragraphs"],
        "additionalProperties": False,
    }
    prompt = definition("cover_letter_generation", schema)
    client = GeminiStructuredClient("primary", "backup", "models/test")
    calls: list[str] = []

    def structured_call(key: str, _: str, __: dict[str, Any], ___: float) -> tuple[dict[str, Any], str, None]:
        calls.append(key)
        if len(calls) == 1:
            return {"paragraphs": ["one"]}, '{"paragraphs":["one"]}', None
        return {"paragraphs": ["one", "two", "three"]}, "", None

    monkeypatch.setattr(client, "_structured_call", structured_call)
    result = client.generate("prompt", prompt)
    assert result["paragraphs"] == ["one", "two", "three"]
    assert calls == ["primary", "backup"]


def test_gemini_all_keys_fail_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    prompt = definition("qualification_scoring", schema)
    client = GeminiStructuredClient("bad", "", "test")

    def fail(*_: object, **__: object) -> object:
        raise RuntimeError("bad")

    monkeypatch.setattr(client, "_structured_call", fail)
    with pytest.raises(ProviderError, match="all configured keys"):
        client.generate("x", prompt)


def test_gemini_workflow_ai_runs_separate_prompt_operations() -> None:
    schemas = {
        "qualification_scoring": {
            "type": "object",
            "properties": {
                "qualification_score": {"type": "integer"},
                "should_apply": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["qualification_score", "should_apply", "reasoning"],
        },
        "cv_project_selection": {
            "type": "object",
            "properties": {"selected_indices": {"type": "array", "items": {"type": "integer"}}},
            "required": ["selected_indices"],
        },
        "cv_project_rewrite": {
            "type": "object",
            "properties": {
                "contents": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                }
            },
            "required": ["contents"],
        },
        "cv_work_experience_selection": {
            "type": "object",
            "properties": {"selected_indices": {"type": "array", "items": {"type": "integer"}}},
            "required": ["selected_indices"],
        },
        "cv_work_experience_rewrite": {
            "type": "object",
            "properties": {
                "contents": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                }
            },
            "required": ["contents"],
        },
        "cv_skills_tailoring": {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "skills": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["label", "skills"],
                    },
                }
            },
            "required": ["groups"],
        },
        "cv_summary_rewrite": {
            "type": "object",
            "properties": {"summary": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary"],
        },
        "cover_letter_generation": {
            "type": "object",
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                }
            },
            "required": ["paragraphs"],
        },
        "job_page_content_extraction": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "job_title": {"type": "string"},
                "job_description": {"type": "string"},
            },
            "required": ["company_name", "job_title", "job_description"],
        },
    }
    prompts = {key: definition(key, schema) for key, schema in schemas.items()}
    calls: list[str] = []

    class Structured:
        def generate(self, _: str, prompt: PromptDefinition) -> dict[str, Any]:
            calls.append(prompt.key)
            outputs: dict[str, dict[str, Any]] = {
                "qualification_scoring": {
                    "qualification_score": 50,
                    "should_apply": False,
                    "reasoning": "score only gate",
                },
                "cv_project_selection": {"selected_indices": [0]},
                "cv_project_rewrite": {"contents": [["Tailored project"]]},
                "cv_work_experience_selection": {"selected_indices": [0]},
                "cv_work_experience_rewrite": {"contents": [["Tailored work"]]},
                "cv_skills_tailoring": {"groups": [{"label": "Programming", "skills": ["Python"]}]},
                "cv_summary_rewrite": {"summary": ["Tailored summary"]},
                "cover_letter_generation": {"paragraphs": ["One", "Two", "Three"]},
                "job_page_content_extraction": {
                    "company_name": "C",
                    "job_title": "T",
                    "job_description": "D",
                },
            }
            return outputs[prompt.key]

    master_cv = {
        "summary": ["Old"],
        "skills": [{"label": "Programming", "value": "Python, R"}],
        "projects": [{"title": "P", "content": ["Old project"]}],
        "work_experience": [{"title": "W", "organization": "O", "content": ["Old work"]}],
    }
    ai = GeminiWorkflowAI(Structured(), prompts, project_selection_count=1)  # type: ignore[arg-type]
    job = Job(source="x", url="https://x/1", company_name="C", title="T", description="D")
    assert ai.qualify(job, master_cv, prompts).score == 50
    tailored = ai.tailor(job, master_cv, prompts)
    assert tailored.cv["summary"] == ["Tailored summary"]
    assert tailored.cover_letter["paragraphs"] == ["One", "Two", "Three"]
    assert ai.extract_job("posting", "https://x/1").identity == "https://x/1"
    assert calls == [
        "qualification_scoring",
        "cv_project_selection",
        "cv_project_rewrite",
        "cv_work_experience_selection",
        "cv_work_experience_rewrite",
        "cv_skills_tailoring",
        "cv_summary_rewrite",
        "cover_letter_generation",
        "job_page_content_extraction",
    ]
