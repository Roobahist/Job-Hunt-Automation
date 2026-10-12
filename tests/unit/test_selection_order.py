from __future__ import annotations

import json
from typing import Any

import pytest

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.integrations.gemini import GeminiWorkflowAI
from job_hunt.integrations.gemini_atlas import AtlasGeminiWorkflowAI


def definition(key: str, template: str) -> PromptDefinition:
    if key.endswith("_selection"):
        schema = {
            "type": "object",
            "properties": {"selected_indices": {"type": "array", "items": {"type": "integer"}}},
            "required": ["selected_indices"],
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "contents": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                }
            },
            "required": ["contents"],
            "additionalProperties": False,
        }
    return PromptDefinition(key=key, version=1, template=template, output_structure=schema, temperature=0)


class Structured:
    def __init__(self, selections: dict[str, list[int]]) -> None:
        self.selections = selections
        self.rewrite_titles: dict[str, list[str]] = {}

    def generate(self, prompt: str, prompt_definition: PromptDefinition) -> dict[str, Any]:
        key = prompt_definition.key
        if key in self.selections:
            return {"selected_indices": list(self.selections[key])}
        if key in {"cv_project_rewrite", "cv_work_experience_rewrite"}:
            items = json.loads(prompt)
            titles = [str(item["title"]) for item in items]
            self.rewrite_titles[key] = titles
            return {"contents": [[f"rewritten:{title}"] for title in titles]}
        raise AssertionError(f"Unexpected prompt key: {key}")


def job() -> Job:
    return Job(source="test", url="https://example.com/job", company_name="C", title="T", description="D")


def test_selection_runtime_schema_requires_exact_count_and_uniqueness() -> None:
    for key in ("cv_project_selection", "cv_work_experience_selection"):
        runtime = GeminiWorkflowAI._runtime_definition(definition(key, "selection"), {"selection_count": 3})
        selected_indices = runtime.output_structure["properties"]["selected_indices"]
        assert selected_indices["minItems"] == 3
        assert selected_indices["maxItems"] == 3
        assert selected_indices["uniqueItems"] is True


def test_nova_projects_and_work_normalize_order_before_rewrite() -> None:
    client = Structured(
        {
            "cv_project_selection": [2, 0],
            "cv_work_experience_selection": [2, 0],
        }
    )
    ai = GeminiWorkflowAI(
        client,  # type: ignore[arg-type]
        project_selection_count=2,
        work_experience_selection_count=2,
    )
    prompts = {
        "cv_project_selection": definition("cv_project_selection", "selection"),
        "cv_project_rewrite": definition("cv_project_rewrite", "[[rewrite_inputs_json]]"),
        "cv_work_experience_selection": definition("cv_work_experience_selection", "selection"),
        "cv_work_experience_rewrite": definition("cv_work_experience_rewrite", "[[rewrite_inputs_json]]"),
    }
    master_cv = {
        "projects": [
            {"title": "P0", "content": ["old P0"]},
            {"title": "P1", "content": ["old P1"]},
            {"title": "P2", "content": ["old P2"]},
        ],
        "work_experience": [
            {"title": "W0", "organization": "O0", "content": ["old W0"]},
            {"title": "W1", "organization": "O1", "content": ["old W1"]},
            {"title": "W2", "organization": "O2", "content": ["old W2"]},
        ],
    }

    project_indices, projects = ai._project_pipeline(job(), master_cv, prompts)
    work_indices, experiences = ai._work_pipeline(job(), master_cv, prompts)

    assert project_indices == [0, 2]
    assert [item["title"] for item in projects] == ["P0", "P2"]
    assert [item["content"] for item in projects] == [["rewritten:P0"], ["rewritten:P2"]]
    assert client.rewrite_titles["cv_project_rewrite"] == ["P0", "P2"]

    assert work_indices == [0, 2]
    assert [item["title"] for item in experiences] == ["W0", "W2"]
    assert [item["content"] for item in experiences] == [["rewritten:W0"], ["rewritten:W2"]]
    assert client.rewrite_titles["cv_work_experience_rewrite"] == ["W0", "W2"]


def test_nova_work_rejects_empty_selection() -> None:
    client = Structured({"cv_work_experience_selection": []})
    ai = GeminiWorkflowAI(client, work_experience_selection_count=2)  # type: ignore[arg-type]
    prompts = {
        "cv_work_experience_selection": definition("cv_work_experience_selection", "selection"),
        "cv_work_experience_rewrite": definition("cv_work_experience_rewrite", "[[rewrite_inputs_json]]"),
    }
    master_cv = {
        "work_experience": [
            {"title": "W0", "organization": "O0", "content": ["old W0"]},
            {"title": "W1", "organization": "O1", "content": ["old W1"]},
        ]
    }

    with pytest.raises(ValueError, match="work-experience selection must contain exactly 2 indices"):
        ai._work_pipeline(job(), master_cv, prompts)


def test_nova_projects_reject_too_few_selections() -> None:
    client = Structured({"cv_project_selection": [1]})
    ai = GeminiWorkflowAI(client, project_selection_count=2)  # type: ignore[arg-type]
    prompts = {
        "cv_project_selection": definition("cv_project_selection", "selection"),
        "cv_project_rewrite": definition("cv_project_rewrite", "[[rewrite_inputs_json]]"),
    }
    master_cv = {
        "projects": [
            {"title": "P0", "content": ["old P0"]},
            {"title": "P1", "content": ["old P1"]},
        ]
    }

    with pytest.raises(ValueError, match="project selection must contain exactly 2 indices"):
        ai._project_pipeline(job(), master_cv, prompts)


def test_atlas_work_normalizes_order_and_preserves_parent_expansion_pairing() -> None:
    client = Structured({"cv_work_experience_selection": [2, 0]})
    ai = AtlasGeminiWorkflowAI(
        client,  # type: ignore[arg-type]
        work_experience_selection_count=2,
    )
    prompts = {
        "cv_work_experience_selection": definition("cv_work_experience_selection", "selection"),
        "cv_work_experience_rewrite": definition("cv_work_experience_rewrite", "[[rewrite_inputs_json]]"),
    }
    source = {
        "sections": [
            {
                "type": "entries",
                "entries": [
                    {"title": "W0", "date": "2026", "content": ["old W0"]},
                    {"title": "Parent", "date": "2025", "content": ["old parent"]},
                    {
                        "title": "Nested",
                        "date": "2025",
                        "parent": "Parent",
                        "nested_group": "Parent",
                        "content": ["old nested"],
                    },
                ],
            }
        ]
    }

    experiences = ai._work_pipeline_sections(job(), source, prompts)

    assert [item["title"] for item in experiences] == ["W0", "Parent", "Nested"]
    assert [item["content"] for item in experiences] == [
        ["rewritten:W0"],
        ["rewritten:Parent"],
        ["rewritten:Nested"],
    ]
    assert client.rewrite_titles["cv_work_experience_rewrite"] == ["W0", "Parent", "Nested"]


def test_atlas_work_rejects_empty_selection() -> None:
    client = Structured({"cv_work_experience_selection": []})
    ai = AtlasGeminiWorkflowAI(
        client,  # type: ignore[arg-type]
        work_experience_selection_count=2,
    )
    prompts = {
        "cv_work_experience_selection": definition("cv_work_experience_selection", "selection"),
        "cv_work_experience_rewrite": definition("cv_work_experience_rewrite", "[[rewrite_inputs_json]]"),
    }
    source = {
        "sections": [
            {
                "type": "entries",
                "entries": [
                    {"title": "W0", "date": "2026", "content": ["old W0"]},
                    {"title": "W1", "date": "2025", "content": ["old W1"]},
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="work-experience selection must contain exactly 2 indices"):
        ai._work_pipeline_sections(job(), source, prompts)
