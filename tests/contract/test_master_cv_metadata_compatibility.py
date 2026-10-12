from __future__ import annotations

from typing import Any

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.integrations.constrained_workflow_ai import ConstrainedParallelGeminiWorkflowAI
from job_hunt.rendering.profiles import NovaCvRenderer


def _definition(key: str, template: str, schema: dict[str, Any]) -> PromptDefinition:
    return PromptDefinition(
        key=key,
        version=1,
        template=template,
        output_structure=schema,
        temperature=0.0,
    )


def _prompts() -> dict[str, PromptDefinition]:
    return {
        "cv_project_selection": _definition(
            "cv_project_selection",
            "PROJECTS: [[project_inputs_json]]",
            {
                "type": "object",
                "properties": {
                    "selected_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
                "required": ["selected_indices"],
            },
        ),
        "cv_project_rewrite": _definition(
            "cv_project_rewrite",
            "PROJECTS: [[rewrite_inputs_json]]",
            {
                "type": "object",
                "properties": {
                    "contents": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    }
                },
                "required": ["contents"],
            },
        ),
        "cv_skills_tailoring": _definition(
            "cv_skills_tailoring",
            "SKILLS: [[skills_json]]",
            {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "skills": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["label", "skills"],
                        },
                    }
                },
                "required": ["groups"],
            },
        ),
    }


class _StructuredClient:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.prompts.append((definition.key, prompt))
        if definition.key == "cv_project_selection":
            return {"selected_indices": [0]}
        if definition.key == "cv_project_rewrite":
            return {"contents": [["Tailored project bullet"]]}
        if definition.key == "cv_skills_tailoring":
            return {
                "groups": [
                    {
                        "label": "Machine Learning & Deep Learning",
                        "skills": ["TensorFlow/Keras"],
                    },
                    {
                        "label": "Visualization & Geospatial",
                        "skills": ["Matplotlib", "sf"],
                    },
                ]
            }
        raise AssertionError(f"Unexpected prompt: {definition.key}")


def _job() -> Job:
    return Job(
        source="test",
        url="https://example.com/job",
        company_name="Example",
        title="Machine Learning Engineer",
        description="Build machine learning systems.",
    )


def _ai(client: _StructuredClient, prompts: dict[str, PromptDefinition]) -> ConstrainedParallelGeminiWorkflowAI:
    return ConstrainedParallelGeminiWorkflowAI(
        client,  # type: ignore[arg-type]
        prompts,
        project_selection_count=1,
        parallelism=1,
    )


def test_enriched_project_metadata_is_available_to_prompts_and_preserved() -> None:
    project = {
        "title": "Automation Platform",
        "content": ["Original project bullet"],
        "status": "production",
        "importance": 9,
        "technologies": ["Python", "FastAPI"],
        "tags": ["llm-engineering", "automation"],
        "related_roles": ["Machine Learning Engineer", "AI Engineer"],
        "related_work_experience": ["Data Scientist, Example"],
    }
    master_cv = {"projects": [project]}
    prompts = _prompts()
    client = _StructuredClient()

    indices, tailored = _ai(client, prompts)._project_pipeline(_job(), master_cv, prompts)

    assert indices == [0]
    assert tailored == [{**project, "content": ["Tailored project bullet"]}]

    selection_prompt = client.prompts[0][1]
    rewrite_prompt = client.prompts[1][1]
    for expected in (
        '"status":"production"',
        '"importance":9',
        '"technologies":["Python","FastAPI"]',
        '"tags":["llm-engineering","automation"]',
        '"related_roles":["Machine Learning Engineer","AI Engineer"]',
        '"related_work_experience":["Data Scientist, Example"]',
    ):
        assert expected in selection_prompt
        assert expected in rewrite_prompt

    rendered = NovaCvRenderer._entries(tailored)
    assert "Automation Platform" in rendered
    assert "Tailored project bullet" in rendered
    assert "production" not in rendered
    assert "llm-engineering" not in rendered


def test_project_metadata_remains_optional_for_legacy_master_cvs() -> None:
    project = {
        "title": "Legacy Project",
        "content": ["Original project bullet"],
    }
    master_cv = {"projects": [project]}
    prompts = _prompts()
    client = _StructuredClient()

    indices, tailored = _ai(client, prompts)._project_pipeline(_job(), master_cv, prompts)

    assert indices == [0]
    assert tailored == [
        {
            "title": "Legacy Project",
            "content": ["Tailored project bullet"],
        }
    ]
    assert "importance" not in client.prompts[0][1]
    assert "technologies" not in client.prompts[0][1]


def test_new_skill_group_names_and_merged_skill_values_are_supported() -> None:
    master_cv = {
        "skills": [
            {
                "label": "Machine Learning & Deep Learning",
                "value": "PyTorch, TensorFlow/Keras, scikit-learn",
            },
            {
                "label": "Visualization & Geospatial",
                "value": "Figma, TikZ, ggplot2, Matplotlib, Plotly, sf",
            },
        ]
    }
    prompts = _prompts()
    client = _StructuredClient()

    tailored = _ai(client, prompts)._skills_pipeline(_job(), master_cv, prompts)

    assert tailored == [
        {
            "label": "Machine Learning & Deep Learning",
            "value": "TensorFlow/Keras",
        },
        {
            "label": "Visualization & Geospatial",
            "value": "Matplotlib, sf",
        },
    ]
    skills_prompt = client.prompts[0][1]
    assert '"TensorFlow/Keras"' in skills_prompt
    assert '"label":"Machine Learning & Deep Learning"' in skills_prompt
    assert '"label":"Visualization & Geospatial"' in skills_prompt
