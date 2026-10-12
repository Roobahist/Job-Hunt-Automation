from __future__ import annotations

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.integrations.gemini import GeminiStructuredClient, GeminiWorkflowAI


class CapturingClient(GeminiStructuredClient):
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, object]:
        self.prompts[definition.key] = prompt
        if definition.key in {"cv_project_selection", "cv_work_experience_selection"}:
            return {"selected_indices": [0]}
        if definition.key in {"cv_project_rewrite", "cv_work_experience_rewrite"}:
            return {"contents": [["Rewritten bullet"]]}
        if definition.key == "cv_summary_rewrite":
            return {"summary": ["Summary"]}
        raise AssertionError(definition.key)


def prompt(key: str, template: str, schema: dict[str, object]) -> PromptDefinition:
    return PromptDefinition(key=key, version=1, template=template, temperature=0, output_structure=schema)


def prompts() -> dict[str, PromptDefinition]:
    selection_schema: dict[str, object] = {
        "type": "object",
        "properties": {"selected_indices": {"type": "array", "items": {"type": "integer"}}},
        "required": ["selected_indices"],
    }
    rewrite_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "contents": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            }
        },
        "required": ["contents"],
    }
    summary_schema: dict[str, object] = {
        "type": "object",
        "properties": {"summary": {"type": "array", "items": {"type": "string"}}},
        "required": ["summary"],
    }
    return {
        "cv_project_selection": prompt("cv_project_selection", "[[project_inputs_json]]", selection_schema),
        "cv_project_rewrite": prompt("cv_project_rewrite", "[[rewrite_inputs_json]]", rewrite_schema),
        "cv_work_experience_selection": prompt(
            "cv_work_experience_selection", "[[experience_inputs_json]]", selection_schema
        ),
        "cv_work_experience_rewrite": prompt("cv_work_experience_rewrite", "[[rewrite_inputs_json]]", rewrite_schema),
        "cv_summary_rewrite": prompt(
            "cv_summary_rewrite", "[[summary_context_json]]\n[[master_cv_json]]", summary_schema
        ),
    }


def test_complete_project_and_work_objects_are_available_to_selection_and_rewrite() -> None:
    client = CapturingClient()
    workflow = GeminiWorkflowAI(client, project_selection_count=1, work_experience_selection_count=1)
    master_cv = {
        "summary": ["Original summary"],
        "projects": [
            {
                "title": "Project",
                "secondary": "Secondary context",
                "organization": "Project Org",
                "date": "Summer 2026",
                "url": "https://example.com/project",
                "icon": "project-icon",
                "content": ["Original project bullet"],
            }
        ],
        "work_experience": [
            {
                "title": "Role",
                "organization": "Employer",
                "location": "Calgary",
                "date": "2026",
                "secondary": "Work context",
                "content": ["Original work bullet"],
            }
        ],
        "education": [{"degree": "MSc Statistics", "school": "University"}],
        "honors_awards": [{"title": "Award"}],
    }
    job = Job(
        source="test",
        url="https://example.com/job",
        company_name="Company",
        title="Title",
        description="Description",
    )

    _, projects = workflow._project_pipeline(job, master_cv, prompts())
    _, experiences = workflow._work_pipeline(job, master_cv, prompts())

    for marker in ("Secondary context", "Project Org", "Summer 2026", "https://example.com/project", "project-icon"):
        assert marker in client.prompts["cv_project_selection"]
        assert marker in client.prompts["cv_project_rewrite"]
    for marker in ("Employer", "Calgary", "2026", "Work context"):
        assert marker in client.prompts["cv_work_experience_selection"]
        assert marker in client.prompts["cv_work_experience_rewrite"]

    assert projects[0]["title"] == "Project"
    assert projects[0]["date"] == "Summer 2026"
    assert projects[0]["secondary"] == "Secondary context"
    assert projects[0]["content"] == ["Rewritten bullet"]
    assert experiences[0]["title"] == "Role"
    assert experiences[0]["organization"] == "Employer"
    assert experiences[0]["location"] == "Calgary"
    assert experiences[0]["content"] == ["Rewritten bullet"]


def test_summary_receives_full_master_cv_in_addition_to_tailored_context() -> None:
    client = CapturingClient()
    workflow = GeminiWorkflowAI(client)
    master_cv = {
        "summary": ["Original summary"],
        "education": [{"degree": "MSc Statistics", "school": "University"}],
        "honors_awards": [{"title": "Award"}],
    }
    job = Job(
        source="test",
        url="https://example.com/job",
        company_name="Company",
        title="Title",
        description="Description",
    )

    result = workflow._summary_pipeline(
        job,
        master_cv,
        tailored_skills=[],
        tailored_projects=[],
        tailored_experiences=[],
        prompts=prompts(),
    )

    assert result == ["Summary"]
    rendered = client.prompts["cv_summary_rewrite"]
    assert "MSc Statistics" in rendered
    assert "University" in rendered
    assert "Award" in rendered
