from __future__ import annotations

from typing import Any

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.integrations.configuration import ATLAS_PROMPT_KEYS
from job_hunt.integrations.gemini_atlas import AtlasGeminiWorkflowAI


def definition(key: str) -> PromptDefinition:
    schemas: dict[str, dict[str, Any]] = {
        "job_compatibility_filter": {
            "type": "object",
            "properties": {
                "compatible": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["compatible", "reasoning"],
            "additionalProperties": False,
        },
        "job_page_content_extraction": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "job_title": {"type": "string"},
                "job_description": {"type": "string"},
                "location": {"type": ["string", "null"]},
                "contract_type": {"type": ["string", "null"]},
            },
            "required": ["company_name", "job_title", "job_description"],
            "additionalProperties": False,
        },
        "qualification_scoring": {
            "type": "object",
            "properties": {
                "qualification_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "should_apply": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["qualification_score", "should_apply", "reasoning"],
            "additionalProperties": False,
        },
        "cv_work_experience_selection": {
            "type": "object",
            "properties": {"selected_indices": {"type": "array", "items": {"type": "integer"}}},
            "required": ["selected_indices"],
            "additionalProperties": False,
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
            "additionalProperties": False,
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
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["groups"],
            "additionalProperties": False,
        },
        "cv_references_inclusion": {
            "type": "object",
            "properties": {"include_references": {"type": "boolean"}},
            "required": ["include_references"],
            "additionalProperties": False,
        },
        "cv_summary_rewrite": {
            "type": "object",
            "properties": {"summary": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        "cover_letter_generation": {
            "type": "object",
            "properties": {"paragraphs": {"type": "array", "items": {"type": "string"}}},
            "required": ["paragraphs"],
            "additionalProperties": False,
        },
    }
    return PromptDefinition(
        key=key,
        version=1,
        template=key,
        output_structure=schemas[key],
        temperature=0.2,
    )


def test_atlas_profile_reuses_shared_operations_and_preserves_sections() -> None:
    prompts = {key: definition(key) for key in ATLAS_PROMPT_KEYS}
    calls: list[str] = []

    class Structured:
        def generate(self, _: str, prompt: PromptDefinition) -> dict[str, Any]:
            calls.append(prompt.key)
            outputs: dict[str, dict[str, Any]] = {
                "cv_work_experience_selection": {"selected_indices": [1]},
                "cv_work_experience_rewrite": {"contents": [["Parent tailored"], ["Nested tailored"]]},
                "cv_skills_tailoring": {"groups": [{"label": "Design", "skills": ["Figma", "Framer"]}]},
                "cv_references_inclusion": {"include_references": False},
                "cv_summary_rewrite": {"summary": ["Tailored summary"]},
                "cover_letter_generation": {"paragraphs": ["One", "Two", "Three"]},
            }
            return outputs[prompt.key]

    master_cv = {
        "sections": [
            {"type": "text", "title": "ABOUT ME", "content": ["Old summary"]},
            {
                "type": "entries",
                "title": "EXPERIENCE",
                "entries": [
                    {"title": "Parent", "parent": None, "content": ["Parent old"]},
                    {"title": "Nested", "parent": "Parent", "content": ["Nested old"]},
                ],
            },
            {"type": "education"},
            {
                "type": "label_rows",
                "title": "SKILLS",
                "rows": [{"label": "Design", "value": "Figma | Framer"}],
            },
            {
                "type": "references",
                "title": "REFERENCES",
                "items": [{"name": "Person"}],
            },
        ]
    }
    ai = AtlasGeminiWorkflowAI(Structured(), prompts, work_experience_selection_count=1)  # type: ignore[arg-type]
    job = Job(source="x", url="https://x/1", company_name="C", title="T", description="D")
    tailored = ai.tailor(job, master_cv, prompts)
    sections = tailored.cv["sections"]
    assert isinstance(sections, list)
    by_type = {section["type"]: section for section in sections}
    assert by_type["text"]["content"] == ["Tailored summary"]
    assert [entry["title"] for entry in by_type["entries"]["entries"]] == ["Parent", "Nested"]
    assert by_type["label_rows"]["rows"][0]["value"] == "Figma | Framer"
    assert by_type["references"]["items"] == []
    assert calls == [
        "cv_work_experience_selection",
        "cv_work_experience_rewrite",
        "cv_skills_tailoring",
        "cv_references_inclusion",
        "cv_summary_rewrite",
        "cover_letter_generation",
    ]
