from __future__ import annotations

import pytest
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations.constrained_workflow_ai import _skill_constrained_definition
from job_hunt.integrations.gemini import _validate_json_schema


def _definition() -> PromptDefinition:
    return PromptDefinition(
        key="cv_skills_tailoring",
        version=1.0,
        template="[[skills_json]]",
        temperature=0.0,
        output_structure={
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
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["groups"],
            "additionalProperties": False,
        },
    )


def _values() -> dict[str, object]:
    return {
        "skills_json": [
            {"label": "Programming", "skills": ["Python", "R", "SQL"]},
            {"label": "Data Science", "skills": ["pandas", "NumPy", "data visualization"]},
        ]
    }


def test_runtime_skill_schema_uses_master_cv_as_closed_whitelist() -> None:
    runtime = _skill_constrained_definition(_definition(), _values())

    skill_items = runtime.output_structure["properties"]["groups"]["items"]["properties"]["skills"]["items"]
    assert skill_items["enum"] == ["Python", "R", "SQL", "pandas", "NumPy", "data visualization"]


def test_runtime_skill_schema_rejects_hallucinated_skill() -> None:
    runtime = _skill_constrained_definition(_definition(), _values())

    with pytest.raises(JsonSchemaValidationError):
        _validate_json_schema(
            {
                "groups": [
                    {
                        "label": "Analytics",
                        "skills": ["SQL", "Power BI", "DAX"],
                    }
                ]
            },
            runtime.output_structure,
        )


def test_runtime_skill_schema_accepts_exact_source_skills() -> None:
    runtime = _skill_constrained_definition(_definition(), _values())

    result = _validate_json_schema(
        {
            "groups": [
                {
                    "label": "Analytics",
                    "skills": ["SQL", "pandas", "NumPy"],
                }
            ]
        },
        runtime.output_structure,
    )

    assert result["groups"][0]["skills"] == ["SQL", "pandas", "NumPy"]


def test_runtime_skill_schema_rejects_duplicate_skill_within_group() -> None:
    runtime = _skill_constrained_definition(_definition(), _values())

    with pytest.raises(JsonSchemaValidationError):
        _validate_json_schema(
            {
                "groups": [
                    {
                        "label": "Analytics",
                        "skills": ["SQL", "SQL"],
                    }
                ]
            },
            runtime.output_structure,
        )
