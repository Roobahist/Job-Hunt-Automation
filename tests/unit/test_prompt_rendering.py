from __future__ import annotations

from typing import Any

import pytest

from job_hunt.application.compatibility import GeminiCompatibilityFilter
from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.errors import ConfigurationError
from job_hunt.prompt_rendering import render_prompt


class StructuredClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {"compatible": True}


def _compatibility_prompt() -> PromptDefinition:
    return PromptDefinition(
        key="job_compatibility_filter",
        version=1.0,
        template="Raw data:\n[[scraped_job_json]]\nCompany:\n[[company_name]]",
        output_structure={
            "type": "object",
            "properties": {"compatible": {"type": "boolean"}},
            "required": ["compatible"],
            "additionalProperties": False,
        },
        temperature=0.0,
    )


def test_inserted_placeholder_like_text_is_preserved_as_data() -> None:
    rendered = render_prompt(
        "Payload: [[payload]]",
        {"payload": {"tracking": "[[req_postHashtag]]"}},
        "example",
    )

    assert "[[req_postHashtag]]" in rendered


def test_inserted_text_is_not_recursively_substituted() -> None:
    rendered = render_prompt(
        "Description: [[description]] Company: [[company_name]]",
        {
            "description": "Literal [[company_name]] from source data",
            "company_name": "Real Company",
        },
        "example",
    )

    assert "Literal [[company_name]] from source data" in rendered
    assert rendered.endswith("Company: Real Company")


def test_missing_template_placeholder_still_fails() -> None:
    with pytest.raises(ConfigurationError, match="missing_value"):
        render_prompt("Value: [[missing_value]]", {}, "example")


def test_compatibility_filter_accepts_linkedin_placeholder_like_provider_data() -> None:
    client = StructuredClient()
    compatibility = GeminiCompatibilityFilter(client)  # type: ignore[arg-type]
    job = assign_identity(
        Job(
            source="linkedin",
            external_id="4457989168",
            url="https://www.linkedin.com/jobs/view/4457989168",
            company_name="Example",
            title="Example role",
            description="Remote role",
            location="Canada",
            provider_data={
                "workplaceType": "REMOTE",
                "tracking": "[[req_postHashtag]]",
            },
        )
    )

    assert compatibility.compatible(job, {"job_compatibility_filter": _compatibility_prompt()}) is True
    assert "[[req_postHashtag]]" in client.prompts[0]
