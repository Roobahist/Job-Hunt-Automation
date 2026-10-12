from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations.gemini import GeminiWorkflowAI


class CaptureClient:
    def __init__(self) -> None:
        self.definition: PromptDefinition | None = None

    def generate(self, _prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.definition = definition
        count = definition.output_structure["properties"]["contents"]["minItems"]
        return {"contents": [[] for _ in range(count)]}


def rewrite_definition(key: str) -> PromptDefinition:
    return PromptDefinition(
        key=key,
        version=1,
        template="INPUTS [[rewrite_inputs_json]]",
        temperature=0,
        output_structure={
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
    )


@pytest.mark.parametrize("key", ["cv_project_rewrite", "cv_work_experience_rewrite"])
def test_rewrite_schema_requires_exact_runtime_input_count(key: str) -> None:
    client = CaptureClient()
    ai = GeminiWorkflowAI(client)  # type: ignore[arg-type]
    definition = rewrite_definition(key)
    inputs = [{"content": ["a"]}, {"content": ["b"]}, {"content": ["c"]}]

    ai._run(definition, {"rewrite_inputs_json": inputs})

    assert client.definition is not None
    contents_schema = client.definition.output_structure["properties"]["contents"]
    assert contents_schema["minItems"] == 3
    assert contents_schema["maxItems"] == 3
    assert "minItems" not in definition.output_structure["properties"]["contents"]

    validator = Draft202012Validator(client.definition.output_structure)
    assert not list(validator.iter_errors({"contents": [[], [], []]}))
    assert list(validator.iter_errors({"contents": [[], []]}))
    assert list(validator.iter_errors({"contents": [[], [], [], []]}))


def test_non_rewrite_schema_is_not_modified() -> None:
    client = CaptureClient()
    ai = GeminiWorkflowAI(client)  # type: ignore[arg-type]
    definition = PromptDefinition(
        key="other_operation",
        version=1,
        template="Prompt",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"contents": {"type": "array", "items": {"type": "string"}}},
            "required": ["contents"],
        },
    )

    class OtherClient:
        def __init__(self) -> None:
            self.definition: PromptDefinition | None = None

        def generate(self, _prompt: str, runtime_definition: PromptDefinition) -> dict[str, Any]:
            self.definition = runtime_definition
            return {"contents": []}

    other = OtherClient()
    ai.client = other  # type: ignore[assignment]
    ai._run(definition, {})

    assert other.definition is definition
