from __future__ import annotations

from typing import Any

import httpx

from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient, _strip_outer_markdown_json_fence


class RepairSpy(GeminiStructuredClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.calls += 1
        raise AssertionError("valid fenced JSON must not invoke the repair model")


def _definition() -> PromptDefinition:
    return PromptDefinition(
        key="cv_project_rewrite",
        version=1.0,
        template="unused",
        temperature=0.0,
        output_structure={
            "type": "object",
            "properties": {"compatible": {"type": "boolean"}},
            "required": ["compatible"],
            "additionalProperties": False,
        },
    )


def _transport(content: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "deepseek/deepseek-v4-flash",
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"total_tokens": 4},
            },
        )

    return httpx.MockTransport(handler)


def test_strip_outer_markdown_json_fence_accepts_json_and_unlabelled_fences() -> None:
    labelled, labelled_removed = _strip_outer_markdown_json_fence('  ```JSON  \r\n{"compatible": true}\r\n```  ')
    unlabelled, unlabelled_removed = _strip_outer_markdown_json_fence('```\n{"compatible": true}\n```')

    assert labelled == '{"compatible": true}'
    assert labelled_removed is True
    assert unlabelled == '{"compatible": true}'
    assert unlabelled_removed is True


def test_strip_outer_markdown_json_fence_does_not_extract_embedded_block() -> None:
    content = 'Here is the result:\n```json\n{"compatible": true}\n```'

    normalized, removed = _strip_outer_markdown_json_fence(content)

    assert normalized == content
    assert removed is False


def test_gateway_parses_valid_fenced_json_without_repair() -> None:
    repair = RepairSpy()
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-powerful",
        repair_client=repair,
        transport=_transport('```json\n{"compatible": true}\n```'),
    )

    assert client.generate("prompt", _definition()) == {"compatible": True}
    assert repair.calls == 0
