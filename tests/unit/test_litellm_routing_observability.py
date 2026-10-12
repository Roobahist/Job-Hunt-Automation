from __future__ import annotations

import httpx
import pytest

from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations import llm_routing
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient, _header_int


def _definition() -> PromptDefinition:
    return PromptDefinition(
        key="qualification_scoring",
        version=1,
        template="unused",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"compatible": {"type": "boolean"}},
            "required": ["compatible"],
        },
    )


def test_header_int_handles_missing_and_invalid_values() -> None:
    request = httpx.Request("GET", "http://litellm")
    response = httpx.Response(200, request=request, headers={"x-good": "3", "x-bad": "nope"})

    assert _header_int(response, "x-good") == 3
    assert _header_int(response, "x-bad") is None
    assert _header_int(response, "x-missing") is None


def test_generation_log_exposes_priority_and_litellm_retry_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class CaptureLogger:
        def info(self, event: str, **kwargs: object) -> None:
            events.append((event, kwargs))

        def warning(self, event: str, **kwargs: object) -> None:
            events.append((event, kwargs))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/model/info":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "job-balanced",
                            "litellm_params": {"model": "openrouter/z-ai/glm-5.2:free", "order": 1},
                            "model_info": {"id": "glm", "supports_response_schema": True},
                        },
                        {
                            "model_name": "job-balanced",
                            "litellm_params": {
                                "model": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                                "order": 2,
                            },
                            "model_info": {"id": "nemotron", "supports_response_schema": False},
                        },
                    ]
                },
            )
        return httpx.Response(
            200,
            headers={
                "x-litellm-model-id": "nemotron",
                "x-litellm-call-id": "call-123",
                "x-litellm-attempted-retries": "0",
                "x-litellm-attempted-fallbacks": "0",
            },
            json={
                "model": "job-balanced",
                "choices": [{"message": {"content": '{"compatible":true}'}}],
                "usage": {"total_tokens": 4},
            },
        )

    monkeypatch.setattr(llm_routing, "logger", lambda: CaptureLogger())
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        transport=httpx.MockTransport(handler),
    )

    assert client.generate("prompt", _definition()) == {"compatible": True}

    generation = next(fields for event, fields in events if event == "llm_generation")
    assert generation["upstream_model"] == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert generation["routing_order"] == 2
    assert generation["higher_priority_models"] == ["openrouter/z-ai/glm-5.2:free"]
    assert generation["higher_priority_unavailable_or_filtered"] is True
    assert generation["litellm_attempted_retries"] == 0
    assert generation["litellm_attempted_fallbacks"] == 0
    assert generation["litellm_call_id"] == "call-123"


def test_generation_log_preserves_litellm_fallback_error_header(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class CaptureLogger:
        def info(self, event: str, **kwargs: object) -> None:
            events.append((event, kwargs))

        def warning(self, event: str, **kwargs: object) -> None:
            events.append((event, kwargs))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/model/info":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "job-balanced",
                            "litellm_params": {"model": "openrouter/z-ai/glm-5.2:free", "order": 1},
                            "model_info": {"id": "glm", "supports_response_schema": True},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            headers={
                "x-litellm-model-id": "glm",
                "x-litellm-attempted-retries": "1",
                "x-litellm-attempted-fallbacks": "1",
                "x-litellm-fallback-errors": "upstream 429",
            },
            json={
                "model": "job-balanced",
                "choices": [{"message": {"content": '{"compatible":true}'}}],
                "usage": {},
            },
        )

    monkeypatch.setattr(llm_routing, "logger", lambda: CaptureLogger())
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        transport=httpx.MockTransport(handler),
    )

    client.generate("prompt", _definition())
    generation = next(fields for event, fields in events if event == "llm_generation")
    assert generation["litellm_attempted_retries"] == 1
    assert generation["litellm_attempted_fallbacks"] == 1
    assert generation["litellm_fallback_errors"] == "upstream 429"
    assert generation["higher_priority_unavailable_or_filtered"] is False
