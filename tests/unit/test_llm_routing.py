from __future__ import annotations

import json

import httpx
import pytest

from job_hunt.config import LlmRoute, Settings
from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ConfigurationError, ErrorKind, ProviderError
from job_hunt.integrations import llm_routing
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.llm_routing import (
    LiteLLMGatewayClient,
    RoutedStructuredClient,
    _error_kind,
    capability_group_for_operation,
)


def definition(key: str = "qualification") -> PromptDefinition:
    return PromptDefinition(
        key=key,
        version=1,
        template="unused",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"compatible": {"type": "boolean"}},
            "required": ["compatible"],
        },
    )


class StubClient(GeminiStructuredClient):
    def __init__(self, result: dict[str, object] | None = None, error: ProviderError | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, object]:
        self.calls += 1
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def gateway_transport(content: str = '{"compatible":true}', status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if status_code >= 400:
            return httpx.Response(status_code, text="upstream failed")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "model": "upstream/provider-model",
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"total_tokens": 4},
            },
        )

    return httpx.MockTransport(handler)


def routes(*pairs: tuple[str, StubClient]) -> list[tuple[LlmRoute, GeminiStructuredClient]]:
    return [(LlmRoute(provider="litellm", model=name), client) for name, client in pairs]


def test_generation_routes_are_litellm_capability_aliases() -> None:
    settings = Settings(llm_routes="job-fast,job-balanced,job-powerful")
    assert [route.model for route in settings.llm_route_specs()] == ["job-fast", "job-balanced", "job-powerful"]


def test_generation_routes_reject_old_provider_model_syntax() -> None:
    settings = Settings(llm_routes="gemini:gemini-3.5-flash")
    with pytest.raises(ConfigurationError):
        settings.llm_route_specs()


def test_repair_routes_are_independent() -> None:
    settings = Settings(llm_routes="job-balanced", llm_repair_routes="repair-fast,repair-balanced")
    assert [route.model for route in settings.llm_repair_route_specs()] == ["repair-fast", "repair-balanced"]


def test_default_operation_policy_uses_capability_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_HUNT_LLM_OPERATION_GROUPS_JSON", raising=False)
    assert capability_group_for_operation("compatibility_filter") == "job-balanced"
    assert capability_group_for_operation("job_page_content_extraction") == "job-fast"
    assert capability_group_for_operation("qualification_scoring") == "job-balanced"
    assert capability_group_for_operation("cover_letter_generation") == "job-powerful"


def test_operation_policy_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_LLM_OPERATION_GROUPS_JSON", '{"compatibility":"job-fast"}')
    assert capability_group_for_operation("compatibility_filter") == "job-fast"


def test_repair_policy_is_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_LLM_REPAIR_GROUP", "repair-fast")
    assert capability_group_for_operation("cover_letter_generation:repair", repair=True) == "repair-fast"


def test_capability_router_calls_only_selected_group() -> None:
    fast = StubClient(result={"compatible": False})
    balanced = StubClient(result={"compatible": False})
    powerful = StubClient(result={"compatible": True})
    client = RoutedStructuredClient(routes(("job-fast", fast), ("job-balanced", balanced), ("job-powerful", powerful)))
    assert client.generate("prompt", definition("qualification_scoring")) == {"compatible": False}
    assert fast.calls == 0
    assert balanced.calls == 1
    assert powerful.calls == 0


def test_repair_router_calls_only_repair_group() -> None:
    fast = StubClient(result={"compatible": True})
    balanced = StubClient(result={"compatible": False})
    client = RoutedStructuredClient(routes(("repair-fast", fast), ("repair-balanced", balanced)), repair=True)
    assert client.generate("prompt", definition("cover_letter_generation:repair")) == {"compatible": True}
    assert fast.calls == 1
    assert balanced.calls == 0


def test_capability_router_enforces_configured_fallback_after_litellm_returns_rate_limit() -> None:
    balanced = StubClient(error=ProviderError("limited", ErrorKind.RATE_LIMIT, retryable=True, provider="litellm"))
    powerful = StubClient(result={"compatible": False})
    fast = StubClient(result={"compatible": True})
    client = RoutedStructuredClient(routes(("job-powerful", powerful), ("job-balanced", balanced), ("job-fast", fast)))

    assert client.generate("prompt", definition("qualification_scoring")) == {"compatible": True}
    assert balanced.calls == 1
    assert fast.calls == 1
    assert powerful.calls == 0


def test_gateway_client_sends_schema_in_prompt_and_native_response_format() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "groq/openai/gpt-oss-120b",
                "choices": [{"message": {"content": '{"compatible":true}'}}],
                "usage": {},
            },
        )

    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-powerful",
        transport=httpx.MockTransport(handler),
    )
    assert client.generate("prompt", definition("cover_letter_generation")) == {"compatible": True}
    request = requests[0]
    payload = json.loads(request.content)
    message = payload["messages"][0]["content"]
    response_format = payload["response_format"]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    assert payload["model"] == "job-powerful"
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "cover_letter_generation"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == definition("cover_letter_generation").output_structure
    assert "OUTPUT CONTRACT" in message
    assert '"compatible"' in message


def test_gateway_client_falls_back_without_response_format_when_native_schema_is_rejected() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if "response_format" in payload:
            return httpx.Response(400, text="response_format json_schema is not supported")
        return httpx.Response(
            200,
            json={
                "model": "mistral/mistral-medium-latest",
                "choices": [{"message": {"content": '{"compatible":true}'}}],
                "usage": {},
            },
        )

    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        transport=httpx.MockTransport(handler),
    )

    assert client.generate("prompt", definition()) == {"compatible": True}
    assert len(requests) == 2
    native_payload = json.loads(requests[0].content)
    fallback_payload = json.loads(requests[1].content)
    assert native_payload["response_format"]["type"] == "json_schema"
    assert "OUTPUT CONTRACT" in native_payload["messages"][0]["content"]
    assert "response_format" not in fallback_payload
    assert "OUTPUT CONTRACT" in fallback_payload["messages"][0]["content"]
    assert '"compatible"' in fallback_payload["messages"][0]["content"]


def test_gateway_logs_actual_deployment_and_schema_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    model_info_requests = 0

    class CaptureLogger:
        def info(self, event: str, **kwargs: object) -> None:
            events.append((event, kwargs))

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_info_requests
        if request.url.path == "/v1/model/info":
            model_info_requests += 1
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "job-powerful",
                            "litellm_params": {"model": "groq/openai/gpt-oss-120b"},
                            "model_info": {
                                "id": "deployment-123",
                                "supports_response_schema": True,
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            headers={"x-litellm-model-id": "deployment-123"},
            json={
                "model": "job-powerful",
                "choices": [{"message": {"content": '{"compatible":true}'}}],
                "usage": {"total_tokens": 4},
            },
        )

    monkeypatch.setattr(llm_routing, "logger", lambda: CaptureLogger())
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-powerful",
        transport=httpx.MockTransport(handler),
    )

    assert client.generate("prompt", definition("cover_letter_generation")) == {"compatible": True}
    assert client.generate("prompt", definition("cover_letter_generation")) == {"compatible": True}
    generation_events = [fields for event, fields in events if event == "llm_generation"]
    assert len(generation_events) == 2
    assert generation_events[0]["model_group"] == "job-powerful"
    assert generation_events[0]["upstream_provider"] == "groq"
    assert generation_events[0]["upstream_model"] == "groq/openai/gpt-oss-120b"
    assert generation_events[0]["deployment_id"] == "deployment-123"
    assert generation_events[0]["supports_native_json_schema"] is True
    assert generation_events[0]["structured_output_mode"] == "native_json_schema"
    assert model_info_requests == 1


def test_gateway_client_uses_independent_repair_route_for_invalid_json() -> None:
    repair = StubClient(result={"compatible": True})
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        repair_client=repair,
        transport=gateway_transport(content="not json"),
    )
    assert client.generate("prompt", definition()) == {"compatible": True}
    assert repair.calls == 1
    assert "ORIGINAL OUTPUT" in repair.prompts[0]


def test_gateway_rate_limit_maps_to_domain_error() -> None:
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        transport=gateway_transport(status_code=429),
    )
    with pytest.raises(ProviderError) as caught:
        client.generate("prompt", definition())
    assert caught.value.kind == ErrorKind.RATE_LIMIT
    assert caught.value.retryable is True


def test_http_status_error_mapping() -> None:
    assert _error_kind(429) == ErrorKind.RATE_LIMIT
    assert _error_kind(401) == ErrorKind.AUTHENTICATION
    assert _error_kind(422) == ErrorKind.MALFORMED_PROVIDER_RESPONSE
    assert _error_kind(503) == ErrorKind.TRANSIENT_PROVIDER
