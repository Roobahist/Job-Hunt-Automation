from __future__ import annotations

from typing import Any

import httpx
import pytest

from job_hunt.config import LlmRoute
from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ErrorKind, GatewayUnavailableError, ProviderError
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient
from job_hunt.integrations.llm_routing_runtime import ObservableCapabilityRoutedStructuredClient


def definition() -> PromptDefinition:
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


class SequenceClient(GeminiStructuredClient):
    def __init__(self, outcomes: list[dict[str, Any] | ProviderError]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def routed_client(
    balanced: GeminiStructuredClient,
    fast: GeminiStructuredClient,
) -> ObservableCapabilityRoutedStructuredClient:
    return ObservableCapabilityRoutedStructuredClient(
        [
            (LlmRoute(provider="litellm", model="job-balanced"), balanced),
            (LlmRoute(provider="litellm", model="job-fast"), fast),
        ],
        fallbacks={"job-balanced": ("job-fast",), "job-fast": ()},
    )


def test_connect_error_is_classified_as_gateway_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(GatewayUnavailableError) as caught:
        client.generate("prompt", definition())

    assert caught.value.kind == ErrorKind.GATEWAY_UNAVAILABLE
    assert caught.value.retryable is True
    assert str(caught.value) == "LiteLLM gateway unavailable: ConnectError: [Errno 111] Connection refused"


def test_read_timeout_is_not_misclassified_as_gateway_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderError) as caught:
        client.generate("prompt", definition())

    assert not isinstance(caught.value, GatewayUnavailableError)
    assert caught.value.kind == ErrorKind.TRANSIENT_PROVIDER
    assert "ReadTimeout" in str(caught.value)


def test_gateway_unavailable_does_not_fall_back_to_another_model_group() -> None:
    balanced = SequenceClient(
        [GatewayUnavailableError("LiteLLM gateway unavailable: ConnectError: connection refused")]
    )
    fast = SequenceClient([{"compatible": True}])

    with pytest.raises(GatewayUnavailableError):
        routed_client(balanced, fast).generate("prompt", definition())

    assert balanced.calls == 1
    assert fast.calls == 0


def test_http_provider_failure_still_uses_existing_cross_group_fallback() -> None:
    provider_error = ProviderError(
        "LiteLLM route job-balanced failed with HTTP 503: upstream unavailable",
        ErrorKind.TRANSIENT_PROVIDER,
        retryable=True,
        provider="litellm",
        status_code=503,
    )
    balanced = SequenceClient([provider_error])
    fast = SequenceClient([{"compatible": True}])

    assert routed_client(balanced, fast).generate("prompt", definition()) == {"compatible": True}
    assert balanced.calls == 1
    assert fast.calls == 1
