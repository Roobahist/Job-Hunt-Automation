from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
import structlog

from job_hunt.config import LlmRoute, Settings
from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations import llm_routing
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient, _gateway_client


def _definition() -> PromptDefinition:
    return PromptDefinition(
        key="cv_summary_rewrite",
        version=1,
        template="unused",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"summary": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary"],
        },
    )


def test_gateway_sends_correlation_metadata_trace_parent_and_meaningful_span_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    span_kwargs: dict[str, Any] = {}
    span_outputs: list[Any] = []

    class Observation:
        def update(self, **kwargs: Any) -> None:
            span_outputs.append(kwargs.get("output"))

    @contextmanager
    def fake_span(**kwargs: Any) -> Iterator[Observation]:
        span_kwargs.update(kwargs)
        yield Observation()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "job-powerful",
                "choices": [{"message": {"content": '{"summary":["ok"]}'}}],
                "usage": {"total_tokens": 2},
            },
        )

    monkeypatch.setattr(llm_routing, "langfuse_span", fake_span)
    monkeypatch.setattr(
        llm_routing,
        "trace_headers",
        lambda **_kwargs: {"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"},
    )
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "tenant-key",
        "job-powerful",
        tenant="nova",
        session_id="run-123",
        langfuse_enabled=True,
        transport=httpx.MockTransport(handler),
    )

    tokens = structlog.contextvars.bind_contextvars(
        tenant="nova",
        run_id="run-123",
        job_id="123",
        job_identity="linkedin:123",
    )
    try:
        assert client.generate("prompt", _definition()) == {"summary": ["ok"]}
    finally:
        structlog.contextvars.reset_contextvars(**tokens)

    payload = json.loads(requests[0].content)
    metadata = payload["metadata"]
    assert metadata["generation_name"] == "cv_summary_rewrite"
    assert metadata["operation"] == "cv_summary_rewrite"
    assert metadata["model_group"] == "job-powerful"
    assert metadata["tenant"] == "nova"
    assert metadata["session_id"] == "nova:123"
    assert metadata["run_id"] == "run-123"
    assert metadata["job_id"] == "123"
    assert metadata["tags"] == [
        "job-hunt",
        "operation:cv_summary_rewrite",
        "tenant:nova",
    ]
    assert requests[0].headers["traceparent"].startswith("00-0123456789abcdef")
    assert requests[0].headers["authorization"] == "Bearer tenant-key"

    assert span_kwargs["session_id"] == "nova:123"
    span_input = span_kwargs["input"]
    assert span_input["model_group"] == "job-powerful"
    assert span_input["temperature"] == 0
    assert span_input["messages"][0]["role"] == "user"
    assert "prompt" in span_input["messages"][0]["content"]
    assert "OUTPUT CONTRACT" in span_input["messages"][0]["content"]
    assert span_input["response_format"]["type"] == "json_schema"
    assert span_outputs == [{"summary": ["ok"]}]


def test_gateway_without_job_context_keeps_run_id_but_has_no_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    @contextmanager
    def no_span(**_kwargs: Any) -> Iterator[None]:
        yield None

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "job-fast",
                "choices": [{"message": {"content": '{"summary":["ok"]}'}}],
                "usage": {"total_tokens": 2},
            },
        )

    monkeypatch.setattr(llm_routing, "langfuse_span", no_span)
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "tenant-key",
        "job-fast",
        tenant="nova",
        session_id="run-123",
        transport=httpx.MockTransport(handler),
    )

    assert client.generate("prompt", _definition()) == {"summary": ["ok"]}
    metadata = json.loads(requests[0].content)["metadata"]
    assert metadata["run_id"] == "run-123"
    assert "session_id" not in metadata


def test_gateway_builder_uses_tenant_virtual_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_LITELLM_TENANT_KEY_NOVA", "nova-virtual-key")
    settings = Settings(litellm_api_key="shared-admin-compatible-key")

    client = _gateway_client(
        settings,
        LlmRoute(provider="litellm", model="job-fast"),
        repair_client=None,
        state=None,
        tenant="nova",
        session_id="run-456",
    )

    assert client.client.headers["authorization"] == "Bearer nova-virtual-key"
    assert client.tenant == "nova"
    assert client.session_id == "run-456"
