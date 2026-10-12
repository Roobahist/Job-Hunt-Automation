from __future__ import annotations

import httpx
import pytest

from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations import llm_routing
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient
from job_hunt.logging import _redact


class RepairStub(GeminiStructuredClient):
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, object]:
        return self.result


class CaptureLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **kwargs: object) -> None:
        self.events.append((event, kwargs))

    def warning(self, event: str, **kwargs: object) -> None:
        self.events.append((event, kwargs))


def _definition() -> PromptDefinition:
    return PromptDefinition(
        key="cv_summary_rewrite",
        version=1,
        template="unused",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )


def _transport(content: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/model/info":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200,
            json={
                "model": "deepseek/deepseek-v4-flash",
                "choices": [{"message": {"content": content}}],
                "usage": {},
            },
        )

    return httpx.MockTransport(handler)


def test_invalid_json_logs_original_output_before_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = CaptureLogger()
    monkeypatch.setattr(llm_routing, "logger", lambda: captured)
    raw = '{"summary":"original DeepSeek text"'
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-powerful",
        repair_client=RepairStub({"summary": "original DeepSeek text"}),
        transport=_transport(raw),
    )

    assert client.generate("prompt", _definition()) == {"summary": "original DeepSeek text"}
    events = [fields for event, fields in captured.events if event == "llm_pre_repair_output"]
    assert len(events) == 1
    assert events[0]["prompt_key"] == "cv_summary_rewrite"
    assert events[0]["model_group"] == "job-powerful"
    assert events[0]["raw_output"] == raw
    assert "validation_error" in events[0]


def test_schema_invalid_output_logs_original_output_before_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = CaptureLogger()
    monkeypatch.setattr(llm_routing, "logger", lambda: captured)
    raw = '{"wrong_key":"keep this exact output"}'
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-powerful",
        repair_client=RepairStub({"summary": "keep this exact output"}),
        transport=_transport(raw),
    )

    assert client.generate("prompt", _definition()) == {"summary": "keep this exact output"}
    events = [fields for event, fields in captured.events if event == "llm_pre_repair_output"]
    assert len(events) == 1
    assert events[0]["raw_output"] == raw


def test_valid_output_does_not_log_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = CaptureLogger()
    monkeypatch.setattr(llm_routing, "logger", lambda: captured)
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-powerful",
        transport=_transport('{"summary":"valid output"}'),
    )

    assert client.generate("prompt", _definition()) == {"summary": "valid output"}
    assert all(event != "llm_pre_repair_output" for event, _ in captured.events)


def test_pre_repair_output_is_not_truncated_by_log_redaction() -> None:
    raw = "x" * 5000
    event = {"raw_output": raw, "other": "y" * 5000}

    redacted = _redact(None, "event", event)

    assert redacted["raw_output"] == raw
    assert redacted["other"] == "y" * 2000 + "…[truncated]"
