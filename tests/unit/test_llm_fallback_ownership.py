from __future__ import annotations

from typing import Any

import pytest

from job_hunt.config import LlmRoute
from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.litellm_runtime_config import (
    application_owned_fallback_config,
    prioritize_group_deployments,
)
from job_hunt.integrations.llm_routing_runtime import ObservableCapabilityRoutedStructuredClient


class FakeClient(GeminiStructuredClient):
    def __init__(self, result: dict[str, Any] | None = None, error: ProviderError | None = None) -> None:
        self.result = result or {}
        self.error = error
        self.calls = 0

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.events.append(("warning", event, kwargs))

    def info(self, event: str, **kwargs: object) -> None:
        self.events.append(("info", event, kwargs))


def _route(group: str, client: GeminiStructuredClient) -> tuple[LlmRoute, GeminiStructuredClient]:
    return (LlmRoute(provider="litellm", model=group), client)


def _definition(key: str) -> PromptDefinition:
    return PromptDefinition(
        key=key,
        version=1.0,
        template="test",
        output_structure={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        temperature=0.0,
    )


def test_proxy_config_has_no_cross_group_fallbacks_but_keeps_in_group_failover() -> None:
    config = {
        "model_list": [
            {"model_name": "job-powerful", "litellm_params": {"model": "deepseek/deepseek-v4-flash"}},
            {"model_name": "job-balanced", "litellm_params": {"model": "gemini/gemini-3.6-flash"}},
        ],
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "enable_weighted_failover": True,
            "max_fallbacks": 7,
            "num_retries": 0,
            "fallbacks": [{"job-powerful": ["job-balanced"]}],
        },
    }

    result = application_owned_fallback_config(config)

    assert "fallbacks" not in result["router_settings"]
    assert result["router_settings"]["enable_weighted_failover"] is True
    assert result["router_settings"]["max_fallbacks"] == 7
    powerful_models = {
        entry["litellm_params"]["model"] for entry in result["model_list"] if entry["model_name"] == "job-powerful"
    }
    assert powerful_models == {"deepseek/deepseek-v4-flash"}


def test_model_priority_exhausts_all_keys_for_stronger_model_before_weaker_model() -> None:
    config = {
        "model_list": [
            {
                "model_name": "job-balanced",
                "litellm_params": {
                    "model": "gemini/gemini-3.7-flash",
                    "api_key": "os.environ/GEMINI_API_KEY_1",
                },
            },
            {
                "model_name": "job-balanced",
                "litellm_params": {
                    "model": "gemini/gemini-3.7-flash",
                    "api_key": "os.environ/GEMINI_API_KEY_2",
                },
            },
            {
                "model_name": "job-balanced",
                "litellm_params": {
                    "model": "gemini/gemini-3.6-flash",
                    "api_key": "os.environ/GEMINI_API_KEY_1",
                },
            },
            {
                "model_name": "job-balanced",
                "litellm_params": {
                    "model": "gemini/gemini-3.6-flash",
                    "api_key": "os.environ/GEMINI_API_KEY_2",
                },
            },
        ],
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "allowed_fails": 0,
            "cooldown_time": 5,
            "num_retries": 9,
        },
    }
    registry = {
        "groups": {
            "job-balanced": [
                {"provider": "gemini", "model": "gemini-3.7-flash"},
                {"provider": "gemini", "model": "gemini-3.6-flash"},
            ]
        }
    }

    result = prioritize_group_deployments(config, registry)

    orders = [entry["litellm_params"]["order"] for entry in result["model_list"]]
    assert orders == [1, 1, 2, 2]
    router = result["router_settings"]
    assert router["enable_pre_call_checks"] is True
    assert router["enable_weighted_failover"] is True
    assert router["num_retries"] == 3
    assert router["allowed_fails"] == 1
    assert router["cooldown_time"] == 65


def test_retry_budget_covers_every_deployment_in_largest_group() -> None:
    config = {
        "model_list": [
            {"model_name": "job-balanced", "litellm_params": {"model": f"provider/balanced-{i}"}} for i in range(15)
        ]
        + [{"model_name": "job-fast", "litellm_params": {"model": f"provider/fast-{i}"}} for i in range(5)],
        "router_settings": {},
    }

    result = prioritize_group_deployments(config, {"groups": {}})

    assert result["router_settings"]["num_retries"] == 14


def test_litellm_router_tuning_preserves_stricter_positive_cooldown_values() -> None:
    config = {
        "model_list": [],
        "router_settings": {
            "allowed_fails": 2,
            "cooldown_time": 180,
        },
    }

    result = prioritize_group_deployments(config, {"groups": {}})

    assert result["router_settings"]["allowed_fails"] == 2
    assert result["router_settings"]["cooldown_time"] == 180


def test_retryable_powerful_failure_falls_back_in_application_and_records_reason(monkeypatch) -> None:
    powerful = FakeClient(
        error=ProviderError(
            "DeepSeek quota exceeded",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="litellm",
            status_code=429,
        )
    )
    balanced = FakeClient(result={"ok": True})
    client = ObservableCapabilityRoutedStructuredClient(
        [_route("job-powerful", powerful), _route("job-balanced", balanced)],
        fallbacks={"job-powerful": ["job-balanced"], "job-balanced": []},
    )
    recorder = RecordingLogger()
    monkeypatch.setattr("job_hunt.integrations.llm_routing_runtime.logger", lambda: recorder)

    result = client.generate("prompt", _definition("cv_project_rewrite"))

    assert result == {"ok": True}
    assert powerful.calls == 1
    assert balanced.calls == 1

    failure = next(data for _level, event, data in recorder.events if event == "llm_group_failed")
    assert failure["requested_group"] == "job-powerful"
    assert failure["attempted_group"] == "job-powerful"
    assert failure["fallback_type"] == "retryable_provider_error"
    assert failure["fallback_targets"] == ["job-balanced"]
    assert failure["error_kind"] == "rate_limit"
    assert failure["status_code"] == 429

    success = next(data for _level, event, data in recorder.events if event == "llm_group_succeeded")
    assert success["requested_group"] == "job-powerful"
    assert success["resolved_group"] == "job-balanced"
    assert success["fallback_used"] is True
    assert success["fallback_from"] == "job-powerful"
    assert success["fallback_type"] == "retryable_provider_error"
    assert success["fallback_error_kind"] == "rate_limit"
    assert success["fallback_status_code"] == 429
    assert success["fallback_chain"] == ["job-powerful", "job-balanced"]
    assert "DeepSeek quota exceeded" in str(success["fallback_reason"])


def test_malformed_powerful_output_falls_back_to_balanced(monkeypatch) -> None:
    powerful = FakeClient(
        error=ProviderError(
            "Repair output repeatedly changed source content",
            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
            provider="repair-guard",
        )
    )
    balanced = FakeClient(result={"ok": True})
    client = ObservableCapabilityRoutedStructuredClient(
        [_route("job-powerful", powerful), _route("job-balanced", balanced)],
        fallbacks={"job-powerful": ["job-balanced"], "job-balanced": []},
    )
    recorder = RecordingLogger()
    monkeypatch.setattr("job_hunt.integrations.llm_routing_runtime.logger", lambda: recorder)

    result = client.generate("prompt", _definition("cv_summary_rewrite"))

    assert result == {"ok": True}
    assert powerful.calls == 1
    assert balanced.calls == 1
    failure = next(data for _level, event, data in recorder.events if event == "llm_group_failed")
    assert failure["fallback_type"] == "malformed_provider_response"
    assert failure["fallback_targets"] == ["job-balanced"]


def test_retryable_repair_failure_uses_configured_repair_fallback() -> None:
    fast = FakeClient(
        error=ProviderError(
            "repair model rate limited",
            ErrorKind.RATE_LIMIT,
            retryable=True,
            provider="litellm",
            status_code=429,
        )
    )
    balanced = FakeClient(result={"ok": True})
    client = ObservableCapabilityRoutedStructuredClient(
        [_route("repair-fast", fast), _route("repair-balanced", balanced)],
        repair=True,
        fallbacks={"repair-fast": ["repair-balanced"], "repair-balanced": []},
    )

    result = client.generate("prompt", _definition("cv_summary_rewrite:repair"))

    assert result == {"ok": True}
    assert fast.calls == 1
    assert balanced.calls == 1


def test_nonretryable_powerful_failure_never_crosses_groups() -> None:
    powerful = FakeClient(
        error=ProviderError(
            "authentication failed",
            ErrorKind.AUTHENTICATION,
            retryable=False,
            provider="litellm",
            status_code=401,
        )
    )
    balanced = FakeClient(result={"ok": True})
    client = ObservableCapabilityRoutedStructuredClient(
        [_route("job-powerful", powerful), _route("job-balanced", balanced)],
        fallbacks={"job-powerful": ["job-balanced"], "job-balanced": []},
    )

    with pytest.raises(ProviderError, match="authentication failed"):
        client.generate("prompt", _definition("cv_project_rewrite"))

    assert powerful.calls == 1
    assert balanced.calls == 0


_WRAPPED_GEMINI_QUOTA_ERROR = (
    'LiteLLM route job-balanced failed with HTTP 403: {"error":{"message":'
    '"litellm.BadRequestError: GeminiException BadRequestError - {\\n '
    '\\"error\\": {\\n \\"code\\": 429,\\n '
    '\\"message\\": \\"You exceeded your current quota, please check your plan and billing details.\\",\\n '
    '\\"status\\": \\"RESOURCE_EXHAUSTED\\"\\n }\\n}"}}'
)


def test_wrapped_403_with_upstream_429_is_retryable_rate_limit() -> None:
    error = ProviderError(
        _WRAPPED_GEMINI_QUOTA_ERROR,
        ErrorKind.AUTHENTICATION,
        retryable=False,
        provider="litellm",
        status_code=403,
    )

    assert error.kind is ErrorKind.RATE_LIMIT
    assert error.retryable is True
    assert error.status_code == 403


def test_wrapped_quota_error_falls_back_to_next_group() -> None:
    balanced = FakeClient(
        error=ProviderError(
            _WRAPPED_GEMINI_QUOTA_ERROR,
            ErrorKind.AUTHENTICATION,
            retryable=False,
            provider="litellm",
            status_code=403,
        )
    )
    fast = FakeClient(result={"ok": True})
    client = ObservableCapabilityRoutedStructuredClient(
        [_route("job-balanced", balanced), _route("job-fast", fast)],
        fallbacks={"job-balanced": ["job-fast"], "job-fast": []},
    )

    result = client.generate("prompt", _definition("qualification_scoring"))

    assert result == {"ok": True}
    assert balanced.calls == 1
    assert fast.calls == 1


def test_real_403_authentication_error_stays_nonretryable() -> None:
    error = ProviderError(
        "LiteLLM route job-balanced failed with HTTP 403: invalid API key",
        ErrorKind.AUTHENTICATION,
        retryable=False,
        provider="litellm",
        status_code=403,
    )

    assert error.kind is ErrorKind.AUTHENTICATION
    assert error.retryable is False

    balanced = FakeClient(error=error)
    fast = FakeClient(result={"ok": True})
    client = ObservableCapabilityRoutedStructuredClient(
        [_route("job-balanced", balanced), _route("job-fast", fast)],
        fallbacks={"job-balanced": ["job-fast"], "job-fast": []},
    )

    with pytest.raises(ProviderError, match="invalid API key"):
        client.generate("prompt", _definition("qualification_scoring"))

    assert balanced.calls == 1
    assert fast.calls == 0
