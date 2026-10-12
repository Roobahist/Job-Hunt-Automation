from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_hunt.config import Settings
from job_hunt.integrations.litellm_config import ConfigGenerationError, build_litellm_config, load_model_limits
from job_hunt.integrations.litellm_runtime_config import prioritize_group_deployments

ROUTING_PATH = Path("config/llm-routing.json")
LIMITS_PATH = Path("config/llm-model-limits.json")


def _routing() -> dict[str, object]:
    return json.loads(ROUTING_PATH.read_text(encoding="utf-8"))


def test_provider_keys_get_stable_non_secret_deployment_ids_and_keep_model_priority() -> None:
    routing = _routing()
    config = build_litellm_config(
        env={
            "GEMINI_API_KEY_1": "first-secret-value",
            "GEMINI_API_KEY_2": "second-secret-value",
        },
        registry=routing,
        model_limits=load_model_limits(LIMITS_PATH),
    )
    config = prioritize_group_deployments(config, routing)

    gemini_35 = [
        entry
        for entry in config["model_list"]
        if entry["model_name"] == "job-fast" and entry["litellm_params"]["model"] == "gemini/gemini-3.5-flash-lite"
    ]
    assert [entry["litellm_params"]["order"] for entry in gemini_35] == [1, 1]
    assert {entry["model_info"]["id"] for entry in gemini_35} == {
        "job-fast--gemini-gemini-3-5-flash-lite--key-1",
        "job-fast--gemini-gemini-3-5-flash-lite--key-2",
    }
    assert all("secret" not in entry["model_info"]["id"] for entry in gemini_35)

    gemini_31 = [
        entry
        for entry in config["model_list"]
        if entry["model_name"] == "job-fast" and entry["litellm_params"]["model"] == "gemini/gemini-3.1-flash-lite"
    ]
    assert [entry["litellm_params"]["order"] for entry in gemini_31] == [2, 2]


def test_langfuse_callback_is_opt_in() -> None:
    config = build_litellm_config(
        env={"GEMINI_API_KEY_1": "gemini-key"},
        registry=_routing(),
        model_limits=load_model_limits(LIMITS_PATH),
    )

    assert config["litellm_settings"] == {"drop_params": True}


def test_langfuse_callback_uses_otel_preset_when_enabled() -> None:
    config = build_litellm_config(
        env={
            "GEMINI_API_KEY_1": "gemini-key",
            "JOB_HUNT_LANGFUSE_ENABLED": "true",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
        },
        registry=_routing(),
        model_limits=load_model_limits(LIMITS_PATH),
    )

    assert config["litellm_settings"] == {
        "drop_params": True,
        "callbacks": ["langfuse_otel"],
    }


def test_langfuse_enabled_requires_both_credentials() -> None:
    with pytest.raises(ConfigGenerationError, match="LANGFUSE_SECRET_KEY"):
        build_litellm_config(
            env={
                "GEMINI_API_KEY_1": "gemini-key",
                "JOB_HUNT_LANGFUSE_ENABLED": "true",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            },
            registry=_routing(),
            model_limits=load_model_limits(LIMITS_PATH),
        )


def test_tenant_virtual_key_overrides_shared_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(litellm_api_key="shared-key")
    monkeypatch.setenv("JOB_HUNT_LITELLM_TENANT_KEY_NOVA", "tenant-key")

    assert settings.litellm_key_for_tenant("nova") == "tenant-key"
    assert settings.litellm_key_for_tenant("atlas") == "shared-key"
    assert settings.litellm_key_for_tenant(None) == "shared-key"
