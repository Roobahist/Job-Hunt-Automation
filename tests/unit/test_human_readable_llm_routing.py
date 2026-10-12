from __future__ import annotations

import json
from pathlib import Path

from job_hunt.integrations.litellm_config import (
    build_litellm_config,
    load_model_limits,
    load_provider_registry,
)

ROUTING_PATH = Path("config/llm-routing.json")
LIMITS_PATH = Path("config/llm-model-limits.json")


def test_production_routing_file_lists_models_by_group() -> None:
    routing = load_provider_registry(ROUTING_PATH)

    assert routing["groups"]["job-fast"] == [
        {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
        {"provider": "gemini", "model": "gemini-3.1-flash-lite"},
        {"provider": "mistral", "model": "mistral-medium-latest"},
    ]
    assert routing["groups"]["job-balanced"] == [
        {"provider": "openrouter", "model": "z-ai/glm-5.2:free"},
        {"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"},
        {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"},
        {"provider": "openrouter", "model": "minimax/minimax-m3:free"},
        {"provider": "openrouter", "model": "google/gemma-4-31b-it:free"},
        {"provider": "gemini", "model": "gemini-3.7-flash"},
        {"provider": "gemini", "model": "gemini-3.6-flash"},
        {"provider": "gemini", "model": "gemini-3.5-flash"},
    ]
    assert routing["groups"]["job-powerful"] == [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
    ]
    assert all(
        entry.get("provider") != "openrouter"
        for group_name in ("job-fast", "repair-fast", "repair-balanced")
        for entry in routing["groups"][group_name]
    )
    banned_auto_route = {"provider": "openrouter", "model": "free"}
    assert not any(banned_auto_route in group for group in routing["groups"].values())
    assert "structured_output_modes" not in routing


def test_production_fallback_policy_has_no_powerful_to_fast_hop() -> None:
    routing = load_provider_registry(ROUTING_PATH)

    assert routing["fallbacks"]["job-powerful"] == ["job-balanced"]
    assert routing["fallbacks"]["job-balanced"] == ["job-fast"]
    assert routing["fallbacks"]["job-fast"] == []
    assert "job-fast" not in routing["fallbacks"]["job-powerful"]


def test_model_catalog_tracks_structured_output_capability_per_model() -> None:
    metadata = load_model_limits(LIMITS_PATH)

    assert metadata["deepseek/deepseek-v4-flash"]["supports_native_json_schema"] is False
    assert metadata["gemini/gemini-3.5-flash"]["supports_native_json_schema"] is False
    assert metadata["gemini/gemini-3.1-flash-lite"]["supports_native_json_schema"] is True
    assert metadata["gemini/gemini-3.6-flash"]["supports_native_json_schema"] is True
    assert metadata["gemini/gemini-3.7-flash"]["supports_native_json_schema"] is True
    assert metadata["openrouter/z-ai/glm-5.2:free"]["supports_native_json_schema"] is True
    assert metadata["openrouter/nvidia/nemotron-3-super-120b-a12b:free"]["supports_native_json_schema"] is True
    assert metadata["openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"]["supports_native_json_schema"] is False
    assert metadata["openrouter/stealth/ox-alpha"]["supports_native_json_schema"] is False
    assert metadata["openrouter/minimax/minimax-m3:free"]["supports_native_json_schema"] is False
    assert metadata["openrouter/google/gemma-4-31b-it:free"]["supports_native_json_schema"] is False
    assert "openrouter/free" not in metadata
    assert "openrouter/auto" not in metadata
    assert "supports_native_json_schema" not in metadata["gemini/gemini-3.5-flash-lite"]


def test_generated_litellm_config_uses_provider_slash_model_and_rate_catalog() -> None:
    routing = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))
    limits = load_model_limits(LIMITS_PATH)
    config = build_litellm_config(
        env={
            "GEMINI_API_KEY_1": "gemini-key",
            "DEEPSEEK_API_KEY_1": "deepseek-key",
            "MISTRAL_API_KEY_1": "mistral-key",
            "OPENROUTER_API_KEY_1": "openrouter-key",
        },
        registry=routing,
        model_limits=limits,
    )

    assert config["router_settings"]["fallbacks"] == [
        {"job-powerful": ["job-balanced"]},
        {"job-balanced": ["job-fast"]},
        {"repair-fast": ["repair-balanced"]},
    ]

    powerful_models = {
        entry["litellm_params"]["model"] for entry in config["model_list"] if entry["model_name"] == "job-powerful"
    }
    assert powerful_models == {"deepseek/deepseek-v4-flash"}

    fast_models = {
        entry["litellm_params"]["model"] for entry in config["model_list"] if entry["model_name"] == "job-fast"
    }
    assert fast_models == {
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "mistral/mistral-medium-latest",
    }

    balanced_models = {
        entry["litellm_params"]["model"] for entry in config["model_list"] if entry["model_name"] == "job-balanced"
    }
    assert balanced_models == {
        "openrouter/z-ai/glm-5.2:free",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/minimax/minimax-m3:free",
        "openrouter/google/gemma-4-31b-it:free",
        "gemini/gemini-3.7-flash",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.5-flash",
    }

    repair_models = {
        entry["litellm_params"]["model"]
        for entry in config["model_list"]
        if entry["model_name"] in {"repair-fast", "repair-balanced"}
    }
    assert not any(model.startswith("openrouter/") for model in repair_models)
    assert all(entry["litellm_params"]["model"] != "openrouter/free" for entry in config["model_list"])

    gemini_limits = {
        entry["litellm_params"]["model"]: (
            entry["litellm_params"]["rpm"],
            entry["litellm_params"]["tpm"],
        )
        for entry in config["model_list"]
        if entry["litellm_params"]["model"].startswith("gemini/")
    }
    assert gemini_limits["gemini/gemini-3.5-flash-lite"] == (15, 250000)
    assert gemini_limits["gemini/gemini-3.1-flash-lite"] == (15, 250000)
    assert gemini_limits["gemini/gemini-3.5-flash"] == (5, 250000)
    assert gemini_limits["gemini/gemini-3.6-flash"] == (5, 250000)
    assert gemini_limits["gemini/gemini-3.7-flash"] == (5, 250000)


def test_generated_deployments_apply_structured_output_capability_per_model() -> None:
    routing = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))
    metadata = load_model_limits(LIMITS_PATH)
    config = build_litellm_config(
        env={
            "GEMINI_API_KEY_1": "gemini-key",
            "DEEPSEEK_API_KEY_1": "deepseek-key",
            "MISTRAL_API_KEY_1": "mistral-key",
            "OPENROUTER_API_KEY_1": "openrouter-key",
        },
        registry=routing,
        model_limits=metadata,
    )

    deployments = {
        entry["litellm_params"]["model"]: entry
        for entry in config["model_list"]
        if entry["model_name"] in {"job-fast", "job-balanced", "job-powerful"}
    }

    deepseek = deployments["deepseek/deepseek-v4-flash"]
    assert "response_format" in deepseek["litellm_params"]["additional_drop_params"]
    assert deepseek["model_info"]["supports_response_schema"] is False

    gemini_35 = deployments["gemini/gemini-3.5-flash"]
    assert "response_format" in gemini_35["litellm_params"]["additional_drop_params"]
    assert gemini_35["model_info"]["supports_response_schema"] is False

    for model in (
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.7-flash",
        "openrouter/z-ai/glm-5.2:free",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    ):
        deployment = deployments[model]
        assert "response_format" not in deployment["litellm_params"].get("additional_drop_params", [])
        assert deployment["model_info"]["supports_response_schema"] is True

    for model in (
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/minimax/minimax-m3:free",
        "openrouter/google/gemma-4-31b-it:free",
    ):
        deployment = deployments[model]
        assert "response_format" in deployment["litellm_params"]["additional_drop_params"]
        assert deployment["model_info"]["supports_response_schema"] is False

    unknown = deployments["gemini/gemini-3.5-flash-lite"]
    assert "response_format" not in unknown["litellm_params"].get("additional_drop_params", [])
    assert "id" in unknown["model_info"]
    assert "supports_response_schema" not in unknown["model_info"]
