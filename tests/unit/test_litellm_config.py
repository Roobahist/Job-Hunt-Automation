from __future__ import annotations

from pathlib import Path

import pytest

from job_hunt.integrations.litellm_config import (
    ConfigGenerationError,
    build_litellm_config,
    classify_model,
    discover_with_key_fallback,
    generate_litellm_config,
    group_models,
    indexed_env_values,
    load_model_limits,
    load_provider_registry,
)


def registry(*providers: dict[str, object]) -> dict[str, object]:
    return {"providers": list(providers)}


def provider(
    name: str,
    key_prefix: str,
    *,
    discovery_url: str | None = None,
    fast: list[str] | None = None,
    balanced: list[str] | None = None,
    powerful: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "api_key_prefix": key_prefix,
        "enabled": True,
        "discovery": {"enabled": discovery_url is not None, "url": discovery_url or ""},
        "models": {"fast": fast or [], "balanced": balanced or [], "powerful": powerful or []},
        "exclude_models": [],
    }


def test_indexed_env_values_orders_numbered_keys() -> None:
    env = {"GROQ_API_KEY_10": "ten", "GROQ_API_KEY_2": "two", "GROQ_API_KEY_1": "one"}
    assert indexed_env_values(env, "GROQ_API_KEY_") == [
        ("GROQ_API_KEY_1", "one"),
        ("GROQ_API_KEY_2", "two"),
        ("GROQ_API_KEY_10", "ten"),
    ]


def test_model_size_heuristic_assigns_capabilities() -> None:
    assert classify_model("openai/gpt-oss-120b") == "job-powerful"
    assert classify_model("qwen/qwen3-32b") == "job-balanced"
    assert classify_model("llama-3.1-8b-instant") == "job-fast"


def test_explicit_classification_and_non_chat_filtering() -> None:
    groups = group_models(
        ["openai/gpt-oss-120b", "whisper-large-v3", "llama-3.1-8b-instant"],
        explicit={"job-fast": ["openai/gpt-oss-120b"]},
    )
    assert "openai/gpt-oss-120b" in groups["job-fast"]
    assert all("whisper" not in model for models in groups.values() for model in models)


def test_specialized_non_generation_models_are_blocked() -> None:
    discovered = [
        "openai/gpt-oss-120b",
        "whisper-large-v3",
        "canopylabs/orpheus-v1-english",
        "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-safeguard-20b",
        "vendor/embedding-3-large",
        "vendor/rerank-v2",
    ]
    groups = group_models(discovered)
    flattened = {model for models in groups.values() for model in models}
    assert "openai/gpt-oss-120b" in flattened
    assert flattened == {"openai/gpt-oss-120b"}


def test_discovery_allowlist_rejects_unapproved_models_and_systems() -> None:
    discovered = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "groq/compound",
        "groq/compound-mini",
        "allam-2-7b",
        "canopylabs/orpheus-v1-english",
    ]
    approved = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
    groups = group_models(discovered, allowlist=approved)
    flattened = {model for models in groups.values() for model in models}
    assert flattened == set(approved)


def test_missing_capability_bucket_borrows_nearest_available_models() -> None:
    groups = group_models(["openai/gpt-oss-120b"])
    assert groups["job-powerful"] == ["openai/gpt-oss-120b"]
    assert groups["job-balanced"] == ["openai/gpt-oss-120b"]
    assert groups["job-fast"] == ["openai/gpt-oss-120b"]


def test_provider_can_disable_missing_group_backfill() -> None:
    groups = group_models(["openai/gpt-oss-120b"], fill_missing=False)
    assert groups["job-powerful"] == ["openai/gpt-oss-120b"]
    assert groups["job-balanced"] == []
    assert groups["job-fast"] == []


def test_excluded_models_never_enter_any_capability_pool() -> None:
    groups = group_models(
        ["openai/gpt-oss-120b", "llama-3.1-8b-instant"],
        excluded=["openai/gpt-oss-120b"],
    )
    assert all("openai/gpt-oss-120b" not in models for models in groups.values())


def test_discovery_falls_through_invalid_keys() -> None:
    calls: list[str] = []

    def discover(key: str) -> list[str]:
        calls.append(key)
        if key == "bad":
            raise RuntimeError("invalid key")
        return ["llama-8b"]

    models = discover_with_key_fallback(
        [("ANY_API_KEY_1", "bad"), ("ANY_API_KEY_2", "good")],
        discover,
    )
    assert models == ["llama-8b"]
    assert calls == ["bad", "good"]


def test_generator_expands_every_key_across_discovered_models() -> None:
    env = {"GROQ_API_KEY_1": "key-one", "GROQ_API_KEY_2": "key-two"}
    config = build_litellm_config(
        env=env,
        registry=registry(
            provider(
                "groq",
                "GROQ_API_KEY_",
                discovery_url="https://example.test/models",
            )
        ),
        discoverer=lambda _key, _url: [
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant",
        ],
    )
    target_model = "groq/openai/gpt-oss-120b"
    powerful = [
        item
        for item in config["model_list"]
        if item["model_name"] == "job-powerful"
        if item["litellm_params"]["model"] == target_model
    ]
    assert len(powerful) == 2
    assert {item["litellm_params"]["api_key"] for item in powerful} == {
        "os.environ/GROQ_API_KEY_1",
        "os.environ/GROQ_API_KEY_2",
    }


def test_provider_registry_allowlist_applies_to_discovery() -> None:
    groq = provider(
        "groq",
        "GROQ_API_KEY_",
        discovery_url="https://example.test/models",
    )
    groq["discovery_allowlist"] = ["openai/gpt-oss-120b"]
    config = build_litellm_config(
        env={"GROQ_API_KEY_1": "key"},
        registry=registry(groq),
        discoverer=lambda _key, _url: [
            "openai/gpt-oss-120b",
            "groq/compound",
            "canopylabs/orpheus-v1-english",
        ],
    )
    models = {item["litellm_params"]["model"] for item in config["model_list"]}
    assert models == {"groq/openai/gpt-oss-120b"}


def test_arbitrary_curated_provider_requires_no_python_adapter() -> None:
    openrouter = provider(
        "openrouter",
        "OPENROUTER_API_KEY_",
        fast=["meta-llama/llama-3.1-8b-instruct:free"],
        balanced=["qwen/qwen3-30b-a3b:free"],
        powerful=["openai/gpt-oss-120b:free"],
    )
    config = build_litellm_config(
        env={"OPENROUTER_API_KEY_1": "key"},
        registry=registry(openrouter),
    )
    models = {item["litellm_params"]["model"] for item in config["model_list"]}
    assert "openrouter/meta-llama/llama-3.1-8b-instruct:free" in models
    assert "openrouter/qwen/qwen3-30b-a3b:free" in models
    assert "openrouter/openai/gpt-oss-120b:free" in models


def test_arbitrary_openai_compatible_provider_uses_real_litellm_provider_name() -> None:
    calls: list[tuple[str, str]] = []

    def discover(key: str, url: str) -> list[str]:
        calls.append((key, url))
        return ["vendor-70b", "vendor-8b"]

    custom_provider = provider(
        "openai",
        "VENDOR_API_KEY_",
        discovery_url="https://vendor.test/v1/models",
    )
    custom_provider["litellm_params"] = {"api_base": "https://vendor.test/v1"}
    config = build_litellm_config(
        env={"VENDOR_API_KEY_1": "secret"},
        registry=registry(custom_provider),
        discoverer=discover,
    )
    assert calls == [("secret", "https://vendor.test/v1/models")]
    models = {item["litellm_params"]["model"] for item in config["model_list"]}
    assert "openai/vendor-70b" in models
    assert "openai/vendor-8b" in models
    api_bases = {item["litellm_params"]["api_base"] for item in config["model_list"]}
    assert api_bases == {"https://vendor.test/v1"}


def test_powerful_only_provider_stays_out_of_lower_capability_groups() -> None:
    mistral = provider(
        "mistral",
        "MISTRAL_API_KEY_",
        powerful=["mistral-large-latest"],
    )
    mistral["fill_missing_groups"] = False
    config = build_litellm_config(
        env={"MISTRAL_API_KEY_1": "key"},
        registry=registry(mistral),
        model_limits={"mistral/mistral-large-latest": {"rpm": 4, "tpm": 250000}},
    )
    mistral_entries = [item for item in config["model_list"] if item["litellm_params"]["model"].startswith("mistral/")]
    assert {item["model_name"] for item in mistral_entries} == {"job-powerful"}
    assert all(item["litellm_params"]["rpm"] == 4 for item in mistral_entries)
    assert all(item["litellm_params"]["tpm"] == 250000 for item in mistral_entries)


def test_provider_without_keys_is_skipped() -> None:
    config = build_litellm_config(
        env={"GEMINI_API_KEY_1": "key"},
        registry=registry(
            provider("unused", "UNUSED_API_KEY_", fast=["unused-8b"]),
            provider("gemini", "GEMINI_API_KEY_", fast=["gemini-lite"]),
        ),
    )
    models = {item["litellm_params"]["model"] for item in config["model_list"]}
    assert all(not model.startswith("unused/") for model in models)
    assert "gemini/gemini-lite" in models


def test_generated_config_contains_generation_and_repair_fallbacks() -> None:
    config = build_litellm_config(
        env={"GROQ_API_KEY_1": "key"},
        registry=registry(
            provider(
                "groq",
                "GROQ_API_KEY_",
                discovery_url="https://example.test/models",
            )
        ),
        discoverer=lambda _key, _url: [
            "openai/gpt-oss-120b",
            "qwen-32b",
            "llama-8b",
        ],
    )
    router = config["router_settings"]
    assert router["routing_strategy"] == "simple-shuffle"
    assert router["enable_weighted_failover"] is True
    assert router["num_retries"] == 0
    assert router["max_fallbacks"] >= 1
    assert router["fallbacks"] == [
        {"job-powerful": ["job-balanced", "job-fast"]},
        {"job-balanced": ["job-fast"]},
        {"repair-fast": ["repair-balanced"]},
    ]


def test_generation_requires_at_least_one_configured_provider_key() -> None:
    groq = provider("groq", "GROQ_API_KEY_", fast=["llama-8b"])
    with pytest.raises(ConfigGenerationError):
        build_litellm_config(env={}, registry=registry(groq))


def test_provider_registry_loads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "providers.json"
    path.write_text('{"providers":[{"name":"test"}]}', encoding="utf-8")
    assert load_provider_registry(path)["providers"][0]["name"] == "test"


def test_model_limits_loads_partial_entries(tmp_path: Path) -> None:
    path = tmp_path / "limits.json"
    path.write_text(
        '{"models":[{"provider":"gemini","model":"flash","rpm":5},{"provider":"deepseek","model":"v4"}]}',
        encoding="utf-8",
    )
    assert load_model_limits(path) == {
        "gemini/flash": {"rpm": 5},
        "deepseek/v4": {},
    }


def test_group_model_rejects_prefixed_model_name() -> None:
    explicit_registry = {
        "groups": {"job-fast": [{"provider": "gemini", "model": "gemini/flash"}]},
        "providers": [{"name": "gemini", "api_key_prefix": "GEMINI_API_KEY_", "enabled": True}],
    }
    with pytest.raises(ConfigGenerationError, match="must omit its provider prefix"):
        build_litellm_config(env={"GEMINI_API_KEY_1": "key"}, registry=explicit_registry)


def test_high_level_generator_writes_runtime_config(tmp_path: Path) -> None:
    destination = tmp_path / "litellm.runtime.yaml"
    custom = provider("custom", "CUSTOM_API_KEY_", fast=["model-8b"])
    config = generate_litellm_config(
        env={"CUSTOM_API_KEY_1": "key"},
        registry=registry(custom),
        destination=destination,
    )
    assert destination.exists()
    assert '"job-fast"' in destination.read_text()
    assert config["model_list"]
