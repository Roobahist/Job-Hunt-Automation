from __future__ import annotations

import pytest

from job_hunt.integrations.litellm_config import ConfigGenerationError, build_litellm_config


def _registry() -> dict[str, object]:
    return {
        "providers": [
            {
                "name": "test",
                "api_key_prefix": "TEST_API_KEY_",
                "enabled": True,
                "discovery": {"enabled": False},
                "models": {
                    "fast": ["fast-model"],
                    "balanced": ["balanced-model"],
                    "powerful": ["powerful-model"],
                },
                "exclude_models": [],
            }
        ]
    }


def _multi_provider_registry() -> dict[str, object]:
    return {
        "providers": [
            {
                "name": "alpha",
                "api_key_prefix": "ALPHA_API_KEY_",
                "enabled": True,
                "discovery": {"enabled": False},
                "models": {"fast": ["alpha-fast"], "balanced": [], "powerful": []},
                "exclude_models": [],
            },
            {
                "name": "beta",
                "api_key_prefix": "BETA_API_KEY_",
                "enabled": True,
                "discovery": {"enabled": False},
                "models": {"fast": ["beta-fast"], "balanced": [], "powerful": []},
                "exclude_models": [],
            },
        ]
    }


def test_router_uses_failover_instead_of_same_deployment_retries() -> None:
    config = build_litellm_config(
        env={"TEST_API_KEY_1": "one", "TEST_API_KEY_2": "two", "TEST_API_KEY_3": "three"},
        registry=_registry(),
    )
    router = config["router_settings"]
    assert router["routing_strategy"] == "simple-shuffle"
    assert router["enable_weighted_failover"] is True
    assert router["num_retries"] == 0
    assert router["allowed_fails"] == 0
    assert router["cooldown_time"] == 65


def test_each_provider_key_is_a_separate_litellm_deployment() -> None:
    config = build_litellm_config(
        env={"TEST_API_KEY_1": "one", "TEST_API_KEY_2": "two", "TEST_API_KEY_3": "three"},
        registry=_registry(),
    )
    job_fast = [entry for entry in config["model_list"] if entry["model_name"] == "job-fast"]
    assert len(job_fast) == 3
    assert {entry["litellm_params"]["api_key"] for entry in job_fast} == {
        "os.environ/TEST_API_KEY_1",
        "os.environ/TEST_API_KEY_2",
        "os.environ/TEST_API_KEY_3",
    }


def test_models_and_keys_from_multiple_providers_share_the_capability_pool() -> None:
    config = build_litellm_config(
        env={
            "ALPHA_API_KEY_1": "a1",
            "ALPHA_API_KEY_2": "a2",
            "BETA_API_KEY_1": "b1",
            "BETA_API_KEY_2": "b2",
            "BETA_API_KEY_3": "b3",
        },
        registry=_multi_provider_registry(),
    )
    job_fast = [entry for entry in config["model_list"] if entry["model_name"] == "job-fast"]
    assert len(job_fast) == 5
    assert {entry["litellm_params"]["model"] for entry in job_fast} == {
        "alpha/alpha-fast",
        "beta/beta-fast",
    }
    assert {entry["litellm_params"]["api_key"] for entry in job_fast} == {
        "os.environ/ALPHA_API_KEY_1",
        "os.environ/ALPHA_API_KEY_2",
        "os.environ/BETA_API_KEY_1",
        "os.environ/BETA_API_KEY_2",
        "os.environ/BETA_API_KEY_3",
    }


def test_failover_budget_scales_with_every_generated_deployment() -> None:
    config = build_litellm_config(
        env={
            "ALPHA_API_KEY_1": "a1",
            "ALPHA_API_KEY_2": "a2",
            "BETA_API_KEY_1": "b1",
            "BETA_API_KEY_2": "b2",
            "BETA_API_KEY_3": "b3",
        },
        registry=_multi_provider_registry(),
    )
    assert config["router_settings"]["max_fallbacks"] == len(config["model_list"]) - 1
    assert config["router_settings"]["max_fallbacks"] > 5


def test_capability_fallback_graph_is_delegated_to_litellm() -> None:
    config = build_litellm_config(
        env={"TEST_API_KEY_1": "one"},
        registry=_registry(),
    )
    assert config["router_settings"]["fallbacks"] == [
        {"job-powerful": ["job-balanced", "job-fast"]},
        {"job-balanced": ["job-fast"]},
        {"repair-fast": ["repair-balanced"]},
    ]


def test_reliability_critical_router_behavior_cannot_be_disabled_by_old_env_values() -> None:
    config = build_litellm_config(
        env={
            "TEST_API_KEY_1": "one",
            "LITELLM_NUM_RETRIES": "7",
            "LITELLM_ROUTING_STRATEGY": "latency-based-routing",
            "LITELLM_ALLOWED_FAILS": "2",
            "LITELLM_COOLDOWN_SECONDS": "90",
        },
        registry=_registry(),
    )
    router = config["router_settings"]
    assert router["num_retries"] == 0
    assert router["routing_strategy"] == "simple-shuffle"
    assert router["enable_weighted_failover"] is True
    assert router["allowed_fails"] == 2
    assert router["cooldown_time"] == 90


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LITELLM_ALLOWED_FAILS", "-1"),
        ("LITELLM_COOLDOWN_SECONDS", "-1"),
        ("LITELLM_ALLOWED_FAILS", "bad"),
    ],
)
def test_invalid_router_tuning_fails_config_generation(name: str, value: str) -> None:
    with pytest.raises(ConfigGenerationError, match=name):
        build_litellm_config(
            env={"TEST_API_KEY_1": "one", name: value},
            registry=_registry(),
        )
