from __future__ import annotations

from job_hunt.integrations.litellm_config import build_litellm_config


def test_repair_only_model_does_not_enter_generation_groups() -> None:
    registry = {
        "providers": [
            {
                "name": "gemini",
                "api_key_prefix": "GEMINI_API_KEY_",
                "enabled": True,
                "fill_missing_groups": False,
                "discovery": {"enabled": False},
                "models": {
                    "fast": ["gemini-lite"],
                    "balanced": ["gemini-flash"],
                    "powerful": [],
                },
                "exclude_models": [],
            },
            {
                "name": "mistral",
                "api_key_prefix": "MISTRAL_API_KEY_",
                "enabled": True,
                "fill_missing_groups": False,
                "discovery": {"enabled": False},
                "models": {"fast": [], "balanced": [], "powerful": []},
                "repair_models": {
                    "fast": ["mistral-small-latest"],
                    "balanced": ["mistral-small-latest"],
                },
                "litellm_params": {"rpm": 50, "tpm": 50000},
                "exclude_models": [],
            },
        ]
    }

    config = build_litellm_config(
        env={
            "GEMINI_API_KEY_1": "gemini-key",
            "MISTRAL_API_KEY_1": "mistral-key",
        },
        registry=registry,
    )

    mistral_entries = [
        entry for entry in config["model_list"] if entry["litellm_params"]["model"] == "mistral/mistral-small-latest"
    ]

    assert len(mistral_entries) == 2
    assert {entry["model_name"] for entry in mistral_entries} == {
        "repair-fast",
        "repair-balanced",
    }
    assert all(entry["litellm_params"]["rpm"] == 50 for entry in mistral_entries)
    assert all(entry["litellm_params"]["tpm"] == 50000 for entry in mistral_entries)

    generation_models = {
        entry["litellm_params"]["model"] for entry in config["model_list"] if entry["model_name"].startswith("job-")
    }
    assert "mistral/mistral-small-latest" not in generation_models

    assert {"repair-fast": ["repair-balanced"]} in config["router_settings"]["fallbacks"]
