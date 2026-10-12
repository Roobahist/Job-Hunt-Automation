from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_hunt.config import (
    LINKEDIN_BASE_SEARCH_URL,
    LINKEDIN_JOB_URL_TEMPLATE,
    Settings,
    decode_config_value,
    load_registry,
    parse_configuration_rows,
    read_seed,
)
from job_hunt.errors import ConfigurationError


def minimum_rows() -> list[dict[str, object]]:
    values: dict[str, tuple[str, object]] = {
        "tenant_key": ("text", "test"),
        "applicant_filename": ("text", "Test"),
        "baserow_table_ids": ("json", {"jobs": 1, "searchCriteria": 2, "prompts": 3}),
        "status_option_ids": ("json", {"new": 1, "dropped": 2}),
        "telegram_chat_id": ("text", "1"),
        "gemini_model": ("text", "legacy-row-is-ignored"),
    }
    return [
        {
            "configKey": key,
            "valueType": kind,
            "value": json.dumps(value) if kind == "json" else value,
            "enabled": True,
        }
        for key, (kind, value) in values.items()
    ]


def test_decode_and_parse_configuration() -> None:
    assert decode_config_value("boolean", "YES") is True
    assert decode_config_value("number", "3") == 3
    assert decode_config_value("number", "3.5") == 3.5
    assert decode_config_value("json", "") == []
    assert decode_config_value("json", "   ") == []
    config = parse_configuration_rows(minimum_rows())
    assert config.baserow_table_ids["jobs"] == 1
    assert config.qualification_threshold == 33
    assert config.company_inclusions == []
    assert config.company_exclusions == []
    assert config.title_inclusions == []
    assert config.title_exclusions == []
    assert config.description_inclusions == []
    assert config.description_exclusions == []
    assert config.linkedin_base_search_url == LINKEDIN_BASE_SEARCH_URL
    assert config.linkedin_job_url_template == LINKEDIN_JOB_URL_TEMPLATE
    assert "linkedin_base_search_url" not in config.__class__.model_fields
    assert "linkedin_job_url_template" not in config.__class__.model_fields
    assert not hasattr(config, "gemini_model")
    assert "apify_actor_ids" not in config.__class__.model_fields
    assert "fillout_form_id" not in config.__class__.model_fields
    assert "fillout_field_ids" not in config.__class__.model_fields
    assert config.fillout_field_ids["entryType"] == "entryType"
    assert config.fillout_field_ids["indeedJobId"] == "demoField2d374ace"


def test_blank_json_configuration_value_defaults_to_empty_list() -> None:
    rows = minimum_rows()
    rows.extend(
        [
            {
                "configKey": "company_inclusions",
                "valueType": "json",
                "value": "",
                "enabled": True,
            },
            {
                "configKey": "description_exclusions",
                "valueType": "json",
                "value": "   ",
                "enabled": True,
            },
        ]
    )

    config = parse_configuration_rows(rows)

    assert config.company_inclusions == []
    assert config.description_exclusions == []


def test_baserow_cannot_override_code_owned_linkedin_urls() -> None:
    rows = minimum_rows()
    rows.extend(
        [
            {
                "configKey": "linkedin_base_search_url",
                "valueType": "text",
                "value": "https://example.invalid/search",
                "enabled": True,
            },
            {
                "configKey": "linkedin_job_url_template",
                "valueType": "text",
                "value": "https://example.invalid/jobs/{jobId}",
                "enabled": True,
            },
        ]
    )

    config = parse_configuration_rows(rows)

    assert config.linkedin_base_search_url == "https://www.linkedin.com/jobs/search/"
    assert config.linkedin_job_url_template == "https://www.linkedin.com/jobs/view/{jobId}"


def test_stale_baserow_actor_ids_are_ignored() -> None:
    rows = minimum_rows()
    rows.append(
        {
            "configKey": "apify_actor_ids",
            "valueType": "json",
            "value": '{"linkedinSearch":"user-controlled/actor"}',
            "enabled": True,
        }
    )

    config = parse_configuration_rows(rows)

    assert "apify_actor_ids" not in config.__class__.model_fields
    assert not hasattr(config, "apify_actor_ids")


def test_shared_settings_are_ordered_and_required() -> None:
    settings = Settings(
        apify_tokens="a1, a2,a3",
        litellm_api_key="llm",
        telegram_bot_token="bot",
        telegram_webhook_secret="webhook",
    )
    assert settings.shared_apify_tokens() == ["a1", "a2", "a3"]
    assert settings.shared_litellm_key() == "llm"
    assert settings.shared_telegram_token() == "bot"
    assert settings.shared_telegram_webhook_secret() == "webhook"

    with pytest.raises(ConfigurationError, match="APIFY_TOKENS"):
        Settings(apify_tokens="").shared_apify_tokens()
    with pytest.raises(ConfigurationError, match="LITELLM_API_KEY"):
        Settings(litellm_api_key="").shared_litellm_key()
    with pytest.raises(ConfigurationError, match="TELEGRAM_BOT_TOKEN"):
        Settings(telegram_bot_token="").shared_telegram_token()
    with pytest.raises(ConfigurationError, match="TELEGRAM_WEBHOOK_SECRET"):
        Settings(telegram_webhook_secret="").shared_telegram_webhook_secret()


def test_operational_limits_are_configurable() -> None:
    settings = Settings(
        litellm_request_timeout_seconds=240,
        telegram_request_timeout_seconds=150,
        baserow_request_timeout_seconds=90,
        latex_compile_timeout_seconds=210,
        job_lock_timeout_seconds=2400,
        task_soft_time_limit_seconds=0,
        task_time_limit_seconds=0,
        task_max_retries=11,
        rate_limit_fallback_seconds=75,
        transient_base_delay_seconds=7,
        transient_max_delay_seconds=420,
        provider_quota_cooldown_seconds=3601,
    )
    assert settings.litellm_request_timeout_seconds == 240
    assert settings.telegram_request_timeout_seconds == 150
    assert settings.baserow_request_timeout_seconds == 90
    assert settings.latex_compile_timeout_seconds == 210
    assert settings.job_lock_timeout_seconds == 2400
    assert settings.task_soft_time_limit_seconds == 0
    assert settings.task_time_limit_seconds == 0
    assert settings.task_max_retries == 11
    assert settings.rate_limit_fallback_seconds == 75
    assert settings.transient_base_delay_seconds == 7
    assert settings.transient_max_delay_seconds == 420
    assert settings.provider_quota_cooldown_seconds == 3601
    assert Settings().litellm_request_timeout_seconds == 360
    assert Settings().provider_quota_cooldown_seconds == 3600


def test_invalid_configuration_is_actionable() -> None:
    with pytest.raises(ConfigurationError, match="Invalid tenant configuration"):
        parse_configuration_rows([])


def test_registry_and_real_seed_files(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = load_registry(Path("config/users.toml"))
    assert set(registry) == {"atlas", "nova"}
    monkeypatch.setenv("ATLAS_BASEROW_TOKEN", "secret")
    assert registry["atlas"].secret("baserow") == "secret"
    assert read_seed(Path("config/seeds/atlas.csv")).tenant_key == "atlas_candidate"
    assert read_seed(Path("config/seeds/nova.csv")).tenant_key == "nova_candidate"
