from __future__ import annotations

import pytest

from job_hunt.config import Settings
from job_hunt.errors import ConfigurationError


def test_discovery_order_defaults_to_roobahist_then_existing_fallbacks() -> None:
    settings = Settings(_env_file=None)
    assert settings.discovery_strategies() == ["roobahist", "brightdata", "coregent", "legacy"]


def test_discovery_order_can_be_changed_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_DISCOVERY_ORDER", "legacy,roobahist,brightdata,coregent")
    settings = Settings(_env_file=None)
    assert settings.discovery_strategies() == ["legacy", "roobahist", "brightdata", "coregent"]


def test_discovery_order_rejects_duplicates_and_unknown_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_DISCOVERY_ORDER", "roobahist,roobahist")
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigurationError, match="duplicate"):
        settings.discovery_strategies()

    monkeypatch.setenv("JOB_HUNT_DISCOVERY_ORDER", "roobahist,unknown")
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigurationError, match="unknown"):
        settings.discovery_strategies()


def test_linkedin_job_fetch_order_defaults_to_roobahist_then_existing_fallbacks() -> None:
    settings = Settings(_env_file=None)
    assert settings.linkedin_job_fetch_strategies() == ["roobahist", "apify", "brightdata"]


def test_linkedin_job_fetch_order_can_be_reversed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_LINKEDIN_JOB_FETCH_ORDER", "brightdata,apify,roobahist")
    settings = Settings(_env_file=None)
    assert settings.linkedin_job_fetch_strategies() == ["brightdata", "apify", "roobahist"]


def test_linkedin_job_fetch_order_rejects_duplicates_and_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_LINKEDIN_JOB_FETCH_ORDER", "roobahist,roobahist")
    with pytest.raises(ConfigurationError, match="duplicate"):
        Settings(_env_file=None).linkedin_job_fetch_strategies()

    monkeypatch.setenv("JOB_HUNT_LINKEDIN_JOB_FETCH_ORDER", "roobahist,unknown")
    with pytest.raises(ConfigurationError, match="unknown"):
        Settings(_env_file=None).linkedin_job_fetch_strategies()


def test_brightdata_api_keys_are_optional_and_csv_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).shared_brightdata_api_keys() == []

    monkeypatch.setenv("JOB_HUNT_BRIGHTDATA_API_KEYS", " key-one, key-two ")
    settings = Settings(_env_file=None)
    assert settings.shared_brightdata_api_keys() == ["key-one", "key-two"]


def test_brightdata_dataset_ids_must_be_explicitly_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigurationError, match="BRIGHTDATA_LINKEDIN_JOBS_DATASET_ID"):
        settings.shared_brightdata_linkedin_jobs_dataset_id()
    with pytest.raises(ConfigurationError, match="BRIGHTDATA_INDEED_JOBS_DATASET_ID"):
        settings.shared_brightdata_indeed_jobs_dataset_id()

    monkeypatch.setenv("JOB_HUNT_BRIGHTDATA_LINKEDIN_JOBS_DATASET_ID", " gd_lpfll7v5hcqtkxl6l ")
    monkeypatch.setenv("JOB_HUNT_BRIGHTDATA_INDEED_JOBS_DATASET_ID", " gd_l4dx9j9sscpvs7no2 ")
    settings = Settings(_env_file=None)
    assert settings.shared_brightdata_linkedin_jobs_dataset_id() == "gd_lpfll7v5hcqtkxl6l"
    assert settings.shared_brightdata_indeed_jobs_dataset_id() == "gd_l4dx9j9sscpvs7no2"


def test_apify_discovery_order_defaults_to_new_actor_then_legacy() -> None:
    settings = Settings(_env_file=None)
    assert settings.apify_discovery_strategies() == ["coregent", "legacy"]


def test_apify_discovery_order_can_be_reversed_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_APIFY_DISCOVERY_ORDER", "legacy,coregent")
    settings = Settings(_env_file=None)
    assert settings.apify_discovery_strategies() == ["legacy", "coregent"]


def test_apify_discovery_order_rejects_duplicate_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_APIFY_DISCOVERY_ORDER", "coregent,coregent")
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigurationError, match="duplicate"):
        settings.apify_discovery_strategies()
