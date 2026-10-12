from pathlib import Path

from job_hunt.integrations.apify_actors import (
    INDEED_SINGLE_JOB_ACTOR,
    LINKEDIN_COREGENT_SEARCH_ACTOR,
    LINKEDIN_LEGACY_SEARCH_ACTOR,
    LINKEDIN_SINGLE_JOB_ACTOR,
)


def test_production_apify_actors_use_one_code_owned_reference_format() -> None:
    actors = {
        "linkedin_legacy_search": LINKEDIN_LEGACY_SEARCH_ACTOR,
        "linkedin_coregent_search": LINKEDIN_COREGENT_SEARCH_ACTOR,
        "linkedin_single_job": LINKEDIN_SINGLE_JOB_ACTOR,
        "indeed_single_job": INDEED_SINGLE_JOB_ACTOR,
    }

    assert actors == {
        "linkedin_legacy_search": "cheap_scraper/linkedin-job-scraper",
        "linkedin_coregent_search": "coregent/linkedin-jobs-scraper",
        "linkedin_single_job": "apt_marble/linkedin-job-listings-scraper-by-url",
        "indeed_single_job": "kaix/indeed-scraper",
    }
    assert all(value.count("/") == 1 for value in actors.values())
    assert all(all(part.strip() for part in value.split("/")) for value in actors.values())


def test_tenant_seed_files_do_not_expose_apify_actor_configuration() -> None:
    for seed in (Path("config/seeds/nova.csv"), Path("config/seeds/atlas.csv")):
        assert "apify_actor_ids" not in seed.read_text(encoding="utf-8")
