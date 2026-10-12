from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from job_hunt.integrations.apify import ApifyProvider


def _client_with_actor_results(results: dict[str, object]) -> tuple[MagicMock, dict[str, MagicMock]]:
    client = MagicMock()
    actors: dict[str, MagicMock] = {}
    for actor_id, result in results.items():
        actor = MagicMock()
        if isinstance(result, Exception):
            actor.call.side_effect = result
        else:
            actor.call.return_value = {"id": f"run-{actor_id}", "defaultDatasetId": f"dataset-{actor_id}"}
            client.dataset(f"dataset-{actor_id}").iterate_items.return_value = iter(result)  # type: ignore[arg-type]
        actors[actor_id] = actor
    client.actor.side_effect = lambda actor_id: actors[actor_id]
    client.log.return_value.get.return_value = ""
    return client, actors


def test_coregent_discovery_uses_only_generated_urls_and_requested_actor_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ApifyProvider._unavailable_until.clear()
    client, actors = _client_with_actor_results({"coregent-actor": [{"jobId": "4450774820"}]})
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _token: client)

    provider = ApifyProvider(
        ["token"],
        "legacy-actor",
        "single-actor",
        search_actor_ids={"coregent": "coregent-actor"},
        discovery_order=("coregent", "legacy"),
        skip_job_ids_provider=lambda: ["4440000001", "4440000002"],
        company_exclusions=["Randstad", "Adecco"],
        title_exclusions=["senior", "manager"],
    )

    rows = list(
        provider.discover(
            [
                "https://www.linkedin.com/jobs/search/?keywords=software+engineer&location=canada&f_TPR=r3600",
            ],
            max_items=20,
        )
    )

    assert rows == [{"jobId": "4450774820"}]
    client.actor.assert_called_once_with("coregent-actor")
    run_input = actors["coregent-actor"].call.call_args.kwargs["run_input"]
    assert run_input == {
        "easyApplyOnly": False,
        "excludeJobsWithoutSalary": False,
        "excludeRecruitmentAgencies": False,
        "expandLargeSearches": False,
        "extractPublicEmails": False,
        "includeCompanyDetails": False,
        "includeRecruiterDetails": False,
        "keepUnconfirmedWorkplace": False,
        "maxResults": 20,
        "proxyConfiguration": {"useApifyProxy": True},
        "removeDuplicates": True,
        "skipJobIds": ["4440000001", "4440000002"],
        "searchUrls": [
            "https://www.linkedin.com/jobs/search/?keywords=software+engineer&location=canada&f_TPR=r3600",
        ],
        "underTenApplicantsOnly": False,
        "excludeCompanies": ["Randstad", "Adecco"],
        "excludeJobTitleKeywords": ["senior", "manager"],
    }


def test_discovery_falls_back_to_legacy_actor_when_primary_actor_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ApifyProvider._unavailable_until.clear()
    client, _actors = _client_with_actor_results(
        {
            "coregent-actor": RuntimeError("actor unavailable"),
            "legacy-actor": [{"id": 9}],
        }
    )
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _token: client)

    provider = ApifyProvider(
        ["token"],
        "legacy-actor",
        "single-actor",
        search_actor_ids={"coregent": "coregent-actor"},
        discovery_order=("coregent", "legacy"),
    )

    assert list(
        provider.discover(
            ["https://www.linkedin.com/jobs/search/?f_TPR=r3600"],
            max_items=5,
        )
    ) == [{"id": 9}]
    assert client.actor.call_args_list == [call("coregent-actor"), call("legacy-actor")]


def test_missing_primary_actor_configuration_keeps_legacy_available(monkeypatch: pytest.MonkeyPatch) -> None:
    ApifyProvider._unavailable_until.clear()
    client, _actors = _client_with_actor_results({"legacy-actor": [{"id": 9}]})
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _token: client)

    provider = ApifyProvider(
        ["token"],
        "legacy-actor",
        "single-actor",
        discovery_order=("coregent", "legacy"),
    )

    assert list(
        provider.discover(
            ["https://www.linkedin.com/jobs/search/?f_TPR=r3600"],
            max_items=5,
        )
    ) == [{"id": 9}]
    client.actor.assert_called_once_with("legacy-actor")
