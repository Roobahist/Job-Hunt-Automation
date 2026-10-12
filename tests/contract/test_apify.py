from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from job_hunt.errors import ProviderError
from job_hunt.integrations.apify import ApifyProvider


def client_with_items(items: list[dict[str, object]], *, run_id: str = "run") -> MagicMock:
    client = MagicMock()
    client.actor.return_value.call.return_value = {"id": run_id, "defaultDatasetId": "dataset"}
    client.dataset.return_value.iterate_items.return_value = iter(items)
    client.log.return_value.get.return_value = ""
    return client


def test_apify_waits_for_actor_and_iterates_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    ApifyProvider._unavailable_until.clear()
    client = client_with_items([{"id": 1}, {"id": 2}])
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _: client)
    provider = ApifyProvider(
        ["token"],
        "search",
        "single",
        company_exclusions=["Randstad", "Adecco"],
        title_exclusions=["senior", "manager"],
    )
    assert list(provider.discover(["https://search"], max_items=2)) == [{"id": 1}, {"id": 2}]
    client.actor.assert_called_once_with("search")
    client.actor.return_value.call.assert_called_once_with(
        run_input={
            "startUrls": [{"url": "https://search"}],
            "maxItems": 2,
            "saveOnlyUniqueItems": True,
            "companyExclude": ["Randstad", "Adecco"],
            "jobTitleExclude": ["senior", "manager"],
        },
        max_items=2,
    )
    client.dataset.assert_called_once_with("dataset")
    client.log.assert_not_called()


def test_roobahist_run_passes_unlimited_timeout_and_price_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    ApifyProvider._unavailable_until.clear()
    client = client_with_items([{"id": 1}])
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _: client)
    provider = ApifyProvider(["token"], "search", "single")
    run_input = {"searchUrls": [{"url": "https://search"}]}

    assert list(
        provider._run(
            "munificent_geastrum/apify-linkedin",
            run_input,
            timeout_secs=0,
            max_total_charge_usd=0.1,
        )
    ) == [{"id": 1}]

    client.actor.assert_called_once_with("munificent_geastrum/apify-linkedin")
    client.actor.return_value.call.assert_called_once_with(
        run_input=run_input,
        max_items=None,
        timeout_secs=0,
        max_total_charge_usd=Decimal("0.1"),
    )


def test_apify_rotates_to_next_shared_token_on_capacity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ApifyProvider._unavailable_until.clear()
    exhausted = MagicMock()
    exhausted.actor.return_value.call.side_effect = RuntimeError("monthly usage limit exceeded")
    available = client_with_items([{"id": 9}])
    clients = {"first": exhausted, "second": available}
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda token: clients[token])

    provider = ApifyProvider(["first", "second"], "search", "single")
    assert list(provider.discover(["https://search"], max_items=1)) == [{"id": 9}]
    exhausted.actor.assert_called_once_with("search")
    available.actor.assert_called_once_with("search")


def test_apify_rotates_when_actor_exits_cleanly_for_charge_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ApifyProvider._unavailable_until.clear()
    exhausted = client_with_items([], run_id="exhausted-run")
    exhausted.log.return_value.get.return_value = (
        "Your Maximum charged results is set to 96 and the minimum is 150. "
        "Please set the Maximum charged results parameter to at least 150."
    )
    available = client_with_items([{"id": 9}], run_id="available-run")
    clients = {"first": exhausted, "second": available}
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda token: clients[token])

    provider = ApifyProvider(["first", "second"], "search", "single")
    assert list(provider.discover(["https://search"], max_items=150)) == [{"id": 9}]
    exhausted.log.assert_called_once_with("exhausted-run")
    available.actor.assert_called_once_with("search")
    assert provider._token_id("first") not in ApifyProvider._unavailable_until


def test_apify_minimum_charge_exhaustion_checks_two_full_loops_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ApifyProvider._unavailable_until.clear()
    first = client_with_items([], run_id="first-run")
    second = client_with_items([], run_id="second-run")
    charge_log = "Your Maximum charged results is set below the minimum is 150."
    first.log.return_value.get.return_value = charge_log
    second.log.return_value.get.return_value = charge_log
    clients = {"first": first, "second": second}
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda token: clients[token])

    provider = ApifyProvider(["first", "second"], "search", "single")
    with pytest.raises(ProviderError, match="Apify is out of access") as caught:
        list(provider.discover(["https://search"], max_items=150))

    assert caught.value.retryable is False
    assert first.actor.return_value.call.call_count == 2
    assert second.actor.return_value.call.call_count == 2
    assert ApifyProvider._unavailable_until == {}


def test_apify_empty_search_without_capacity_log_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ApifyProvider._unavailable_until.clear()
    client = client_with_items([], run_id="empty-run")
    client.log.return_value.get.return_value = "Finished successfully. No jobs matched the search."
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _: client)

    provider = ApifyProvider(["token"], "search", "single")
    assert list(provider.discover(["https://search"], max_items=150)) == []
    client.log.assert_called_once_with("empty-run")


def test_apify_fetches_one_linkedin_job_with_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    ApifyProvider._unavailable_until.clear()
    client = client_with_items([{"id": 9}])
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _: client)
    provider = ApifyProvider(["token"], "search", "single")
    assert provider.fetch_linkedin(9, country="ca", max_concurrency=2) == {"id": 9}
    run_input = client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["proxy"]["apifyProxyCountry"] == "CA"


def test_apify_empty_single_job_is_business_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ApifyProvider._unavailable_until.clear()
    client = client_with_items([])
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _: client)
    provider = ApifyProvider(["token"], "search", "single")
    with pytest.raises(ProviderError, match="not found"):
        provider.fetch_linkedin(9, country="ca", max_concurrency=1)


def test_apify_missing_dataset_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    ApifyProvider._unavailable_until.clear()
    client = MagicMock()
    client.actor.return_value.call.return_value = {}
    monkeypatch.setattr("job_hunt.integrations.apify.ApifyClient", lambda _: client)
    provider = ApifyProvider(["token"], "search", "single")
    with pytest.raises(ProviderError, match="no dataset"):
        list(provider.discover(["https://search"], max_items=1))
