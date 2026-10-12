from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from job_hunt.errors import ProviderError
from job_hunt.integrations.apify_linkedin_roobahist import RoobahistLinkedInProvider


class FakeRunner:
    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.calls: list[tuple[str, dict[str, Any], int | None, int | None, float | None]] = []

    def _run(
        self,
        actor_id: str,
        run_input: Mapping[str, Any],
        max_items: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
    ) -> Iterable[dict[str, Any]]:
        self.calls.append((actor_id, dict(run_input), max_items, timeout_secs, max_total_charge_usd))
        return list(self.rows)


def _actor_row(job_id: str = "4450774820") -> dict[str, Any]:
    return {
        "jobId": job_id,
        "jobUrl": f"https://www.linkedin.com/jobs/view/{job_id}",
        "title": "Data Scientist",
        "companyName": "Example Inc",
        "location": "Calgary, AB",
        "postedAt": "2026-09-05T20:00:00.000Z",
        "jobDescription": "Build production machine-learning systems.",
        "jobEmploymentType": "Full-time",
    }


def test_discovery_uses_search_urls_actor_filters_and_only_required_fields() -> None:
    runner = FakeRunner([_actor_row()])
    provider = RoobahistLinkedInProvider(
        runner,
        "roobahist/apify-linkedin",
        company_inclusions=["Example Inc", "OpenAI"],
        company_exclusions=["Randstad", "Adecco"],
        title_inclusions=["data", "machine learning"],
        title_exclusions=["senior", "manager"],
        description_inclusions=["python", "machine learning"],
        description_exclusions=["security clearance", "french"],
    )
    search_url = "https://www.linkedin.com/jobs/search/?keywords=data&location=Canada&f_TPR=r3600"

    rows = provider.discover(
        [search_url],
        max_items=25,
        skip_job_ids=["4440000001", "4440000001", "4440000002"],
    )

    assert rows == [
        {
            "id": "4450774820",
            "url": "https://www.linkedin.com/jobs/view/4450774820",
            "companyName": "Example Inc",
            "title": "Data Scientist",
            "description": "Build production machine-learning systems.",
            "location": "Calgary, AB",
            "employmentType": "Full-time",
            "postedAt": "2026-09-05T20:00:00.000Z",
        }
    ]
    assert len(runner.calls) == 1
    actor_id, run_input, platform_max_items, timeout_secs, max_total_charge_usd = runner.calls[0]
    assert actor_id == "roobahist/apify-linkedin"
    assert platform_max_items is None
    assert timeout_secs == 900
    assert max_total_charge_usd is None
    assert "accessSecret" not in run_input
    assert run_input["searchUrls"] == [{"url": search_url, "inferGeoId": True}]
    assert "inferGeoIdFromSearchUrls" not in run_input
    assert run_input["enrichJobInfo"] is True
    assert run_input["enrichCompanyInfo"] is False
    assert run_input["jobDetailConcurrency"] == 3
    assert run_input["removeDuplicates"] is True
    assert run_input["maxResults"] == 25
    assert run_input["skipJobIds"] == ["4440000001", "4440000002"]
    assert run_input["includeCompanyNames"] == ["Example Inc", "OpenAI"]
    assert run_input["excludeCompanyNames"] == ["Randstad", "Adecco"]
    assert run_input["includeJobTitleKeywords"] == ["data", "machine learning"]
    assert run_input["excludeJobTitleKeywords"] == ["senior", "manager"]
    assert run_input["includeJobDescriptionKeywords"] == ["python", "machine learning"]
    assert run_input["excludeJobDescriptionKeywords"] == ["security clearance", "french"]
    assert "skipCompanyNames" not in run_input
    assert "skipJobTitleKeywords" not in run_input
    assert {name for name, enabled in run_input["searchFields"].items() if enabled} == {
        "jobId",
        "jobUrl",
        "title",
        "companyName",
        "location",
        "postedAt",
    }
    assert {name for name, enabled in run_input["jobFields"].items() if enabled} == {
        "jobDescription",
        "jobEmploymentType",
    }


def test_discovery_can_use_end_of_day_run_limits() -> None:
    runner = FakeRunner([_actor_row()])
    provider = RoobahistLinkedInProvider(
        runner,
        "roobahist/apify-linkedin",
        run_timeout_seconds=0,
        run_max_total_charge_usd=0.1,
    )

    provider.discover(["https://www.linkedin.com/jobs/search/?keywords=data"], max_items=10)

    assert runner.calls[0][3] == 0
    assert runner.calls[0][4] == 0.1
    assert runner.calls[0][1]["jobDetailConcurrency"] == 3


def test_direct_fetch_uses_direct_job_ids_and_preserves_manual_submission_semantics() -> None:
    runner = FakeRunner([_actor_row("4450774821")])
    provider = RoobahistLinkedInProvider(
        runner,
        "roobahist/apify-linkedin",
        company_inclusions=["Another Company"],
        company_exclusions=["Example Inc"],
        title_inclusions=["designer"],
        title_exclusions=["data"],
        description_inclusions=["design systems"],
        description_exclusions=["machine learning"],
    )

    row = provider.fetch_linkedin(4450774821, country="ca", max_concurrency=4)

    assert row["id"] == "4450774821"
    assert row["description"] == "Build production machine-learning systems."
    _, run_input, platform_max_items, timeout_secs, max_total_charge_usd = runner.calls[0]
    assert platform_max_items is None
    assert timeout_secs == 900
    assert max_total_charge_usd is None
    assert run_input["jobIds"] == ["4450774821"]
    assert run_input["skipJobIds"] == []
    assert run_input["includeCompanyNames"] == []
    assert run_input["excludeCompanyNames"] == []
    assert run_input["includeJobTitleKeywords"] == []
    assert run_input["excludeJobTitleKeywords"] == []
    assert run_input["includeJobDescriptionKeywords"] == []
    assert run_input["excludeJobDescriptionKeywords"] == []
    assert "skipCompanyNames" not in run_input
    assert "skipJobTitleKeywords" not in run_input
    assert run_input["maxResults"] == 1
    assert run_input["jobDetailConcurrency"] == 3
    assert run_input["removeDuplicates"] is True
    assert run_input["enrichJobInfo"] is True
    assert run_input["enrichCompanyInfo"] is False


def test_discovery_skips_rows_that_cannot_satisfy_canonical_job_contract() -> None:
    incomplete = _actor_row()
    incomplete["jobDescription"] = None
    runner = FakeRunner([incomplete, _actor_row("4450774822")])
    provider = RoobahistLinkedInProvider(runner, "roobahist/apify-linkedin")

    rows = provider.discover(["https://www.linkedin.com/jobs/search/?keywords=data"], max_items=10)

    assert [row["id"] for row in rows] == ["4450774822"]


def test_direct_fetch_rejects_incomplete_enrichment_so_router_can_fall_back() -> None:
    incomplete = _actor_row()
    incomplete["jobDescription"] = ""
    provider = RoobahistLinkedInProvider(FakeRunner([incomplete]), "roobahist/apify-linkedin")

    with pytest.raises(ProviderError, match="incomplete enriched data") as caught:
        provider.fetch_linkedin(4450774820, country="ca", max_concurrency=1)

    assert caught.value.retryable is True
