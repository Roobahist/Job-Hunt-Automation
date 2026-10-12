from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from job_hunt.integrations.discovery_filters import FilteredDiscoveryProvider


class FakeDiscoveryProvider:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.linkedin_fetches: list[int] = []
        self.indeed_fetches: list[str] = []

    def discover(self, urls: Sequence[str], *, max_items: int) -> list[Mapping[str, Any]]:
        del urls, max_items
        return list(self.rows)

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        del country, max_concurrency
        self.linkedin_fetches.append(job_id)
        return {
            "jobId": str(job_id),
            "companyName": "Blocked Corp",
            "title": "Senior Manager",
            "description": "Blocked description",
        }

    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]:
        self.indeed_fetches.append(job_id)
        return {
            "jobId": job_id,
            "companyName": "Blocked Corp",
            "title": "Senior Manager",
            "description": "Blocked description",
        }


def test_post_filters_use_exact_company_matching_and_keyword_rules() -> None:
    provider = FakeDiscoveryProvider(
        [
            {
                "jobId": "1",
                "companyName": "OpenAI",
                "title": "Machine Learning Engineer",
                "description": "Build Python machine learning systems",
            },
            {
                "jobId": "2",
                "companyName": "OpenAI Research",
                "title": "Machine Learning Engineer",
                "description": "Build Python machine learning systems",
            },
            {
                "jobId": "3",
                "companyName": "OpenAI",
                "title": "Senior Machine Learning Engineer",
                "description": "Build Python machine learning systems",
            },
            {
                "jobId": "4",
                "companyName": "OpenAI",
                "title": "Machine Learning Engineer",
                "description": "Build Java services",
            },
            {
                "jobId": "5",
                "companyName": "OpenAI",
                "title": "Machine Learning Engineer",
                "description": "Build Python systems requiring security clearance",
            },
        ]
    )
    filtered = FilteredDiscoveryProvider(
        provider,
        company_inclusions=[" openai "],
        company_exclusions=["Example Labs"],
        title_inclusions=["machine learning", "data"],
        title_exclusions=["senior", "manager"],
        description_inclusions=["python", "pytorch"],
        description_exclusions=["security clearance", "french"],
    )

    rows = list(filtered.discover(["https://linkedin.example/search"], max_items=20))

    assert [row["jobId"] for row in rows] == ["1"]


def test_exclusions_take_precedence_over_inclusions() -> None:
    provider = FakeDiscoveryProvider(
        [
            {
                "jobId": "1",
                "companyName": "OpenAI",
                "title": "Data Scientist",
                "description": "Python analytics",
            }
        ]
    )
    filtered = FilteredDiscoveryProvider(
        provider,
        company_inclusions=["OpenAI"],
        company_exclusions=["OpenAI"],
        title_inclusions=["data"],
        description_inclusions=["python"],
    )

    assert list(filtered.discover(["https://linkedin.example/search"], max_items=20)) == []


def test_filters_support_provider_specific_field_names() -> None:
    provider = FakeDiscoveryProvider(
        [
            {
                "job_posting_id": "1",
                "company_name": "OpenAI",
                "job_title": "Data Scientist",
                "descriptionText": "Analyze data with Python",
            },
            {
                "id": "2",
                "company": "Other Co",
                "jobTitle": "Data Scientist",
                "jobDescription": "Analyze data with Python",
            },
        ]
    )
    filtered = FilteredDiscoveryProvider(
        provider,
        company_inclusions=["OpenAI"],
        title_inclusions=["data"],
        description_inclusions=["python"],
    )

    rows = list(filtered.discover(["https://linkedin.example/search"], max_items=20))

    assert [row["job_posting_id"] for row in rows] == ["1"]


def test_keyword_filters_use_boundary_aware_suffix_matching_for_title_and_description() -> None:
    provider = FakeDiscoveryProvider(
        [
            {
                "jobId": "1",
                "companyName": "Example",
                "title": "Product Designer",
                "description": "Designed dashboards in Excel",
            },
            {
                "jobId": "2",
                "companyName": "Example",
                "title": "Designated Representative",
                "description": "Known for excellence in reporting",
            },
            {
                "jobId": "3",
                "companyName": "Example",
                "title": "Product Design Specialist",
                "description": "Uses Excel daily",
            },
            {
                "jobId": "4",
                "companyName": "Example",
                "title": "Product Designed Specialist",
                "description": "Uses Excel daily",
            },
            {
                "jobId": "5",
                "companyName": "Example",
                "title": "C++ Developer",
                "description": "Node.js services",
            },
            {
                "jobId": "6",
                "companyName": "Example",
                "title": "C++17 Developer",
                "description": "Node.jsx services",
            },
        ]
    )

    simple = FilteredDiscoveryProvider(
        provider,
        title_inclusions=["design"],
        description_inclusions=["excel"],
    )
    assert [row["jobId"] for row in simple.discover(["https://linkedin.example/search"], max_items=20)] == [
        "1",
        "3",
        "4",
    ]

    phrase = FilteredDiscoveryProvider(provider, title_inclusions=["product design"])
    assert [row["jobId"] for row in phrase.discover(["https://linkedin.example/search"], max_items=20)] == [
        "3"
    ]

    punctuation = FilteredDiscoveryProvider(
        provider,
        title_inclusions=["c++"],
        description_inclusions=["node.js"],
    )
    assert [row["jobId"] for row in punctuation.discover(["https://linkedin.example/search"], max_items=20)] == [
        "5"
    ]


def test_direct_fetches_bypass_scheduled_discovery_filters() -> None:
    provider = FakeDiscoveryProvider([])
    filtered = FilteredDiscoveryProvider(
        provider,
        company_exclusions=["Blocked Corp"],
        title_exclusions=["senior"],
        description_exclusions=["blocked"],
    )

    linkedin = filtered.fetch_linkedin(123456, country="ca", max_concurrency=1)
    indeed = filtered.fetch_indeed("abc123")

    assert linkedin["companyName"] == "Blocked Corp"
    assert indeed["companyName"] == "Blocked Corp"
    assert provider.linkedin_fetches == [123456]
    assert provider.indeed_fetches == ["abc123"]
