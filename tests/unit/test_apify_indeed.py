# ruff: noqa: I001
from __future__ import annotations

from typing import Any

import pytest

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.apify_indeed import ApifyIndeedProvider


JOB_ID = "48ddfac3828876b4"


def _record() -> dict[str, Any]:
    return {
        "id": JOB_ID,
        "title": {"text": "Data Scientist", "normalized": "data scientist"},
        "urls": {
            "indeed": f"https://ca.indeed.com/viewjob?jk={JOB_ID}",
            "external": "https://example.com/apply",
        },
        "description": {"text": "Build production machine learning systems.", "html": "<p>Build systems.</p>"},
        "classification": {"jobType": ["fulltime"]},
        "location": {"formattedShort": "Calgary, AB"},
        "company": {"name": "Example Inc."},
        "dates": {"posted": "2026-09-05"},
        "salary": {"text": "$100,000 a year"},
        "apply": {"url": "https://example.com/apply"},
    }


def test_fetch_indeed_uses_known_job_lookup_and_maps_actor_output(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApifyIndeedProvider(["token"], actor_id="kavin/indeed-scraper")
    calls: list[tuple[str, dict[str, Any], int | None]] = []

    def fake_run(
        actor_id: str,
        run_input: dict[str, Any],
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        calls.append((actor_id, run_input, max_items))
        return [_record()]

    monkeypatch.setattr(provider, "_run", fake_run)

    result = provider.fetch_indeed(JOB_ID, country="ca")

    assert calls == [
        (
            "kavin/indeed-scraper",
            {"jobKeys": [JOB_ID], "country": "CA", "searchMode": "basic"},
            1,
        )
    ]
    assert result["jobId"] == JOB_ID
    assert result["jobUrl"] == f"https://www.indeed.com/viewjob?jk={JOB_ID}"
    assert result["title"] == "Data Scientist"
    assert result["companyName"] == "Example Inc."
    assert result["descriptionText"] == "Build production machine learning systems."
    assert result["location"] == "Calgary, AB"
    assert result["employmentType"] == "fulltime"
    assert result["_jobFetchProvider"] == "apify_indeed"


def test_fetch_indeed_rejects_mismatched_or_incomplete_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ApifyIndeedProvider(["token"])

    def fake_run(
        _actor_id: str,
        _run_input: dict[str, Any],
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        assert max_items == 1
        return [
            {"id": "different", "title": {"text": "Wrong"}},
            {"id": JOB_ID, "title": {"text": "Missing company and description"}},
        ]

    monkeypatch.setattr(provider, "_run", fake_run)

    with pytest.raises(ProviderError, match="not found") as caught:
        provider.fetch_indeed(JOB_ID, country="CA")

    assert caught.value.kind is ErrorKind.BUSINESS
    assert caught.value.provider == "apify"
