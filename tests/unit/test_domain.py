from __future__ import annotations

import pytest
from pydantic import ValidationError

from job_hunt.domain.identity import (
    MAX_SAFE_INTEGER,
    assign_identity,
    canonicalize_url,
    stable_job_id,
)
from job_hunt.domain.models import (
    AiContentSubmission,
    ExternalSubmission,
    Job,
    Qualification,
    UrlSubmission,
)


def job(**updates: object) -> Job:
    values: dict[str, object] = {
        "source": "LinkedIn",
        "external_id": "123",
        "url": "HTTPS://Example.COM/jobs/1/?utm_source=x#top",
        "company_name": "Example",
        "title": "Engineer",
        "description": "Build things",
    }
    values.update(updates)
    return Job.model_validate(values)


def test_canonical_url_and_external_identity() -> None:
    assert canonicalize_url("HTTPS://Example.COM:443/jobs/1/?x=1#part") == "https://example.com/jobs/1"
    identified = assign_identity(job())
    assert identified.identity == "linkedin:123"
    assert 0 < identified.internal_id <= MAX_SAFE_INTEGER


def test_url_fallback_and_collision_rehash() -> None:
    candidate = assign_identity(job(external_id=None))
    collided = assign_identity(job(external_id=None), lambda value, _: value == candidate.internal_id)
    assert candidate.identity == "https://example.com/jobs/1"
    assert collided.internal_id != candidate.internal_id
    assert stable_job_id("x") == stable_job_id("x")


def test_submission_urls_default_to_https_when_scheme_is_missing() -> None:
    external = ExternalSubmission(
        entry_type="external",
        company_name="Example",
        job_title="Engineer",
        job_description="Build things",
        job_url="example.com/jobs/1",
    )
    url = UrlSubmission(entry_type="url", job_url="example.com/jobs/2")
    ai = AiContentSubmission(
        entry_type="ai_content",
        page_content="Job description",
        source_url="example.com/jobs/3",
    )

    assert str(external.job_url) == "https://example.com/jobs/1"
    assert str(url.job_url) == "https://example.com/jobs/2"
    assert str(ai.source_url) == "https://example.com/jobs/3"


def test_submission_url_normalization_preserves_existing_scheme_and_rejects_unsupported_scheme() -> None:
    https_url = UrlSubmission(entry_type="url", job_url="https://example.com/job")
    http_url = UrlSubmission(entry_type="url", job_url="http://example.com/job")

    assert str(https_url.job_url) == "https://example.com/job"
    assert str(http_url.job_url) == "http://example.com/job"

    with pytest.raises(ValidationError):
        UrlSubmission(entry_type="url", job_url="ftp://example.com/job")


def test_qualification_uses_score_gate_and_force() -> None:
    result = Qualification(score=90, should_apply=False, reasoning="Mismatch")
    assert result.passes(33)
    assert result.passes(33, force=True)
    assert Qualification(score=33, should_apply=False, reasoning="Threshold met").passes(33)
    assert not Qualification(score=32, should_apply=True, reasoning="Low").passes(33)
