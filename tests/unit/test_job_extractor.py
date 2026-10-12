# ruff: noqa: I001
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.integrations import job_extractor as job_extractor_module
from job_hunt.integrations.job_extractor import StructuredJobExtractor


SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "job_title": {"type": "string"},
        "job_description": {"type": "string"},
        "job_url": {"type": "string"},
        "location": {"type": "string"},
        "contract_type": {
            "type": "string",
            "enum": ["", "Full-time", "Contract", "Part-time", "Temporary", "Internship", "Volunteer"],
        },
        "published_at": {"type": "string"},
        "external_job_id": {"type": "string", "pattern": "^$|^[0-9]+$"},
    },
    "required": [
        "company_name",
        "job_title",
        "job_description",
        "job_url",
        "location",
        "contract_type",
        "published_at",
        "external_job_id",
    ],
    "additionalProperties": False,
}


class Client:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str, _: PromptDefinition) -> dict[str, object]:
        self.prompt = prompt
        return self.response


def definition() -> PromptDefinition:
    return PromptDefinition(
        key="job_page_content_extraction",
        version=1,
        template="Page Content:\n[[page_content]]",
        output_structure=SCHEMA,
        temperature=0,
    )


def response(**updates: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "company_name": "",
        "job_title": "",
        "job_description": "Build and maintain AI systems.",
        "job_url": "",
        "location": "",
        "contract_type": "",
        "published_at": "",
        "external_job_id": "",
    }
    payload.update(updates)
    return payload


def test_content_only_extraction_accepts_unknown_company_and_title() -> None:
    client = Client(response())
    job = StructuredJobExtractor(client, definition()).extract_job("pasted description", "")

    assert job.company_name == ""
    assert job.title == ""
    assert job.description == "Build and maintain AI systems."
    assert job.url == ""
    assert job.identity.startswith("web:content:")
    assert "pasted description" in client.prompt


def test_extractor_maps_schema_field_names_and_optional_date() -> None:
    client = Client(
        response(
            company_name="Example",
            job_title="AI Engineer",
            job_url="https://example.com/jobs/42?source=form",
            location="Remote",
            contract_type="Full-time",
            published_at="2026-08-24",
            external_job_id="42",
        )
    )
    job = StructuredJobExtractor(client, definition()).extract_job("posting", "")

    assert job.external_id == "42"
    assert job.published_at == datetime(2026, 8, 24, tzinfo=UTC)
    assert job.contract_type == "Full-time"
    assert job.location == "Remote"
    assert job.identity == "web:42"


def test_extractor_falls_back_to_submitted_url_when_model_url_is_invalid() -> None:
    client = Client(response(job_url="/"))
    job = StructuredJobExtractor(client, definition()).extract_job(
        "posting",
        "https://example.com/jobs/42?source=form",
    )

    assert job.url == "https://example.com/jobs/42"
    assert job.identity == "https://example.com/jobs/42"


def test_extractor_reports_empty_description_without_requiring_optional_metadata() -> None:
    client = Client(response(job_description=""))

    try:
        StructuredJobExtractor(client, definition()).extract_job("posting", "")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected empty job_description to fail")

    assert "empty required job_description" in message
    assert "company_name" in message
    assert "job_title" in message
    assert "job_description" in message


def test_ai_extraction_joins_job_application_trace_and_captures_io(monkeypatch: pytest.MonkeyPatch) -> None:
    active = False
    trace_seeds: list[str] = []
    span_calls: list[dict[str, Any]] = []
    span_outputs: list[Any] = []

    class TracingClient(Client):
        def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, object]:
            assert active
            return super().generate(prompt, definition)

    class Observation:
        def update(self, **kwargs: Any) -> None:
            span_outputs.append(kwargs.get("output"))

    def fake_trace_context(*, enabled: bool, seed: str) -> dict[str, str] | None:
        assert enabled
        trace_seeds.append(seed)
        return {"trace_id": "1" * 32}

    @contextmanager
    def fake_span(**kwargs: Any) -> Iterator[Observation]:
        nonlocal active
        span_calls.append(kwargs)
        active = True
        try:
            yield Observation()
        finally:
            active = False

    monkeypatch.setattr(job_extractor_module, "langfuse_trace_context", fake_trace_context)
    monkeypatch.setattr(job_extractor_module, "langfuse_span", fake_span)

    client = TracingClient(
        response(
            company_name="Example",
            job_title="AI Engineer",
            job_url="https://example.com/jobs/42",
            location="Calgary",
            contract_type="Full-time",
            external_job_id="42",
        )
    )
    job = StructuredJobExtractor(
        client,
        definition(),
        tenant="nova",
        session_id="run-123",
        langfuse_enabled=True,
    ).extract_job("full pasted job description", "https://example.com/jobs/42")

    assert trace_seeds == ["job-hunt:nova:run-123"]
    assert len(span_calls) == 1
    span = span_calls[0]
    assert span["name"] == "job-information-extraction"
    assert span["trace_name"] == "job-application"
    assert span["trace_context"] == {"trace_id": "1" * 32}
    assert "session_id" not in span
    assert span["user_id"] == "nova"
    assert span["input"]["page_content"] == "full pasted job description"
    assert "full pasted job description" in span["input"]["rendered_prompt"]
    assert span_outputs[0]["model_output"]["job_title"] == "AI Engineer"
    assert span_outputs[0]["job"]["description"] == "Build and maintain AI systems."
    assert job.identity == "web:42"


def test_job_domain_requires_description_but_not_optional_metadata() -> None:
    job = Job(source="web", url="", company_name="", title="", description="Role details")
    assert job.description == "Role details"

    try:
        Job(source="web", url="", company_name="", title="", description="")
    except ValidationError as exc:
        assert "description is required" in str(exc)
    else:
        raise AssertionError("expected empty description to fail")
