from __future__ import annotations

from typing import Any

import httpx
import pytest

from job_hunt.application.normalization import linkedin_job_id_from_url
from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.apify import ApifyProvider
from job_hunt.integrations.baserow import BaserowJobRepository
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient


def test_linkedin_detection_rejects_lookalike_hostnames() -> None:
    assert linkedin_job_id_from_url("https://evil-linkedin.com/jobs/view/example-4452378707") is None
    assert linkedin_job_id_from_url("https://ca.linkedin.com/jobs/view/example-4452378707") == 4452378707


def test_forced_reset_refreshes_source_fields_without_touching_artifacts() -> None:
    updates: list[tuple[int, int, dict[str, Any]]] = []

    class Client:
        def update_row(self, table_id: int, row_id: int, values: dict[str, Any]) -> dict[str, Any]:
            updates.append((table_id, row_id, values))
            return {"id": row_id, **values}

    repository = BaserowJobRepository(  # type: ignore[arg-type]
        Client(),
        123,
        {"new": 1, "dropped": 2},
        {"fullTime": 10},
    )
    job = Job(
        source="linkedin",
        external_id="4452378707",
        url="https://www.linkedin.com/jobs/view/4452378707",
        company_name="Updated Company",
        title="Updated Title",
        description="Updated description",
        location="Calgary",
        contract_type="Full-time",
    )

    result = repository.reset(42, job)

    assert result["id"] == 42
    assert updates[0][0:2] == (123, 42)
    fields = updates[0][2]
    assert fields["Company Name"] == "Updated Company"
    assert fields["Title"] == "Updated Title"
    assert fields["Job Description"] == "Updated description"
    assert fields["Contract Type"] == 10
    assert "CV" not in fields
    assert "Cover Letter" not in fields
    assert "Score" not in fields


def test_apify_does_not_probe_tokens_while_all_are_cooling_down() -> None:
    class State:
        def available(self, _scope: str, _resource: str) -> bool:
            return False

        def remaining_cooldown(self, _scope: str, resource: str) -> int:
            return 42 if resource else 99

    provider = ApifyProvider(
        ["token-one", "token-two"],
        "search",
        "single",
        state=State(),  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderError) as caught:
        provider._available_tokens()

    assert caught.value.kind == ErrorKind.RATE_LIMIT
    assert caught.value.retryable is True
    assert caught.value.retry_after == 42


def test_litellm_gateway_regenerates_identical_requests() -> None:
    calls = 0

    class State:
        def get_checkpoint(self, _digest: str) -> dict[str, Any] | None:
            return {"compatible": False}

        def set_checkpoint(self, _digest: str, _value: dict[str, Any], *, ttl_seconds: int) -> None:
            raise AssertionError(f"LLM output checkpoints must not be written, got ttl={ttl_seconds}")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": "mistral/mistral-medium-latest",
                "choices": [{"message": {"content": '{"compatible":true}'}}],
                "usage": {"total_tokens": 4},
            },
        )

    definition = PromptDefinition(
        key="qualification_scoring",
        version=2,
        template="unused",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"compatible": {"type": "boolean"}},
            "required": ["compatible"],
        },
    )
    client = LiteLLMGatewayClient(
        "http://litellm:4000",
        "secret",
        "job-balanced",
        state=State(),  # type: ignore[arg-type]
        checkpoint_ttl_seconds=321,
        transport=httpx.MockTransport(handler),
    )

    assert client.generate("same prompt", definition) == {"compatible": True}
    assert client.generate("same prompt", definition) == {"compatible": True}
    assert calls == 2
