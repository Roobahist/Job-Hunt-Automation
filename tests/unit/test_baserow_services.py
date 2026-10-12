from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, Qualification
from job_hunt.errors import ConfigurationError
from job_hunt.integrations.baserow import BaserowClient, BaserowJobRepository
from job_hunt.integrations.configuration import (
    COMMON_PROMPT_KEYS,
    ATLAS_PROMPT_KEYS,
    NOVA_PROMPT_KEYS,
    BaserowConfigurationRepository,
    validate_prompt_contract,
)


class Client:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.deleted: list[int] = []
        self.fields = [
            {"name": name}
            for name in {
                "Job ID",
                "Company Name",
                "Title",
                "Job Description",
                "Link",
                "Status",
                "Score",
                "CV",
                "Cover Letter",
                "Log",
                "Date",
                "Location",
                "Contract Type",
            }
        ]
        self.fields.append({"name": "Created On", "type": "created_on"})

    def find_equal(self, _: int, field: str, value: object) -> dict[str, Any] | None:
        return next((row for row in self.rows if row.get(field) == value), None)

    def get_row(self, _: int, row_id: int) -> dict[str, Any]:
        return next(row for row in self.rows if row.get("id") == row_id)

    def create_row(self, _: int, values: dict[str, Any]) -> dict[str, Any]:
        return {"id": 3, **values}

    def update_row(self, _: int, row_id: int, values: dict[str, Any]) -> dict[str, Any]:
        self.updates.append(values)
        return {"id": row_id, **values}

    def delete_row(self, _: int, row_id: int) -> None:
        self.deleted.append(row_id)

    def iter_rows(self, _: int) -> object:
        return iter(self.rows)

    def list_fields(self, _: int) -> list[dict[str, Any]]:
        return self.fields


def sample_job() -> Job:
    return assign_identity(
        Job(
            source="linkedin",
            external_id="4452378707",
            url="https://linkedin.com/jobs/view/4452378707",
            company_name="C",
            title="T",
            description="D",
            published_at=datetime(2026, 8, 19, tzinfo=UTC),
        )
    )


def prompt_row(key: str, *, version: float = 1, status: str = "Active") -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    return {
        "Prompt Key": key,
        "Version": version,
        "Prompt Template": f"prompt for {key}",
        "Output Structure": json.dumps(schema),
        "Temperature": 0.2,
        "Status": status,
        "Enabled": True,
    }


def test_real_baserow_client_supports_configuration_contract() -> None:
    assert callable(getattr(BaserowClient, "list_fields", None))


def test_repository_create_find_status_and_narrow_updates() -> None:
    client = Client()
    repo = BaserowJobRepository(
        client,
        1,
        {"new": 10, "dropped": 11, "toApply": 12, "applied": 13},
        {"fullTime": 14},
        to_apply_score_threshold=70,
    )  # type: ignore[arg-type]
    job = sample_job()
    created = repo.create(job)
    assert created["Status"] == 10
    assert created["Job ID"] == "4452378707"
    assert created["Date"] == "2026-08-19"
    assert created["Link"] == job.url

    client.rows = [{"id": 3, "Job ID": "4452378707", "Link": job.url, "Status": {"id": 11, "value": "Dropped"}}]
    assert repo.find(job)["id"] == 3  # type: ignore[index]
    assert repo.has_status(3, "dropped")
    assert not repo.has_status(3, "applied")

    reset = repo.reset(3, job)
    assert reset["id"] == 3
    assert client.updates[-1]["Job ID"] == "4452378707"
    assert client.updates[-1]["Link"] == job.url
    assert "Score" not in client.updates[-1]
    assert "CV" not in client.updates[-1]
    assert "Cover Letter" not in client.updates[-1]
    assert "Status" not in client.updates[-1]

    repo.save_qualification(3, Qualification(score=20, reasoning="low"))
    assert client.updates[-1] == {"Score": 20}
    assert "Link" not in client.updates[-1]
    assert "Status" not in client.updates[-1]

    repo.save_qualification(3, Qualification(score=80, reasoning="high"))
    assert client.updates[-1] == {"Score": 80, "Status": 12}

    repo.save_artifacts(3, {"CV": [{"name": "cv.pdf"}], "Cover Letter": [{"name": "cl.pdf"}]})
    assert set(client.updates[-1]) == {"CV", "Cover Letter"}
    assert "Link" not in client.updates[-1]
    assert "Status" not in client.updates[-1]

    repo.set_status(3, "applied")
    assert client.updates[-1]["Status"] == 13


def test_repository_deletes_only_rows_with_requested_status() -> None:
    client = Client()
    repo = BaserowJobRepository(
        client,
        1,
        {"new": 10, "dropped": 11, "toApply": 12},
    )  # type: ignore[arg-type]
    client.rows = [
        {"id": 1, "Status": {"id": 11, "value": "Dropped"}},
        {"id": 2, "Status": {"id": 12, "value": "To Apply"}},
        {"id": 3, "Status": 11},
    ]

    assert repo.delete_status("dropped") == 2
    assert client.deleted == [1, 3]


def test_repository_preserves_alphanumeric_external_job_id_as_text() -> None:
    client = Client()
    repo = BaserowJobRepository(client, 1, {"new": 10})  # type: ignore[arg-type]
    job = assign_identity(
        Job(
            source="indeed",
            external_id="a4d97fa27223cc84",
            url="https://www.indeed.com/viewjob?jk=a4d97fa27223cc84",
            company_name="C",
            title="T",
            description="D",
        )
    )

    created = repo.create(job)
    assert created["Job ID"] == "a4d97fa27223cc84"

    client.rows = [{"id": 4, "Job ID": "a4d97fa27223cc84", "Link": job.url}]
    assert repo.find(job)["id"] == 4  # type: ignore[index]


def test_configuration_reads_live_prompt_contract_and_validates_fields() -> None:
    client = Client()
    repository = BaserowConfigurationRepository(client, 1)  # type: ignore[arg-type]
    all_profile_keys = ATLAS_PROMPT_KEYS | NOVA_PROMPT_KEYS
    client.rows = [prompt_row(key) for key in sorted(all_profile_keys)]
    client.rows.append(prompt_row("cv_summary_rewrite", version=99, status="Draft"))
    prompts = repository.prompts(2)
    summary = prompts["cv_summary_rewrite"]
    assert summary.version == 1
    assert summary.temperature == 0.2
    assert summary.output_structure["additionalProperties"] is False
    validate_prompt_contract(prompts, ATLAS_PROMPT_KEYS, "atlas")
    validate_prompt_contract(prompts, NOVA_PROMPT_KEYS, "nova")
    repository.validate_job_table(1)

    client.rows.append(prompt_row("cv_summary_rewrite", version=2))
    with pytest.raises(ConfigurationError, match="Duplicate active prompt"):
        repository.prompts(2)

    client.fields = [{"name": "Job ID"}]
    with pytest.raises(ConfigurationError, match="missing fields"):
        repository.validate_job_table(1)


def test_configuration_requires_created_on_timestamp_for_duplicate_window() -> None:
    client = Client()
    repository = BaserowConfigurationRepository(client, 1)  # type: ignore[arg-type]
    client.fields = [field for field in client.fields if field.get("type") != "created_on"]

    with pytest.raises(ConfigurationError, match="Created on field"):
        repository.validate_job_table(1)


def test_configuration_rejects_missing_common_or_invalid_schema() -> None:
    client = Client()
    repository = BaserowConfigurationRepository(client, 1)  # type: ignore[arg-type]
    client.rows = [prompt_row(key) for key in sorted(ATLAS_PROMPT_KEYS)]
    client.rows[0]["Output Structure"] = "not json"
    with pytest.raises(ConfigurationError, match="invalid Output Structure"):
        repository.prompts(2)

    client.rows = [prompt_row(key) for key in sorted(COMMON_PROMPT_KEYS - {"qualification_scoring"})]
    with pytest.raises(ConfigurationError, match="Missing active prompts"):
        repository.prompts(2)


def test_profile_contracts_are_independent() -> None:
    client = Client()
    repository = BaserowConfigurationRepository(client, 1)  # type: ignore[arg-type]

    client.rows = [prompt_row(key) for key in sorted(ATLAS_PROMPT_KEYS)]
    atlas_prompts = repository.prompts(2)
    validate_prompt_contract(atlas_prompts, ATLAS_PROMPT_KEYS, "atlas")
    assert "cv_project_selection" not in ATLAS_PROMPT_KEYS
    assert "cv_project_rewrite" not in ATLAS_PROMPT_KEYS
    with pytest.raises(ConfigurationError, match="nova"):
        validate_prompt_contract(atlas_prompts, NOVA_PROMPT_KEYS, "nova")

    client.rows = [prompt_row(key) for key in sorted(NOVA_PROMPT_KEYS)]
    nova_prompts = repository.prompts(2)
    validate_prompt_contract(nova_prompts, NOVA_PROMPT_KEYS, "nova")
    with pytest.raises(ConfigurationError, match="atlas"):
        validate_prompt_contract(nova_prompts, ATLAS_PROMPT_KEYS, "atlas")
