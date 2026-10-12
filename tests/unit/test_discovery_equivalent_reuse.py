from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from job_hunt.domain.models import Job, Qualification
from job_hunt.integrations.baserow import BaserowJobRepository
from job_hunt.integrations.discovery_skip import recent_baserow_rows


class Client:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.rows: list[dict[str, Any]] = []
        self.updates: list[tuple[int, dict[str, Any]]] = []
        self.uploaded: list[Path] = []
        self.uploaded_contents: dict[str, str] = {}

    def find_equal(self, _: int, field: str, value: object) -> dict[str, Any] | None:
        return next((row for row in self.rows if row.get(field) == value), None)

    def get_row(self, _: int, row_id: int) -> dict[str, Any]:
        return next(row for row in self.rows if row["id"] == row_id)

    def create_row(self, _: int, values: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": max((int(existing["id"]) for existing in self.rows), default=0) + 1,
            "Created On": self.now.isoformat(),
            "Score": 0,
            **values,
        }
        self.rows.append(row)
        return row

    def update_row(self, _: int, row_id: int, values: dict[str, Any]) -> dict[str, Any]:
        row = self.get_row(0, row_id)
        row.update(values)
        self.updates.append((row_id, dict(values)))
        return row

    def iter_rows(self, _: int, **__: object) -> Iterator[dict[str, Any]]:
        yield from sorted(self.rows, key=lambda row: str(row.get("Created On") or ""), reverse=True)

    def list_fields(self, _: int) -> list[dict[str, Any]]:
        return [{"name": "Created On", "type": "created_on"}]

    def upload_file(self, path: Path) -> dict[str, Any]:
        self.uploaded.append(path)
        self.uploaded_contents[path.name] = path.read_text(encoding="utf-8")
        return {"name": path.name, "url": f"https://files.invalid/{path.name}"}


def _job(job_id: str, location: str) -> Job:
    return Job(
        source="linkedin",
        external_id=job_id,
        url=f"https://www.linkedin.com/jobs/view/{job_id}",
        company_name="Example AI",
        title=f"Machine Learning Engineer - {location}",
        description=(
            f"Join our {location} team to build production machine learning systems, "
            "design model-serving APIs, improve inference performance, and collaborate "
            "with product and data teams on reliable AI features."
        ),
        location=location,
        contract_type="Full-time",
        published_at=datetime(2026, 9, 18, 8, tzinfo=UTC),
        provider_data={"_scheduled_discovery": True},
    )


def _repository(client: Client) -> BaserowJobRepository:
    return BaserowJobRepository(
        client,  # type: ignore[arg-type]
        1,
        {"new": 10, "dropped": 11, "toApply": 12, "applied": 13},
        {"fullTime": 20},
        to_apply_score_threshold=70,
        equivalence_lookback_seconds=7200,
    )


def test_same_run_duplicate_is_reserved_and_inherits_representative_result(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    client = Client(now)
    repository = _repository(client)
    representative = _job("100001", "Calgary, Alberta, Canada")
    duplicate = _job("100002", "Toronto, Ontario, Canada")

    assert repository.find(representative) is None
    duplicate_row = repository.find(duplicate)
    assert duplicate_row is not None
    assert duplicate_row["Job ID"] == "100002"
    assert duplicate_row["Location"] == "Toronto, Ontario, Canada"
    assert duplicate_row["Score"] == 0

    representative_row = repository.create(representative)
    representative_id = int(representative_row["id"])
    duplicate_id = int(duplicate_row["id"])

    repository.save_qualification(representative_id, Qualification(score=86))
    repository.save_artifacts(
        representative_id,
        {
            "CV": [{"name": "cv.pdf", "url": "https://files.invalid/cv.pdf"}],
            "Cover Letter": [{"name": "cover.pdf", "url": "https://files.invalid/cover.pdf"}],
            "Application ZIP": [{"name": "application.zip", "url": "https://files.invalid/application.zip"}],
        },
    )
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    repository.save_log(representative_id, trace)

    copied = client.get_row(1, duplicate_id)
    assert copied["Score"] == 86
    assert copied["Status"] == 12
    assert copied["CV"] == [{"name": "cv.pdf"}]
    assert copied["Cover Letter"] == [{"name": "cover.pdf"}]
    assert copied["Application ZIP"] == [{"name": "application.zip"}]
    assert copied["Log"] == [{"name": "job-100002-reuse.log", "url": "https://files.invalid/job-100002-reuse.log"}]
    assert copied["Job ID"] == "100002"
    assert copied["Location"] == "Toronto, Ontario, Canada"
    assert client.get_row(1, representative_id)["Log"] == [
        {"name": "trace.json", "url": "https://files.invalid/trace.json"}
    ]
    reuse_log = client.uploaded_contents["job-100002-reuse.log"]
    assert "source_job_id=100001" in reuse_log
    assert "target_job_id=100002" in reuse_log
    assert "not independently processed" in reuse_log


def test_prior_processed_equivalent_is_copied_without_reprocessing() -> None:
    now = datetime.now(UTC)
    client = Client(now)
    source = _job("100001", "Calgary, Alberta, Canada")
    client.rows.append(
        {
            "id": 4,
            "Created On": (now - timedelta(minutes=70)).isoformat(),
            "Job ID": source.external_id,
            "Company Name": source.company_name,
            "Title": source.title,
            "Job Description": source.description,
            "Location": source.location,
            "Contract Type": {"id": 20, "value": "Full-time"},
            "Link": source.url,
            "Date": "2026-09-18",
            "Status": {"id": 12, "value": "To Apply"},
            "Score": 92,
            "CV": [{"name": "cv.pdf", "url": "https://files.invalid/cv.pdf"}],
            "Cover Letter": [{"name": "cover.pdf", "url": "https://files.invalid/cover.pdf"}],
            "Application ZIP": [{"name": "application.zip", "url": "https://files.invalid/application.zip"}],
            "Log": [{"name": "trace.json", "url": "https://files.invalid/trace.json"}],
        }
    )
    repository = _repository(client)
    duplicate = _job("100002", "Vancouver, British Columbia, Canada")
    second_duplicate = _job("100003", "Toronto, Ontario, Canada")

    found = repository.find(duplicate)
    second_found = repository.find(second_duplicate)

    assert found is not None
    assert found["Job ID"] == "100002"
    assert found["Location"] == "Vancouver, British Columbia, Canada"
    assert found["Score"] == 92
    assert found["Status"] == 12
    assert found["CV"] == [{"name": "cv.pdf"}]
    assert found["Cover Letter"] == [{"name": "cover.pdf"}]
    assert found["Application ZIP"] == [{"name": "application.zip"}]
    assert found["Log"] == [{"name": "job-100002-reuse.log", "url": "https://files.invalid/job-100002-reuse.log"}]
    assert second_found is not None
    assert second_found["Score"] == 92
    assert second_found["Log"] == [
        {"name": "job-100003-reuse.log", "url": "https://files.invalid/job-100003-reuse.log"}
    ]
    assert found["Log"] != second_found["Log"]
    assert "source_row_id=4" in client.uploaded_contents["job-100002-reuse.log"]
    assert "not independently processed" in client.uploaded_contents["job-100003-reuse.log"]


def test_unprocessed_prior_equivalent_does_not_block_new_representative() -> None:
    now = datetime.now(UTC)
    client = Client(now)
    source = _job("100001", "Calgary, Alberta, Canada")
    client.rows.append(
        {
            "id": 4,
            "Created On": (now - timedelta(minutes=70)).isoformat(),
            "Job ID": source.external_id,
            "Company Name": source.company_name,
            "Title": source.title,
            "Job Description": source.description,
            "Location": source.location,
            "Contract Type": {"id": 20, "value": "Full-time"},
            "Link": source.url,
            "Date": "2026-09-18",
            "Status": {"id": 10, "value": "New"},
            "Score": None,
        }
    )
    repository = _repository(client)
    new_job = _job("100002", "Vancouver, British Columbia, Canada")

    assert repository.find(new_job) is None


def test_recent_rows_uses_exact_two_hour_created_on_window() -> None:
    now = datetime(2026, 9, 18, 8, 30, tzinfo=UTC)
    client = Client(now)
    client.rows = [
        {"id": 1, "Created On": (now - timedelta(minutes=10)).isoformat()},
        {"id": 2, "Created On": (now - timedelta(minutes=90)).isoformat()},
        {"id": 3, "Created On": (now - timedelta(hours=2, seconds=1)).isoformat()},
    ]

    rows = recent_baserow_rows(client, 1, lookback_seconds=7200, now=now)

    assert [row["id"] for row in rows] == [1, 2]
