from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import job_hunt.worker as worker


class FakeStore:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []

    def update(self, _run_id: object, **values: object) -> None:
        self.updates.append(dict(values))

    def save(self, _status: object) -> None:
        return None

    def merge_notification(self, _run_id: object, **_values: object) -> None:
        return None

    def save_request(self, _run_id: object, request: dict[str, object]) -> None:
        self.requests.append(request)


class FakeState:
    def set_snapshot(self, *_args: object, **_kwargs: object) -> None:
        return None


class FakeBaserow:
    def iter_rows(self, table_id: int) -> list[dict[str, object]]:
        assert table_id == 1
        return [
            {
                "id": 11,
                "Active": True,
                "Keywords": "data scientist",
                "Location": "alberta",
                "Experience Level": [{"value": "Entry Level"}],
                "Workplace Type": [{"value": "Remote"}],
            }
        ]


class FakeRepository:
    def __init__(self) -> None:
        self.deleted_statuses: list[str] = []

    def find(self, _job: object) -> None:
        return None

    def delete_status(self, status: str) -> int:
        self.deleted_statuses.append(status)
        return 4


class FakeEndOfDayDiscovery:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.max_items: int | None = None
        self.skip_job_ids: list[str] = []

    def discover(
        self,
        urls: list[str],
        *,
        max_items: int,
        skip_job_ids: list[str],
    ) -> list[dict[str, object]]:
        self.urls = list(urls)
        self.max_items = max_items
        self.skip_job_ids = list(skip_job_ids)
        return [
            {
                "id": "300",
                "url": "https://linkedin.com/jobs/view/300",
                "companyName": "New Company",
                "title": "Data Scientist",
                "description": "A newly recovered posting",
                "location": "Calgary, Alberta, Canada",
                "employmentType": "Full-time",
            }
        ]


class FakeServices:
    def __init__(self) -> None:
        self.baserow = FakeBaserow()
        self.repository = FakeRepository()
        self.end_of_day_discovery = FakeEndOfDayDiscovery()
        self.config = SimpleNamespace(
            baserow_table_ids={"searchCriteria": 1, "jobs": 2},
            linkedin_base_search_url="https://linkedin.example/search",
            linkedin_schedule_interval_hours=1,
            linkedin_max_items=150,
            company_exclusions=[],
            title_exclusions=[],
            end_of_day_dropped_cleanup_enabled=True,
        )

    def snapshot(self) -> dict[str, object]:
        return {}


class FakeContainer:
    def __init__(self, services: FakeServices) -> None:
        self.services = services

    def tenant(self, _tenant: str) -> FakeServices:
        return self.services


def test_end_of_day_catchup_uses_24_hour_roobahist_search_and_skip_ids(monkeypatch: object) -> None:
    store = FakeStore()
    state = FakeState()
    services = FakeServices()
    container = FakeContainer(services)
    queued_headers: list[list[object]] = []
    callbacks: list[object] = []

    monkeypatch.setattr(worker, "_store", lambda: store)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "_state", lambda: state)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "Container", lambda _settings: container)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker,
        "recent_baserow_linkedin_job_ids",
        lambda *_args, **kwargs: ["100", "200"] if kwargs["lookback_seconds"] == 86_820 else [],
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker.process_submission,
        "s",
        lambda *args: ("process", args),
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker.finish_end_of_day_tenant,
        "s",
        lambda *args: ("finish", args),
    )  # type: ignore[attr-defined]

    def fake_chord(headers: list[object]) -> object:
        queued_headers.append(headers)

        def apply(callback: object) -> None:
            callbacks.append(callback)

        return apply

    monkeypatch.setattr(worker, "chord", fake_chord)  # type: ignore[attr-defined]

    run_id = str(uuid4())
    result = worker.end_of_day_catchup_tenant.run("nova", run_id, 86_820)

    expected_url = (
        "https://linkedin.example/search?keywords=data+scientist&location=alberta&f_E=2&f_WT=2&f_TPR=r86820&sortBy=R"
    )
    assert services.end_of_day_discovery.urls == [expected_url]
    assert services.end_of_day_discovery.max_items == 150
    assert services.end_of_day_discovery.skip_job_ids == ["100", "200"]
    assert result["skip_job_ids"] == 2
    assert result["missed_jobs"] == 1
    assert result["queued"] == 1
    assert len(queued_headers) == 1
    assert len(queued_headers[0]) == 1
    assert callbacks == [("finish", ("nova", run_id, result))]
    assert store.requests[0]["kind"] == "end-of-day-job"


def test_end_of_day_cleanup_deletes_dropped_rows_when_enabled(monkeypatch: object) -> None:
    store = FakeStore()
    services = FakeServices()
    container = FakeContainer(services)
    monkeypatch.setattr(worker, "_store", lambda: store)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "Container", lambda _settings: container)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker.settings, "end_of_day_dropped_cleanup_enabled", True)

    result = worker.finish_end_of_day_tenant.run([], "nova", str(uuid4()), {"queued": 1})

    assert services.repository.deleted_statuses == ["dropped"]
    assert result["dropped_deleted"] == 4


def test_end_of_day_cleanup_can_be_disabled_independently(monkeypatch: object) -> None:
    store = FakeStore()
    services = FakeServices()
    services.config.end_of_day_dropped_cleanup_enabled = False
    container = FakeContainer(services)
    monkeypatch.setattr(worker, "_store", lambda: store)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "Container", lambda _settings: container)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker.settings, "end_of_day_dropped_cleanup_enabled", True)

    result = worker.finish_end_of_day_tenant.run([], "nova", str(uuid4()), {"queued": 1})

    assert services.repository.deleted_statuses == []
    assert result["dropped_deleted"] == 0
