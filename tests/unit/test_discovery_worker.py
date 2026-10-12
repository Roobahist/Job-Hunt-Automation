from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import job_hunt.worker as worker


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.deleted: list[str] = []

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.values.pop(key, None)


class FakeStore:
    def __init__(self) -> None:
        self.redis = FakeRedis()
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


class FakeRepository:
    def find(self, job: object) -> dict[str, int] | None:
        external_id = getattr(job, "external_id", None)
        return {"id": 99} if external_id == "1" else None


class FakeBaserow:
    def __init__(self) -> None:
        self.row_updates: list[tuple[int, int, dict[str, object]]] = []

    def iter_rows(self, _table_id: int) -> list[dict[str, object]]:
        return [
            {"id": 10, "Active": False, "Generated URL": "https://linkedin.example/inactive"},
            {
                "id": 11,
                "Active": True,
                "Generated URL": "https://linkedin.example/stale?keywords=old&f_TPR=r86400",
                "Keywords": "data scientist",
                "Location": "alberta",
                "Experience Level": [{"value": "Entry Level"}],
                "Workplace Type": [{"value": "Remote"}],
            },
        ]

    def update_row(self, table_id: int, row_id: int, values: dict[str, object]) -> dict[str, object]:
        self.row_updates.append((table_id, row_id, dict(values)))
        return {"id": row_id, **values}


class FakeDiscovery:
    def __init__(self, baserow: FakeBaserow) -> None:
        self.baserow = baserow
        self.urls: list[str] = []

    def discover(self, urls: list[str], *, max_items: int) -> list[dict[str, object]]:
        assert self.baserow.row_updates, "Generated URL must be written to Baserow before Apify runs"
        self.urls = urls
        assert max_items == 25
        return [
            {
                "id": 1,
                "url": "https://linkedin.com/jobs/view/1",
                "companyName": "Existing",
                "title": "Existing Job",
                "description": "Already tracked",
            },
            {
                "id": 2,
                "url": "https://linkedin.com/jobs/view/2",
                "companyName": "New",
                "title": "New Job",
                "description": "New posting",
            },
        ]


class FakeServices:
    def __init__(self) -> None:
        self.baserow = FakeBaserow()
        self.repository = FakeRepository()
        self.discovery = FakeDiscovery(self.baserow)
        self.config = SimpleNamespace(
            baserow_table_ids={"searchCriteria": 1},
            linkedin_base_search_url="https://linkedin.example/search",
            linkedin_schedule_interval_hours=1,
            company_exclusions=[],
            title_exclusions=[],
            linkedin_max_items=25,
            telegram_chat_id="42",
        )

    def snapshot(self) -> dict[str, object]:
        return {}


class FakeContainer:
    def __init__(self, services: FakeServices) -> None:
        self.services = services

    def tenant(self, _tenant: str) -> FakeServices:
        return self.services


def _install_worker_fakes(
    monkeypatch: object,
    store: FakeStore,
    state: FakeState,
    services: FakeServices,
    notification_jobs: list[str],
    submission_jobs: list[str],
) -> None:
    container = FakeContainer(services)
    monkeypatch.setattr(worker, "_store", lambda: store)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "_state", lambda: state)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "Container", lambda _settings: container)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker.init_job_notification,
        "delay",
        lambda _tenant, _run_id, job, _chat: notification_jobs.append(str(job["external_id"])),
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker.process_submission,
        "delay",
        lambda _tenant, payload, *_args: submission_jobs.append(str(payload["external_job_id"])),
    )  # type: ignore[attr-defined]


def test_discovery_rebuilds_and_writes_active_generated_urls_before_apify(monkeypatch: object) -> None:
    store = FakeStore()
    state = FakeState()
    services = FakeServices()
    notification_jobs: list[str] = []
    submission_jobs: list[str] = []
    _install_worker_fakes(monkeypatch, store, state, services, notification_jobs, submission_jobs)

    result = worker.discover_tenant.run("nova", str(uuid4()))

    expected_url = (
        "https://linkedin.example/search?keywords=data+scientist&location=alberta&f_E=2&f_WT=2&f_TPR=r3600&sortBy=R"
    )
    assert services.discovery.urls == [expected_url]
    assert services.baserow.row_updates == [(1, 11, {"Generated URL": expected_url})]
    assert result["criteria_total"] == 2
    assert result["criteria_active"] == 1
    assert result["discovered"] == 2
    assert result["duplicates"] == 1
    assert result["queued"] == 1
    # Discovery must not expose a Telegram message before process_submission
    # persists the corresponding Baserow row.
    assert notification_jobs == []
    assert submission_jobs == ["2"]


def test_next_discovery_consumes_pending_recovery_window(monkeypatch: object) -> None:
    store = FakeStore()
    state = FakeState()
    services = FakeServices()
    notification_jobs: list[str] = []
    submission_jobs: list[str] = []
    _install_worker_fakes(monkeypatch, store, state, services, notification_jobs, submission_jobs)

    marker = {
        "last_slot": "2026-08-25T12-0600",
        "detected_at": "2026-08-25T14:30:00-06:00",
        "missed_count": 2,
        "missed_slots": ["2026-08-25T13-0600", "2026-08-25T14-0600"],
        "lookback_hours": 6,
    }
    cleared: list[dict[str, object]] = []
    notified: list[int] = []
    monkeypatch.setattr(worker, "load_pending_recovery", lambda _redis, _tenant: marker)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "recovery_window_seconds", lambda *_args, **_kwargs: 10800)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker,
        "clear_pending_recovery",
        lambda _redis, _tenant, pending: cleared.append(dict(pending)),
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker,
        "notify_recovery_complete",
        lambda **kwargs: notified.append(int(kwargs["window_seconds"])),
    )  # type: ignore[attr-defined]

    worker.discover_tenant.run("nova", str(uuid4()))

    expected_url = (
        "https://linkedin.example/search?keywords=data+scientist&location=alberta&f_E=2&f_WT=2&f_TPR=r10800&sortBy=R"
    )
    assert services.discovery.urls == [expected_url]
    assert services.baserow.row_updates == [(1, 11, {"Generated URL": expected_url})]
    assert cleared == [marker]
    assert notified == [10800]


def test_recovery_marker_is_retained_when_notification_fails(monkeypatch: object) -> None:
    store = FakeStore()
    state = FakeState()
    services = FakeServices()
    notification_jobs: list[str] = []
    submission_jobs: list[str] = []
    _install_worker_fakes(monkeypatch, store, state, services, notification_jobs, submission_jobs)

    marker = {
        "last_slot": "2026-08-25T12-0600",
        "detected_at": "2026-08-25T14:30:00-06:00",
        "missed_count": 1,
        "missed_slots": ["2026-08-25T13-0600"],
        "lookback_hours": 6,
    }
    cleared: list[dict[str, object]] = []
    monkeypatch.setattr(worker, "load_pending_recovery", lambda _redis, _tenant: marker)  # type: ignore[attr-defined]
    monkeypatch.setattr(worker, "recovery_window_seconds", lambda *_args, **_kwargs: 7200)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        worker,
        "clear_pending_recovery",
        lambda _redis, _tenant, pending: cleared.append(dict(pending)),
    )  # type: ignore[attr-defined]

    def fail_notification(**_kwargs: object) -> None:
        raise RuntimeError("telegram unavailable")

    monkeypatch.setattr(worker, "notify_recovery_complete", fail_notification)  # type: ignore[attr-defined]

    result = worker.discover_tenant.run("nova", str(uuid4()))

    assert result["queued"] == 1
    assert cleared == []
