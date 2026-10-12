from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from job_hunt.application.runs import RunCoordinator
from job_hunt.domain.models import RunStatus


class Store:
    def __init__(self) -> None:
        self.statuses: dict[UUID, RunStatus] = {}
        self.requests: dict[UUID, dict[str, object]] = {}

    def save(self, status: RunStatus) -> None:
        self.statuses[status.run_id] = status

    def get(self, run_id: UUID) -> RunStatus | None:
        return self.statuses.get(run_id)

    def save_request(self, run_id: UUID, request: dict[str, object]) -> None:
        self.requests[run_id] = request

    def get_request(self, run_id: UUID) -> dict[str, object] | None:
        return self.requests.get(run_id)

    def update(self, run_id: UUID, **fields: object) -> RunStatus:
        status = self.statuses[run_id]
        for key, value in fields.items():
            setattr(status, key, value)
        return status


class Queue:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def submission(
        self,
        tenant: str,
        payload: dict[str, Any],
        run_id: UUID,
        force: bool,
        snapshot_id: str | None = None,
        checkpoint_namespace: str | None = None,
        resume_row_id: int | None = None,
    ) -> str:
        self.calls.append(
            (
                "submission",
                tenant,
                payload,
                run_id,
                force,
                snapshot_id,
                checkpoint_namespace,
                resume_row_id,
            )
        )
        return "submission-task"

    def discovery(self, tenant: str, run_id: UUID) -> str:
        self.calls.append(("discovery", tenant, run_id))
        return "discovery-task"


def coordinator() -> tuple[RunCoordinator, Store, Queue, list[str]]:
    store, queue, tenants = Store(), Queue(), []
    runs = RunCoordinator(store, queue, lambda tenant: tenants.append(tenant))  # type: ignore[arg-type]
    return runs, store, queue, tenants


def test_enqueue_submission_and_discovery() -> None:
    runs, store, queue, tenants = coordinator()
    response = runs.enqueue_submission("atlas", {"entry_type": "url"}, "manual", force=True)
    assert tenants == ["atlas"]
    assert store.statuses[response.run_id].task_id == "submission-task"
    request = store.requests[response.run_id]
    assert request["force"] is True
    assert request["checkpoint_namespace"] == str(response.run_id)
    assert queue.calls[0][-2] == str(response.run_id)
    assert queue.calls[0][-1] is None

    discovery = runs.enqueue_discovery("nova")
    assert store.statuses[discovery.run_id].task_id == "discovery-task"
    assert queue.calls[-1][0] == "discovery"


def test_retry_preserves_snapshot_checkpoint_and_persisted_row_ownership() -> None:
    runs, store, queue, _ = coordinator()
    original = RunStatus(tenant="atlas", kind="scheduled-job")
    original.notification = {"row_id": 4758}
    store.save(original)
    store.save_request(
        original.run_id,
        {
            "kind": "scheduled-job",
            "payload": {"entry_type": "external"},
            "force": False,
            "snapshot_id": "batch-1",
            "checkpoint_namespace": "lineage-1",
        },
    )
    result = runs.retry(original.run_id)
    assert store.statuses[result.run_id].original_run_id == original.run_id
    assert queue.calls[-1][-3] == "batch-1"
    assert queue.calls[-1][-2] == "lineage-1"
    assert queue.calls[-1][-1] == 4758
    assert store.requests[result.run_id]["checkpoint_namespace"] == "lineage-1"
    assert store.requests[result.run_id]["resume_row_id"] == 4758


def test_repeated_retry_preserves_row_ownership_from_replay_request() -> None:
    runs, store, queue, _ = coordinator()
    original = RunStatus(tenant="nova", kind="manual")
    store.save(original)
    store.save_request(
        original.run_id,
        {
            "kind": "manual",
            "payload": {"entry_type": "external"},
            "force": False,
            "checkpoint_namespace": "lineage-1",
            "resume_row_id": 91,
        },
    )

    result = runs.retry(original.run_id)

    assert queue.calls[-1][-1] == 91
    assert store.requests[result.run_id]["resume_row_id"] == 91


def test_fresh_retry_starts_new_checkpoint_lineage_and_drops_resume_ownership() -> None:
    runs, store, queue, _ = coordinator()
    original = RunStatus(tenant="atlas", kind="manual")
    original.notification = {"row_id": 77}
    store.save(original)
    store.save_request(
        original.run_id,
        {
            "kind": "manual",
            "payload": {"entry_type": "external"},
            "force": True,
            "checkpoint_namespace": "old-lineage",
            "resume_row_id": 77,
        },
    )
    result = runs.retry(original.run_id, fresh=True)
    expected = str(result.run_id)
    assert store.requests[result.run_id]["checkpoint_namespace"] == expected
    assert "resume_row_id" not in store.requests[result.run_id]
    assert queue.calls[-1][-2] == expected
    assert queue.calls[-1][-1] is None


def test_retry_missing_run_raises() -> None:
    runs, _, _, _ = coordinator()
    try:
        runs.retry(uuid4())
    except KeyError:
        pass
    else:
        raise AssertionError("missing run should raise KeyError")
