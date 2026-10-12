from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from job_hunt.domain.models import EnqueueResponse, RetryResponse, RunStatus
from job_hunt.run_store import RunStore

RunCreated = Callable[[UUID], None]


class Queue(Protocol):
    def submission(
        self,
        tenant: str,
        payload: dict[str, Any],
        run_id: UUID,
        force: bool,
        snapshot_id: str | None = None,
        checkpoint_namespace: str | None = None,
        resume_row_id: int | None = None,
    ) -> str: ...

    def discovery(self, tenant: str, run_id: UUID) -> str: ...


class RunCoordinator:
    def __init__(
        self,
        store: RunStore,
        queue: Queue,
        tenant_exists: Callable[[str], object],
    ) -> None:
        self.store = store
        self.queue = queue
        self.tenant_exists = tenant_exists

    def _queue_submission(
        self,
        tenant: str,
        payload: dict[str, Any],
        run_id: UUID,
        force: bool,
        snapshot_id: str | None,
        checkpoint_namespace: str,
        resume_row_id: int | None = None,
    ) -> str:
        return self.queue.submission(
            tenant,
            payload,
            run_id,
            force,
            snapshot_id=snapshot_id,
            checkpoint_namespace=checkpoint_namespace,
            resume_row_id=resume_row_id,
        )

    def enqueue_submission(
        self,
        tenant: str,
        payload: dict[str, Any],
        kind: str,
        *,
        force: bool,
        snapshot_id: str | None = None,
        on_created: RunCreated | None = None,
    ) -> EnqueueResponse:
        self.tenant_exists(tenant)
        run = RunStatus(tenant=tenant, kind=kind)
        checkpoint_namespace = str(run.run_id)
        self.store.save(run)
        request: dict[str, object] = {
            "tenant": tenant,
            "payload": payload,
            "kind": kind,
            "force": force,
            "checkpoint_namespace": checkpoint_namespace,
        }
        if snapshot_id:
            request["snapshot_id"] = snapshot_id
        self.store.save_request(run.run_id, request)

        if on_created is not None:
            on_created(run.run_id)

        task_id = self._queue_submission(
            tenant,
            payload,
            run.run_id,
            force,
            snapshot_id,
            checkpoint_namespace,
        )
        self.store.update(run.run_id, task_id=task_id)
        return EnqueueResponse(run_id=run.run_id)

    def enqueue_discovery(self, tenant: str) -> EnqueueResponse:
        self.tenant_exists(tenant)
        run = RunStatus(tenant=tenant, kind="discovery")
        self.store.save(run)
        self.store.save_request(run.run_id, {"tenant": tenant, "kind": "discovery"})
        task_id = self.queue.discovery(tenant, run.run_id)
        self.store.update(run.run_id, task_id=task_id)
        return EnqueueResponse(run_id=run.run_id)

    @staticmethod
    def _row_id(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @classmethod
    def _persisted_row_id(cls, status: RunStatus) -> int | None:
        notification = status.notification if isinstance(status.notification, dict) else {}
        return cls._row_id(notification.get("row_id"))

    def retry(self, run_id: UUID, *, fresh: bool = False) -> RetryResponse:
        original = self.store.get(run_id)
        request = self.store.get_request(run_id)
        if not original or not request:
            raise KeyError(str(run_id))
        retry = RunStatus(tenant=original.tenant, kind=original.kind, original_run_id=run_id)
        replay = dict(request)
        resume_row_id: int | None = None
        if replay["kind"] != "discovery":
            replay["checkpoint_namespace"] = (
                str(retry.run_id) if fresh else str(replay.get("checkpoint_namespace") or original.run_id)
            )
            if not fresh:
                # Prefer the durable row recorded by the run itself. A replay may
                # also already carry ownership from an earlier replay in the chain,
                # which keeps repeated operator retries safe before the task reaches
                # its persistence callback again.
                resume_row_id = self._persisted_row_id(original) or self._row_id(replay.get("resume_row_id"))
                if resume_row_id is not None:
                    replay["resume_row_id"] = resume_row_id
            else:
                replay.pop("resume_row_id", None)
        self.store.save(retry)
        self.store.save_request(retry.run_id, replay)
        if replay["kind"] == "discovery":
            task_id = self.queue.discovery(original.tenant, retry.run_id)
        else:
            payload = replay.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Stored run replay payload must be an object")
            task_id = self._queue_submission(
                original.tenant,
                dict(payload),
                retry.run_id,
                fresh or bool(replay.get("force", False)),
                str(replay["snapshot_id"]) if replay.get("snapshot_id") else None,
                str(replay["checkpoint_namespace"]),
                resume_row_id=resume_row_id,
            )
        self.store.update(retry.run_id, task_id=task_id)
        return RetryResponse(original_run_id=run_id, run_id=retry.run_id)
