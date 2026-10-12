from __future__ import annotations

from typing import Any
from uuid import UUID

from job_hunt.worker import discover_tenant, process_submission


class CeleryQueue:
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
        task = process_submission.delay(
            tenant,
            payload,
            str(run_id),
            force,
            snapshot_id,
            checkpoint_namespace,
            resume_row_id,
        )
        return str(task.id)

    def discovery(self, tenant: str, run_id: UUID) -> str:
        task = discover_tenant.delay(tenant, str(run_id))
        return str(task.id)
