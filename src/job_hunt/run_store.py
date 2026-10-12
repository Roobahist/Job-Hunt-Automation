from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from redis import Redis, WatchError

from job_hunt.domain.models import RunState, RunStatus
from job_hunt.redis_client import configure_redis_retry


class RunStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 604800) -> None:
        self.redis = configure_redis_retry(redis)
        self.ttl_seconds = ttl_seconds

    def save(self, status: RunStatus) -> None:
        status.updated_at = datetime.now(UTC)
        self.redis.setex(self._key(status.run_id), self.ttl_seconds, status.model_dump_json())

    def get(self, run_id: UUID) -> RunStatus | None:
        raw = cast(str | bytes | None, self.redis.get(self._key(run_id)))
        return RunStatus.model_validate_json(raw) if raw else None

    def save_request(self, run_id: UUID, request: dict[str, object]) -> None:
        self.redis.setex(f"{self._key(run_id)}:request", self.ttl_seconds, json.dumps(request, default=str))

    def get_request(self, run_id: UUID) -> dict[str, object] | None:
        raw = cast(str | bytes | None, self.redis.get(f"{self._key(run_id)}:request"))
        return json.loads(raw) if raw else None

    def update(
        self,
        run_id: UUID,
        *,
        state: RunState | None = None,
        stage: str | None = None,
        **fields: object,
    ) -> RunStatus:
        key = self._key(run_id)
        for _ in range(5):
            pipeline = self.redis.pipeline()
            try:
                pipeline.watch(key)  # type: ignore[no-untyped-call]
                raw = cast(str | bytes | None, pipeline.get(key))
                if raw is None:
                    raise KeyError(str(run_id))
                status = RunStatus.model_validate_json(raw)
                values = status.model_dump()
                if state is not None:
                    values["state"] = state
                if stage is not None:
                    values["stage"] = stage
                values.update(fields)
                updated = RunStatus.model_validate(values)
                updated.updated_at = datetime.now(UTC)
                pipeline.multi()
                pipeline.setex(key, self.ttl_seconds, updated.model_dump_json())
                pipeline.execute()
                return updated
            except WatchError:
                continue
            finally:
                pipeline.reset()
        raise RuntimeError(f"Run {run_id} changed too frequently to update safely")

    def merge_notification(self, run_id: UUID, **fields: object) -> RunStatus:
        """Merge notification metadata without losing progress written by another worker."""
        key = self._key(run_id)
        for _ in range(5):
            pipeline = self.redis.pipeline()
            try:
                pipeline.watch(key)  # type: ignore[no-untyped-call]
                raw = cast(str | bytes | None, pipeline.get(key))
                if raw is None:
                    raise KeyError(str(run_id))
                status = RunStatus.model_validate_json(raw)
                notification = dict(status.notification or {})
                incoming = dict(fields)
                timeline_value = incoming.pop("timeline", None)
                if isinstance(timeline_value, dict):
                    stored_timeline = notification.get("timeline")
                    timeline = dict(stored_timeline) if isinstance(stored_timeline, dict) else {}
                    for stage, stage_value in timeline_value.items():
                        stored_stage = timeline.get(str(stage))
                        existing_stage = dict(stored_stage) if isinstance(stored_stage, dict) else {}
                        if isinstance(stage_value, dict):
                            existing_stage.update(stage_value)
                        else:
                            existing_stage["value"] = stage_value
                        timeline[str(stage)] = existing_stage
                    notification["timeline"] = timeline
                notification.update(incoming)
                values = status.model_dump()
                values["notification"] = notification
                updated = RunStatus.model_validate(values)
                updated.updated_at = datetime.now(UTC)
                pipeline.multi()
                pipeline.setex(key, self.ttl_seconds, updated.model_dump_json())
                pipeline.execute()
                return updated
            except WatchError:
                continue
            finally:
                pipeline.reset()
        raise RuntimeError(f"Run {run_id} changed too frequently to update safely")

    @staticmethod
    def _key(run_id: UUID) -> str:
        return f"job-hunt:run:{run_id}"
