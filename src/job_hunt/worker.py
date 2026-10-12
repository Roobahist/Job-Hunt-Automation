from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID
from zoneinfo import ZoneInfo

from celery import Celery, chord  # type: ignore[import-untyped]
from celery.schedules import crontab  # type: ignore[import-untyped]
from pydantic import TypeAdapter
from redis import Redis

from job_hunt.application.discovery import build_search_url, normalize_discovery, search_criteria_active
from job_hunt.config import Settings, load_registry
from job_hunt.container import Container
from job_hunt.domain.models import Job, JobSubmission, RunState, RunStatus
from job_hunt.errors import ErrorKind, GatewayUnavailableError, WorkflowError
from job_hunt.integrations.discovery_skip import recent_baserow_linkedin_job_ids
from job_hunt.integrations.telegram import TelegramNotifier
from job_hunt.logging import configure_logging, logger
from job_hunt.recovery import (
    clear_pending_recovery,
    load_pending_recovery,
    notify_recovery_complete,
    recovery_window_seconds,
)
from job_hunt.retry import retry_transient
from job_hunt.run_store import RunStore
from job_hunt.state import RedisState

settings = Settings()
configure_logging(json_logs=settings.environment != "development")
celery_app = Celery("job_hunt", broker=settings.redis_url, backend=settings.redis_url)
celery_config: dict[str, object] = {
    "task_acks_late": True,
    "task_reject_on_worker_lost": True,
    "worker_prefetch_multiplier": 1,
    "result_expires": settings.run_ttl_seconds,
    "worker_send_task_events": True,
    "task_send_sent_event": True,
    "timezone": settings.scheduler_timezone,
    "enable_utc": True,
    "beat_schedule": {
        "dispatch-due-tenants": {
            "task": "job_hunt.dispatch_due_tenants",
            "schedule": crontab(minute=0),
        },
    },
    "task_routes": {
        "job_hunt.discover_tenant": {"queue": "fast"},
        "job_hunt.dispatch_due_tenants": {"queue": "fast"},
        "job_hunt.dispatch_end_of_day_tenants": {"queue": "fast"},
        "job_hunt.wait_for_hourly_persistence_then_eod": {"queue": "fast"},
        "job_hunt.end_of_day_catchup_tenant": {"queue": "fast"},
        "job_hunt.finish_end_of_day_tenant": {"queue": "fast"},
        "job_hunt.process_submission": {"queue": "fast"},
        "job_hunt.generate_documents": {"queue": "documents"},
        "job_hunt.init_job_notification": {"queue": "notifications"},
        "job_hunt.update_job_notification": {"queue": "notifications"},
        "job_hunt.notify_documents": {"queue": "notifications"},
    },
    "task_default_queue": "fast",
}
if settings.task_soft_time_limit_seconds > 0:
    celery_config["task_soft_time_limit"] = settings.task_soft_time_limit_seconds
if settings.task_time_limit_seconds > 0:
    celery_config["task_time_limit"] = settings.task_time_limit_seconds
celery_app.conf.update(**celery_config)

_TASK_MAX_RETRIES = settings.task_max_retries
_GATEWAY_UNAVAILABLE_RETRY_SECONDS = 60
_RATE_LIMIT_FALLBACK_SECONDS = settings.rate_limit_fallback_seconds
_TRANSIENT_BASE_DELAY_SECONDS = settings.transient_base_delay_seconds
_TRANSIENT_MAX_DELAY_SECONDS = settings.transient_max_delay_seconds
_END_OF_DAY_WINDOW_SECONDS = 24 * 60 * 60
_END_OF_DAY_PERSISTENCE_POLL_SECONDS = 5

_STAGE_ORDER = (
    "discovery",
    "normalization",
    "persistence",
    "compatibility_filter",
    "qualification",
    "documents_queue",
    "document_check",
    "tailoring",
    "rendering",
    "artifact_upload",
    "notification_queue",
    "telegram_finalization",
)
_STAGE_LABELS = {
    "discovery": "Discovered",
    "normalization": "Normalize job",
    "persistence": "Persist to Baserow",
    "compatibility_filter": "Compatibility filter",
    "qualification": "Qualification",
    "documents_queue": "Waiting for document worker",
    "document_check": "Check current Baserow status",
    "tailoring": "Tailor CV and cover letter",
    "rendering": "Render application files",
    "artifact_upload": "Upload artifacts",
    "notification_queue": "Waiting for Telegram worker",
    "telegram_finalization": "Finalize Telegram message",
}


def _redis() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


def _store() -> RunStore:
    return RunStore(_redis(), settings.run_ttl_seconds)


def _state() -> RedisState:
    return RedisState(_redis())


def _notifier() -> TelegramNotifier:
    return TelegramNotifier(
        settings.shared_telegram_token(),
        timeout_seconds=settings.telegram_request_timeout_seconds,
    )


def _notification_caption(
    job: Job,
    score: int | None,
    notification: dict[str, object] | None = None,
) -> str:
    published = job.published_at.strftime("%B %d, %Y") if job.published_at else "Not specified"
    job_id = job.external_id or str(job.internal_id)
    match_score = f"{score}/100" if score is not None else "Pending"
    meta = notification or {}
    current_stage = str(meta.get("current_stage") or "discovery")
    processing_state = str(meta.get("processing_state") or "processing")
    state_label = {
        "complete": "✅ Complete",
        "dropped": "⛔ Dropped",
        "failed": "❌ Failed",
        "deferred": "⏳ Waiting to retry",
    }.get(processing_state, f"⚙️ {_STAGE_LABELS.get(current_stage, current_stage)}")
    lines = [
        job.title,
        f"🏢 {job.company_name}",
        "",
        f"📍 Location: {job.location or 'Not specified'}",
        f"💼 Contract: {job.contract_type or 'Not specified'}",
        f"🎯 Match score: {match_score}",
        f"🆔 Job ID: {job_id}",
        f"🗓 Published: {published}",
        "",
        f"Status: {state_label}",
    ]
    timeline = meta.get("timeline")
    if isinstance(timeline, dict):
        for stage in _STAGE_ORDER:
            raw = timeline.get(stage)
            if not isinstance(raw, dict):
                continue
            label = _STAGE_LABELS.get(stage, stage)
            duration = raw.get("duration_seconds")
            if isinstance(duration, (int, float)):
                lines.append(f"✓ {label}: {float(duration):.1f}s")
                continue
            started = raw.get("started_at")
            if stage == current_stage and isinstance(started, str):
                try:
                    local = datetime.fromisoformat(started).astimezone(ZoneInfo(settings.scheduler_timezone))
                    lines.append(f"▶ {label}: started {local.strftime('%I:%M:%S %p').lstrip('0')}")
                except ValueError:
                    lines.append(f"▶ {label}: running")
    retry_in = meta.get("retry_in_seconds")
    if processing_state == "deferred" and isinstance(retry_in, (int, float)):
        lines.append(f"Retry in: {int(retry_in)}s")
    error_message = meta.get("error_message")
    if processing_state == "failed" and isinstance(error_message, str):
        lines.append(f"Error: {error_message[:160]}")
    return "\n".join(lines)[:1024]


def _retry_countdown(exc: WorkflowError, retries: int) -> int:
    if isinstance(exc, GatewayUnavailableError):
        return _GATEWAY_UNAVAILABLE_RETRY_SECONDS
    if exc.retry_after is not None and exc.retry_after > 0:
        return max(1, min(_TRANSIENT_MAX_DELAY_SECONDS, int(exc.retry_after)))
    if exc.kind == ErrorKind.RATE_LIMIT:
        return _RATE_LIMIT_FALLBACK_SECONDS
    exponential = _TRANSIENT_BASE_DELAY_SECONDS * (2**retries)
    return max(1, min(_TRANSIENT_MAX_DELAY_SECONDS, exponential))


def _retry_plan(task: Any, exc: Exception) -> tuple[int, int] | None:
    if not isinstance(exc, WorkflowError) or not exc.retryable:
        return None
    retries = int(getattr(task.request, "retries", 0))
    if isinstance(exc, GatewayUnavailableError):
        return retries, _retry_countdown(exc, retries)
    if retries >= _TASK_MAX_RETRIES:
        return None
    return retries, _retry_countdown(exc, retries)


def _resume_row_id(store: RunStore, run_id: UUID, retries: int) -> int | None:
    """Return the persisted row owned by this run only when Celery is retrying it."""
    if retries <= 0:
        return None
    status = store.get(run_id)
    notification = status.notification if status and isinstance(status.notification, dict) else {}
    row_id = notification.get("row_id")
    return row_id if isinstance(row_id, int) and not isinstance(row_id, bool) else None


def _defer_task(
    task: Any,
    exc: WorkflowError,
    *,
    tenant: str,
    run_id: str,
    stage: str,
    retries: int,
    countdown: int,
) -> NoReturn:
    gateway_unavailable = isinstance(exc, GatewayUnavailableError)
    retry_limit: int | None = None if gateway_unavailable else _TASK_MAX_RETRIES
    logger().warning(
        "task_deferred",
        tenant=tenant,
        run_id=run_id,
        stage=stage,
        error_kind=exc.kind,
        reason=str(exc),
        retry_in_seconds=countdown,
        retry_number=retries + 1,
        max_retries="unlimited" if gateway_unavailable else _TASK_MAX_RETRIES,
    )
    raise task.retry(exc=exc, countdown=countdown, max_retries=retry_limit)


def _log_terminal_failure(event: str, exc: Exception, **context: object) -> None:
    if isinstance(exc, WorkflowError):
        logger().error(
            event,
            **context,
            error_type=type(exc).__name__,
            error_kind=exc.kind,
            error_message=str(exc),
            retryable=exc.retryable,
        )
        return
    logger().exception(event, **context, error_type=type(exc).__name__)


def _automatic_discovery_slot(now: datetime, interval_hours: int) -> str | None:
    local = now.astimezone(ZoneInfo(settings.scheduler_timezone))
    if local.minute != 0 or local.hour % interval_hours != 0:
        return None
    return local.strftime("%Y-%m-%dT%H%z")


def _completed_day_slot(now: datetime) -> str | None:
    local = now.astimezone(ZoneInfo(settings.scheduler_timezone))
    if local.hour != 0 or local.minute != 0:
        return None
    return (local.date() - timedelta(days=1)).isoformat()


def _end_of_day_window_seconds(day_slot: str, now: datetime | None = None) -> int:
    timezone = ZoneInfo(settings.scheduler_timezone)
    current = (now or datetime.now(timezone)).astimezone(timezone)
    completed_day_midnight = datetime.fromisoformat(day_slot).replace(tzinfo=timezone)
    next_midnight = completed_day_midnight + timedelta(days=1)
    extra_seconds = max(
        0,
        int((current.astimezone(UTC) - next_midnight.astimezone(UTC)).total_seconds()),
    )
    return _END_OF_DAY_WINDOW_SECONDS + extra_seconds


def _end_of_day_flags(services: Any) -> tuple[bool, bool]:
    catchup_enabled = settings.end_of_day_catchup_enabled and services.config.linkedin_end_of_day_catchup_enabled
    cleanup_enabled = settings.end_of_day_dropped_cleanup_enabled and services.config.end_of_day_dropped_cleanup_enabled
    return catchup_enabled, cleanup_enabled


def _empty_end_of_day_counts() -> dict[str, int]:
    return {
        "criteria_total": 0,
        "criteria_active": 0,
        "skip_job_ids": 0,
        "discovered": 0,
        "duplicates": 0,
        "missed_jobs": 0,
        "queued": 0,
    }


def _queue_end_of_day_maintenance(
    tenant: str,
    day_slot: str,
    *,
    services: Any | None = None,
    redis: Redis | None = None,
    window_seconds: int = _END_OF_DAY_WINDOW_SECONDS,
) -> bool:
    resolved_services = services or Container(settings).tenant(tenant)
    catchup_enabled, cleanup_enabled = _end_of_day_flags(resolved_services)
    if not catchup_enabled and not cleanup_enabled:
        return False

    resolved_redis = redis or _redis()
    due_key = f"job-hunt:last-end-of-day-slot:{tenant}"
    if resolved_redis.get(due_key) == day_slot:
        return False

    run = RunStatus(tenant=tenant, kind="end-of-day")
    _store().save(run)
    if catchup_enabled:
        end_of_day_catchup_tenant.delay(tenant, str(run.run_id), window_seconds)
    else:
        finish_end_of_day_tenant.delay([], tenant, str(run.run_id), _empty_end_of_day_counts())
    resolved_redis.set(due_key, day_slot)
    logger().info(
        "end_of_day_maintenance_queued",
        tenant=tenant,
        run_id=str(run.run_id),
        schedule_timezone=settings.scheduler_timezone,
        schedule_day=day_slot,
        window_seconds=window_seconds,
        catchup_enabled=catchup_enabled,
        cleanup_enabled=cleanup_enabled,
    )
    return True


def _scheduled_job_payload(job: Job) -> dict[str, Any]:
    return {
        "entry_type": "external",
        "source": job.source,
        "external_job_id": job.external_id,
        "company_name": job.company_name,
        "job_title": job.title,
        "job_description": job.description,
        "job_url": job.url,
        "location": job.location,
        "contract_type": job.contract_type,
        "published_at": job.published_at.isoformat() if job.published_at else None,
    }


def _prepare_scheduled_submission(
    store: RunStore,
    tenant: str,
    job: Job,
    snapshot_id: str,
    *,
    kind: str = "scheduled-job",
) -> tuple[RunStatus, dict[str, Any], str]:
    child = RunStatus(tenant=tenant, kind=kind)
    checkpoint_namespace = str(child.run_id)
    store.save(child)
    now = datetime.now(UTC).isoformat()
    store.merge_notification(
        child.run_id,
        processing_state="processing",
        current_stage="discovery",
        timeline={
            "discovery": {
                "started_at": now,
                "completed_at": now,
                "duration_seconds": 0.0,
            }
        },
    )
    payload = _scheduled_job_payload(job)
    store.save_request(
        child.run_id,
        {
            "tenant": tenant,
            "payload": payload,
            "kind": kind,
            "force": False,
            "snapshot_id": snapshot_id,
            "checkpoint_namespace": checkpoint_namespace,
        },
    )
    return child, payload, checkpoint_namespace


def _progress_start(
    tenant: str,
    run_id: UUID,
    job: Job,
    chat_id: str,
    stage: str,
    *,
    row_id: int | None = None,
    score: int | None = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    fields: dict[str, object] = {
        "processing_state": "processing",
        "current_stage": stage,
        "retry_in_seconds": None,
        "timeline": {stage: {"started_at": now}},
    }
    if row_id is not None:
        fields["row_id"] = row_id
    if score is not None:
        fields["score"] = score
    _store().merge_notification(run_id, **fields)
    update_job_notification.delay(tenant, str(run_id), job.model_dump(mode="json"), chat_id)


def _progress_finish(run_id: UUID, stage: str) -> None:
    store = _store()
    status = store.get(run_id)
    started_at: str | None = None
    if status and isinstance(status.notification, dict):
        timeline = status.notification.get("timeline")
        if isinstance(timeline, dict):
            entry = timeline.get(stage)
            if isinstance(entry, dict) and isinstance(entry.get("started_at"), str):
                started_at = str(entry["started_at"])
    now = datetime.now(UTC)
    stage_update: dict[str, object] = {"completed_at": now.isoformat()}
    if started_at is not None:
        try:
            started = datetime.fromisoformat(started_at)
            stage_update["duration_seconds"] = max(0.0, (now - started).total_seconds())
        except ValueError:
            pass
    store.merge_notification(run_id, timeline={stage: stage_update})


def _progress_deferred(
    tenant: str,
    run_id: UUID,
    job: Job | None,
    chat_id: str | None,
    stage: str,
    countdown: int,
) -> None:
    _store().merge_notification(
        run_id,
        processing_state="deferred",
        current_stage=stage,
        retry_in_seconds=countdown,
    )
    if job is not None and chat_id is not None:
        update_job_notification.delay(tenant, str(run_id), job.model_dump(mode="json"), chat_id)


def _progress_terminal(
    tenant: str,
    run_id: UUID,
    job: Job | None,
    chat_id: str | None,
    state: str,
    *,
    row_id: int | None = None,
    score: int | None = None,
    error_message: str | None = None,
) -> None:
    fields: dict[str, object] = {"processing_state": state, "current_stage": state}
    if row_id is not None:
        fields["row_id"] = row_id
    if score is not None:
        fields["score"] = score
    if error_message is not None:
        fields["error_message"] = error_message
    _store().merge_notification(run_id, **fields)
    if job is not None and chat_id is not None:
        update_job_notification.delay(tenant, str(run_id), job.model_dump(mode="json"), chat_id)


def _ensure_notification(tenant: str, run_id: UUID, job: Job, chat_id: str) -> None:
    store = _store()
    status = store.get(run_id)
    notification = status.notification if status and isinstance(status.notification, dict) else {}
    if notification.get("init_queued"):
        return
    store.merge_notification(run_id, init_queued=True)
    init_job_notification.delay(tenant, str(run_id), job.model_dump(mode="json"), chat_id)


@celery_app.task(
    name="job_hunt.init_job_notification",
    bind=True,
    max_retries=_TASK_MAX_RETRIES,
    throws=(WorkflowError,),
)  # type: ignore[untyped-decorator]
def init_job_notification(
    self: Any,
    tenant: str,
    run_id: str,
    job_data: dict[str, Any],
    chat_id: str,
) -> dict[str, str]:
    identifier = UUID(run_id)
    store = _store()
    try:
        status = store.get(identifier)
        notification = status.notification if status and isinstance(status.notification, dict) else {}
        existing = notification.get("message_id")
        if existing:
            return {"state": "exists", "message_id": str(existing)}
        job = Job.model_validate(job_data)
        score = notification.get("score") if isinstance(notification.get("score"), int) else None
        row_id = notification.get("row_id") if isinstance(notification.get("row_id"), int) else None
        caption = _notification_caption(job, score, notification)
        message_id = _notifier().send_processing_message(
            chat_id,
            caption=caption,
            job_url=job.url,
            row_id=row_id,
        )
        store.merge_notification(
            identifier,
            message_id=message_id,
            state="processing",
            last_caption=caption,
        )
        return {"state": "sent", "message_id": message_id}
    except Exception as exc:
        plan = _retry_plan(self, exc)
        if plan is not None and isinstance(exc, WorkflowError):
            retries, countdown = plan
            _defer_task(
                self,
                exc,
                tenant=tenant,
                run_id=run_id,
                stage="telegram_init",
                retries=retries,
                countdown=countdown,
            )
        logger().warning("telegram_progress_init_failed", tenant=tenant, run_id=run_id, error=str(exc))
        return {"state": "failed", "message_id": ""}


@celery_app.task(
    name="job_hunt.update_job_notification",
    bind=True,
    max_retries=_TASK_MAX_RETRIES,
    throws=(WorkflowError,),
)  # type: ignore[untyped-decorator]
def update_job_notification(
    self: Any,
    tenant: str,
    run_id: str,
    job_data: dict[str, Any],
    chat_id: str,
) -> dict[str, str]:
    identifier = UUID(run_id)
    store = _store()
    try:
        status = store.get(identifier)
        notification = status.notification if status and isinstance(status.notification, dict) else {}
        job = Job.model_validate(job_data)
        score = notification.get("score") if isinstance(notification.get("score"), int) else None
        row_id = notification.get("row_id") if isinstance(notification.get("row_id"), int) else None
        message_id = notification.get("message_id")
        if not message_id:
            return {"state": "pending_init", "message_id": ""}
        caption = _notification_caption(job, score, notification)
        if notification.get("last_caption") == caption:
            return {"state": "unchanged", "message_id": str(message_id)}
        _notifier().edit_processing_message(
            chat_id,
            str(message_id),
            caption=caption,
            job_url=job.url,
            row_id=row_id,
            run_id=run_id,
        )
        store.merge_notification(identifier, last_caption=caption)
        return {"state": "updated", "message_id": str(message_id)}
    except Exception as exc:
        plan = _retry_plan(self, exc)
        if plan is not None and isinstance(exc, WorkflowError):
            retries, countdown = plan
            _defer_task(
                self,
                exc,
                tenant=tenant,
                run_id=run_id,
                stage="telegram_progress",
                retries=retries,
                countdown=countdown,
            )
        logger().warning("telegram_progress_update_failed", tenant=tenant, run_id=run_id, error=str(exc))
        return {"state": "failed", "message_id": ""}


@celery_app.task(
    name="job_hunt.notify_documents",
    bind=True,
    max_retries=_TASK_MAX_RETRIES,
    throws=(WorkflowError,),
)  # type: ignore[untyped-decorator]
def notify_documents(
    self: Any,
    tenant: str,
    run_id: str,
    paths: list[str],
    caption: str,
    chat_id: str,
    row_id: int,
    job_url: str,
    job_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    store = _store()
    identifier = UUID(run_id)
    job = Job.model_validate(job_data) if job_data is not None else None
    try:
        _progress_finish(identifier, "notification_queue")
        if job is not None:
            _progress_start(tenant, identifier, job, chat_id, "telegram_finalization", row_id=row_id)
        artifacts = [Path(path) for path in paths]
        archives = [path for path in artifacts if path.suffix.casefold() == ".zip"]
        if len(archives) != 1:
            raise ValueError("Final Telegram notification requires exactly one ZIP archive")
        status = store.get(identifier)
        notification = status.notification if status and isinstance(status.notification, dict) else {}
        message_id = notification.get("message_id")
        score = notification.get("score") if isinstance(notification.get("score"), int) else None
        notifier = _notifier()
        if message_id and job is not None:
            action_id = notifier.finalize_processing_message(
                chat_id,
                str(message_id),
                archives[0],
                caption=_notification_caption(job, score, notification),
                job_url=job_url,
                row_id=row_id,
                run_id=run_id,
            )
        else:
            action_id = retry_transient(
                notifier.send_application_bundle,
                chat_id,
                artifacts,
                caption=caption,
                job_url=job_url,
                row_id=row_id,
                run_id=run_id,
            )
        _progress_finish(identifier, "telegram_finalization")
        store.merge_notification(
            identifier,
            processing_state="complete",
            current_stage="complete",
            state="sent",
            message_id=action_id,
            action_message_id=action_id,
            artifact_count=1,
            row_id=row_id,
        )
        final_caption = caption
        if job is not None:
            final_status = store.get(identifier)
            final_notification = (
                final_status.notification if final_status and isinstance(final_status.notification, dict) else {}
            )
            final_score = final_notification.get("score") if isinstance(final_notification.get("score"), int) else None
            final_caption = _notification_caption(job, final_score, final_notification)
            notifier.edit_final_caption(
                chat_id,
                action_id,
                caption=final_caption,
                job_url=job_url,
                row_id=row_id,
                run_id=run_id,
            )
        store.merge_notification(identifier, last_caption=final_caption)
        store.update(identifier, state=RunState.SUCCEEDED, stage="complete", error=None)
        return {"state": "sent", "message_id": action_id}
    except Exception as exc:
        plan = _retry_plan(self, exc)
        if plan is not None and isinstance(exc, WorkflowError):
            retries, countdown = plan
            _progress_deferred(tenant, identifier, job, chat_id, "telegram_finalization", countdown)
            _defer_task(
                self,
                exc,
                tenant=tenant,
                run_id=run_id,
                stage="notification",
                retries=retries,
                countdown=countdown,
            )
        store.merge_notification(
            identifier,
            state="failed",
            processing_state="failed",
            error_message=str(exc),
        )
        store.update(
            identifier,
            state=RunState.FAILED,
            stage="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _log_terminal_failure("notification_failed", exc, tenant=tenant, run_id=run_id, stage="notification")
        raise


@celery_app.task(
    name="job_hunt.generate_documents",
    bind=True,
    max_retries=None,
    throws=(WorkflowError,),
)  # type: ignore[untyped-decorator]
def generate_documents(
    self: Any,
    tenant: str,
    job_data: dict[str, Any],
    run_id: str,
    row_id: int,
    score: int,
    snapshot_id: str | None = None,
    checkpoint_namespace: str | None = None,
) -> dict[str, Any]:
    store = _store()
    identifier = UUID(run_id)
    job = Job.model_validate(job_data)
    chat_id: str | None = None
    store.update(
        identifier,
        state=RunState.RUNNING,
        stage="documents",
        task_id=self.request.id,
        error=None,
    )
    try:
        snapshot = _state().get_snapshot(snapshot_id) if snapshot_id else None
        services = Container(settings).tenant(
            tenant,
            snapshot=snapshot,
            checkpoint_namespace=checkpoint_namespace or run_id,
        )
        chat_id = services.config.telegram_chat_id
        store.merge_notification(identifier, row_id=row_id, score=score)
        _progress_finish(identifier, "documents_queue")
        _progress_start(
            tenant,
            identifier,
            job,
            chat_id,
            "document_check",
            row_id=row_id,
            score=score,
        )

        def progress(stage: str, event: str) -> None:
            if event == "start":
                if stage == "tailoring":
                    _progress_finish(identifier, "document_check")
                _progress_start(
                    tenant,
                    identifier,
                    job,
                    chat_id or "",
                    stage,
                    row_id=row_id,
                    score=score,
                )
            else:
                _progress_finish(identifier, stage)

        result = services.workflow.generate_documents(
            job,
            run_id=identifier,
            row_id=row_id,
            score=score,
            master_cv=services.context.master_cv,
            prompts=services.prompts,
            applicant_filename=services.config.applicant_filename,
            progress=progress,
        )
        if not result.passed:
            _progress_finish(identifier, "document_check")
            _progress_terminal(
                tenant,
                identifier,
                job,
                chat_id,
                "dropped",
                row_id=row_id,
                score=score,
            )
            store.update(identifier, state=RunState.SKIPPED, stage="complete", error=None)
            return {"state": RunState.SKIPPED, "row_id": row_id, "published": False}

        _progress_start(
            tenant,
            identifier,
            job,
            chat_id,
            "notification_queue",
            row_id=row_id,
            score=score,
        )
        store.update(identifier, state=RunState.RUNNING, stage="notification_queued", error=None)
        if result.notification_paths:
            notify_documents.delay(
                tenant,
                run_id,
                list(result.notification_paths),
                _notification_caption(job, result.score),
                chat_id,
                result.row_id,
                job.url,
                job.model_dump(mode="json"),
            )
        return {
            "state": "notification_queued",
            "row_id": result.row_id,
            "published": result.artifacts_published,
        }
    except Exception as exc:
        plan = _retry_plan(self, exc)
        if plan is not None and isinstance(exc, WorkflowError):
            retries, countdown = plan
            _progress_deferred(tenant, identifier, job, chat_id, "documents", countdown)
            store.update(
                identifier,
                state=RunState.RUNNING,
                stage="documents_deferred",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "retry_in_seconds": countdown,
                },
            )
            _defer_task(
                self,
                exc,
                tenant=tenant,
                run_id=run_id,
                stage="documents",
                retries=retries,
                countdown=countdown,
            )
        _progress_terminal(
            tenant,
            identifier,
            job,
            chat_id,
            "failed",
            row_id=row_id,
            score=score,
            error_message=str(exc),
        )
        store.update(
            identifier,
            state=RunState.FAILED,
            stage="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _log_terminal_failure("document_generation_failed", exc, tenant=tenant, run_id=run_id, stage="documents")
        raise


@celery_app.task(
    name="job_hunt.process_submission",
    bind=True,
    max_retries=None,
    throws=(WorkflowError,),
)  # type: ignore[untyped-decorator]
def process_submission(
    self: Any,
    tenant: str,
    submission: dict[str, Any],
    run_id: str,
    force: bool = False,
    snapshot_id: str | None = None,
    checkpoint_namespace: str | None = None,
    resume_row_id: int | None = None,
) -> dict[str, Any]:
    store = _store()
    identifier = UUID(run_id)
    redis = store.redis
    job: Job | None = None
    chat_id: str | None = None
    active_stage = "normalization"
    retry_number = int(getattr(self.request, "retries", 0))
    owned_row_id = resume_row_id if resume_row_id is not None else _resume_row_id(store, identifier, retry_number)
    store.update(
        identifier,
        state=RunState.RUNNING,
        stage=active_stage,
        task_id=self.request.id,
        error=None,
    )
    try:
        snapshot = _state().get_snapshot(snapshot_id) if snapshot_id else None
        services = Container(settings).tenant(
            tenant,
            snapshot=snapshot,
            checkpoint_namespace=checkpoint_namespace or run_id,
        )
        chat_id = services.config.telegram_chat_id
        parsed: Any = TypeAdapter(JobSubmission).validate_python(submission)
        now = datetime.now(UTC).isoformat()
        store.merge_notification(
            identifier,
            processing_state="processing",
            current_stage=active_stage,
            retry_in_seconds=None,
            timeline={"normalization": {"started_at": now}},
        )
        job = services.normalizer.normalize(
            parsed,
            country=services.config.apify_proxy_country,
            max_concurrency=services.config.apify_max_concurrency,
        )
        _progress_finish(identifier, "normalization")

        active_stage = "persistence"
        lock = redis.lock(
            f"job-hunt:lock:{tenant}:{job.identity}",
            timeout=settings.job_lock_timeout_seconds,
            blocking_timeout=settings.job_lock_timeout_seconds,
        )
        if not lock.acquire(blocking=True):
            raise TimeoutError("Timed out waiting for an in-progress run of the same job")
        try:
            store.update(identifier, stage=active_stage)

            def progress(stage: str, event: str) -> None:
                nonlocal active_stage
                if event == "start":
                    active_stage = stage
                    _progress_start(tenant, identifier, job, chat_id or "", stage)
                else:
                    _progress_finish(identifier, stage)

            def persisted(row_id: int) -> None:
                # The row must exist before Telegram is allowed to expose the job.
                # This row_id also establishes same-run ownership for Celery retry.
                store.merge_notification(identifier, row_id=row_id)
                _ensure_notification(tenant, identifier, job, chat_id or "")

            qualification = services.workflow.persist_and_qualify(
                job,
                run_id=identifier,
                master_cv=services.context.master_cv,
                prompts=services.prompts,
                threshold=services.config.qualification_threshold,
                force=force,
                progress=progress,
                persisted=persisted,
                resume_row_id=owned_row_id,
            )
        finally:
            if lock.owned():
                lock.release()

        store.merge_notification(identifier, row_id=qualification.row_id, score=qualification.score)
        update_job_notification.delay(tenant, run_id, job.model_dump(mode="json"), chat_id)
        if not qualification.passed:
            _progress_terminal(
                tenant,
                identifier,
                job,
                chat_id,
                "dropped",
                row_id=qualification.row_id,
                score=qualification.score,
            )
            store.update(identifier, state=RunState.SKIPPED, stage="complete", error=None)
            return {
                "state": RunState.SKIPPED,
                "row_id": qualification.row_id,
                "published": False,
            }

        if qualification.score is None:
            raise RuntimeError("Passing qualification result must include a score")

        active_stage = "documents_queue"
        store.update(identifier, state=RunState.RUNNING, stage="documents_queued", error=None)
        _progress_start(
            tenant,
            identifier,
            job,
            chat_id,
            "documents_queue",
            row_id=qualification.row_id,
            score=qualification.score,
        )
        generate_documents.delay(
            tenant,
            job.model_dump(mode="json"),
            run_id,
            qualification.row_id,
            qualification.score,
            snapshot_id,
            checkpoint_namespace or run_id,
        )
        return {
            "state": "documents_queued",
            "row_id": qualification.row_id,
            "published": False,
        }
    except Exception as exc:
        plan = _retry_plan(self, exc)
        if plan is not None and isinstance(exc, WorkflowError):
            retries, countdown = plan
            _progress_deferred(tenant, identifier, job, chat_id, active_stage, countdown)
            store.update(
                identifier,
                state=RunState.RUNNING,
                stage=f"{active_stage}_deferred",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "retry_in_seconds": countdown,
                },
            )
            _defer_task(
                self,
                exc,
                tenant=tenant,
                run_id=run_id,
                stage=active_stage,
                retries=retries,
                countdown=countdown,
            )
        _progress_terminal(
            tenant,
            identifier,
            job,
            chat_id,
            "failed",
            error_message=str(exc),
        )
        store.update(
            identifier,
            state=RunState.FAILED,
            stage="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _log_terminal_failure("job_failed", exc, tenant=tenant, run_id=run_id, stage=active_stage)
        raise


@celery_app.task(
    name="job_hunt.wait_for_hourly_persistence_then_eod",
    bind=True,
    max_retries=None,
)  # type: ignore[untyped-decorator]
def wait_for_hourly_persistence_then_eod(
    self: Any,
    tenant: str,
    day_slot: str,
    child_run_ids: list[str],
) -> dict[str, int | bool]:
    store = _store()
    persisted = 0
    failed_before_persistence = 0
    pending: list[str] = []

    for child_run_id in dict.fromkeys(child_run_ids):
        status = store.get(UUID(child_run_id))
        if status is None:
            pending.append(child_run_id)
            continue
        notification = status.notification if isinstance(status.notification, dict) else {}
        row_id = notification.get("row_id")
        if isinstance(row_id, int) and not isinstance(row_id, bool):
            persisted += 1
            continue
        if status.state == RunState.FAILED:
            failed_before_persistence += 1
            continue
        pending.append(child_run_id)

    if pending:
        logger().info(
            "end_of_day_waiting_for_hourly_persistence",
            tenant=tenant,
            schedule_day=day_slot,
            child_runs=len(child_run_ids),
            persisted=persisted,
            failed_before_persistence=failed_before_persistence,
            pending=len(pending),
        )
        raise self.retry(countdown=_END_OF_DAY_PERSISTENCE_POLL_SECONDS, max_retries=None)

    window_seconds = _end_of_day_window_seconds(day_slot)
    queued = _queue_end_of_day_maintenance(tenant, day_slot, window_seconds=window_seconds)
    logger().info(
        "hourly_persistence_barrier_released",
        tenant=tenant,
        schedule_day=day_slot,
        child_runs=len(child_run_ids),
        persisted=persisted,
        failed_before_persistence=failed_before_persistence,
        window_seconds=window_seconds,
        end_of_day_queued=queued,
    )
    return {
        "child_runs": len(child_run_ids),
        "persisted": persisted,
        "failed_before_persistence": failed_before_persistence,
        "window_seconds": window_seconds,
        "end_of_day_queued": queued,
    }


@celery_app.task(name="job_hunt.discover_tenant", throws=(WorkflowError,))  # type: ignore[untyped-decorator]
def discover_tenant(tenant: str, run_id: str, end_of_day_slot: str | None = None) -> dict[str, int]:
    store = _store()
    identifier = UUID(run_id)
    pending_recovery: dict[str, Any] | None = None
    selected_window_seconds: int | None = None
    try:
        store.update(identifier, state=RunState.RUNNING, stage="discovery", error=None)
        services = Container(settings).tenant(tenant)
        pending_recovery = load_pending_recovery(store.redis, tenant)
        if pending_recovery is not None:
            selected_window_seconds = recovery_window_seconds(
                pending_recovery,
                now=datetime.now(UTC),
                timezone=ZoneInfo(settings.scheduler_timezone),
            )
        effective_window_seconds = selected_window_seconds or services.config.linkedin_schedule_interval_hours * 3600
        logger().warning(
            "automatic_discovery_window_selected",
            tenant=tenant,
            run_id=run_id,
            recovery=pending_recovery is not None,
            interval_hours=services.config.linkedin_schedule_interval_hours,
            window_seconds=effective_window_seconds,
            f_tpr=f"r{effective_window_seconds}",
            missed_count=int(pending_recovery.get("missed_count", 0)) if pending_recovery else 0,
            missed_slots=pending_recovery.get("missed_slots", []) if pending_recovery else [],
            lookback_hours=int(pending_recovery.get("lookback_hours", 0)) if pending_recovery else 0,
        )
        snapshot_id = run_id
        _state().set_snapshot(
            snapshot_id,
            services.snapshot(),
            ttl_seconds=settings.discovery_snapshot_ttl_seconds,
        )
        search_criteria_table_id = services.config.baserow_table_ids["searchCriteria"]
        all_criteria = list(services.baserow.iter_rows(search_criteria_table_id))
        criteria = [row for row in all_criteria if search_criteria_active(row)]
        urls: list[str] = []
        for row in criteria:
            url = build_search_url(
                services.config.linkedin_base_search_url,
                row,
                services.config.linkedin_schedule_interval_hours,
                window_seconds=selected_window_seconds,
            )
            row_id = row.get("id")
            if not isinstance(row_id, int) or isinstance(row_id, bool):
                raise ValueError("Active Search Criteria row is missing a valid Baserow row id")
            retry_transient(
                services.baserow.update_row,
                search_criteria_table_id,
                row_id,
                {"Generated URL": url},
            )
            row["Generated URL"] = url
            urls.append(url)
        rows = services.discovery.discover(urls, max_items=services.config.linkedin_max_items)
        jobs = normalize_discovery(
            rows,
            services.config.company_exclusions,
            services.config.title_exclusions,
        )

        new_jobs: list[Job] = []
        duplicates = 0
        for job in jobs:
            existing = retry_transient(services.repository.find, job)
            if existing is not None:
                duplicates += 1
                logger().info(
                    "discovery_duplicate_skipped",
                    tenant=tenant,
                    run_id=run_id,
                    job_identity=job.identity,
                    row_id=existing.get("id"),
                )
                continue
            new_jobs.append(job)

        child_run_ids: list[str] = []
        for job in new_jobs:
            child, payload, checkpoint_namespace = _prepare_scheduled_submission(
                store,
                tenant,
                job,
                snapshot_id,
            )
            child_run_id = str(child.run_id)
            child_run_ids.append(child_run_id)
            process_submission.delay(
                tenant,
                payload,
                child_run_id,
                False,
                snapshot_id,
                checkpoint_namespace,
            )
        counts = {
            "criteria_total": len(all_criteria),
            "criteria_active": len(criteria),
            "discovered": len(jobs),
            "duplicates": duplicates,
            "queued": len(new_jobs),
        }
        store.update(
            identifier,
            state=RunState.SUCCEEDED,
            stage="dispatched",
            counts=counts,
            error=None,
        )
        logger().info("discovery_dispatched", tenant=tenant, run_id=run_id, **counts)
        if end_of_day_slot is not None:
            wait_for_hourly_persistence_then_eod.delay(tenant, end_of_day_slot, child_run_ids)
            logger().info(
                "end_of_day_persistence_barrier_queued",
                tenant=tenant,
                run_id=run_id,
                schedule_day=end_of_day_slot,
                child_runs=len(child_run_ids),
            )
        if pending_recovery is not None and selected_window_seconds is not None:
            logger().warning(
                "missed_discovery_recovery_applied",
                tenant=tenant,
                run_id=run_id,
                missed_count=int(pending_recovery.get("missed_count", 0)),
                missed_slots=pending_recovery.get("missed_slots", []),
                window_seconds=selected_window_seconds,
                f_tpr=f"r{selected_window_seconds}",
                duplicates=duplicates,
                queued=len(new_jobs),
            )
            try:
                retry_transient(
                    notify_recovery_complete,
                    settings=settings,
                    chat_id=services.config.telegram_chat_id,
                    tenant=tenant,
                    marker=pending_recovery,
                    window_seconds=selected_window_seconds,
                    counts=counts,
                )
            except Exception as exc:
                logger().warning(
                    "missed_discovery_recovery_notification_failed",
                    tenant=tenant,
                    run_id=run_id,
                    error=str(exc),
                    marker_retained=True,
                )
            else:
                clear_pending_recovery(store.redis, tenant, pending_recovery)
                logger().info(
                    "missed_discovery_recovery_notification_sent",
                    tenant=tenant,
                    run_id=run_id,
                    missed_count=int(pending_recovery.get("missed_count", 0)),
                )
        return counts
    except Exception as exc:
        store.update(
            identifier,
            state=RunState.FAILED,
            stage="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _log_terminal_failure("discovery_failed", exc, tenant=tenant, run_id=run_id, stage="discovery")
        raise


@celery_app.task(name="job_hunt.finish_end_of_day_tenant", throws=(WorkflowError,))  # type: ignore[untyped-decorator]
def finish_end_of_day_tenant(
    _: list[dict[str, Any]],
    tenant: str,
    run_id: str,
    counts: dict[str, int],
) -> dict[str, int]:
    store = _store()
    identifier = UUID(run_id)
    try:
        services = Container(settings).tenant(tenant)
        cleanup_enabled = (
            settings.end_of_day_dropped_cleanup_enabled and services.config.end_of_day_dropped_cleanup_enabled
        )
        deleted = retry_transient(services.repository.delete_status, "dropped") if cleanup_enabled else 0
        final_counts = {**counts, "dropped_deleted": deleted}
        store.update(
            identifier,
            state=RunState.SUCCEEDED,
            stage="complete",
            counts=final_counts,
            error=None,
        )
        logger().info(
            "end_of_day_maintenance_completed",
            tenant=tenant,
            run_id=run_id,
            cleanup_enabled=cleanup_enabled,
            **final_counts,
        )
        return final_counts
    except Exception as exc:
        store.update(
            identifier,
            state=RunState.FAILED,
            stage="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _log_terminal_failure("end_of_day_cleanup_failed", exc, tenant=tenant, run_id=run_id, stage="cleanup")
        raise


@celery_app.task(name="job_hunt.end_of_day_catchup_tenant", throws=(WorkflowError,))  # type: ignore[untyped-decorator]
def end_of_day_catchup_tenant(
    tenant: str,
    run_id: str,
    window_seconds: int = _END_OF_DAY_WINDOW_SECONDS,
) -> dict[str, int]:
    store = _store()
    identifier = UUID(run_id)
    try:
        store.update(identifier, state=RunState.RUNNING, stage="end_of_day_discovery", error=None)
        services = Container(settings).tenant(tenant)
        effective_window_seconds = max(_END_OF_DAY_WINDOW_SECONDS, int(window_seconds))
        snapshot_id = run_id
        _state().set_snapshot(
            snapshot_id,
            services.snapshot(),
            ttl_seconds=settings.discovery_snapshot_ttl_seconds,
        )
        search_criteria_table_id = services.config.baserow_table_ids["searchCriteria"]
        all_criteria = list(services.baserow.iter_rows(search_criteria_table_id))
        criteria = [row for row in all_criteria if search_criteria_active(row)]
        urls = [
            build_search_url(
                services.config.linkedin_base_search_url,
                row,
                services.config.linkedin_schedule_interval_hours,
                window_seconds=effective_window_seconds,
            )
            for row in criteria
        ]
        skip_job_ids = recent_baserow_linkedin_job_ids(
            services.baserow,
            services.config.baserow_table_ids["jobs"],
            lookback_seconds=effective_window_seconds,
        )
        rows = services.end_of_day_discovery.discover(
            urls,
            max_items=services.config.linkedin_max_items,
            skip_job_ids=skip_job_ids,
        )
        jobs = normalize_discovery(
            rows,
            services.config.company_exclusions,
            services.config.title_exclusions,
        )

        new_jobs: list[Job] = []
        duplicates = 0
        for job in jobs:
            existing = retry_transient(services.repository.find, job)
            if existing is not None:
                duplicates += 1
                continue
            new_jobs.append(job)

        signatures: list[Any] = []
        for job in new_jobs:
            child, payload, checkpoint_namespace = _prepare_scheduled_submission(
                store,
                tenant,
                job,
                snapshot_id,
                kind="end-of-day-job",
            )
            signatures.append(
                process_submission.s(
                    tenant,
                    payload,
                    str(child.run_id),
                    False,
                    snapshot_id,
                    checkpoint_namespace,
                )
            )

        counts = {
            "criteria_total": len(all_criteria),
            "criteria_active": len(criteria),
            "skip_job_ids": len(skip_job_ids),
            "discovered": len(jobs),
            "duplicates": duplicates,
            "missed_jobs": len(new_jobs),
            "queued": len(new_jobs),
        }
        store.update(
            identifier,
            state=RunState.RUNNING,
            stage="waiting_for_qualification",
            counts=counts,
            error=None,
        )
        logger().info(
            "end_of_day_catchup_dispatched",
            tenant=tenant,
            run_id=run_id,
            actor="munificent_geastrum/apify-linkedin",
            window_seconds=effective_window_seconds,
            f_tpr=f"r{effective_window_seconds}",
            **counts,
        )
        if signatures:
            chord(signatures)(finish_end_of_day_tenant.s(tenant, run_id, counts))
        else:
            finish_end_of_day_tenant.delay([], tenant, run_id, counts)
        return counts
    except Exception as exc:
        store.update(
            identifier,
            state=RunState.FAILED,
            stage="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _log_terminal_failure(
            "end_of_day_catchup_failed",
            exc,
            tenant=tenant,
            run_id=run_id,
            stage="end_of_day_discovery",
        )
        raise


@celery_app.task(name="job_hunt.dispatch_end_of_day_tenants", throws=(WorkflowError,))  # type: ignore[untyped-decorator]
def dispatch_end_of_day_tenants() -> dict[str, int]:
    """Manual recovery entry point for the last completed Calgary day.

    Normal end-of-day maintenance is sequenced from the midnight scheduled discovery
    after its discovered jobs have reached Baserow.
    """
    redis = _redis()
    queued = 0
    failed = 0
    skipped = 0
    local_now = datetime.now(ZoneInfo(settings.scheduler_timezone))
    day_slot = (local_now.date() - timedelta(days=1)).isoformat()
    for key, bootstrap in load_registry(settings.registry_path).items():
        if not bootstrap.enabled:
            continue
        try:
            services = Container(settings).tenant(key)
            if _queue_end_of_day_maintenance(key, day_slot, services=services, redis=redis):
                queued += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            _log_terminal_failure("end_of_day_dispatch_failed", exc, tenant=key, stage="end_of_day_dispatch")
    return {"queued": queued, "failed": failed, "skipped": skipped}


@celery_app.task(name="job_hunt.dispatch_due_tenants", throws=(WorkflowError,))  # type: ignore[untyped-decorator]
def dispatch_due_tenants() -> dict[str, int]:
    redis = _redis()
    queued = 0
    failed = 0
    local_now = datetime.now(ZoneInfo(settings.scheduler_timezone)).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    completed_day_slot = _completed_day_slot(local_now)
    for key, bootstrap in load_registry(settings.registry_path).items():
        if not bootstrap.enabled:
            continue
        try:
            services = Container(settings).tenant(key)
            interval_hours = services.config.linkedin_schedule_interval_hours
            slot = _automatic_discovery_slot(local_now, interval_hours)
            if slot is None:
                continue
            due_key = f"job-hunt:last-auto-discovery-slot:{key}"
            if redis.get(due_key) == slot:
                continue
            redis.set(due_key, slot)
            run = RunStatus(tenant=key, kind="discovery")
            _store().save(run)
            end_of_day_slot: str | None = None
            if completed_day_slot is not None:
                catchup_enabled, cleanup_enabled = _end_of_day_flags(services)
                if catchup_enabled or cleanup_enabled:
                    end_of_day_slot = completed_day_slot
            discover_tenant.delay(key, str(run.run_id), end_of_day_slot)
            queued += 1
            logger().info(
                "automatic_discovery_queued",
                tenant=key,
                run_id=str(run.run_id),
                schedule_timezone=settings.scheduler_timezone,
                schedule_slot=slot,
                interval_hours=interval_hours,
                end_of_day_after_persistence=end_of_day_slot is not None,
                end_of_day_slot=end_of_day_slot,
            )
        except Exception as exc:
            failed += 1
            _log_terminal_failure("tenant_dispatch_failed", exc, tenant=key, stage="dispatch")
    return {"queued": queued, "failed": failed}
