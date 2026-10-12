from __future__ import annotations

import hmac
import logging
from collections.abc import Callable, Mapping
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from redis import Redis

from job_hunt.application.fillout_reuse import resolve_existing_fillout
from job_hunt.application.normalization import fillout_submission
from job_hunt.application.runs import Queue, RunCoordinator
from job_hunt.config import Settings
from job_hunt.container import Container, TelegramRoute
from job_hunt.domain.models import (
    EnqueueResponse,
    EntryType,
    ExternalSubmission,
    JobSubmission,
    RetryResponse,
    RunStatus,
)
from job_hunt.errors import ConfigurationError, ProviderError, WorkflowError
from job_hunt.logging import configure_logging
from job_hunt.queueing import CeleryQueue
from job_hunt.run_store import RunStore
from job_hunt.security import verify_bearer

logger = logging.getLogger(__name__)


def _telegram_message_job_url(message: dict[str, object]) -> str:
    markup = message.get("reply_markup")
    keyboard = markup.get("inline_keyboard") if isinstance(markup, dict) else None
    if not isinstance(keyboard, list):
        return ""
    for row in keyboard:
        if not isinstance(row, list):
            continue
        for button in row:
            if isinstance(button, dict) and isinstance(button.get("url"), str):
                return str(button["url"])
    return ""


def _dropped_processing_caption(caption: str) -> str:
    lines: list[str] = []
    found_status = False
    for line in caption.splitlines():
        if line.startswith("Status:"):
            lines.append("Status: ⛔ Dropped manually")
            found_status = True
        elif line.startswith("Retry in:"):
            continue
        elif line.startswith("▶ "):
            stage = line[2:].split(":", 1)[0].strip()
            lines.append(f"■ {stage}: stopped manually")
        else:
            lines.append(line)
    if not found_status:
        lines.extend(["", "Status: ⛔ Dropped manually"])
    return "\n".join(lines)[:1024]


def _fillout_validation_context(payload: Mapping[str, object], field_ids: Mapping[str, str]) -> dict[str, object]:
    source = payload.get("submission", payload)
    if not isinstance(source, Mapping):
        source = payload

    flattened: dict[str, object] = dict(source)
    questions = source.get("questions", [])
    if isinstance(questions, list):
        for question in questions:
            if not isinstance(question, Mapping):
                continue
            question_id = question.get("id") or question.get("questionId")
            if question_id:
                flattened[str(question_id)] = question.get("value")

    def value(name: str) -> object | None:
        configured_id = field_ids.get(name)
        if configured_id and configured_id in flattened:
            return flattened[configured_id]
        return flattened.get(name)

    def nonempty(raw: object | None) -> bool:
        if raw is None:
            return False
        if isinstance(raw, str):
            return bool(raw.strip())
        if isinstance(raw, (list, tuple, dict, set)):
            return bool(raw)
        return True

    tracked_fields = (
        "entryType",
        "linkedinJobId",
        "companyName",
        "jobTitle",
        "jobDescription",
        "jobUrl",
        "location",
        "contractType",
        "publishedAt",
        "externalJobId",
        "pageContent",
    )
    return {
        "entry_type": str(value("entryType") or ""),
        "fields_nonempty": {name: nonempty(value(name)) for name in tracked_fields},
    }


def _fillout_intake_caption(payload: dict[str, object], run_id: UUID) -> tuple[str, str | None]:
    entry_type = str(payload.get("entry_type") or "")
    title = "Job submission received"
    company = "Pending normalization"
    job_id = "Pending"
    job_url: str | None = None
    status_text = "Received, starting normalization"

    if entry_type == "linkedin":
        raw_id = payload.get("linkedin_job_id")
        if isinstance(raw_id, int):
            job_id = str(raw_id)
            title = f"LinkedIn job {raw_id} received"
            job_url = f"https://www.linkedin.com/jobs/view/{raw_id}"
    elif entry_type == "external":
        title = str(payload.get("job_title") or title)
        company = str(payload.get("company_name") or company)
        job_id = str(payload.get("external_job_id") or "Pending")
        raw_url = payload.get("job_url")
        job_url = str(raw_url) if raw_url else None
    elif entry_type == "url":
        title = "Job URL received"
        status_text = "Fetching job page and extracting details with AI"
        raw_url = payload.get("job_url")
        job_url = str(raw_url) if raw_url else None
    elif entry_type == "ai_content":
        title = "Job content received"
        status_text = "Preparing content and extracting job details with AI"
        raw_url = payload.get("source_url")
        job_url = str(raw_url) if raw_url else None

    context_lines: list[str] = []
    if entry_type == "url" and job_url:
        context_lines.append(f"🔗 URL: {job_url}")

    caption = "\n".join(
        [
            title,
            f"🏢 {company}",
            *context_lines,
            "",
            "📥 Source: Fillout",
            f"🆔 Job ID: {job_id}",
            f"🔄 Run ID: {run_id}",
            "",
            f"Status: ⚙️ {status_text}",
        ]
    )
    return caption[:1024], job_url


def create_app(
    *,
    settings: Settings | None = None,
    queue: Queue | None = None,
    store: RunStore | None = None,
    container_factory: Callable[[], Container] | None = None,
) -> FastAPI:
    configured = settings or Settings()
    configure_logging(json_logs=configured.environment != "development")
    run_store = store or RunStore(
        Redis.from_url(configured.redis_url, decode_responses=True), configured.run_ttl_seconds
    )
    task_queue = queue or CeleryQueue()
    container = container_factory or (lambda: Container(configured))
    coordinator = RunCoordinator(run_store, task_queue, lambda tenant: container().registry.get(tenant))
    app = FastAPI(title="Job Hunt Automation", version="0.2.0")

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(
            "Request validation failed",
            extra={
                "event": "request_validation_failed",
                "method": request.method,
                "path": request.url.path,
                "validation_errors": exc.errors(),
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(ValidationError)
    async def model_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        logger.warning(
            "Submitted model values are invalid",
            extra={
                "event": "model_validation_failed",
                "method": request.method,
                "path": request.url.path,
                "validation_errors": exc.errors(include_url=False),
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Submitted values are incomplete or invalid",
                "details": exc.errors(include_url=False),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        labels = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            429: "rate_limit_exceeded",
            503: "service_unavailable",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": labels.get(exc.status_code, "request_error"),
                "message": str(exc.detail),
            },
            headers=exc.headers,
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "configuration_error", "message": str(exc)})

    @app.exception_handler(WorkflowError)
    async def workflow_error(_: Request, exc: WorkflowError) -> JSONResponse:
        code = 400 if exc.kind.value == "validation" else 502
        return JSONResponse(status_code=code, content={"error": exc.kind.value, "message": str(exc)})

    def operator(authorization: Annotated[str | None, Header()] = None) -> None:
        if not verify_bearer(authorization, configured.operator_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid operator bearer token")

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness() -> dict[str, str]:
        try:
            run_store.redis.ping()
        except Exception as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Redis is unavailable") from exc
        return {"status": "ready"}

    @app.post("/webhooks/fillout/{tenant}", response_model=EnqueueResponse, status_code=202)
    def fillout_webhook(
        tenant: str,
        payload: dict[str, object],
        authorization: Annotated[str | None, Header()] = None,
    ) -> EnqueueResponse:
        services = container().tenant(tenant)
        if not verify_bearer(authorization, services.context.bootstrap.secret("fillout")):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Fillout bearer token")
        try:
            submission = fillout_submission(payload, services.config.fillout_field_ids)
        except (ValidationError, ValueError, TypeError) as exc:
            context = _fillout_validation_context(payload, services.config.fillout_field_ids)
            logger.warning(
                "Fillout submission validation failed",
                extra={
                    "event": "fillout_submission_validation_failed",
                    "tenant": tenant,
                    "entry_type": context["entry_type"],
                    "fields_nonempty": context["fields_nonempty"],
                    "validation_error": str(exc),
                },
            )
            raise

        intake_payload = submission.model_dump(mode="json")
        existing = resolve_existing_fillout(
            submission,
            services.repository,
            services.normalizer.linkedin_template,
        )
        if existing.row is not None:
            if existing.job is not None:
                job = existing.job
                submission = ExternalSubmission.model_validate(
                    {
                        "entry_type": EntryType.EXTERNAL,
                        "company_name": job.company_name,
                        "job_title": job.title,
                        "job_description": job.description,
                        "job_url": job.url,
                        "location": job.location,
                        "contract_type": job.contract_type,
                        "published_at": job.published_at,
                        "external_job_id": job.external_id,
                        "source": job.source,
                        "reuse_existing_score": True,
                        "reuse_existing_data": True,
                    }
                )
            else:
                submission = submission.model_copy(update={"reuse_existing_score": True})

        submission_payload = submission.model_dump(mode="json")

        def acknowledge(run_id: UUID) -> None:
            caption, job_url = _fillout_intake_caption(intake_payload, run_id)
            run_store.merge_notification(
                run_id,
                init_queued=True,
                processing_state="processing",
                current_stage="normalization",
                retry_in_seconds=None,
            )
            document_content: str | None = None
            if intake_payload.get("entry_type") == "ai_content":
                raw_content = intake_payload.get("page_content")
                if isinstance(raw_content, str) and raw_content.strip():
                    document_content = raw_content
            try:
                message_id = services.notifier.send_processing_message(
                    services.config.telegram_chat_id,
                    caption=caption,
                    job_url=job_url,
                    row_id=None,
                    document_content=document_content,
                )
            except Exception as exc:
                run_store.merge_notification(
                    run_id,
                    init_queued=False,
                    intake_notification_error=str(exc),
                )
                logger.warning(
                    "Immediate Fillout Telegram acknowledgement failed",
                    extra={"tenant": tenant, "run_id": str(run_id), "error": str(exc)},
                )
                return
            run_store.merge_notification(
                run_id,
                message_id=message_id,
                state="processing",
                last_caption=caption,
            )

        return coordinator.enqueue_submission(
            tenant,
            submission_payload,
            "fillout",
            force=True,
            on_created=acknowledge,
        )

    def telegram_route(chat_id: str) -> TelegramRoute | None:
        # Telegram callback volume is low. Resolve against current Baserow configuration
        # so changing a tenant chat ID does not require an API restart to invalidate a cache.
        return container().telegram_route(chat_id)

    @app.post("/webhooks/telegram")
    def telegram_webhook(
        payload: dict[str, object],
        x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, bool]:
        expected = configured.shared_telegram_webhook_secret()
        if not x_telegram_bot_api_secret_token or not hmac.compare_digest(expected, x_telegram_bot_api_secret_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Telegram webhook secret")
        callback = payload.get("callback_query")
        if not isinstance(callback, dict):
            return {"ok": True}
        message = callback.get("message")
        chat = message.get("chat") if isinstance(message, dict) else None
        chat_id = str(chat.get("id")) if isinstance(chat, dict) and chat.get("id") is not None else ""
        routed = telegram_route(chat_id) if chat_id else None
        if routed is None:
            return {"ok": True}
        callback_id = str(callback.get("id") or "")
        data = str(callback.get("data") or "")
        response_text = "Action ignored"
        try:
            if data.startswith("status:"):
                _, status_key, raw_row_id = data.split(":", 2)
                row_id = int(raw_row_id)
                routed.repository.set_status(row_id, status_key)
                response_text = "Status updated"
                if status_key == "dropped" and isinstance(message, dict):
                    message_id = message.get("message_id")
                    caption = message.get("caption")
                    job_url = _telegram_message_job_url(message)
                    if message_id is not None and isinstance(caption, str) and job_url:
                        try:
                            routed.notifier.edit_processing_message(
                                chat_id,
                                str(message_id),
                                caption=_dropped_processing_caption(caption),
                                job_url=job_url,
                                row_id=None,
                            )
                        except ProviderError as exc:
                            logger.warning(
                                "Telegram dropped-state message update failed",
                                extra={
                                    "tenant": routed.tenant,
                                    "chat_id": chat_id,
                                    "row_id": row_id,
                                    "error": str(exc),
                                },
                            )
            elif data.startswith("retry:"):
                run_id = UUID(data.split(":", 1)[1])
                run = run_store.get(run_id)
                if run is None or run.tenant != routed.tenant:
                    raise ValueError("Run does not belong to Telegram chat tenant")
                coordinator.retry(run_id, fresh=True)
                response_text = "Regeneration queued"
        except (KeyError, ValueError):
            response_text = "Action could not be applied"
        if callback_id:
            try:
                routed.notifier.answer_callback(callback_id, response_text)
            except ProviderError as exc:
                logger.warning(
                    "Telegram callback acknowledgement failed",
                    extra={
                        "tenant": routed.tenant,
                        "chat_id": chat_id,
                        "error": str(exc),
                    },
                )
        return {"ok": True}

    @app.post(
        "/v1/tenants/{tenant}/jobs",
        response_model=EnqueueResponse,
        status_code=202,
        dependencies=[Depends(operator)],
    )
    def submit_job(tenant: str, submission: JobSubmission, force: bool = False) -> EnqueueResponse:
        return coordinator.enqueue_submission(
            tenant,
            submission.model_dump(mode="json"),
            "manual",
            force=force,
        )

    @app.post(
        "/v1/tenants/{tenant}/discoveries",
        response_model=EnqueueResponse,
        status_code=202,
        dependencies=[Depends(operator)],
    )
    def submit_discovery(tenant: str) -> EnqueueResponse:
        return coordinator.enqueue_discovery(tenant)

    @app.get("/v1/runs/{run_id}", response_model=RunStatus, dependencies=[Depends(operator)])
    def get_run(run_id: UUID) -> RunStatus:
        found = run_store.get(run_id)
        if not found:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
        return found

    @app.post(
        "/v1/runs/{run_id}/retry",
        response_model=RetryResponse,
        status_code=202,
        dependencies=[Depends(operator)],
    )
    def retry_run(run_id: UUID) -> RetryResponse:
        try:
            return coordinator.retry(run_id)
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run or replay data not found") from exc

    return app
