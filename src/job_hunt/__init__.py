"""Multi-tenant job-hunt automation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from celery.signals import task_failure  # type: ignore[import-untyped]
from redis import Redis

__version__ = "0.1.0"

_LOGGER = logging.getLogger(__name__)
_FAILURE_TASK_ARGUMENTS = {
    "job_hunt.process_submission": (0, 2),
    "job_hunt.generate_documents": (0, 2),
    "job_hunt.notify_documents": (0, 1),
}


def _task_value(
    args: Sequence[object] | None,
    kwargs: dict[str, object] | None,
    key: str,
    position: int,
) -> object | None:
    if kwargs and key in kwargs:
        return kwargs[key]
    if args is not None and len(args) > position:
        return args[position]
    return None


def _failure_caption(caption: str, exception: BaseException) -> str:
    lines = [line for line in caption.splitlines() if not line.startswith("Error:")]
    status_line = "Status: ❌ Failed"
    status_index = next((index for index, line in enumerate(lines) if line.startswith("Status:")), None)
    if status_index is None:
        lines.extend(("", status_line))
    else:
        lines[status_index] = status_line

    error_text = f"Error: {type(exception).__name__}: {exception}"
    base = "\n".join(lines).rstrip()
    remaining = 1024 - len(base) - 1
    if remaining <= 0:
        return base[:1024]
    return f"{base}\n{error_text[:remaining]}"


def _notification_context(
    task_name: str,
    args: Sequence[object] | None,
    kwargs: dict[str, object] | None,
) -> tuple[str, UUID] | None:
    positions = _FAILURE_TASK_ARGUMENTS.get(task_name)
    if positions is None:
        return None
    tenant = _task_value(args, kwargs, "tenant", positions[0])
    run_id = _task_value(args, kwargs, "run_id", positions[1])
    if not isinstance(tenant, str) or not tenant:
        return None
    try:
        identifier = UUID(str(run_id))
    except (TypeError, ValueError):
        return None
    return tenant, identifier


def _update_failed_placeholder(
    task_name: str,
    exception: BaseException,
    args: Sequence[object] | None,
    kwargs: dict[str, object] | None,
) -> None:
    context = _notification_context(task_name, args, kwargs)
    if context is None:
        return
    tenant, run_id = context

    try:
        # Imports stay local so package initialization remains lightweight and avoids circular imports.
        from job_hunt.config import Settings
        from job_hunt.container import Container
        from job_hunt.integrations.telegram import TelegramNotifier
        from job_hunt.run_store import RunStore

        settings = Settings()
        store = RunStore(
            Redis.from_url(settings.redis_url, decode_responses=True),
            settings.run_ttl_seconds,
        )
        status = store.get(run_id)
        notification = status.notification if status and isinstance(status.notification, dict) else {}
        message_id = notification.get("message_id")
        last_caption = notification.get("last_caption")
        if not message_id or not isinstance(last_caption, str):
            return
        if "Status: ❌ Failed" in last_caption and "Error:" in last_caption:
            return

        services = Container(settings).tenant(tenant)
        caption = _failure_caption(last_caption, exception)
        TelegramNotifier(
            settings.shared_telegram_token(),
            timeout_seconds=settings.telegram_request_timeout_seconds,
        ).edit_processing_message(
            services.config.telegram_chat_id,
            str(message_id),
            caption=caption,
            job_url=None,
            row_id=None,
        )
        store.merge_notification(
            run_id,
            state="failed",
            processing_state="failed",
            error_message=str(exception),
            last_caption=caption,
        )
    except Exception:
        _LOGGER.exception(
            "Failed to update Telegram placeholder after task failure",
            extra={"task_name": task_name, "tenant": tenant, "run_id": str(run_id)},
        )


@task_failure.connect  # type: ignore[untyped-decorator]
def _on_task_failure(
    sender: Any = None,
    exception: BaseException | None = None,
    args: Sequence[object] | None = None,
    kwargs: dict[str, object] | None = None,
    **_: object,
) -> None:
    if exception is None:
        return
    task_name = str(getattr(sender, "name", ""))
    _update_failed_placeholder(task_name, exception, args, kwargs)
