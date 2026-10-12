from __future__ import annotations

import logging
import re
from collections.abc import MutableMapping
from typing import Any, cast

import structlog

from job_hunt.job_trace import write_job_trace

_SECRET_PATTERN = re.compile(r"(authorization|token|api[_-]?key|secret|password)", re.I)
_UNTRUNCATED_LOG_FIELDS = frozenset({"raw_output"})


def _redact(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in list(event):
        if _SECRET_PATTERN.search(key):
            event[key] = "[REDACTED]"
        elif key not in _UNTRUNCATED_LOG_FIELDS and isinstance(event[key], str) and len(event[key]) > 2000:
            event[key] = event[key][:2000] + "…[truncated]"
    return event


def configure_logging(*, json_logs: bool) -> None:
    # Some application modules use the standard-library logging API while
    # structured pipeline events use structlog. Ensure both paths have a
    # usable INFO-level handler. basicConfig is intentionally non-forcing so
    # Uvicorn and Celery can keep their own handlers when they already exist.
    logging.basicConfig(level=logging.INFO)

    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            write_job_trace,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def logger() -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger())
