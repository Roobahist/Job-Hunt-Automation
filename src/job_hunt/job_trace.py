from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def trace_path(root: Path, run_id: str, job_id: str) -> Path:
    safe_job_id = _SAFE_ID.sub("-", job_id).strip("-") or "unknown"
    return root / run_id / f"job-{safe_job_id}.log"


@contextmanager
def job_trace_context(path: Path, **context: object) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = structlog.contextvars.bind_contextvars(job_trace_path=str(path), **context)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


def _short(value: object, limit: int = 240) -> str:
    if isinstance(value, str):
        text = " ".join(value.split())
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fallback_target(event: Mapping[str, Any]) -> str:
    targets = event.get("fallback_targets")
    if isinstance(targets, list) and targets:
        return str(targets[0])
    target = event.get("fallback_to") or event.get("resolved_group")
    return str(target) if target else "next configured fallback"


def _format_event(event: Mapping[str, Any]) -> str:
    name = str(event.get("event") or "event")
    level = str(event.get("level") or "info").upper()
    timestamp = str(event.get("timestamp") or "")
    prefix = f"{timestamp} {level}".strip()

    if name == "llm_generation":
        usage_value = event.get("usage")
        usage: Mapping[str, Any] = usage_value if isinstance(usage_value, Mapping) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        details_value = usage.get("prompt_tokens_details")
        token_details: Mapping[str, Any] = details_value if isinstance(details_value, Mapping) else {}
        detail_cached = token_details.get("cached_tokens", 0)
        cached = int(usage.get("prompt_cache_hit_tokens") or detail_cached or 0)
        model = event.get("upstream_model") or event.get("model_group") or "unknown-model"
        repaired = "yes" if event.get("repaired") else "no"
        return (
            f"{prefix} LLM {event.get('prompt_key')}: {model} via {event.get('model_group')} "
            f"input={prompt_tokens} cached={cached} output={output_tokens} repair={repaired} "
            f"latency={int(event.get('latency_ms') or 0)}ms"
        )

    if name == "llm_group_failed":
        status = event.get("status_code") or event.get("fallback_status_code")
        status_text = f", HTTP {status}" if status else ""
        group = event.get("attempted_group") or event.get("group") or event.get("requested_group")
        reason = _short(event.get("reason") or event.get("fallback_reason") or "")
        return (
            f"{prefix} LLM {event.get('prompt_key')}: {group} unavailable "
            f"({event.get('error_kind') or 'provider_error'}{status_text}); "
            f"trying {_fallback_target(event)}; reason={reason}"
        )

    if name == "llm_group_succeeded" and event.get("fallback_used"):
        return (
            f"{prefix} LLM {event.get('prompt_key')}: fallback succeeded on "
            f"{event.get('resolved_group') or event.get('group')}"
        )

    if name == "repair_preservation_checked":
        verdict = "accepted" if event.get("accepted") else "rejected"
        return (
            f"{prefix} REPAIR {event.get('prompt_key')} attempt={event.get('attempt')} {verdict}: "
            f"token_delta={event.get('token_delta')} similarity={event.get('token_similarity')} "
            f"size_ratio={event.get('size_ratio')}"
        )

    if name == "repair_retrying":
        return (
            f"{prefix} REPAIR {event.get('prompt_key')}: repair changed content too much; "
            f"retrying with stricter preservation instructions"
        )

    if name == "task_deferred":
        return (
            f"{prefix} RETRY stage={event.get('stage')} in={event.get('retry_in_seconds')}s "
            f"attempt={event.get('retry_number')}/{event.get('max_retries')} "
            f"reason={_short(event.get('reason') or '')}"
        )

    if name.endswith("failed") or name.endswith("error"):
        return (
            f"{prefix} ERROR {name}: "
            f"{_short(event.get('error_message') or event.get('reason') or event.get('error') or '')}"
        )

    keys = [
        key
        for key in (
            "stage",
            "row_id",
            "score",
            "passed",
            "compatible",
            "forced",
            "reprocessed",
            "reason",
        )
        if key in event
    ]
    event_details = " ".join(f"{key}={_short(event[key])}" for key in keys)
    return f"{prefix} {name}{(' ' + event_details) if event_details else ''}"


def write_job_trace(_: Any, __: str, event: MutableMapping[str, Any]) -> Mapping[str, Any]:
    raw_path = event.get("job_trace_path")
    if not isinstance(raw_path, str) or not raw_path:
        return event
    path = Path(raw_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = _format_event(event)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8", errors="replace"))
        finally:
            os.close(fd)
    except OSError:
        # Per-job tracing is observability only and must never break job processing.
        pass
    event.pop("job_trace_path", None)
    return event
