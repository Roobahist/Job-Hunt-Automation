from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from job_hunt import worker
from job_hunt.errors import ErrorKind, GatewayUnavailableError, ProviderError


class RetryCalled(Exception):
    pass


class FakeTask:
    def __init__(self, retries: int) -> None:
        self.request = SimpleNamespace(retries=retries)
        self.retry_kwargs: dict[str, Any] | None = None

    def retry(self, **kwargs: Any) -> None:
        self.retry_kwargs = kwargs
        raise RetryCalled


def test_gateway_unavailable_retry_plan_ignores_normal_retry_limit() -> None:
    task = FakeTask(worker._TASK_MAX_RETRIES + 100)
    exc = GatewayUnavailableError("LiteLLM gateway unavailable: connection refused")

    assert worker._retry_plan(task, exc) == (
        worker._TASK_MAX_RETRIES + 100,
        worker._GATEWAY_UNAVAILABLE_RETRY_SECONDS,
    )


def test_normal_retryable_provider_error_still_stops_at_configured_limit() -> None:
    task = FakeTask(worker._TASK_MAX_RETRIES)
    exc = ProviderError(
        "provider unavailable",
        ErrorKind.TRANSIENT_PROVIDER,
        retryable=True,
        provider="litellm",
        status_code=503,
    )

    assert worker._retry_plan(task, exc) is None


def test_gateway_defer_uses_task_default_unlimited_retry_limit() -> None:
    task = FakeTask(25)
    exc = GatewayUnavailableError("LiteLLM gateway unavailable: connection refused")

    with pytest.raises(RetryCalled):
        worker._defer_task(
            task,
            exc,
            tenant="test",
            run_id="00000000-0000-0000-0000-000000000000",
            stage="compatibility_filter",
            retries=25,
            countdown=worker._GATEWAY_UNAVAILABLE_RETRY_SECONDS,
        )

    assert task.retry_kwargs is not None
    assert task.retry_kwargs["max_retries"] is None
    assert task.retry_kwargs["countdown"] == 60


def test_llm_tasks_have_unlimited_task_default_for_gateway_deferral() -> None:
    assert worker.process_submission.max_retries is None
    assert worker.generate_documents.max_retries is None
