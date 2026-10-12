from __future__ import annotations


def test_api_application_imports() -> None:
    """Catch broken API import graphs before a deployment reaches container startup."""
    from job_hunt.api.app import create_app

    assert callable(create_app)


def test_worker_application_imports() -> None:
    """Catch worker-only import failures before queue consumers are replaced."""
    from job_hunt.worker import celery_app

    assert celery_app is not None


def test_runtime_contract_symbols_are_available() -> None:
    """Keep cross-module runtime contracts synchronized with their concrete implementations."""
    from job_hunt.domain.models import EnqueueResponse, RetryResponse, RunState, RunStatus
    from job_hunt.integrations.baserow import BaserowClient

    assert RunState.QUEUED.value == "queued"
    assert all(symbol is not None for symbol in (RunStatus, EnqueueResponse, RetryResponse))
    assert callable(getattr(BaserowClient, "list_fields", None))
