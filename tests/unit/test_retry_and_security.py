from __future__ import annotations

import socket

import pytest

from job_hunt.errors import ErrorKind, WorkflowError
from job_hunt.retry import retry_transient
from job_hunt.security import validate_public_url, verify_bearer


def test_bearer_uses_exact_scheme_and_value() -> None:
    assert verify_bearer("Bearer correct", "correct")
    assert not verify_bearer("bearer correct", "correct")
    assert not verify_bearer("Bearer wrong", "correct")
    assert not verify_bearer(None, "correct")


def test_ssrf_blocks_private_and_accepts_public() -> None:
    def private(*_: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    def public(*_: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

    with pytest.raises(WorkflowError, match="Private"):
        validate_public_url("http://localhost/x", private)
    validate_public_url("https://example.com/x", public)
    with pytest.raises(WorkflowError, match="HTTP"):
        validate_public_url("file:///etc/passwd", public)


def test_retry_transient_only_retries_retryable_errors() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise WorkflowError("later", ErrorKind.TRANSIENT_PROVIDER, retryable=True, retry_after=0)
        return "ok"

    assert retry_transient(operation, attempts=3, sleep=sleeps.append) == "ok"
    assert calls == 3
    with pytest.raises(WorkflowError):
        retry_transient(lambda: (_ for _ in ()).throw(WorkflowError("bad", ErrorKind.VALIDATION)))
