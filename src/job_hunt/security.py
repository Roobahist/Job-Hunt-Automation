from __future__ import annotations

import hmac
import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from job_hunt.errors import ErrorKind, WorkflowError


def verify_bearer(authorization: str | None, expected: str) -> bool:
    if not authorization or not authorization.startswith("Bearer ") or not expected:
        return False
    return hmac.compare_digest(authorization[7:], expected)


def validate_public_url(
    url: str,
    resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise WorkflowError("Only public HTTP(S) URLs are accepted", ErrorKind.VALIDATION)
    if parts.username is not None or parts.password is not None:
        raise WorkflowError("Credential-bearing URLs are not accepted", ErrorKind.VALIDATION)
    try:
        records = resolver(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as exc:
        raise WorkflowError("URL host could not be resolved", ErrorKind.VALIDATION) from exc
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if not address.is_global:
            raise WorkflowError("Private or reserved network URLs are not accepted", ErrorKind.VALIDATION)


def fetch_public_text(
    url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = 2_000_000,
    max_redirects: int = 5,
) -> str:
    owned = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(20.0), follow_redirects=False)
    current = url
    try:
        for _ in range(max_redirects + 1):
            validate_public_url(current)
            with http.stream("GET", current, headers={"User-Agent": "job-hunt-automation/0.1"}) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise WorkflowError(
                            "Redirect omitted Location header",
                            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                        )
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not any(value in content_type for value in ("text/", "application/xhtml+xml", "application/json")):
                    raise WorkflowError("URL did not return a supported text document", ErrorKind.VALIDATION)
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise WorkflowError("URL response exceeds the size limit", ErrorKind.VALIDATION)
                    chunks.append(chunk)
                return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        raise WorkflowError("URL exceeded the redirect limit", ErrorKind.VALIDATION)
    finally:
        if owned:
            http.close()
