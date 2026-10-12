from __future__ import annotations

import hashlib
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

from job_hunt.domain.models import Job

MAX_SAFE_INTEGER = (1 << 53) - 1


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, "", ""))


def _usable_http_url(url: str) -> bool:
    parts = urlsplit(url.strip())
    return parts.scheme.lower() in {"http", "https"} and bool(parts.hostname)


def job_identity(source: str, external_id: str | None, url: str, *, fallback_text: str = "") -> str:
    if external_id and external_id.strip():
        return f"{source.strip().lower()}:{external_id.strip()}"
    if _usable_http_url(url):
        return canonicalize_url(url)
    normalized_fallback = fallback_text.strip()
    if normalized_fallback:
        digest = hashlib.sha256(normalized_fallback.encode()).hexdigest()
        return f"{source.strip().lower()}:content:{digest}"
    return f"{source.strip().lower()}:content:empty"


def stable_job_id(identity: str, *, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{identity}:{salt}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") & MAX_SAFE_INTEGER
    return value or 1


def assign_identity(job: Job, collision: Callable[[int, str], bool] | None = None) -> Job:
    fallback_text = "\n".join((job.company_name, job.title, job.description))
    identity = job_identity(job.source, job.external_id, job.url, fallback_text=fallback_text)
    salt = 0
    candidate = stable_job_id(identity, salt=salt)
    while collision and collision(candidate, identity):
        salt += 1
        candidate = stable_job_id(identity, salt=salt)
    canonical_url = canonicalize_url(job.url) if _usable_http_url(job.url) else ""
    return job.model_copy(update={"identity": identity, "internal_id": candidate, "url": canonical_url})
