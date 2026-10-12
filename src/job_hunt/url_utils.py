from __future__ import annotations

from urllib.parse import urlsplit


def valid_http_url(value: object) -> str | None:
    """Return a normalized HTTP(S) URL, or None when the value is not usable as a link."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = urlsplit(text)
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return None
    return text
