from __future__ import annotations

import re
from html.parser import HTMLParser

_WHITESPACE = re.compile(r"\s+")
_BLOCKED_TAGS = {"script", "style", "noscript", "svg", "template"}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._blocked_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in _BLOCKED_TAGS:
            self._blocked_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in _BLOCKED_TAGS and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._blocked_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "\n".join(self._parts)


def prepare_page_content(content: str) -> str:
    """Remove non-visible page markup without clipping visible source content."""
    candidate = content
    if "<" in content and ">" in content:
        parser = _VisibleTextParser()
        try:
            parser.feed(content)
            parser.close()
            visible = parser.text()
            if visible.strip():
                candidate = visible
        except Exception:
            candidate = content

    return _WHITESPACE.sub(" ", candidate).strip()
