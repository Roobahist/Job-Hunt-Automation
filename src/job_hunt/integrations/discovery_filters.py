from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from job_hunt.ports import DiscoveryProvider

_ALLOWED_SUFFIXES = ("", "s", "es", "ed", "ing", "er", "ers")


def _normalize_filter_text(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _has_exact_bounded_match(value: str, keyword: str) -> bool:
    start = 0
    while True:
        index = value.find(keyword, start)
        if index < 0:
            return False
        end = index + len(keyword)
        starts_cleanly = index == 0 or not _is_word_char(value[index - 1])
        ends_cleanly = end == len(value) or not _is_word_char(value[end])
        if starts_cleanly and ends_cleanly:
            return True
        start = index + 1


def _keyword_matches(value: str, keyword: str) -> bool:
    if keyword not in value:
        return False

    if not keyword.isalnum():
        return _has_exact_bounded_match(value, keyword)

    start = 0
    while True:
        index = value.find(keyword, start)
        if index < 0:
            return False
        end = index + len(keyword)
        if index > 0 and _is_word_char(value[index - 1]):
            start = index + 1
            continue

        token_end = end
        while token_end < len(value) and _is_word_char(value[token_end]):
            token_end += 1
        if value[end:token_end] in _ALLOWED_SUFFIXES:
            return True
        start = index + 1


def _row_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


class FilteredDiscoveryProvider:
    """Apply tenant discovery rules after provider retrieval without affecting direct fetches."""

    def __init__(
        self,
        provider: DiscoveryProvider,
        *,
        company_inclusions: Sequence[str] = (),
        company_exclusions: Sequence[str] = (),
        title_inclusions: Sequence[str] = (),
        title_exclusions: Sequence[str] = (),
        description_inclusions: Sequence[str] = (),
        description_exclusions: Sequence[str] = (),
    ) -> None:
        self.provider = provider
        self.company_inclusions = frozenset(
            normalized for value in company_inclusions if (normalized := _normalize_filter_text(value))
        )
        self.company_exclusions = frozenset(
            normalized for value in company_exclusions if (normalized := _normalize_filter_text(value))
        )
        self.title_inclusions = tuple(
            dict.fromkeys(normalized for value in title_inclusions if (normalized := _normalize_filter_text(value)))
        )
        self.title_exclusions = tuple(
            dict.fromkeys(normalized for value in title_exclusions if (normalized := _normalize_filter_text(value)))
        )
        self.description_inclusions = tuple(
            dict.fromkeys(
                normalized for value in description_inclusions if (normalized := _normalize_filter_text(value))
            )
        )
        self.description_exclusions = tuple(
            dict.fromkeys(
                normalized for value in description_exclusions if (normalized := _normalize_filter_text(value))
            )
        )

    @staticmethod
    def _matches_keywords(value: str, *, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
        if any(_keyword_matches(value, keyword) for keyword in exclude):
            return False
        return not include or any(_keyword_matches(value, keyword) for keyword in include)

    def _matches(self, row: Mapping[str, Any]) -> bool:
        company = _normalize_filter_text(_row_text(row, "companyName", "company_name", "company"))
        if company in self.company_exclusions:
            return False
        if self.company_inclusions and company not in self.company_inclusions:
            return False

        title = _normalize_filter_text(_row_text(row, "title", "jobTitle", "job_title"))
        if not self._matches_keywords(title, include=self.title_inclusions, exclude=self.title_exclusions):
            return False

        description = _normalize_filter_text(
            _row_text(row, "description", "descriptionText", "jobDescription", "job_description")
        )
        return self._matches_keywords(
            description,
            include=self.description_inclusions,
            exclude=self.description_exclusions,
        )

    def discover(self, urls: Sequence[str], *, max_items: int) -> Iterable[Mapping[str, Any]]:
        return [row for row in self.provider.discover(urls, max_items=max_items) if self._matches(row)]

    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]:
        return self.provider.fetch_linkedin(job_id, country=country, max_concurrency=max_concurrency)

    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]:
        return self.provider.fetch_indeed(job_id)
