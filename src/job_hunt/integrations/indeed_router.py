from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.apify_indeed import ApifyIndeedProvider
from job_hunt.integrations.brightdata_indeed import BrightDataIndeedProvider
from job_hunt.logging import logger

_SUPPORTED_INDEED_FETCH_STRATEGIES = {"apify", "brightdata"}


class IndeedFetchRouter:
    """Fetch Indeed jobs using the configured provider order."""

    def __init__(
        self,
        apify: ApifyIndeedProvider,
        *,
        brightdata: BrightDataIndeedProvider | None,
        country: str,
        order: Sequence[str] = ("apify",),
    ) -> None:
        self.apify = apify
        self.brightdata = brightdata
        self.country = str(country).strip().upper() or "US"
        self.order = tuple(str(strategy).strip() for strategy in order if str(strategy).strip())
        if not self.order:
            raise ValueError("At least one Indeed job fetch strategy is required")
        if len(set(self.order)) != len(self.order):
            raise ValueError("Indeed job fetch strategies must not contain duplicates")
        unknown = set(self.order) - _SUPPORTED_INDEED_FETCH_STRATEGIES
        if unknown:
            raise ValueError(f"Unknown Indeed job fetch strategies: {sorted(unknown)}")

    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]:
        normalized_id = str(job_id).strip()
        if not normalized_id:
            raise ValueError("Indeed job ID is required")

        last_error: ProviderError | None = None
        for strategy in self.order:
            if strategy == "brightdata" and self.brightdata is None:
                logger().warning(
                    "indeed_job_fetch_strategy_unconfigured",
                    strategy=strategy,
                    job_id=normalized_id,
                    configured_order=list(self.order),
                )
                continue

            logger().info(
                "indeed_job_fetch_strategy_started",
                strategy=strategy,
                job_id=normalized_id,
                configured_order=list(self.order),
            )
            try:
                if strategy == "apify":
                    row = self.apify.fetch_indeed(normalized_id, country=self.country)
                elif strategy == "brightdata":
                    if self.brightdata is None:
                        continue
                    row = self.brightdata.fetch_indeed(normalized_id)
                else:
                    raise ValueError(f"Unsupported Indeed job fetch strategy: {strategy}")
            except ProviderError as exc:
                last_error = exc
                logger().warning(
                    "indeed_job_fetch_strategy_failed",
                    strategy=strategy,
                    job_id=normalized_id,
                    error_kind=exc.kind,
                    retryable=exc.retryable,
                    error_message=str(exc),
                )
                continue

            logger().info(
                "indeed_job_fetch_strategy_succeeded",
                strategy=strategy,
                job_id=normalized_id,
            )
            return row

        if last_error is not None:
            raise last_error
        raise ProviderError(
            f"No configured Indeed job fetch strategy is available for {list(self.order)}",
            ErrorKind.CONFIGURATION,
            provider="indeed_job_fetch",
        )
