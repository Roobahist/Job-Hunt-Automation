from __future__ import annotations

from typing import Any

from job_hunt.config import Settings
from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import (
    ConfigurationError,
    ErrorKind,
    GatewayUnavailableError,
    ProviderError,
    is_context_length_error,
    is_rate_limit_error,
)
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.llm_routing import (
    CapabilityRoutedStructuredClient,
    _build_gateway_routes,
    _build_repair_routes,
    _group_fallbacks,
    capability_group_for_operation,
)
from job_hunt.integrations.preserving_repair import PreservingRepairClient
from job_hunt.logging import logger
from job_hunt.state import RedisState


def _error_kind_value(kind: ErrorKind) -> str:
    return getattr(kind, "value", str(kind))


def _effective_failure(exc: ProviderError) -> tuple[ErrorKind, bool]:
    if exc.kind == ErrorKind.RATE_LIMIT or is_rate_limit_error(exc):
        return ErrorKind.RATE_LIMIT, True
    return exc.kind, exc.retryable


def _failure_record(
    group: str,
    exc: ProviderError,
    fallback_type: str | None,
    *,
    error_kind: ErrorKind,
    retryable: bool,
) -> dict[str, Any]:
    return {
        "group": group,
        "error_kind": _error_kind_value(error_kind),
        "reason": str(exc),
        "status_code": exc.status_code,
        "retryable": retryable,
        "fallback_type": fallback_type,
    }


class ObservableCapabilityRoutedStructuredClient(CapabilityRoutedStructuredClient):
    """Application-owned cross-group routing with explicit fallback observability."""

    def _configured_retry_fallbacks(self, group: str) -> tuple[str, ...]:
        return tuple(candidate for candidate in self.fallbacks.get(group, ()) if candidate in self.routes)

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        primary = capability_group_for_operation(definition.key, repair=self.repair)
        if primary not in self.routes:
            mode = "repair" if self.repair else "generation"
            raise ConfigurationError(f"LiteLLM capability group {primary!r} is not configured for {mode}")

        pending = [primary]
        attempted: set[str] = set()
        attempt_chain: list[str] = []
        context_failures: list[str] = []
        failure_records: list[dict[str, Any]] = []
        last_fallback_error: ProviderError | None = None

        while pending:
            group = pending.pop(0)
            if group in attempted:
                continue
            attempted.add(group)
            attempt_chain.append(group)
            client = self.routes[group]
            try:
                result = client.generate(prompt, definition)
            except GatewayUnavailableError as exc:
                logger().warning(
                    "llm_gateway_unavailable",
                    prompt_key=definition.key,
                    requested_group=primary,
                    attempted_group=group,
                    fallback_targets=[],
                    error_kind=exc.kind,
                    retryable=True,
                    reason=str(exc),
                    attempt_chain=list(attempt_chain),
                    router="application",
                )
                raise
            except ProviderError as exc:
                effective_kind, effective_retryable = _effective_failure(exc)
                context_failure = is_context_length_error(exc)
                malformed_failure = effective_kind == ErrorKind.MALFORMED_PROVIDER_RESPONSE and not context_failure
                retry_failure = effective_retryable and not context_failure
                fallback_type = (
                    "context_length"
                    if context_failure
                    else "retryable_provider_error"
                    if retry_failure
                    else "malformed_provider_response"
                    if malformed_failure
                    else None
                )

                if context_failure:
                    candidates = [
                        candidate for candidate in self._context_fallbacks(group) if candidate not in attempted
                    ]
                elif retry_failure or malformed_failure:
                    candidates = [
                        candidate for candidate in self._configured_retry_fallbacks(group) if candidate not in attempted
                    ]
                else:
                    candidates = []

                failure = _failure_record(
                    group,
                    exc,
                    fallback_type,
                    error_kind=effective_kind,
                    retryable=effective_retryable,
                )
                failure["fallback_targets"] = candidates
                failure_records.append(failure)
                logger().warning(
                    "llm_group_failed",
                    prompt_key=definition.key,
                    requested_group=primary,
                    attempted_group=group,
                    fallback_type=fallback_type,
                    fallback_targets=candidates,
                    error_kind=failure["error_kind"],
                    status_code=exc.status_code,
                    retryable=effective_retryable,
                    reason=str(exc),
                    attempt_chain=list(attempt_chain),
                    router="application",
                )

                if context_failure:
                    context_failures.append(f"{group}: {exc}")
                if context_failure or retry_failure or malformed_failure:
                    last_fallback_error = exc
                    if candidates:
                        pending.extend(candidates)
                        continue
                raise

            fallback_used = group != primary
            fallback = failure_records[-1] if fallback_used and failure_records else None
            logger().info(
                "llm_group_succeeded",
                prompt_key=definition.key,
                requested_group=primary,
                resolved_group=group,
                group=group,
                repair=self.repair,
                fallback_used=fallback_used,
                fallback_type=fallback.get("fallback_type") if fallback else None,
                fallback_from=fallback.get("group") if fallback else None,
                fallback_reason=fallback.get("reason") if fallback else None,
                fallback_error_kind=fallback.get("error_kind") if fallback else None,
                fallback_status_code=fallback.get("status_code") if fallback else None,
                fallback_chain=list(attempt_chain),
                context_fallback=bool(context_failures) and fallback_used,
                retry_fallback=bool(
                    fallback_used and fallback and fallback.get("fallback_type") == "retryable_provider_error"
                ),
                router="application",
            )
            return result

        if last_fallback_error is not None:
            raise last_fallback_error
        raise ProviderError(
            "No configured LLM route could satisfy this request. " + "; ".join(context_failures),
            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
            provider="litellm",
        )


def build_routed_structured_client(
    settings: Settings,
    *,
    state: RedisState | None = None,
    tenant: str | None = None,
    session_id: str | None = None,
) -> ObservableCapabilityRoutedStructuredClient:
    fallbacks = _group_fallbacks()
    repair_routes = _build_repair_routes(settings, state=state, tenant=tenant, session_id=session_id)
    routed_repair_client: GeminiStructuredClient = ObservableCapabilityRoutedStructuredClient(
        repair_routes,
        repair=True,
        fallbacks=fallbacks,
    )
    repair_client: GeminiStructuredClient = PreservingRepairClient(routed_repair_client)
    generation_routes = _build_gateway_routes(
        settings,
        settings.llm_route_specs(),
        repair_client=repair_client,
        state=state,
        tenant=tenant,
        session_id=session_id,
    )
    return ObservableCapabilityRoutedStructuredClient(
        generation_routes,
        fallbacks=fallbacks,
    )
