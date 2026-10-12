from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

_CONTEXT_ERROR_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "max context",
    "prompt too long",
    "input too long",
    "too many input tokens",
    "token limit",
)

_RATE_LIMIT_ERROR_MARKERS = (
    "quota exceeded",
    "exceeded your current quota",
    "resource_exhausted",
    "too many requests",
    "rate limit",
    "rate_limit",
    "ratelimiterror",
)


def is_context_length_error(error: object) -> bool:
    text = str(error).casefold()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


def is_rate_limit_error(error: object) -> bool:
    text = str(error).casefold()
    has_rate_limit_signal = any(marker in text for marker in _RATE_LIMIT_ERROR_MARKERS)
    return has_rate_limit_signal and (
        "429" in text
        or "quota exceeded" in text
        or "exceeded your current quota" in text
        or "resource_exhausted" in text
    )


class ErrorKind(StrEnum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    RATE_LIMIT = "rate_limit"
    GATEWAY_UNAVAILABLE = "gateway_unavailable"
    TRANSIENT_PROVIDER = "transient_provider"
    MALFORMED_PROVIDER_RESPONSE = "malformed_provider_response"
    DOCUMENT_RENDERING = "document_rendering"
    BUSINESS = "business"


@dataclass(slots=True)
class WorkflowError(Exception):
    message: str
    kind: ErrorKind
    retryable: bool = False
    provider: str | None = None
    status_code: int | None = None
    retry_after: float | None = None

    def __str__(self) -> str:
        return self.message


class ConfigurationError(WorkflowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorKind.CONFIGURATION)


class DocumentRenderingError(WorkflowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorKind.DOCUMENT_RENDERING)


class ProviderError(WorkflowError):
    def __init__(
        self,
        message: str,
        kind: ErrorKind,
        retryable: bool = False,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        normalized_kind = kind
        normalized_retryable = retryable
        if kind is ErrorKind.AUTHENTICATION and is_rate_limit_error(message):
            normalized_kind = ErrorKind.RATE_LIMIT
            normalized_retryable = True

        super().__init__(
            message,
            normalized_kind,
            normalized_retryable,
            provider,
            status_code,
            retry_after,
        )


class GatewayUnavailableError(ProviderError):
    """The application could not establish a connection to the shared LiteLLM gateway."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            ErrorKind.GATEWAY_UNAVAILABLE,
            retryable=True,
            provider="litellm",
        )
