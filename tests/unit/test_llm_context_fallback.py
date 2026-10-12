from __future__ import annotations

import pytest

from job_hunt.config import LlmRoute
from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.integrations.llm_routing import RoutedStructuredClient

FALLBACKS = {
    "job-powerful": ["job-balanced"],
    "job-balanced": ["job-fast"],
    "job-fast": [],
}


class StubClient(GeminiStructuredClient):
    def __init__(self, *, result: dict[str, object] | None = None, error: ProviderError | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, object]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def definition(key: str = "job_page_content_extraction") -> PromptDefinition:
    return PromptDefinition(
        key=key,
        version=1,
        template="unused",
        temperature=0,
        output_structure={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )


def route(name: str, client: StubClient) -> tuple[LlmRoute, GeminiStructuredClient]:
    return LlmRoute(provider="litellm", model=name), client


def context_error(group: str) -> ProviderError:
    return ProviderError(
        f"LLM input exceeds the context window for route {group}",
        ErrorKind.MALFORMED_PROVIDER_RESPONSE,
        provider="litellm",
        status_code=400,
    )


def retryable_error(group: str, kind: ErrorKind = ErrorKind.RATE_LIMIT, status_code: int = 429) -> ProviderError:
    return ProviderError(
        f"LiteLLM route {group} failed",
        kind,
        retryable=True,
        provider="litellm",
        status_code=status_code,
    )


def non_retryable_error(group: str) -> ProviderError:
    return ProviderError(
        f"LiteLLM route {group} failed",
        ErrorKind.AUTHENTICATION,
        retryable=False,
        provider="litellm",
        status_code=401,
    )


def routed_client(fast: StubClient, balanced: StubClient, powerful: StubClient) -> RoutedStructuredClient:
    return RoutedStructuredClient(
        [
            route("job-fast", fast),
            route("job-balanced", balanced),
            route("job-powerful", powerful),
        ],
        fallbacks=FALLBACKS,
    )


def test_context_failure_escalates_to_larger_capability_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_HUNT_LLM_CONTEXT_GROUP_ORDER", raising=False)
    fast = StubClient(error=context_error("job-fast"))
    balanced = StubClient(result={"ok": True})
    powerful = StubClient(result={"ok": False})
    client = routed_client(fast, balanced, powerful)

    assert client.generate("complete prompt", definition()) == {"ok": True}
    assert fast.calls == 1
    assert balanced.calls == 1
    assert powerful.calls == 0


def test_context_failure_raises_after_allowed_context_groups_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JOB_HUNT_LLM_CONTEXT_GROUP_ORDER", raising=False)
    fast = StubClient(error=context_error("job-fast"))
    balanced = StubClient(error=context_error("job-balanced"))
    powerful = StubClient(error=context_error("job-powerful"))
    client = routed_client(fast, balanced, powerful)

    with pytest.raises(ProviderError, match="too large for every configured context fallback group"):
        client.generate("complete prompt", definition())

    assert fast.calls == 1
    assert balanced.calls == 1
    assert powerful.calls == 0


def test_balanced_rate_limit_falls_back_to_fast() -> None:
    fast = StubClient(result={"ok": True})
    balanced = StubClient(error=retryable_error("job-balanced"))
    powerful = StubClient(result={"ok": False})
    client = routed_client(fast, balanced, powerful)

    assert client.generate("complete prompt", definition("qualification_scoring")) == {"ok": True}
    assert balanced.calls == 1
    assert fast.calls == 1
    assert powerful.calls == 0


def test_balanced_transient_provider_failure_falls_back_to_fast() -> None:
    fast = StubClient(result={"ok": True})
    balanced = StubClient(error=retryable_error("job-balanced", ErrorKind.TRANSIENT_PROVIDER, 503))
    powerful = StubClient(result={"ok": False})
    client = routed_client(fast, balanced, powerful)

    assert client.generate("complete prompt", definition("compatibility")) == {"ok": True}
    assert balanced.calls == 1
    assert fast.calls == 1
    assert powerful.calls == 0


def test_powerful_rate_limit_uses_balanced_before_fast() -> None:
    fast = StubClient(result={"ok": False})
    balanced = StubClient(result={"ok": True})
    powerful = StubClient(error=retryable_error("job-powerful"))
    client = routed_client(fast, balanced, powerful)

    assert client.generate("complete prompt", definition("cover_letter_generation")) == {"ok": True}
    assert powerful.calls == 1
    assert balanced.calls == 1
    assert fast.calls == 0


def test_powerful_reaches_fast_only_after_balanced_also_fails_retryably() -> None:
    fast = StubClient(result={"ok": True})
    balanced = StubClient(error=retryable_error("job-balanced"))
    powerful = StubClient(error=retryable_error("job-powerful"))
    client = routed_client(fast, balanced, powerful)

    assert client.generate("complete prompt", definition("cover_letter_generation")) == {"ok": True}
    assert powerful.calls == 1
    assert balanced.calls == 1
    assert fast.calls == 1


def test_powerful_never_skips_failed_balanced_with_non_retryable_error() -> None:
    fast = StubClient(result={"ok": True})
    balanced = StubClient(error=non_retryable_error("job-balanced"))
    powerful = StubClient(error=retryable_error("job-powerful"))
    client = routed_client(fast, balanced, powerful)

    with pytest.raises(ProviderError, match="job-balanced failed"):
        client.generate("complete prompt", definition("cover_letter_generation"))

    assert powerful.calls == 1
    assert balanced.calls == 1
    assert fast.calls == 0


def test_fast_retryable_failure_has_no_lower_group_to_fall_back_to() -> None:
    fast_error = retryable_error("job-fast")
    fast = StubClient(error=fast_error)
    balanced = StubClient(result={"ok": True})
    powerful = StubClient(result={"ok": True})
    client = routed_client(fast, balanced, powerful)

    with pytest.raises(ProviderError) as exc_info:
        client.generate("complete prompt", definition())

    assert exc_info.value is fast_error
    assert fast.calls == 1
    assert balanced.calls == 0
    assert powerful.calls == 0
