from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from typing import Any

from langfuse import get_client, propagate_attributes
from opentelemetry.propagate import inject

_LOGGER = logging.getLogger(__name__)


def _metadata_strings(metadata: Mapping[str, object] | None) -> dict[str, str]:
    if not metadata:
        return {}
    result: dict[str, str] = {}
    for key, value in metadata.items():
        normalized_key = "".join(character for character in str(key) if character.isalnum())
        if not normalized_key:
            continue
        text = str(value)
        if text:
            result[normalized_key] = text[:200]
    return result


def langfuse_trace_context(*, enabled: bool, seed: str) -> dict[str, str] | None:
    """Return a deterministic Langfuse trace context so separate workers can join one run trace."""
    if not enabled:
        return None
    try:
        trace_id = get_client().create_trace_id(seed=seed)
    except Exception as exc:
        _LOGGER.warning("Langfuse trace-id creation failed; continuing without cross-task trace joining: %s", exc)
        return None
    return {"trace_id": trace_id}


def update_langfuse_output(observation: Any | None, output: Any) -> None:
    """Attach useful output without allowing observability failures to break application work."""
    if observation is None:
        return
    try:
        observation.update(output=output)
    except Exception as exc:
        _LOGGER.warning("Langfuse output update failed; application result is unchanged: %s", exc)


@contextmanager
def langfuse_span(
    *,
    enabled: bool,
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_name: str | None = None,
    trace_context: Mapping[str, str] | None = None,
    input: Any = None,
    metadata: Mapping[str, object] | None = None,
    tags: Sequence[str] = (),
) -> Iterator[Any | None]:
    """Create a Langfuse span without allowing observability setup/export failures to break application work."""
    if not enabled:
        yield None
        return

    stack = ExitStack()
    try:
        langfuse = get_client()
        observation_kwargs: dict[str, Any] = {"as_type": "span", "name": name}
        if input is not None:
            observation_kwargs["input"] = input
        if trace_context:
            observation_kwargs["trace_context"] = dict(trace_context)
        observation = stack.enter_context(langfuse.start_as_current_observation(**observation_kwargs))
        attributes: dict[str, Any] = {}
        if user_id:
            attributes["user_id"] = user_id
        if session_id:
            attributes["session_id"] = session_id
        if trace_name:
            attributes["trace_name"] = trace_name
        safe_metadata = _metadata_strings(metadata)
        if safe_metadata:
            attributes["metadata"] = safe_metadata
        if tags:
            attributes["tags"] = list(tags)
        if attributes:
            stack.enter_context(propagate_attributes(**attributes))
    except Exception as exc:
        with suppress(Exception):
            stack.close()
        _LOGGER.warning("Langfuse span setup failed; continuing without application tracing: %s", exc)
        yield None
        return

    try:
        yield observation
    except BaseException:
        exc_info = sys.exc_info()
        try:
            stack.__exit__(*exc_info)
        except Exception as instrumentation_error:
            _LOGGER.warning("Langfuse span shutdown failed: %s", instrumentation_error)
        raise
    else:
        try:
            stack.close()
        except Exception as exc:
            _LOGGER.warning("Langfuse span shutdown failed; application result is unchanged: %s", exc)


def trace_headers(*, enabled: bool) -> dict[str, str]:
    """Inject the active W3C trace context so LiteLLM can continue the Langfuse trace."""
    if not enabled:
        return {}
    headers: dict[str, str] = {}
    try:
        inject(headers)
    except Exception as exc:
        _LOGGER.warning("OpenTelemetry trace-context injection failed; continuing without parent linkage: %s", exc)
        return {}
    return headers
