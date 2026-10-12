from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import structlog
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from job_hunt.config import LlmRoute, Settings
from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import (
    ConfigurationError,
    ErrorKind,
    GatewayUnavailableError,
    ProviderError,
    is_context_length_error,
)
from job_hunt.integrations.gemini import GeminiStructuredClient, _validate_json_schema
from job_hunt.logging import logger
from job_hunt.observability import langfuse_span, trace_headers, update_langfuse_output
from job_hunt.state import RedisState

_DEFAULT_OPERATION_GROUPS = {
    "compatibility": "job-balanced",
    "content_extraction": "job-fast",
    "qualification": "job-balanced",
    "project_selection": "job-balanced",
    "project_rewrite": "job-powerful",
    "work_experience_selection": "job-balanced",
    "work_experience_rewrite": "job-powerful",
    "skills": "job-balanced",
    "summary": "job-powerful",
    "cover_letter": "job-powerful",
}
_DEFAULT_CONTEXT_GROUP_ORDER = ("job-fast", "job-balanced")
_NATIVE_SCHEMA_FALLBACK_STATUS_CODES = {400, 422}


def _strip_outer_markdown_json_fence(content: str) -> tuple[str, bool]:
    stripped = content.strip()
    lines = stripped.splitlines()
    if len(lines) < 3:
        return content, False
    opening = lines[0].strip().casefold()
    if opening not in {"```", "```json"} or lines[-1].strip() != "```":
        return content, False
    return "\n".join(lines[1:-1]).strip(), True


def _repair_prompt(raw_output: str, schema: dict[str, Any], validation_error: str) -> str:
    return (
        "Repair the following model output so it conforms exactly to the supplied JSON Schema. "
        "Preserve the original meaning and facts. Do not add facts that were not present. "
        "Return only the repaired structured result.\n\n"
        f"VALIDATION ERROR:\n{validation_error}\n\n"
        f"JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"ORIGINAL OUTPUT:\n{raw_output}"
    )


def _structured_prompt(prompt: str, schema: Mapping[str, Any]) -> str:
    return (
        f"{prompt}\n\n"
        "OUTPUT CONTRACT:\n"
        "Return exactly one JSON object and no surrounding prose or Markdown. "
        "The JSON object must conform to this JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def _schema_name(operation: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", operation).strip("_")
    return (normalized or "structured_output")[:64]


def _native_response_format(definition: PromptDefinition) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _schema_name(definition.key),
            "strict": True,
            "schema": definition.output_structure,
        },
    }


def _repair_definition(operation: str, schema: dict[str, Any]) -> PromptDefinition:
    return PromptDefinition(
        key=f"{operation}:repair",
        version=1.0,
        template="repair structured output",
        output_structure=schema,
        temperature=0.0,
    )


def _error_kind(status_code: int) -> ErrorKind:
    if status_code == 429:
        return ErrorKind.RATE_LIMIT
    if status_code in {401, 403}:
        return ErrorKind.AUTHENTICATION
    if status_code in {400, 404, 422}:
        return ErrorKind.MALFORMED_PROVIDER_RESPONSE
    return ErrorKind.TRANSIENT_PROVIDER


def _routing_config() -> Mapping[str, Any]:
    path = Path(os.getenv("LITELLM_ROUTING_CONFIG_PATH", "config/llm-routing.json"))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"LiteLLM routing config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"LiteLLM routing config is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("LiteLLM routing config must be a JSON object")
    return payload


def _operation_groups() -> dict[str, str]:
    configured = os.getenv("JOB_HUNT_LLM_OPERATION_GROUPS_JSON", "").strip()
    if not configured:
        return dict(_DEFAULT_OPERATION_GROUPS)
    try:
        parsed = json.loads(configured)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("JOB_HUNT_LLM_OPERATION_GROUPS_JSON must be valid JSON") from exc
    valid_mapping = isinstance(parsed, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
    )
    if not valid_mapping:
        raise ConfigurationError("JOB_HUNT_LLM_OPERATION_GROUPS_JSON must map operation names to LiteLLM aliases")
    return {**_DEFAULT_OPERATION_GROUPS, **parsed}


def _context_group_order() -> tuple[str, ...]:
    configured = os.getenv("JOB_HUNT_LLM_CONTEXT_GROUP_ORDER", "").strip()
    if not configured:
        return _DEFAULT_CONTEXT_GROUP_ORDER
    groups = tuple(group.strip() for group in configured.split(",") if group.strip())
    if not groups:
        raise ConfigurationError("JOB_HUNT_LLM_CONTEXT_GROUP_ORDER must contain at least one model group")
    return groups


def _group_fallbacks() -> dict[str, tuple[str, ...]]:
    raw = _routing_config().get("fallbacks")
    if not isinstance(raw, Mapping):
        raise ConfigurationError("LiteLLM routing config must define a fallbacks object")

    result: dict[str, tuple[str, ...]] = {}
    for group, targets in raw.items():
        valid_targets = isinstance(targets, list) and all(isinstance(item, str) for item in targets)
        if not isinstance(group, str) or not valid_targets:
            raise ConfigurationError("LiteLLM fallback entries must map group names to lists of group names")
        result[group] = tuple(targets)
    return result


def capability_group_for_operation(operation: str, *, repair: bool = False) -> str:
    if repair:
        return os.getenv("JOB_HUNT_LLM_REPAIR_GROUP", "repair-fast").strip() or "repair-fast"
    overrides = _operation_groups()
    normalized = operation.lower().replace("-", "_")
    for marker, group in overrides.items():
        if marker.lower().replace("-", "_") in normalized:
            return group
    return os.getenv("JOB_HUNT_LLM_DEFAULT_GROUP", "job-balanced").strip() or "job-balanced"


def _header_int(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _context_text(context: Mapping[str, Any], key: str) -> str | None:
    value = context.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class LiteLLMGatewayClient(GeminiStructuredClient):
    """OpenAI-compatible client for one logical capability group exposed by LiteLLM Proxy."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_seconds: int = 180,
        repair_client: GeminiStructuredClient | None = None,
        state: RedisState | None = None,
        checkpoint_ttl_seconds: int = 604800,
        transport: httpx.BaseTransport | None = None,
        tenant: str | None = None,
        session_id: str | None = None,
        langfuse_enabled: bool = False,
    ) -> None:
        if not model:
            raise ValueError("A LiteLLM model alias is required")
        self.model = model
        self.repair_client = repair_client
        self.tenant = tenant
        self.session_id = session_id
        self.langfuse_enabled = langfuse_enabled
        # State and checkpoint TTL remain accepted for construction compatibility. LLM outputs
        # are intentionally not reused: every generation request must reach the model.
        self.state = state
        self.checkpoint_ttl_seconds = checkpoint_ttl_seconds
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )
        self._deployment_catalog: dict[str, tuple[str | None, str, bool | None, int | None, str | None]] | None = None

    def _repair_via_routes(
        self,
        raw_output: str,
        schema: dict[str, Any],
        validation_error: str,
        *,
        operation: str,
    ) -> dict[str, Any]:
        logger().warning(
            "llm_pre_repair_output",
            prompt_key=operation,
            model_group=self.model,
            validation_error=validation_error,
            raw_output=raw_output,
            router="litellm-proxy",
        )
        if self.repair_client is None:
            raise ProviderError(
                f"Structured output failed validation: {validation_error}",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="litellm",
            )
        return self.repair_client.generate(
            _repair_prompt(raw_output, schema, validation_error),
            _repair_definition(operation, schema),
        )

    def _load_deployment_catalog(
        self,
    ) -> dict[str, tuple[str | None, str, bool | None, int | None, str | None]]:
        if self._deployment_catalog is not None:
            return self._deployment_catalog

        catalog: dict[str, tuple[str | None, str, bool | None, int | None, str | None]] = {}
        try:
            response = self.client.get("/v1/model/info")
            response.raise_for_status()
            payload = response.json()
            entries = payload.get("data", []) if isinstance(payload, Mapping) else []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                model_info = entry.get("model_info", {})
                params = entry.get("litellm_params", {})
                if not isinstance(model_info, Mapping) or not isinstance(params, Mapping):
                    continue
                deployment_id = str(model_info.get("id", "")).strip()
                provider_model = str(params.get("model", "")).strip()
                if not deployment_id or not provider_model:
                    continue
                provider = provider_model.split("/", 1)[0] if "/" in provider_model else None
                raw_support = model_info.get("supports_response_schema")
                supports_schema = raw_support if isinstance(raw_support, bool) else None
                raw_order = params.get("order")
                order = raw_order if isinstance(raw_order, int) and not isinstance(raw_order, bool) else None
                raw_group = entry.get("model_name")
                group = str(raw_group).strip() if raw_group is not None else None
                catalog[deployment_id] = (provider, provider_model, supports_schema, order, group)
        except (httpx.HTTPError, ValueError, TypeError):
            # Observability must never turn a successful LLM completion into a failed job.
            catalog = {}

        self._deployment_catalog = catalog
        return catalog

    def _upstream_identity(
        self,
        response: httpx.Response,
        payload: Mapping[str, Any],
    ) -> tuple[str | None, str | None, str | None, bool | None, int | None]:
        deployment_id = response.headers.get("x-litellm-model-id")
        if deployment_id:
            identity = self._load_deployment_catalog().get(deployment_id)
            if identity is not None:
                provider, model, supports_schema, order, _group = identity
                return provider, model, deployment_id, supports_schema, order

        payload_model = str(payload.get("model", "")).strip()
        if payload_model and payload_model != self.model:
            provider = payload_model.split("/", 1)[0] if "/" in payload_model else None
            return provider, payload_model, deployment_id, None, None
        return None, None, deployment_id, None, None

    def _higher_priority_models(self, served_order: int | None) -> list[str]:
        if served_order is None or served_order <= 1:
            return []
        models: dict[int, str] = {}
        for _deployment_id, (_provider, model, _schema, order, group) in self._load_deployment_catalog().items():
            if group == self.model and order is not None and order < served_order:
                models.setdefault(order, model)
        return [models[order] for order in sorted(models)]

    def _routing_evidence(self, response: httpx.Response, served_order: int | None) -> dict[str, object]:
        attempted_retries = _header_int(response, "x-litellm-attempted-retries")
        attempted_fallbacks = _header_int(response, "x-litellm-attempted-fallbacks")
        fallback_errors = response.headers.get("x-litellm-fallback-errors")
        higher_priority_models = self._higher_priority_models(served_order)
        return {
            "routing_order": served_order,
            "higher_priority_models": higher_priority_models,
            "higher_priority_unavailable_or_filtered": bool(higher_priority_models) and attempted_retries in {None, 0},
            "litellm_attempted_retries": attempted_retries,
            "litellm_attempted_fallbacks": attempted_fallbacks,
            "litellm_fallback_errors": fallback_errors[:2000] if fallback_errors else None,
            "litellm_call_id": response.headers.get("x-litellm-call-id"),
        }

    def _request_context(self, definition: PromptDefinition) -> tuple[str | None, str | None, dict[str, Any]]:
        context = structlog.contextvars.get_contextvars()
        tenant = _context_text(context, "tenant") or self.tenant
        run_id = _context_text(context, "run_id") or self.session_id
        job_id = _context_text(context, "job_id")
        session_id = f"{tenant}:{job_id}" if tenant and job_id else None
        metadata: dict[str, Any] = {
            "generation_name": definition.key,
            "operation": definition.key,
            "model_group": self.model,
        }
        if tenant:
            metadata["tenant"] = tenant
        if session_id:
            metadata["session_id"] = session_id
        if run_id:
            metadata["run_id"] = run_id
        for field in ("job_id", "job_identity", "row_id"):
            value = _context_text(context, field)
            if value:
                metadata[field] = value
        if self.langfuse_enabled:
            tags = ["job-hunt", f"operation:{definition.key}"]
            if tenant:
                tags.append(f"tenant:{tenant}")
            metadata["tags"] = tags
        return tenant, session_id, metadata

    def _completion_payload(
        self,
        prompt: str,
        definition: PromptDefinition,
        *,
        include_response_format: bool,
    ) -> dict[str, Any]:
        _tenant, _session_id, metadata = self._request_context(definition)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _structured_prompt(prompt, definition.output_structure),
                }
            ],
            "temperature": definition.temperature,
            "metadata": metadata,
        }
        if include_response_format:
            payload["response_format"] = _native_response_format(definition)
        return payload

    def _post_completion(
        self,
        prompt: str,
        definition: PromptDefinition,
        *,
        include_response_format: bool,
    ) -> httpx.Response:
        return self.client.post(
            "/v1/chat/completions",
            json=self._completion_payload(
                prompt,
                definition,
                include_response_format=include_response_format,
            ),
            headers=trace_headers(enabled=self.langfuse_enabled),
        )

    def _provider_error(self, response: httpx.Response) -> ProviderError:
        kind = _error_kind(response.status_code)
        body = response.text
        if is_context_length_error(body):
            message = f"LLM input exceeds the context window for route {self.model}: {body[:500]}"
        else:
            message = f"LiteLLM route {self.model} failed with HTTP {response.status_code}: {body[:500]}"
        return ProviderError(
            message,
            kind,
            retryable=kind in {ErrorKind.RATE_LIMIT, ErrorKind.TRANSIENT_PROVIDER},
            provider="litellm",
            status_code=response.status_code,
        )

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        tenant, session_id, metadata = self._request_context(definition)
        span_metadata = {
            "operation": definition.key,
            "modelgroup": self.model,
        }
        for field in ("job_id", "job_identity", "row_id"):
            value = metadata.get(field)
            if value is not None:
                span_metadata[field] = value
        span_input = {
            "model_group": self.model,
            "temperature": definition.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": _structured_prompt(prompt, definition.output_structure),
                }
            ],
            "response_format": _native_response_format(definition),
        }
        with langfuse_span(
            enabled=self.langfuse_enabled,
            name=definition.key,
            user_id=tenant,
            session_id=session_id,
            input=span_input,
            metadata=span_metadata,
            tags=("job-hunt", "llm-operation"),
        ) as observation:
            result = self._generate(prompt, definition)
            update_langfuse_output(observation, result)
            return result

    def _generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        started = time.perf_counter()
        structured_output_mode = "model_capability"
        try:
            response = self._post_completion(prompt, definition, include_response_format=True)
            if response.status_code in _NATIVE_SCHEMA_FALLBACK_STATUS_CODES and not is_context_length_error(
                response.text
            ):
                logger().info(
                    "llm_structured_output_fallback",
                    prompt_key=definition.key,
                    model_group=self.model,
                    status_code=response.status_code,
                    router="litellm-proxy",
                )
                structured_output_mode = "runtime_prompt_schema_fallback"
                response = self._post_completion(prompt, definition, include_response_format=False)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise GatewayUnavailableError(
                f"LiteLLM gateway unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"LiteLLM gateway request failed: {type(exc).__name__}: {exc}",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="litellm",
            ) from exc

        if response.is_error:
            raise self._provider_error(response)

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "LiteLLM gateway returned a malformed chat-completions response",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="litellm",
            ) from exc

        raw_output = json.dumps(content, ensure_ascii=False) if isinstance(content, Mapping) else str(content or "")
        repaired = False
        markdown_fence_removed = False
        repair_output = raw_output
        if isinstance(content, Mapping):
            parsed: object = dict(content)
        elif isinstance(content, str):
            parse_content, markdown_fence_removed = _strip_outer_markdown_json_fence(content)
            repair_output = parse_content if markdown_fence_removed else raw_output
            try:
                parsed = json.loads(parse_content)
            except json.JSONDecodeError as exc:
                result = self._repair_via_routes(
                    repair_output,
                    definition.output_structure,
                    str(exc),
                    operation=definition.key,
                )
                repaired = True
                parsed = result
        else:
            raise ProviderError(
                "LiteLLM gateway returned empty structured output",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="litellm",
            )

        if repaired:
            result = _validate_json_schema(parsed, definition.output_structure)
        else:
            try:
                result = _validate_json_schema(parsed, definition.output_structure)
            except JsonSchemaValidationError as exc:
                result = self._repair_via_routes(
                    repair_output,
                    definition.output_structure,
                    str(exc),
                    operation=definition.key,
                )
                repaired = True

        upstream_provider, upstream_model, deployment_id, supports_schema, served_order = self._upstream_identity(
            response, payload
        )
        if structured_output_mode == "model_capability":
            if supports_schema is True:
                structured_output_mode = "native_json_schema"
            elif supports_schema is False:
                structured_output_mode = "prompt_schema_model_metadata"
        logger().info(
            "llm_generation",
            prompt_key=definition.key,
            provider="litellm",
            model_group=self.model,
            upstream_provider=upstream_provider,
            upstream_model=upstream_model,
            deployment_id=deployment_id,
            supports_native_json_schema=supports_schema,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=payload.get("usage", {}),
            repaired=repaired,
            markdown_fence_removed=markdown_fence_removed,
            structured_output_mode=structured_output_mode,
            router="litellm-proxy",
            **self._routing_evidence(response, served_order),
        )
        return result


class CapabilityRoutedStructuredClient(GeminiStructuredClient):
    """Select a capability group and enforce configured cross-group fallbacks."""

    def __init__(
        self,
        routes: Sequence[tuple[LlmRoute, GeminiStructuredClient]],
        *,
        repair: bool = False,
        fallbacks: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.routes = {route.model: client for route, client in routes}
        self.repair = repair
        self.fallbacks = {
            group: tuple(targets)
            for group, targets in (fallbacks if fallbacks is not None else _group_fallbacks()).items()
        }
        if not self.routes:
            raise ConfigurationError("No usable LiteLLM capability groups are configured")

    def _context_fallbacks(self, group: str) -> tuple[str, ...]:
        if self.repair:
            return ()
        order = _context_group_order()
        if group not in order:
            return ()
        return tuple(candidate for candidate in order[order.index(group) + 1 :] if candidate in self.routes)

    def _retry_fallbacks(self, group: str) -> tuple[str, ...]:
        if self.repair:
            return ()
        return tuple(candidate for candidate in self.fallbacks.get(group, ()) if candidate in self.routes)

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        primary = capability_group_for_operation(definition.key, repair=self.repair)
        if primary not in self.routes:
            mode = "repair" if self.repair else "generation"
            raise ConfigurationError(f"LiteLLM capability group {primary!r} is not configured for {mode}")

        pending = [primary]
        attempted: set[str] = set()
        context_failures: list[str] = []
        last_retryable_error: ProviderError | None = None

        while pending:
            group = pending.pop(0)
            if group in attempted:
                continue
            attempted.add(group)
            client = self.routes[group]
            try:
                result = client.generate(prompt, definition)
            except ProviderError as exc:
                if isinstance(exc, GatewayUnavailableError):
                    raise
                logger().warning(
                    "llm_group_failed",
                    prompt_key=definition.key,
                    group=group,
                    error_kind=exc.kind,
                    reason=str(exc),
                    router="litellm-proxy",
                )
                if is_context_length_error(exc) and not self.repair:
                    context_failures.append(f"{group}: {exc}")
                    pending.extend(
                        candidate for candidate in self._context_fallbacks(group) if candidate not in attempted
                    )
                    continue
                if exc.retryable and not self.repair:
                    last_retryable_error = exc
                    pending.extend(
                        candidate for candidate in self._retry_fallbacks(group) if candidate not in attempted
                    )
                    continue
                raise
            logger().info(
                "llm_group_succeeded",
                prompt_key=definition.key,
                group=group,
                repair=self.repair,
                context_fallback=bool(context_failures) and group != primary,
                retry_fallback=last_retryable_error is not None and group != primary,
                router="litellm-proxy",
            )
            return result

        if last_retryable_error is not None:
            raise last_retryable_error
        raise ProviderError(
            "LLM input is too large for every configured context fallback group. " + "; ".join(context_failures),
            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
            provider="litellm",
        )


RoutedStructuredClient = CapabilityRoutedStructuredClient


def _gateway_client(
    settings: Settings,
    route: LlmRoute,
    *,
    repair_client: GeminiStructuredClient | None,
    state: RedisState | None,
    tenant: str | None = None,
    session_id: str | None = None,
) -> LiteLLMGatewayClient:
    return LiteLLMGatewayClient(
        settings.litellm_base_url,
        settings.litellm_key_for_tenant(tenant),
        route.model,
        timeout_seconds=settings.litellm_request_timeout_seconds,
        repair_client=repair_client,
        state=state,
        checkpoint_ttl_seconds=settings.llm_checkpoint_ttl_seconds,
        tenant=tenant,
        session_id=session_id,
        langfuse_enabled=settings.langfuse_enabled,
    )


def _build_gateway_routes(
    settings: Settings,
    specs: Sequence[LlmRoute],
    *,
    repair_client: GeminiStructuredClient | None,
    state: RedisState | None,
    tenant: str | None = None,
    session_id: str | None = None,
) -> list[tuple[LlmRoute, GeminiStructuredClient]]:
    return [
        (
            route,
            _gateway_client(
                settings,
                route,
                repair_client=repair_client,
                state=state,
                tenant=tenant,
                session_id=session_id,
            ),
        )
        for route in specs
    ]


def _build_repair_routes(
    settings: Settings,
    *,
    state: RedisState | None,
    tenant: str | None = None,
    session_id: str | None = None,
) -> list[tuple[LlmRoute, GeminiStructuredClient]]:
    """Build repair tiers so schema-invalid repair output can escalate to the next tier.

    LiteLLM handles deployment rotation inside a group. The application generation router
    separately enforces configured cross-group fallbacks for retryable provider failures.
    This repair chain covers schema-invalid HTTP 200 outputs.
    """
    specs = settings.llm_repair_route_specs()
    validation_fallback: GeminiStructuredClient | None = None
    reversed_routes: list[tuple[LlmRoute, GeminiStructuredClient]] = []
    for route in reversed(specs):
        client = _gateway_client(
            settings,
            route,
            repair_client=validation_fallback,
            state=state,
            tenant=tenant,
            session_id=session_id,
        )
        reversed_routes.append((route, client))
        validation_fallback = client
    return list(reversed(reversed_routes))


def build_routed_structured_client(
    settings: Settings,
    *,
    state: RedisState | None = None,
    tenant: str | None = None,
    session_id: str | None = None,
) -> CapabilityRoutedStructuredClient:
    fallbacks = _group_fallbacks()
    repair_client = CapabilityRoutedStructuredClient(
        _build_repair_routes(settings, state=state, tenant=tenant, session_id=session_id),
        repair=True,
        fallbacks=fallbacks,
    )
    return CapabilityRoutedStructuredClient(
        _build_gateway_routes(
            settings,
            settings.llm_route_specs(),
            repair_client=repair_client,
            state=state,
            tenant=tenant,
            session_id=session_id,
        ),
        fallbacks=fallbacks,
    )