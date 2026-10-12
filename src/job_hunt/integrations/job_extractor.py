from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.observability import langfuse_span, langfuse_trace_context, update_langfuse_output
from job_hunt.url_utils import valid_http_url

_PLACEHOLDER = re.compile(r"\[\[([A-Za-z0-9_]+)\]\]")


class StructuredGenerator(Protocol):
    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]: ...


def _render(template: str, values: dict[str, object]) -> str:
    rendered = template
    for key, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        rendered = rendered.replace(f"[[{key}]]", text)
    unresolved = sorted(set(_PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise ValueError(f"job_page_content_extraction has unresolved placeholders: {unresolved}")
    return rendered


def _text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()
        if value is not None:
            return str(value).strip()
    return ""


def _optional_text(payload: dict[str, Any], *keys: str) -> str | None:
    value = _text(payload, *keys)
    return value or None


def _parse_published_at(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class StructuredJobExtractor:
    """Convert configured extraction output into a Job without inventing absent metadata."""

    def __init__(
        self,
        client: StructuredGenerator,
        definition: PromptDefinition,
        *,
        tenant: str | None = None,
        session_id: str | None = None,
        langfuse_enabled: bool = False,
    ) -> None:
        self.client = client
        self.definition = definition
        self.tenant = tenant
        self.session_id = session_id
        self.langfuse_enabled = langfuse_enabled

    def _langfuse_trace_context(self) -> dict[str, str] | None:
        if not self.session_id:
            return None
        tenant = self.tenant or "unknown"
        return langfuse_trace_context(
            enabled=self.langfuse_enabled,
            seed=f"job-hunt:{tenant}:{self.session_id}",
        )

    def extract_job(self, content: str, source_url: str) -> Job:
        prompt = _render(
            self.definition.template,
            {
                "page_content": content,
                "content": content,
                "source_url": source_url,
                "job_url": source_url,
            },
        )
        span_input = {
            "source_url": source_url,
            "page_content": content,
            "rendered_prompt": prompt,
            "prompt_version": self.definition.version,
        }
        with langfuse_span(
            enabled=self.langfuse_enabled,
            name="job-information-extraction",
            trace_name="job-application",
            trace_context=self._langfuse_trace_context(),
            user_id=self.tenant,
            input=span_input,
            metadata={
                "operation": self.definition.key,
                "promptVersion": self.definition.version,
            },
            tags=("job-hunt", "normalization", "information-extraction"),
        ) as observation:
            generated = self.client.generate(prompt, self.definition)

            description = _text(generated, "job_description", "description")
            if not description:
                empty_fields = [
                    name
                    for name, value in (
                        ("company_name", _text(generated, "company_name", "company")),
                        ("job_title", _text(generated, "job_title", "title")),
                        ("job_description", description),
                    )
                    if not value
                ]
                raise ValueError(
                    "job_page_content_extraction returned an empty required job_description; "
                    f"empty fields: {', '.join(empty_fields)}"
                )

            extracted_url = valid_http_url(_text(generated, "job_url", "url")) or valid_http_url(source_url) or ""
            job = Job(
                source="web",
                external_id=_optional_text(generated, "external_job_id", "external_id"),
                url=extracted_url,
                company_name=_text(generated, "company_name", "company"),
                title=_text(generated, "job_title", "title"),
                description=description,
                location=_optional_text(generated, "location"),
                contract_type=_optional_text(generated, "contract_type"),
                published_at=_parse_published_at(generated.get("published_at")),
                provider_data=dict(generated),
            )
            identified = assign_identity(job)
            update_langfuse_output(
                observation,
                {
                    "model_output": generated,
                    "job": identified.model_dump(mode="json"),
                },
            )
            return identified
