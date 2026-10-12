from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.errors import ConfigurationError
from job_hunt.ports import StructuredClient
from job_hunt.prompt_rendering import render_prompt


class GeminiCompatibilityFilter:
    def __init__(self, client: StructuredClient) -> None:
        self.client = client

    @staticmethod
    def _definition(prompts: Mapping[str, PromptDefinition]) -> PromptDefinition:
        try:
            return prompts["job_compatibility_filter"]
        except KeyError as exc:
            raise ConfigurationError("Missing active prompt: job_compatibility_filter") from exc

    def compatible(self, job: Job, prompts: Mapping[str, PromptDefinition]) -> bool:
        definition = self._definition(prompts)
        normalized = job.model_dump(mode="json")
        scraped: dict[str, Any] = job.provider_data or normalized
        values: dict[str, object] = {
            "job_description": job.description,
            "job_title": job.title,
            "company_name": job.company_name,
            "job_url": job.url,
            "location": job.location or "",
            "contract_type": job.contract_type or "",
            "published_at": normalized.get("published_at") or "",
            "job_json": normalized,
            "scraped_job_json": scraped,
        }
        generated = self.client.generate(
            render_prompt(definition.template, values, definition.key),
            definition,
        )
        compatible = generated.get("compatible")
        if not isinstance(compatible, bool):
            raise ValueError("job_compatibility_filter must return compatible as a boolean")
        return compatible
