from __future__ import annotations

from typing import Any

from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.gemini import GeminiStructuredClient
from job_hunt.logging import logger
from job_hunt.repair_guard import assess_repair, repair_max_attempts

_ORIGINAL_MARKER = "ORIGINAL OUTPUT:\n"

_STRICT_REPAIR_RULES = """\
PRESERVATION RULES FOR THIS REPAIR:
- This is a syntax/schema repair, not a rewriting task.
- Preserve every existing textual value verbatim whenever possible.
- Do not summarize, expand, shorten, paraphrase, polish, or improve prose.
- Do not add or remove facts, bullets, sentences, skills, accomplishments, or qualifications.
- Only change JSON syntax, escaping, required wrapper/property names, and the minimum structure required by the schema.
- If a required schema key is missing, add the key around the existing value instead of rewriting that value.
"""


class PreservingRepairClient(GeminiStructuredClient):
    """Retry schema repair independently when a model changes source content too much."""

    def __init__(self, delegate: GeminiStructuredClient) -> None:
        self.delegate = delegate

    @staticmethod
    def _source_output(prompt: str) -> str:
        if _ORIGINAL_MARKER not in prompt:
            raise ProviderError(
                "Repair request did not contain the original model output",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="repair-guard",
            )
        return prompt.split(_ORIGINAL_MARKER, 1)[1]

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        source = self._source_output(prompt)
        guarded_prompt = f"{_STRICT_REPAIR_RULES}\n{prompt}"
        attempts = repair_max_attempts()
        last_reason = "repair preservation check failed"

        for attempt in range(1, attempts + 1):
            result = self.delegate.generate(guarded_prompt, definition)
            assessment = assess_repair(source, result)
            last_reason = assessment.reason
            logger().info(
                "repair_preservation_checked",
                prompt_key=definition.key.removesuffix(":repair"),
                attempt=attempt,
                max_attempts=attempts,
                accepted=assessment.accepted,
                source_tokens=assessment.source_tokens,
                repaired_tokens=assessment.repaired_tokens,
                token_delta=assessment.token_delta,
                token_similarity=round(assessment.token_similarity, 3),
                size_ratio=round(assessment.size_ratio, 3),
                reason=assessment.reason,
            )
            if assessment.accepted:
                return result
            if attempt < attempts:
                logger().warning(
                    "repair_retrying",
                    prompt_key=definition.key.removesuffix(":repair"),
                    attempt=attempt,
                    reason=assessment.reason,
                )

        raise ProviderError(
            f"Repair output repeatedly changed source content beyond the allowed preservation limits: {last_reason}",
            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
            provider="repair-guard",
        )
