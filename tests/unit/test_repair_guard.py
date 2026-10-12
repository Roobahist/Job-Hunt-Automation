from __future__ import annotations

from typing import Any

import pytest

from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ProviderError
from job_hunt.integrations.preserving_repair import PreservingRepairClient
from job_hunt.repair_guard import assess_repair


class SequenceRepairClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def _definition() -> PromptDefinition:
    return PromptDefinition(
        key="cv_summary_rewrite:repair",
        version=1.0,
        template="repair",
        output_structure={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        temperature=0.0,
    )


def test_structural_repair_preserves_content() -> None:
    raw = '```json\n{"summary":"Data scientist building reliable Python workflows"\n```'
    repaired = {"summary": "Data scientist building reliable Python workflows"}

    assessment = assess_repair(raw, repaired)

    assert assessment.accepted
    assert assessment.token_delta <= 20


def test_preserving_repair_retries_from_same_canonical_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_REPAIR_MAX_ATTEMPTS", "3")
    source_text = " ".join(f"source{i}" for i in range(45))
    rewritten = {"summary": "completely different replacement wording " * 4}
    preserved = {"summary": source_text}
    delegate = SequenceRepairClient([rewritten, preserved])
    client = PreservingRepairClient(delegate)  # type: ignore[arg-type]
    prompt = f"VALIDATION ERROR:\ninvalid json\n\nORIGINAL OUTPUT:\n{source_text}"

    result = client.generate(prompt, _definition())

    assert result == preserved
    assert len(delegate.prompts) == 2
    assert delegate.prompts[0] == delegate.prompts[1]
    assert "PREVIOUS REPAIR WAS REJECTED" not in delegate.prompts[1]
    assert "Do not summarize, expand, shorten, paraphrase" in delegate.prompts[0]
    assert prompt in delegate.prompts[0]


def test_preserving_repair_rejects_repeated_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_HUNT_REPAIR_MAX_ATTEMPTS", "2")
    source_text = " ".join(f"source{i}" for i in range(50))
    rewrite = {"summary": "different content " * 5}
    delegate = SequenceRepairClient([rewrite, rewrite])
    client = PreservingRepairClient(delegate)  # type: ignore[arg-type]
    prompt = f"VALIDATION ERROR:\ninvalid json\n\nORIGINAL OUTPUT:\n{source_text}"

    with pytest.raises(ProviderError, match="changed source content"):
        client.generate(prompt, _definition())

    assert len(delegate.prompts) == 2
    assert delegate.prompts[0] == delegate.prompts[1]
