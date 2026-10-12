from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+%#@-]+")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


@dataclass(frozen=True, slots=True)
class RepairAssessment:
    accepted: bool
    source_tokens: int
    repaired_tokens: int
    token_delta: int
    token_similarity: float
    size_ratio: float
    reason: str


def repair_max_attempts() -> int:
    return max(1, int(os.getenv("JOB_HUNT_REPAIR_MAX_ATTEMPTS", "3")))


def repair_max_token_delta() -> int:
    return max(0, int(os.getenv("JOB_HUNT_REPAIR_MAX_TOKEN_DELTA", "20")))


def repair_min_token_similarity() -> float:
    return min(1.0, max(0.0, float(os.getenv("JOB_HUNT_REPAIR_MIN_TOKEN_SIMILARITY", "0.90"))))


def repair_min_size_ratio() -> float:
    return max(0.0, float(os.getenv("JOB_HUNT_REPAIR_MIN_SIZE_RATIO", "0.82")))


def repair_max_size_ratio() -> float:
    return max(repair_min_size_ratio(), float(os.getenv("JOB_HUNT_REPAIR_MAX_SIZE_RATIO", "1.18")))


def _normalized_text(value: str) -> str:
    return _FENCE_RE.sub("", value.strip())


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(_normalized_text(value))]


def assess_repair(raw_output: str, repaired: dict[str, Any]) -> RepairAssessment:
    repaired_text = json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))
    source_tokens = _tokens(raw_output)
    repaired_tokens = _tokens(repaired_text)
    source_count = len(source_tokens)
    repaired_count = len(repaired_tokens)
    token_delta = abs(repaired_count - source_count)
    similarity = SequenceMatcher(None, source_tokens, repaired_tokens, autojunk=False).ratio()
    source_size = max(1, len(_normalized_text(raw_output)))
    size_ratio = len(repaired_text) / source_size

    reasons: list[str] = []
    if token_delta > repair_max_token_delta():
        reasons.append(f"token delta {token_delta} exceeds {repair_max_token_delta()}")
    if similarity < repair_min_token_similarity():
        reasons.append(f"token similarity {similarity:.3f} is below {repair_min_token_similarity():.3f}")
    if size_ratio < repair_min_size_ratio() or size_ratio > repair_max_size_ratio():
        reasons.append(
            f"size ratio {size_ratio:.3f} is outside {repair_min_size_ratio():.3f}-{repair_max_size_ratio():.3f}"
        )

    return RepairAssessment(
        accepted=not reasons,
        source_tokens=source_count,
        repaired_tokens=repaired_count,
        token_delta=token_delta,
        token_similarity=similarity,
        size_ratio=size_ratio,
        reason="; ".join(reasons) if reasons else "repair preserved the original content",
    )
