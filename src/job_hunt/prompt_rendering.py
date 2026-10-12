from __future__ import annotations

import json
import re
from collections.abc import Mapping

from job_hunt.errors import ConfigurationError

_PLACEHOLDER = re.compile(r"\[\[([A-Za-z0-9_]+)\]\]")


def _compact(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def render_prompt(template: str, values: Mapping[str, object], prompt_key: str) -> str:
    """Render placeholders present in the template without interpreting inserted data."""

    missing = sorted({match.group(1) for match in _PLACEHOLDER.finditer(template)} - values.keys())
    if missing:
        raise ConfigurationError(f"Prompt '{prompt_key}' contains unresolved placeholders: {missing}")

    return _PLACEHOLDER.sub(lambda match: _compact(values[match.group(1)]), template)
