from __future__ import annotations

import pytest

from job_hunt.errors import DocumentRenderingError
from job_hunt.rendering.latex import latex_value


def test_bare_text_preserves_safe_inline_latex_commands() -> None:
    assert latex_value(r"Built \textbf{production} systems with A&B") == r"Built \textbf{production} systems with A\&B"
    assert latex_value(r"Used \textit{statistical modeling} and \underline{validation}") == (
        r"Used \textit{statistical modeling} and \underline{validation}"
    )


def test_auto_mode_preserves_nested_safe_commands_and_escapes_plain_text() -> None:
    value = {"text": r"\textbf{Python & \textit{SQL}}", "format": "auto"}
    assert latex_value(value) == r"\textbf{Python \& \textit{SQL}}"


def test_plain_mode_escapes_all_commands() -> None:
    value = {"text": r"\textbf{literal}", "format": "plain"}
    assert latex_value(value) == r"\textbackslash{}textbf\{literal\}"


def test_explicit_latex_mode_remains_available_for_trusted_content() -> None:
    value = {"text": r"\customcommand{trusted}", "format": "latex"}
    assert latex_value(value) == r"\customcommand{trusted}"


def test_unknown_command_is_not_executed_in_auto_mode() -> None:
    rendered = latex_value(r"\input{secret}")
    assert rendered == r"\textbackslash{}input\{secret\}"


def test_unknown_text_format_is_rejected() -> None:
    with pytest.raises(DocumentRenderingError, match="Text format must be one of"):
        latex_value({"text": "value", "format": "unsupported"})
