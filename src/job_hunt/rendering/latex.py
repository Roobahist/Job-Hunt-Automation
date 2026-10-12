from __future__ import annotations

import re
import subprocess
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from job_hunt.errors import DocumentRenderingError

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

SAFE_LATEX_COMMANDS = {
    "emph",
    "textbf",
    "textit",
    "textnormal",
    "textrm",
    "textsf",
    "textsl",
    "textsc",
    "texttt",
    "underline",
    "mbox",
    "hbox",
    "href",
    "url",
    "nolinkurl",
    "textsuperscript",
    "textsubscript",
    "small",
    "footnotesize",
    "scriptsize",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "newline",
    "linebreak",
    "ldots",
    "dots",
    "copyright",
    "textcopyright",
    "pounds",
    "textregistered",
    "texttrademark",
    "textdegree",
    "textasciitilde",
    "textasciicircum",
    "textbackslash",
}


def _normalized_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value or ""))


def escape_latex(value: object) -> str:
    text = _normalized_text(value)
    return "".join(LATEX_ESCAPES.get(character, character) for character in text)


def _balanced_group_end(text: str, start: int, opener: str, closer: str) -> int | None:
    if start >= len(text) or text[start] != opener:
        return None
    depth = 0
    index = start
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _render_safe_command(text: str, start: int) -> tuple[str, int] | None:
    match = re.match(r"\\([A-Za-z@]+|.)", text[start:])
    if match is None:
        return None

    command = match.group(1)
    command_end = start + match.end()
    if len(command) == 1 and command in "#$%&_{}~^\\":
        return text[start:command_end], command_end
    if command not in SAFE_LATEX_COMMANDS:
        return None

    rendered = [text[start:command_end]]
    index = command_end

    while index < len(text) and text[index].isspace() and text[index] != "\n":
        rendered.append(text[index])
        index += 1

    if index < len(text) and text[index] == "[":
        optional_end = _balanced_group_end(text, index, "[", "]")
        if optional_end is None:
            return None
        rendered.append(text[index:optional_end])
        index = optional_end

    while index < len(text) and text[index] == "{":
        group_end = _balanced_group_end(text, index, "{", "}")
        if group_end is None:
            return None
        inner = text[index + 1 : group_end - 1]
        rendered.append("{" + auto_latex(inner) + "}")
        index = group_end

    return "".join(rendered), index


def auto_latex(value: object) -> str:
    """Escape normal text while preserving a conservative set of inline LaTeX commands."""
    text = _normalized_text(value)
    output: list[str] = []
    index = 0

    while index < len(text):
        if text[index] == "\\":
            rendered = _render_safe_command(text, index)
            if rendered is not None:
                command_text, index = rendered
                output.append(command_text)
                continue
            output.append(LATEX_ESCAPES["\\"])
            index += 1
            continue

        character = text[index]
        output.append(LATEX_ESCAPES.get(character, character))
        index += 1

    return "".join(output)


def latex_value(value: object) -> str:
    if isinstance(value, Mapping):
        allowed = {"text", "format"}
        unknown = set(value) - allowed
        if unknown:
            raise DocumentRenderingError(f"Unknown text-format keys: {sorted(unknown)}")

        text = value.get("text", "")
        text_format = value.get("format", "auto")
        if text_format == "latex":
            return _normalized_text(text)
        if text_format in {"plain", "text"}:
            return escape_latex(text)
        if text_format == "auto":
            return auto_latex(text)
        raise DocumentRenderingError("Text format must be one of: auto, plain, text, latex")

    return auto_latex(value)


def inject_once(template: str, replacements: Mapping[str, str]) -> str:
    output = template
    for marker, content in replacements.items():
        count = output.count(marker)
        if count != 1:
            raise DocumentRenderingError(f"Template marker {marker!r} must occur once; found {count}")
        output = output.replace(marker, content)
    unresolved = re.findall(r"%%__[A-Z0-9_]+__%%", output)
    if unresolved:
        raise DocumentRenderingError(f"Unresolved template markers: {sorted(set(unresolved))}")
    return output


class TexCompiler(Protocol):
    def compile(self, tex_path: Path) -> Path: ...


class PdfLatexCompiler:
    def __init__(self, executable: str = "pdflatex", timeout_seconds: int = 120) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def compile(self, tex_path: Path) -> Path:
        command = [
            self.executable,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]
        for _ in range(2):
            try:
                result = subprocess.run(
                    command,
                    cwd=tex_path.parent,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DocumentRenderingError(f"LaTeX compiler failed to start or timed out: {exc}") from exc
            if result.returncode:
                output = (result.stdout + "\n" + result.stderr)[-4000:]
                raise DocumentRenderingError(f"LaTeX compilation failed:\n{output}")
        pdf = tex_path.with_suffix(".pdf")
        if not pdf.exists():
            raise DocumentRenderingError("LaTeX reported success without producing a PDF")
        for suffix in (".aux", ".log", ".out"):
            tex_path.with_suffix(suffix).unlink(missing_ok=True)
        return pdf
