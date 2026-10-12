from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from job_hunt.errors import DocumentRenderingError
from job_hunt.rendering.latex import inject_once, latex_value


def _content(items: object, command: str = "CVContent") -> str:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return ""
    rendered: list[str] = []
    for item in items:
        bullet = True
        value = item
        if isinstance(item, Mapping):
            value = item.get("text", "")
            bullet = bool(item.get("bullet", True))
        text = latex_value(value)
        if not text:
            continue
        optional = "" if bullet else "[false]"
        rendered.append(f"\\{command}{optional}{{{text}}}")
    return "\n".join(rendered)


def _atlas_content(items: object, *, nested: bool = False) -> str:
    """Render structured Atlas content without allowing JSON to define LaTeX structure."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return ""

    bullet_command = "CVBullet"
    list_command = "CVNestedBulletList" if nested else "CVBulletList"
    text_command = "CVNestedText" if nested else "CVEntryText"
    rendered: list[str] = []
    pending_bullets: list[str] = []

    def flush_bullets() -> None:
        if not pending_bullets:
            return
        body = "\n".join(pending_bullets)
        rendered.append(f"\\{list_command}{{\n{body}\n}}")
        pending_bullets.clear()

    for item in items:
        bullet = True
        value = item
        if isinstance(item, Mapping):
            value = item.get("text", "")
            bullet = bool(item.get("bullet", True))
        text = latex_value(value)
        if not text:
            continue
        if bullet:
            pending_bullets.append(f"\\{bullet_command}{{{text}}}")
        else:
            flush_bullets()
            rendered.append(f"\\{text_command}{{{text}}}")

    flush_bullets()
    return "\n".join(rendered)


def _label_row(row: Mapping[str, Any]) -> str:
    label = latex_value(row.get("label", ""))
    value = latex_value(row.get("value", ""))
    return f"\\CVLabelRow{{{label}}}{{{value}}}"


def _wrapped_entries(entries: Sequence[str]) -> str:
    rendered = [entry for entry in entries if entry]
    if not rendered:
        return ""
    return "\\CVEntries{\n" + "\n".join(rendered) + "\n}"


class CvRenderer(ABC):
    @abstractmethod
    def render(self, template: str, data: Mapping[str, Any]) -> str: ...


class NovaCvRenderer(CvRenderer):
    markers: ClassVar[dict[str, str]] = {
        "summary": "%%__SUMMARY__%%",
        "skills": "%%__SKILLS__%%",
        "projects": "%%__PROJECTS__%%",
        "work_experience": "%%__WORK_EXPERIENCE__%%",
        "awards": "%%__AWARDS__%%",
    }

    def render(self, template: str, data: Mapping[str, Any]) -> str:
        missing = set(self.markers) - data.keys()
        if missing:
            raise DocumentRenderingError(f"Nova CV missing sections: {sorted(missing)}")
        replacements = {
            self.markers["summary"]: self._summary(data["summary"]),
            self.markers["skills"]: self._skills(data["skills"]),
            self.markers["projects"]: self._entries(data["projects"]),
            self.markers["work_experience"]: self._entries(data["work_experience"]),
            self.markers["awards"]: self._entries(data["awards"]),
        }
        return inject_once(template, replacements)

    @staticmethod
    def _summary(value: object) -> str:
        values = value if isinstance(value, list) else [value]
        rendered = [latex_value(item) for item in values]
        return "\n".join(f"\\CVText{{{item}}}" for item in rendered if item)

    @staticmethod
    def _skills(value: object) -> str:
        if not isinstance(value, list):
            raise DocumentRenderingError("skills must be a list")
        rows = "\n".join(_label_row(row) for row in value if isinstance(row, Mapping))
        return f"\\CVLabelRows{{\n{rows}\n}}" if rows else ""

    @staticmethod
    def _entries(value: object) -> str:
        if not isinstance(value, list):
            raise DocumentRenderingError("entry section must be a list")
        entries: list[str] = []
        for entry in value:
            if not isinstance(entry, Mapping):
                continue
            args = [
                latex_value(entry.get("title", "")),
                latex_value(entry.get("secondary", "")),
                latex_value(entry.get("url", "")),
                latex_value(entry.get("icon", "")),
                latex_value(entry.get("organization", "")),
                latex_value(entry.get("date", "")),
                _content(entry.get("content", [])),
            ]
            # The template ignores an icon by itself. If every field that can
            # produce visible content is blank, emitting CVEntry would make the
            # surrounding itemize structurally non-empty in TeX source but leave
            # it with no actual \item at expansion time.
            if not any(args[index] for index in (0, 1, 2, 4, 5, 6)):
                continue
            entries.append("\\CVEntry" + "".join(f"{{{arg}}}" for arg in args))
        return _wrapped_entries(entries)


class AtlasCvRenderer(CvRenderer):
    marker = "%%__SECTIONS__%%"

    def render(self, template: str, data: Mapping[str, Any]) -> str:
        sections = data.get("sections")
        if not isinstance(sections, list):
            raise DocumentRenderingError("Atlas CV requires a sections list")
        education_count = sum(
            isinstance(section, Mapping) and section.get("type") == "education" for section in sections
        )
        if education_count != 1:
            raise DocumentRenderingError("Atlas CV requires exactly one education section")
        rendered = [self._section(section) for section in sections if isinstance(section, Mapping)]
        return inject_once(template, {self.marker: "\n\n".join(filter(None, rendered))})

    def _section(self, section: Mapping[str, Any]) -> str:
        kind = section.get("type")
        title = latex_value(section.get("title", ""))
        heading = f"\\CVSection{{{title}}}\n" if title else ""
        if kind == "education":
            return "\\CVFixedEducation"
        if kind == "text":
            content = section.get("content", [])
            values = content if isinstance(content, list) else [content]
            body = "\n".join(
                f"\\CVText{{{rendered}}}" for rendered in (latex_value(value) for value in values) if rendered
            )
            return heading + body if body else ""
        if kind == "label_rows":
            rows = section.get("rows", [])
            body = "\n".join(_label_row(row) for row in rows if isinstance(row, Mapping))
            return heading + f"\\CVLabelRows{{\n{body}\n}}" if body else ""
        if kind == "references":
            refs = section.get("items", [])
            body = "\n".join(
                "\\CVReference"
                f"{{{latex_value(ref.get('name', ''))}}}"
                f"{{{latex_value(ref.get('title', ''))}}}"
                f"{{{latex_value(ref.get('contact', ''))}}}"
                for ref in refs
                if isinstance(ref, Mapping)
                and any(latex_value(ref.get(field, "")) for field in ("name", "title", "contact"))
            )
            if not body:
                return ""
            return heading + f"\\CVReferences{{\n{body}\n}}"
        if kind == "entries":
            body = self._entries(section.get("entries", []))
            return heading + body if body else ""
        raise DocumentRenderingError(f"Unknown Atlas section type: {kind!r}")

    @staticmethod
    def _entries(value: object) -> str:
        if not isinstance(value, list):
            raise DocumentRenderingError("entries must be a list")
        parents = [item for item in value if isinstance(item, Mapping) and not item.get("parent")]
        children = [item for item in value if isinstance(item, Mapping) and item.get("parent")]
        output: list[str] = []
        for entry in parents:
            nested_entries: list[tuple[Mapping[str, Any], str]] = []
            for child in children:
                if child.get("parent") != entry.get("title"):
                    continue
                child_args = [
                    latex_value(child.get("title", "")),
                    latex_value(child.get("date", "")),
                    latex_value(child.get("secondary_right", "")),
                    latex_value(child.get("url", "")),
                    latex_value(child.get("icon", "")),
                    _atlas_content(child.get("content", []), nested=True),
                ]
                if not any(child_args[index] for index in (0, 1, 2, 3, 5)):
                    continue
                nested_entries.append(
                    (child, "\\CVNestedEntry" + "".join(f"{{{arg}}}" for arg in child_args))
                )

            nested_group = ""
            if nested_entries:
                label = latex_value(nested_entries[0][0].get("nested_group", ""))
                nested_body = "\n".join(rendered for _, rendered in nested_entries)
                nested_group = f"\\CVNestedGroup{{{label}}}{{{nested_body}}}"

            args = [
                latex_value(entry.get("title", "")),
                latex_value(entry.get("date", "")),
                latex_value(entry.get("secondary", entry.get("secondary_left", ""))),
                latex_value(entry.get("secondary_right", "")),
                latex_value(entry.get("url", "")),
                latex_value(entry.get("icon", "")),
                _atlas_content(entry.get("content", [])),
                nested_group,
            ]
            # Atlas's CVEntry ignores the icon by itself, but nested content can
            # legitimately make an otherwise header-less parent renderable.
            if not any(args[index] for index in (0, 1, 2, 3, 4, 6, 7)):
                continue
            output.append("\\CVEntry" + "".join(f"{{{arg}}}" for arg in args))
        return _wrapped_entries(output)


class CoverLetterRenderer:
    markers: ClassVar[dict[str, str]] = {
        "date": "%%__DATE__%%",
        "company_name": "%%__COMPANY_NAME__%%",
        "paragraph_1": "%%__PARAGRAPH_1__%%",
        "paragraph_2": "%%__PARAGRAPH_2__%%",
        "paragraph_3": "%%__PARAGRAPH_3__%%",
    }

    def render(self, template: str, data: Mapping[str, Any]) -> str:
        paragraphs = data.get("paragraphs")
        if paragraphs is None:
            paragraphs = [data.get(f"paragraph_{index}") for index in range(1, 4)]
        if not isinstance(paragraphs, list) or len(paragraphs) != 3 or not all(paragraphs):
            raise DocumentRenderingError("Cover letter requires exactly three non-empty paragraphs")
        replacements = {
            self.markers["date"]: latex_value(data.get("date", "")),
            self.markers["company_name"]: latex_value(data.get("company_name", "")),
            self.markers["paragraph_1"]: f"\\CLParagraph{{{latex_value(paragraphs[0])}}}",
            self.markers["paragraph_2"]: f"\\CLParagraph{{{latex_value(paragraphs[1])}}}",
            self.markers["paragraph_3"]: f"\\CLParagraph{{{latex_value(paragraphs[2])}}}",
        }
        if not replacements[self.markers["date"]] or not replacements[self.markers["company_name"]]:
            raise DocumentRenderingError("Cover letter date and company_name are required")
        return inject_once(template, replacements)


class AtlasCoverLetterRenderer(CoverLetterRenderer):
    """Atlas-specific strategy boundary; the signature remains in Atlas's template."""


class NovaCoverLetterRenderer(CoverLetterRenderer):
    """Nova-specific strategy boundary; the signature remains in Nova's template."""
