from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from job_hunt.domain.models import TailoredContent
from job_hunt.errors import DocumentRenderingError
from job_hunt.rendering.documents import TenantDocumentRenderer
from job_hunt.rendering.latex import escape_latex, inject_once, latex_value
from job_hunt.rendering.profiles import (
    AtlasCoverLetterRenderer,
    AtlasCvRenderer,
    NovaCoverLetterRenderer,
    NovaCvRenderer,
)


class FakeCompiler:
    def compile(self, tex_path: Path) -> Path:
        pdf = tex_path.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4 test")
        return pdf


def cover(company: str = "A&B") -> dict[str, object]:
    return {
        "date": "August 19, 2026",
        "company_name": company,
        "paragraphs": ["First.", "Second.", "Third."],
    }


def test_shared_latex_safety_and_injection() -> None:
    assert escape_latex("A&B_#") == r"A\&B\_\#"
    assert latex_value({"text": r"\textbf{Safe}", "format": "latex"}) == r"\textbf{Safe}"
    assert inject_once("x %%__A__%%", {"%%__A__%%": "y"}) == "x y"
    with pytest.raises(DocumentRenderingError, match="must occur once"):
        inject_once("none", {"%%__A__%%": "y"})


@pytest.mark.parametrize(
    "tenant,cv_renderer,cover_renderer",
    [
        ("atlas", AtlasCvRenderer(), AtlasCoverLetterRenderer()),
        ("nova", NovaCvRenderer(), NovaCoverLetterRenderer()),
    ],
)
def test_real_tenant_templates_and_master_cv_render(
    tmp_path: Path, tenant: str, cv_renderer: object, cover_renderer: object
) -> None:
    root = Path("tenants") / tenant
    master = json.loads((root / "master_cv.json").read_text())
    renderer = TenantDocumentRenderer(
        cv_renderer,  # type: ignore[arg-type]
        cover_renderer,  # type: ignore[arg-type]
        root / "templates/cv_template.tex",
        root / "templates/cover_letter_template.tex",
        FakeCompiler(),
    )
    bundle = renderer.render(
        TailoredContent(cv=master, cover_letter=cover()),
        tmp_path / tenant,
        tenant,
        applicant_filename="Applicant",
    )
    assert bundle.cv_pdf.read_bytes().startswith(b"%PDF")
    assert bundle.cover_letter_pdf.exists() and bundle.archive.exists()
    assert bundle.cv_json.name == "Applicant_CV.json"
    assert bundle.cv_tex.name == "Applicant_CV.tex"
    assert bundle.cv_pdf.name == "Applicant_CV.pdf"
    assert bundle.cover_letter_json.name == "Applicant_CL.json"
    assert bundle.cover_letter_tex.name == "Applicant_CL.tex"
    assert bundle.cover_letter_pdf.name == "Applicant_CL.pdf"
    with zipfile.ZipFile(bundle.archive) as archive:
        assert set(archive.namelist()) == {
            "Applicant_CV.json",
            "Applicant_CV.tex",
            "Applicant_CV.pdf",
            "Applicant_CL.json",
            "Applicant_CL.tex",
            "Applicant_CL.pdf",
        }
    assert "%%__" not in bundle.cv_tex.read_text()
    assert r"A\&B" in bundle.cover_letter_tex.read_text()


def test_nova_renderer_does_not_emit_empty_entry_lists() -> None:
    assert NovaCvRenderer._entries([]) == ""
    assert NovaCvRenderer._entries([{}]) == ""
    assert NovaCvRenderer._entries([{"icon": "link", "content": ["", {"text": ""}]}]) == ""


def test_nova_renderer_keeps_visible_entries_and_drops_empty_neighbors() -> None:
    rendered = NovaCvRenderer._entries(
        [
            {},
            {"title": "Visible", "content": ["", {"text": "Did useful work"}]},
            {"icon": "link"},
        ]
    )
    assert rendered.startswith("\\CVEntries{")
    assert rendered.count("\\CVEntry{") == 1
    assert "Visible" in rendered
    assert "\\CVContent{Did useful work}" in rendered
    assert "\\CVContent{}" not in rendered


def test_atlas_renderer_wraps_parent_bullets_in_itemize_structure() -> None:
    rendered = AtlasCvRenderer().render(
        "%%__SECTIONS__%%",
        {
            "sections": [
                {
                    "type": "entries",
                    "title": "EXPERIENCE",
                    "entries": [
                        {
                            "title": "Designer",
                            "date": "2026",
                            "content": [
                                {"text": "First bullet", "bullet": True},
                                {"text": "Second bullet", "bullet": True},
                            ],
                        }
                    ],
                },
                {"type": "education"},
            ]
        },
    )
    assert "\\CVBulletList{" in rendered
    assert "\\CVBullet{First bullet}" in rendered
    assert "\\CVBullet{Second bullet}" in rendered


def test_atlas_renderer_wraps_nested_bullets_and_renders_plain_text_safely() -> None:
    rendered = AtlasCvRenderer().render(
        "%%__SECTIONS__%%",
        {
            "sections": [
                {
                    "type": "entries",
                    "title": "PROJECTS",
                    "entries": [
                        {
                            "title": "Parent",
                            "content": [{"text": "Parent note", "bullet": False}],
                        },
                        {
                            "title": "Child",
                            "parent": "Parent",
                            "nested_group": "Selected work",
                            "content": [
                                {"text": "Nested bullet", "bullet": True},
                                {"text": "Nested note", "bullet": False},
                            ],
                        },
                    ],
                },
                {"type": "education"},
            ]
        },
    )
    assert "\\CVEntryText{Parent note}" in rendered
    assert "\\CVNestedBulletList{" in rendered
    assert "\\CVBullet{Nested bullet}" in rendered
    assert "\\CVNestedText{Nested note}" in rendered
    assert "\\CVBullet[false]" not in rendered


def test_atlas_renderer_omits_empty_entry_sections_and_nested_entries() -> None:
    rendered = AtlasCvRenderer().render(
        "%%__SECTIONS__%%",
        {
            "sections": [
                {
                    "type": "entries",
                    "title": "PROJECTS",
                    "entries": [
                        {},
                        {"title": "Parent"},
                        {"parent": "Parent", "icon": "link", "content": [{"text": ""}]},
                    ],
                },
                {"type": "entries", "title": "EMPTY", "entries": [{}]},
                {"type": "education"},
            ]
        },
    )
    assert "\\CVSection{PROJECTS}" in rendered
    assert "\\CVEntry{Parent}" in rendered
    assert "\\CVNestedEntry" not in rendered
    assert "\\CVSection{EMPTY}" not in rendered


def test_atlas_renderer_omits_empty_references_section() -> None:
    rendered = AtlasCvRenderer().render(
        "%%__SECTIONS__%%",
        {
            "sections": [
                {"type": "education"},
                {"type": "references", "title": "REFERENCES", "items": []},
            ]
        },
    )
    assert "\\CVSection{REFERENCES}" not in rendered
    assert "\\CVReferences{" not in rendered


def test_cover_letter_requires_exactly_three_paragraphs() -> None:
    template = " ".join(AtlasCoverLetterRenderer.markers.values())
    with pytest.raises(DocumentRenderingError, match="exactly three"):
        AtlasCoverLetterRenderer().render(template, {"date": "x", "company_name": "y", "paragraphs": ["one"]})


def test_atlas_requires_one_education_marker() -> None:
    with pytest.raises(DocumentRenderingError, match="exactly one"):
        AtlasCvRenderer().render("%%__SECTIONS__%%", {"sections": []})
