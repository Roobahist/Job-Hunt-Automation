from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from job_hunt.domain.models import TailoredContent
from job_hunt.rendering.documents import TenantDocumentRenderer
from job_hunt.rendering.profiles import (
    AtlasCoverLetterRenderer,
    AtlasCvRenderer,
    NovaCoverLetterRenderer,
    NovaCvRenderer,
)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
@pytest.mark.parametrize(
    "tenant,cv_renderer,cover_renderer",
    [
        ("atlas", AtlasCvRenderer(), AtlasCoverLetterRenderer()),
        ("nova", NovaCvRenderer(), NovaCoverLetterRenderer()),
    ],
)
def test_real_templates_compile_to_pdf(
    tmp_path: Path,
    tenant: str,
    cv_renderer: object,
    cover_renderer: object,
) -> None:
    root = Path("tenants") / tenant
    renderer = TenantDocumentRenderer(
        cv_renderer,  # type: ignore[arg-type]
        cover_renderer,  # type: ignore[arg-type]
        root / "templates/cv_template.tex",
        root / "templates/cover_letter_template.tex",
    )
    content = TailoredContent(
        cv=json.loads((root / "master_cv.json").read_text(encoding="utf-8")),
        cover_letter={
            "date": "August 19, 2026",
            "company_name": "Example Company",
            "paragraphs": ["First paragraph.", "Second paragraph.", "Third paragraph."],
        },
    )
    result = renderer.render(
        content,
        tmp_path / tenant,
        tenant,
        applicant_filename="Applicant",
    )
    assert result.cv_pdf.stat().st_size > 1000
    assert result.cover_letter_pdf.stat().st_size > 1000
