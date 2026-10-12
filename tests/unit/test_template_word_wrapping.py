from __future__ import annotations

from pathlib import Path

import pytest

NOVA_CV = Path("tenants/nova/templates/cv_template.tex")
TEMPLATES = [
    NOVA_CV,
    Path("tenants/nova/templates/cover_letter_template.tex"),
    Path("tenants/atlas/templates/cv_template.tex"),
    Path("tenants/atlas/templates/cover_letter_template.tex"),
]


@pytest.mark.parametrize("template_path", TEMPLATES)
def test_template_disables_word_hyphenation_globally(template_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")

    if template_path == NOVA_CV:
        assert "\\pretolerance=10000" not in template
    else:
        assert "\\pretolerance=10000" in template
    assert "\\hyphenpenalty=10000" in template
    assert "\\exhyphenpenalty=10000" in template
    assert "\\setlength{\\emergencystretch}{3em}" in template
