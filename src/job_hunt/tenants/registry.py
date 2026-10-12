from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from job_hunt.config import TenantBootstrap
from job_hunt.errors import ConfigurationError
from job_hunt.rendering.documents import TenantDocumentRenderer
from job_hunt.rendering.latex import PdfLatexCompiler
from job_hunt.rendering.profiles import (
    CoverLetterRenderer,
    CvRenderer,
    AtlasCoverLetterRenderer,
    AtlasCvRenderer,
    NovaCoverLetterRenderer,
    NovaCvRenderer,
)


@dataclass(frozen=True, slots=True)
class TenantContext:
    bootstrap: TenantBootstrap
    master_cv: dict[str, Any]
    renderer: TenantDocumentRenderer


class TenantRegistry:
    def __init__(
        self,
        bootstraps: dict[str, TenantBootstrap],
        project_root: Path = Path("."),
        *,
        latex_compile_timeout_seconds: int = 180,
    ) -> None:
        self.bootstraps = bootstraps
        self.project_root = project_root
        self.latex_compile_timeout_seconds = latex_compile_timeout_seconds

    def get(self, key: str) -> TenantContext:
        try:
            bootstrap = self.bootstraps[key]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown tenant: {key}") from exc
        if not bootstrap.enabled:
            raise ConfigurationError(f"Tenant is disabled: {key}")
        root = self.project_root / bootstrap.tenant_root
        try:
            master_cv = json.loads((root / "master_cv.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Could not load master CV for {key}: {exc}") from exc
        if bootstrap.renderer == "atlas":
            cv: CvRenderer = AtlasCvRenderer()
            cover: CoverLetterRenderer = AtlasCoverLetterRenderer()
        elif bootstrap.renderer == "nova":
            cv, cover = NovaCvRenderer(), NovaCoverLetterRenderer()
        else:
            raise ConfigurationError(f"Unknown renderer profile: {bootstrap.renderer}")
        renderer = TenantDocumentRenderer(
            cv,
            cover,
            root / "templates/cv_template.tex",
            root / "templates/cover_letter_template.tex",
            compiler=PdfLatexCompiler(timeout_seconds=self.latex_compile_timeout_seconds),
        )
        return TenantContext(bootstrap, master_cv, renderer)
