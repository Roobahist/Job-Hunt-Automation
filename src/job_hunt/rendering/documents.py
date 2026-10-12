from __future__ import annotations

import json
import shutil
from pathlib import Path

from job_hunt.domain.models import ArtifactBundle, TailoredContent
from job_hunt.rendering.latex import PdfLatexCompiler, TexCompiler
from job_hunt.rendering.profiles import CoverLetterRenderer, CvRenderer


class TenantDocumentRenderer:
    def __init__(
        self,
        cv_renderer: CvRenderer,
        cover_letter_renderer: CoverLetterRenderer,
        cv_template: Path,
        cover_letter_template: Path,
        compiler: TexCompiler | None = None,
    ) -> None:
        self.cv_renderer = cv_renderer
        self.cover_letter_renderer = cover_letter_renderer
        self.cv_template = cv_template
        self.cover_letter_template = cover_letter_template
        self.compiler = compiler or PdfLatexCompiler()

    def render(
        self,
        content: TailoredContent,
        output_directory: Path,
        basename: str,
        *,
        applicant_filename: str,
    ) -> ArtifactBundle:
        # The per-job trace is stored in the run directory before rendering starts,
        # so that directory may already exist. Artifact filenames remain deterministic
        # and are replaced by the current render when a run is resumed.
        output_directory.mkdir(parents=True, exist_ok=True)
        cv_json = output_directory / f"{applicant_filename}_CV.json"
        cover_json = output_directory / f"{applicant_filename}_CL.json"
        cv_tex = cv_json.with_suffix(".tex")
        cover_tex = cover_json.with_suffix(".tex")
        cv_json.write_text(json.dumps(content.cv, indent=2, ensure_ascii=False), encoding="utf-8")
        cover_json.write_text(json.dumps(content.cover_letter, indent=2, ensure_ascii=False), encoding="utf-8")
        cv_tex.write_text(
            self.cv_renderer.render(self.cv_template.read_text(encoding="utf-8"), content.cv),
            encoding="utf-8",
        )
        cover_tex.write_text(
            self.cover_letter_renderer.render(
                self.cover_letter_template.read_text(encoding="utf-8"), content.cover_letter
            ),
            encoding="utf-8",
        )
        cv_pdf = self.compiler.compile(cv_tex)
        cover_pdf = self.compiler.compile(cover_tex)
        temporary_archive = Path(
            shutil.make_archive(
                str(output_directory.parent / f".{basename}-bundle"),
                "zip",
                output_directory,
            )
        )
        archive = output_directory / f"{basename}.zip"
        temporary_archive.replace(archive)
        return ArtifactBundle(
            run_directory=output_directory,
            cv_json=cv_json,
            cv_tex=cv_tex,
            cv_pdf=cv_pdf,
            cover_letter_json=cover_json,
            cover_letter_tex=cover_tex,
            cover_letter_pdf=cover_pdf,
            archive=archive,
        )
