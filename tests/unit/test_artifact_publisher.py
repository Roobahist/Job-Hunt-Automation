from __future__ import annotations

from pathlib import Path

from job_hunt.domain.models import ArtifactBundle
from job_hunt.integrations.artifacts import APPLICATION_ZIP_FIELD, BaserowArtifactPublisher


def _bundle(tmp_path: Path) -> ArtifactBundle:
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "cv_json": "cv.json",
            "cv_tex": "cv.tex",
            "cv_pdf": "cv.pdf",
            "cover_letter_json": "cl.json",
            "cover_letter_tex": "cl.tex",
            "cover_letter_pdf": "cl.pdf",
            "archive": "application.zip",
        }.items()
    }
    for path in paths.values():
        path.write_text("artifact")

    return ArtifactBundle(run_directory=tmp_path, **paths)


def test_baserow_publisher_uploads_pdfs_and_complete_zip(tmp_path: Path) -> None:
    uploaded: list[str] = []

    class BaserowStub:
        def upload_file(self, path: Path) -> dict[str, str]:
            uploaded.append(path.name)
            return {"name": path.name}

    result = BaserowArtifactPublisher(BaserowStub()).publish(_bundle(tmp_path))  # type: ignore[arg-type]

    assert uploaded == ["cv.pdf", "cl.pdf", "application.zip"]
    assert result == {
        "CV": [{"name": "cv.pdf"}],
        "Cover Letter": [{"name": "cl.pdf"}],
        APPLICATION_ZIP_FIELD: [{"name": "application.zip"}],
    }
