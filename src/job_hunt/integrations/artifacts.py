from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from job_hunt.domain.models import ArtifactBundle

if TYPE_CHECKING:
    from job_hunt.integrations.baserow import BaserowClient


APPLICATION_ZIP_FIELD = "Application ZIP"


class BaserowArtifactPublisher:
    """Persist final application artifacts directly in Baserow.

    The Jobs table receives the application-ready CV and cover-letter PDFs plus the
    complete ZIP bundle. JSON and TeX sources remain grouped inside that ZIP for
    debugging and reproducibility.
    """

    def __init__(self, baserow: BaserowClient) -> None:
        self.baserow = baserow

    def publish(self, artifacts: ArtifactBundle) -> Mapping[str, Any]:
        return {
            "CV": [self.baserow.upload_file(artifacts.cv_pdf)],
            "Cover Letter": [self.baserow.upload_file(artifacts.cover_letter_pdf)],
            APPLICATION_ZIP_FIELD: [self.baserow.upload_file(artifacts.archive)],
        }
