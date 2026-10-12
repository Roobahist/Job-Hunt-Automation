from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from job_hunt.domain.models import (
    ArtifactBundle,
    Job,
    PromptDefinition,
    Qualification,
    TailoredContent,
)


class StructuredClient(Protocol):
    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]: ...


class JobExtractor(Protocol):
    def extract_job(self, content: str, source_url: str) -> Job: ...


class JobRepository(Protocol):
    def find(self, job: Job) -> Mapping[str, Any] | None: ...
    def create(self, job: Job) -> Mapping[str, Any]: ...
    def reset(self, row_id: int, job: Job) -> Mapping[str, Any]: ...
    def clear_qualification(self, row_id: int) -> None: ...
    def save_qualification(self, row_id: int, result: Qualification) -> None: ...
    def save_artifacts(self, row_id: int, uploaded_files: Mapping[str, Any]) -> None: ...
    def save_log(self, row_id: int, path: Path) -> None: ...
    def set_status(self, row_id: int, status_key: str) -> None: ...
    def has_status(self, row_id: int, status_key: str) -> bool: ...


class CompatibilityFilter(Protocol):
    def compatible(self, job: Job, prompts: Mapping[str, PromptDefinition]) -> bool: ...


class Qualifier(Protocol):
    def qualify(
        self,
        job: Job,
        master_cv: Mapping[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> Qualification: ...


class Tailor(Protocol):
    def tailor(
        self,
        job: Job,
        master_cv: Mapping[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> TailoredContent: ...


class DocumentRenderer(Protocol):
    def render(
        self,
        content: TailoredContent,
        output_directory: Path,
        basename: str,
        *,
        applicant_filename: str,
    ) -> ArtifactBundle: ...


class ArtifactPublisher(Protocol):
    def publish(self, artifacts: ArtifactBundle) -> Mapping[str, Any]: ...


class Notifier(Protocol):
    def send_documents(self, chat_id: str, artifacts: Iterable[Path], caption: str) -> str: ...


class DiscoveryProvider(Protocol):
    def discover(self, urls: Sequence[str], *, max_items: int) -> Iterable[Mapping[str, Any]]: ...
    def fetch_linkedin(self, job_id: int, *, country: str, max_concurrency: int) -> Mapping[str, Any]: ...
    def fetch_indeed(self, job_id: str) -> Mapping[str, Any]: ...
