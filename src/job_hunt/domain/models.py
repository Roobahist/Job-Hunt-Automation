from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

_HTTP_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _normalize_http_url_input(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    if text.startswith("//"):
        return f"https:{text}"
    if _HTTP_SCHEME_PATTERN.match(text):
        return text
    return f"https://{text}"


class EntryType(StrEnum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    EXTERNAL = "external"
    AI_CONTENT = "ai_content"
    URL = "url"


class SubmissionBase(BaseModel):
    reuse_existing_score: bool = False
    reuse_existing_data: bool = False


class LinkedInSubmission(SubmissionBase):
    entry_type: Literal[EntryType.LINKEDIN]
    linkedin_job_id: int = Field(gt=0)


class IndeedSubmission(SubmissionBase):
    entry_type: Literal[EntryType.INDEED]
    indeed_job_id: str = Field(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ExternalSubmission(SubmissionBase):
    entry_type: Literal[EntryType.EXTERNAL]
    company_name: str = Field(min_length=1)
    job_title: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    job_url: HttpUrl
    location: str | None = None
    contract_type: str | None = None
    published_at: datetime | None = None
    external_job_id: str | None = None
    source: str = "external"

    @field_validator("job_url", mode="before")
    @classmethod
    def normalize_job_url(cls, value: Any) -> Any:
        return _normalize_http_url_input(value)


class AiContentSubmission(SubmissionBase):
    entry_type: Literal[EntryType.AI_CONTENT]
    page_content: str = Field(min_length=1)
    source_url: HttpUrl | None = None

    @field_validator("source_url", mode="before")
    @classmethod
    def normalize_source_url(cls, value: Any) -> Any:
        return _normalize_http_url_input(value)


class UrlSubmission(SubmissionBase):
    entry_type: Literal[EntryType.URL]
    job_url: HttpUrl

    @field_validator("job_url", mode="before")
    @classmethod
    def normalize_job_url(cls, value: Any) -> Any:
        return _normalize_http_url_input(value)


JobSubmission = Annotated[
    LinkedInSubmission | IndeedSubmission | ExternalSubmission | AiContentSubmission | UrlSubmission,
    Field(discriminator="entry_type"),
]


class Job(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    source: str
    external_id: str | None = None
    url: str
    company_name: str
    title: str
    description: str
    location: str | None = None
    contract_type: str | None = None
    published_at: datetime | None = None
    identity: str = ""
    internal_id: int = 0
    provider_data: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def require_description(self) -> Job:
        if not self.description:
            raise ValueError("description is required")
        return self


class PromptDefinition(BaseModel):
    key: str = Field(min_length=1)
    version: float
    template: str = Field(min_length=1)
    output_structure: dict[str, Any]
    temperature: float = Field(ge=0, le=2)

    @model_validator(mode="after")
    def require_object_schema(self) -> PromptDefinition:
        if self.output_structure.get("type") != "object":
            raise ValueError("output_structure must define an object JSON Schema")
        properties = self.output_structure.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError("output_structure must define object properties")
        return self


class Qualification(BaseModel):
    score: int = Field(ge=0, le=100)
    # Legacy compatibility metadata only. Workflow decisions are based exclusively
    # on configured score thresholds; prompts are not required to return this field.
    should_apply: bool = False
    reasoning: str = Field(default="Qualification scored", min_length=1)

    def passes(self, threshold: int, *, force: bool = False) -> bool:
        # Manual/reprocessed submissions intentionally bypass the automatic
        # qualification cutoff while still retaining their actual score.
        return force or self.score >= threshold


class TailoredContent(BaseModel):
    cv: dict[str, Any]
    cover_letter: dict[str, Any]


class ArtifactBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_directory: Path
    cv_json: Path
    cv_tex: Path
    cv_pdf: Path
    cover_letter_json: Path
    cover_letter_tex: Path
    cover_letter_pdf: Path
    archive: Path

    def notification_paths(self) -> list[Path]:
        return [self.archive]


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    tenant: str
    kind: str
    state: RunState = RunState.QUEUED
    stage: str = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    original_run_id: UUID | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    error: dict[str, object] | None = None
    notification: dict[str, object] | None = None


class EnqueueResponse(BaseModel):
    run_id: UUID
    status: Literal["queued"] = "queued"


class RetryResponse(BaseModel):
    original_run_id: UUID
    run_id: UUID
    status: Literal["queued"] = "queued"


class RunRecord(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    tenant: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    source: str
    total_seen: int = 0
    total_new: int = 0
    total_passed: int = 0
    total_failed: int = 0
