from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from job_hunt.domain.models import Job, PromptDefinition
from job_hunt.errors import ErrorKind, WorkflowError
from job_hunt.job_trace import job_trace_context, trace_path
from job_hunt.logging import logger
from job_hunt.observability import langfuse_span, langfuse_trace_context, update_langfuse_output
from job_hunt.ports import ArtifactPublisher, CompatibilityFilter, DocumentRenderer, JobRepository, Qualifier, Tailor
from job_hunt.retry import retry_transient

ProgressCallback = Callable[[str, str], None]
PersistedCallback = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    row_id: int
    passed: bool
    artifacts_published: bool
    score: int | None = None
    notification_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationResult:
    row_id: int
    passed: bool
    score: int | None
    attach_trace: bool = True


class ApplicationWorkflow:
    def __init__(
        self,
        repository: JobRepository,
        qualifier: Qualifier,
        tailor: Tailor,
        renderer: DocumentRenderer,
        publisher: ArtifactPublisher,
        artifact_root: Path,
        *,
        compatibility_filter: CompatibilityFilter | None = None,
        tenant_key: str | None = None,
        langfuse_enabled: bool = False,
    ) -> None:
        self.repository = repository
        self.compatibility_filter = compatibility_filter
        self.qualifier = qualifier
        self.tailor = tailor
        self.renderer = renderer
        self.publisher = publisher
        self.artifact_root = artifact_root
        self.tenant_key = tenant_key
        self.langfuse_enabled = langfuse_enabled

    @staticmethod
    def _progress(callback: ProgressCallback | None, stage: str, event: str) -> None:
        if callback is not None:
            callback(stage, event)

    @staticmethod
    def _stored_score(row: Mapping[str, Any] | None) -> int | None:
        if row is None:
            return None
        value = row.get("Score")
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            score = int(value)
        elif isinstance(value, str) and value.strip():
            try:
                score = int(float(value.strip()))
            except ValueError:
                return None
        else:
            return None
        return score if 0 <= score <= 100 else None

    def _dropped(self, row_id: int) -> bool:
        return bool(retry_transient(self.repository.has_status, row_id, "dropped"))

    @staticmethod
    def _skipped(row_id: int, score: int) -> WorkflowResult:
        return WorkflowResult(row_id=row_id, passed=False, artifacts_published=False, score=score)

    def _cancelled(self, log: Any, row_id: int, stage: str, score: int | None = None) -> QualificationResult:
        log.info("job_cancelled_by_status", stage=stage, row_id=row_id)
        return QualificationResult(row_id=row_id, passed=False, score=score)

    @staticmethod
    def _job_id(job: Job) -> str:
        return job.external_id or str(job.internal_id)

    def _langfuse_session_id(self, job: Job) -> str | None:
        if not self.tenant_key:
            return None
        return f"{self.tenant_key}:{self._job_id(job)}"

    def _trace_path(self, job: Job, run_id: UUID) -> Path:
        return trace_path(self.artifact_root, str(run_id), self._job_id(job))

    def _trace_context(self, job: Job, run_id: UUID, path: Path, *, row_id: int | None = None) -> dict[str, object]:
        context: dict[str, object] = {
            "run_id": str(run_id),
            "job_id": self._job_id(job),
            "job_identity": job.identity,
        }
        if self.tenant_key:
            context["tenant"] = self.tenant_key
        if row_id is not None:
            context["row_id"] = row_id
        return context

    def _langfuse_metadata(self, job: Job, *, row_id: int | None = None) -> dict[str, object]:
        metadata: dict[str, object] = {
            "jobId": self._job_id(job),
            "jobIdentity": job.identity,
        }
        if row_id is not None:
            metadata["rowId"] = row_id
        return metadata

    def _langfuse_trace_context(self, run_id: UUID) -> dict[str, str] | None:
        tenant = self.tenant_key or "unknown"
        return langfuse_trace_context(
            enabled=self.langfuse_enabled,
            seed=f"job-hunt:{tenant}:{run_id}",
        )

    @staticmethod
    def _prompt_versions(prompts: Mapping[str, PromptDefinition]) -> dict[str, float]:
        return {key: definition.version for key, definition in prompts.items()}

    def _save_trace(self, row_id: int, path: Path) -> None:
        save_log = getattr(self.repository, "save_log", None)
        if not callable(save_log) or not path.exists():
            return
        logger().info("job_trace_snapshot_ready", row_id=row_id, path=path.name)
        try:
            retry_transient(save_log, row_id, path)
        except Exception as exc:
            # Observability must never turn an otherwise successful application into a failed job.
            logger().warning(
                "job_trace_attach_failed",
                row_id=row_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def persist_and_qualify(
        self,
        job: Job,
        *,
        run_id: UUID,
        master_cv: Mapping[str, Any],
        prompts: Mapping[str, PromptDefinition],
        threshold: int,
        force: bool,
        progress: ProgressCallback | None = None,
        persisted: PersistedCallback | None = None,
        resume_row_id: int | None = None,
        reuse_existing_score: bool = False,
        reuse_existing_data: bool = False,
    ) -> QualificationResult:
        reuse_existing_score = reuse_existing_score or bool(job.provider_data.get("reuse_existing_score"))
        reuse_existing_data = reuse_existing_data or bool(job.provider_data.get("reuse_existing_data"))
        path = self._trace_path(job, run_id)
        job_context = self._trace_context(job, run_id, path)
        trace_context = self._langfuse_trace_context(run_id)
        span_input = {
            "job": job.model_dump(mode="json"),
            "master_cv": dict(master_cv),
            "qualification_threshold": threshold,
            "force": force,
            "resume_row_id": resume_row_id,
            "reuse_existing_score": reuse_existing_score,
            "reuse_existing_data": reuse_existing_data,
            "prompt_versions": self._prompt_versions(prompts),
        }
        with (
            job_trace_context(path, **job_context),
            langfuse_span(
                enabled=self.langfuse_enabled,
                name="qualification-pipeline",
                trace_name="job-application",
                trace_context=trace_context,
                user_id=self.tenant_key,
                session_id=self._langfuse_session_id(job),
                input=span_input,
                metadata=self._langfuse_metadata(job),
                tags=("job-hunt", "qualification"),
            ) as observation,
        ):
            logger().info("job_processing_started", stage="qualification_pipeline")
            try:
                result = self._persist_and_qualify_impl(
                    job,
                    run_id=run_id,
                    master_cv=master_cv,
                    prompts=prompts,
                    threshold=threshold,
                    force=force,
                    progress=progress,
                    persisted=persisted,
                    resume_row_id=resume_row_id,
                    reuse_existing_score=reuse_existing_score,
                    reuse_existing_data=reuse_existing_data,
                )
            except Exception as exc:
                logger().error(
                    "job_processing_failed",
                    stage="qualification_pipeline",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                try:
                    existing = retry_transient(self.repository.find, job)
                    if existing is not None:
                        self._save_trace(int(existing["id"]), path)
                except Exception:
                    pass
                raise
            update_langfuse_output(
                observation,
                {
                    "row_id": result.row_id,
                    "passed": result.passed,
                    "score": result.score,
                    "attach_trace": result.attach_trace,
                },
            )
            if result.attach_trace:
                self._save_trace(result.row_id, path)
            return result

    def _persist_and_qualify_impl(
        self,
        job: Job,
        *,
        run_id: UUID,
        master_cv: Mapping[str, Any],
        prompts: Mapping[str, PromptDefinition],
        threshold: int,
        force: bool,
        progress: ProgressCallback | None = None,
        persisted: PersistedCallback | None = None,
        resume_row_id: int | None = None,
        reuse_existing_score: bool = False,
        reuse_existing_data: bool = False,
    ) -> QualificationResult:
        log = logger().bind(run_id=str(run_id), job_identity=job.identity)
        self._progress(progress, "persistence", "start")
        existing = retry_transient(self.repository.find, job)
        existing_row_id = int(existing["id"]) if existing is not None else None
        existing_score = self._stored_score(existing)
        resuming = resume_row_id is not None

        if existing is not None and not force and not resuming:
            row_id = int(existing["id"])
            if persisted is not None:
                persisted(row_id)
            self._progress(progress, "persistence", "finish")
            log.info("job_already_tracked", stage="persist", row_id=row_id, score=existing_score)
            return QualificationResult(row_id=row_id, passed=False, score=existing_score, attach_trace=False)

        if resuming:
            if existing_row_id != resume_row_id:
                raise WorkflowError(
                    f"Cannot resume run on Baserow row {resume_row_id}: matching row is {existing_row_id}",
                    ErrorKind.BUSINESS,
                )
            row = existing
        elif existing_row_id is not None and reuse_existing_data:
            row = existing
        else:
            row = (
                retry_transient(self.repository.reset, existing_row_id, job)
                if existing_row_id is not None
                else retry_transient(self.repository.create, job)
            )
        if row is None:
            raise WorkflowError("Baserow persistence returned no row", ErrorKind.MALFORMED_PROVIDER_RESPONSE)
        row_id = int(row["id"])
        structlog_context = {"row_id": row_id}
        # Lower layers such as the LiteLLM gateway inherit this row id through
        # structlog contextvars without needing job-specific method parameters.
        import structlog

        row_tokens = structlog.contextvars.bind_contextvars(**structlog_context)
        try:
            if existing is not None and not resuming and not reuse_existing_score:
                retry_transient(self.repository.clear_qualification, row_id)
            if force and not resuming:
                retry_transient(self.repository.set_status, row_id, "new")
            if persisted is not None:
                persisted(row_id)
            self._progress(progress, "persistence", "finish")
            log.info(
                "job_persisted",
                stage="persist",
                row_id=row_id,
                reprocessed=bool(existing),
                forced=force,
                resumed=resuming,
                reused_existing_data=bool(existing and reuse_existing_data),
            )

            if self._dropped(row_id):
                return self._cancelled(log, row_id, "pre_compatibility")

            if self.compatibility_filter is not None and not force:
                self._progress(progress, "compatibility_filter", "start")
                compatible = self.compatibility_filter.compatible(job, prompts)
                self._progress(progress, "compatibility_filter", "finish")
                if self._dropped(row_id):
                    return self._cancelled(log, row_id, "post_compatibility")
                log.info("job_compatibility_checked", stage="compatibility_filter", compatible=compatible)
                if not compatible:
                    retry_transient(self.repository.set_status, row_id, "dropped")
                    log.info("job_compatibility_filtered", stage="compatibility_filter", row_id=row_id)
                    return QualificationResult(row_id=row_id, passed=False, score=None)
            elif force:
                log.info("job_compatibility_bypassed", stage="compatibility_filter", row_id=row_id)

            if self._dropped(row_id):
                return self._cancelled(log, row_id, "pre_qualification")

            if reuse_existing_score and existing_score is not None:
                self._progress(progress, "qualification", "start")
                self._progress(progress, "qualification", "finish")
                passed = force or existing_score >= threshold
                if not passed and not force:
                    retry_transient(self.repository.set_status, row_id, "dropped")
                log.info(
                    "job_qualification_reused",
                    stage="qualification",
                    score=existing_score,
                    passed=passed,
                    forced=force,
                )
                return QualificationResult(row_id=row_id, passed=passed, score=existing_score)

            self._progress(progress, "qualification", "start")
            qualification = self.qualifier.qualify(job, master_cv, prompts)
            self._progress(progress, "qualification", "finish")
            if self._dropped(row_id):
                return self._cancelled(log, row_id, "post_qualification", qualification.score)

            passed = qualification.passes(threshold, force=force)
            retry_transient(self.repository.save_qualification, row_id, qualification)
            if not passed and not force:
                retry_transient(self.repository.set_status, row_id, "dropped")
            log.info(
                "job_qualified",
                stage="qualification",
                score=qualification.score,
                passed=passed,
                forced=force,
            )
            return QualificationResult(row_id=row_id, passed=passed, score=qualification.score)
        finally:
            structlog.contextvars.reset_contextvars(**row_tokens)

    def generate_documents(
        self,
        job: Job,
        *,
        run_id: UUID,
        row_id: int,
        score: int,
        master_cv: Mapping[str, Any],
        prompts: Mapping[str, PromptDefinition],
        applicant_filename: str,
        progress: ProgressCallback | None = None,
    ) -> WorkflowResult:
        path = self._trace_path(job, run_id)
        job_context = self._trace_context(job, run_id, path, row_id=row_id)
        trace_context = self._langfuse_trace_context(run_id)
        span_input = {
            "job": job.model_dump(mode="json"),
            "master_cv": dict(master_cv),
            "row_id": row_id,
            "qualification_score": score,
            "applicant_filename": applicant_filename,
            "prompt_versions": self._prompt_versions(prompts),
        }
        with (
            job_trace_context(path, **job_context),
            langfuse_span(
                enabled=self.langfuse_enabled,
                name="document-pipeline",
                trace_name="job-application",
                trace_context=trace_context,
                user_id=self.tenant_key,
                session_id=self._langfuse_session_id(job),
                input=span_input,
                metadata=self._langfuse_metadata(job, row_id=row_id),
                tags=("job-hunt", "documents"),
            ) as observation,
        ):
            logger().info("job_processing_started", stage="document_pipeline", row_id=row_id)
            try:
                result = self._generate_documents_impl(
                    job,
                    run_id=run_id,
                    row_id=row_id,
                    score=score,
                    master_cv=master_cv,
                    prompts=prompts,
                    applicant_filename=applicant_filename,
                    progress=progress,
                )
            except Exception as exc:
                logger().error(
                    "job_processing_failed",
                    stage="document_pipeline",
                    row_id=row_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                self._save_trace(row_id, path)
                raise
            update_langfuse_output(
                observation,
                {
                    "row_id": result.row_id,
                    "passed": result.passed,
                    "artifacts_published": result.artifacts_published,
                    "score": result.score,
                    "notification_paths": list(result.notification_paths),
                },
            )
            logger().info("job_processing_completed", stage="document_pipeline", row_id=row_id)
            self._save_trace(row_id, path)
            return result

    def _generate_documents_impl(
        self,
        job: Job,
        *,
        run_id: UUID,
        row_id: int,
        score: int,
        master_cv: Mapping[str, Any],
        prompts: Mapping[str, PromptDefinition],
        applicant_filename: str,
        progress: ProgressCallback | None = None,
    ) -> WorkflowResult:
        log = logger().bind(run_id=str(run_id), job_identity=job.identity)
        if self._dropped(row_id):
            log.info("job_documents_skipped", stage="documents", row_id=row_id, reason="status_dropped")
            return self._skipped(row_id, score)

        self._progress(progress, "tailoring", "start")
        tailored = self.tailor.tailor(job, master_cv, prompts)
        self._progress(progress, "tailoring", "finish")
        if self._dropped(row_id):
            log.info("job_documents_skipped", stage="post_tailoring", row_id=row_id, reason="status_dropped")
            return self._skipped(row_id, score)

        source_id = self._job_id(job)
        archive_basename = f"{applicant_filename}-{job.company_name}-{source_id}".replace("/", "-")
        self._progress(progress, "rendering", "start")
        artifacts = self.renderer.render(
            tailored,
            self.artifact_root / str(run_id),
            archive_basename,
            applicant_filename=applicant_filename,
        )
        self._progress(progress, "rendering", "finish")
        if self._dropped(row_id):
            log.info("job_documents_skipped", stage="post_rendering", row_id=row_id, reason="status_dropped")
            return self._skipped(row_id, score)

        self._progress(progress, "artifact_upload", "start")
        uploaded = retry_transient(self.publisher.publish, artifacts)
        retry_transient(self.repository.save_artifacts, row_id, uploaded)
        self._progress(progress, "artifact_upload", "finish")
        log.info("job_documents_ready", stage="publish", row_id=row_id)
        return WorkflowResult(
            row_id=row_id,
            passed=True,
            artifacts_published=True,
            score=score,
            notification_paths=tuple(str(path) for path in artifacts.notification_paths()),
        )
