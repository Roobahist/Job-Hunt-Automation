from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import httpx
import typer
from pydantic import TypeAdapter
from redis import Redis

from job_hunt.application.cleanup import purge_rows_by_status
from job_hunt.application.runs import RunCoordinator
from job_hunt.application.triage import triage_new_jobs
from job_hunt.config import Settings, load_registry, read_seed
from job_hunt.container import Container
from job_hunt.domain.models import JobSubmission, TailoredContent
from job_hunt.integrations.baserow import BaserowClient
from job_hunt.integrations.configuration import BaserowConfigurationRepository
from job_hunt.queueing import CeleryQueue
from job_hunt.run_store import RunStore

app = typer.Typer(no_args_is_help=True, help="Operate the multi-tenant job-hunt workflow.")
config_app = typer.Typer(help="Validate local and Baserow configuration.")
app.add_typer(config_app, name="config")


def _settings() -> Settings:
    return Settings()


def _store(settings: Settings) -> RunStore:
    return RunStore(Redis.from_url(settings.redis_url, decode_responses=True), settings.run_ttl_seconds)


def _coordinator(settings: Settings) -> RunCoordinator:
    container = Container(settings)
    return RunCoordinator(_store(settings), CeleryQueue(), container.registry.get)


def _validate_litellm_groups(settings: Settings) -> None:
    expected = {route.model for route in [*settings.llm_route_specs(), *settings.llm_repair_route_specs()]}
    response = httpx.get(
        f"{settings.litellm_base_url.rstrip('/')}/v1/models",
        headers={"Authorization": f"Bearer {settings.shared_litellm_key()}"},
        timeout=settings.litellm_request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    available = {str(item.get("id", "")) for item in data if isinstance(item, dict) and str(item.get("id", "")).strip()}
    missing = sorted(expected - available)
    if missing:
        raise ValueError(f"LiteLLM is missing configured capability groups: {missing}")


@app.command()
def submit(
    tenant: str,
    input_file: Annotated[Path, typer.Option("--input", exists=True, readable=True)],
    force: bool = False,
) -> None:
    settings = _settings()
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    submission: Any = TypeAdapter(JobSubmission).validate_python(payload)
    dumped = TypeAdapter(JobSubmission).dump_python(submission, mode="json")
    result = _coordinator(settings).enqueue_submission(
        tenant,
        dumped,
        "manual",
        force=force,
    )
    typer.echo(str(result.run_id))


@app.command()
def discover(tenant: str) -> None:
    result = _coordinator(_settings()).enqueue_discovery(tenant)
    typer.echo(str(result.run_id))


@app.command("purge-dropped")
def purge_dropped(
    tenant: str,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Permanently delete Dropped rows after rechecking each row immediately before deletion.",
        ),
    ] = False,
) -> None:
    settings = _settings()
    registry = load_registry(settings.registry_path)
    try:
        bootstrap = registry[tenant]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise typer.BadParameter(f"Unknown tenant {tenant!r}. Available tenants: {available}") from exc

    baserow = BaserowClient(
        bootstrap.secret("baserow"),
        bootstrap.baserow_base_url,
        timeout_seconds=settings.baserow_request_timeout_seconds,
    )
    config = BaserowConfigurationRepository(baserow, bootstrap.config_table_id).load()
    jobs_table_id = config.baserow_table_ids["jobs"]
    try:
        dropped_status_id = config.status_option_ids["dropped"]
    except KeyError as exc:
        raise typer.BadParameter(f"Tenant {tenant!r} has no configured dropped status option") from exc

    summary = purge_rows_by_status(
        rows=baserow.iter_rows(jobs_table_id),
        target_status_id=dropped_status_id,
        get_row=lambda row_id: baserow.get_row(jobs_table_id, row_id),
        delete_row=lambda row_id: baserow.delete_row(jobs_table_id, row_id),
        execute=execute,
    )
    typer.echo(
        "SUMMARY "
        f"tenant={tenant} scanned={summary.scanned} dropped={summary.matched} "
        f"rechecked={summary.rechecked} skipped_changed={summary.skipped_changed} "
        f"deleted={summary.deleted} executed={summary.executed}"
    )
    if not execute and summary.matched:
        typer.echo(f"DRY RUN: rerun with --execute to recheck and delete up to {summary.matched} Dropped rows.")


@app.command("triage-new")
def triage_new(
    tenant: str,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Evaluate rows without changing Baserow.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Process at most this many New rows.")] = None,
    after_id: Annotated[
        int | None,
        typer.Option(
            "--after-id",
            min=0,
            help="Only process New rows whose Baserow row ID is greater than this cursor.",
        ),
    ] = None,
) -> None:
    services = Container(_settings()).tenant(tenant, checkpoint_namespace=f"triage-new:{tenant}")
    compatibility_filter = services.workflow.compatibility_filter
    if compatibility_filter is None:
        raise typer.BadParameter("This tenant has no compatibility filter configured")

    rows = services.baserow.iter_rows(services.config.baserow_table_ids["jobs"])
    summary = triage_new_jobs(
        rows=rows,
        new_status_id=services.config.status_option_ids["new"],
        threshold=services.config.qualification_threshold,
        compatibility_filter=compatibility_filter,
        qualifier=services.workflow.qualifier,
        repository=services.repository,
        prompts=services.prompts,
        master_cv=services.context.master_cv,
        dry_run=dry_run,
        limit=limit,
        after_id=after_id,
    )

    for decision in summary.decisions:
        score = f" score={decision.score}" if decision.score is not None else ""
        typer.echo(
            f"row={decision.row_id} action={decision.action}{score} "
            f"company={decision.company_name!r} title={decision.title!r} reason={decision.reason}"
        )

    next_cursor = str(summary.next_after_id) if summary.next_after_id is not None else "none"
    typer.echo(
        "SUMMARY "
        f"scanned={summary.scanned} processed={summary.processed} kept={summary.kept} "
        f"dropped={summary.dropped} compatibility_dropped={summary.compatibility_dropped} "
        f"qualification_dropped={summary.qualification_dropped} scored={summary.scored} "
        f"reused_scores={summary.reused_scores} errors={summary.errors} dry_run={dry_run} "
        f"next_after_id={next_cursor}"
    )
    if summary.next_after_id is not None:
        typer.echo(f"NEXT: job-hunt triage-new {tenant} --limit {limit or 100} --after-id {summary.next_after_id}")
    if summary.errors:
        raise typer.Exit(2)


@app.command()
def status(run_id: UUID) -> None:
    found = _store(_settings()).get(run_id)
    if not found:
        raise typer.BadParameter("Run not found")
    typer.echo(found.model_dump_json(indent=2))


@app.command()
def retry(run_id: UUID) -> None:
    try:
        result = _coordinator(_settings()).retry(run_id)
    except KeyError as exc:
        raise typer.BadParameter("Run or replay data not found") from exc
    typer.echo(str(result.run_id))


@app.command()
def render(
    tenant: str,
    input_file: Annotated[Path, typer.Option("--input", exists=True)],
    output_directory: Annotated[Path, typer.Option("--output")],
    basename: str = "application",
) -> None:
    context = Container(_settings()).registry.get(tenant)
    content = TailoredContent.model_validate_json(input_file.read_text(encoding="utf-8"))
    result = context.renderer.render(
        content,
        output_directory,
        basename,
        applicant_filename=basename,
    )
    typer.echo(result.model_dump_json(indent=2))


@config_app.command("validate")
def validate_configuration(tenant: str | None = None, live: bool = False) -> None:
    settings = _settings()
    registry = load_registry(settings.registry_path)
    selected = [tenant] if tenant else list(registry)
    failures: list[str] = []
    for key in selected:
        try:
            bootstrap = registry[key]
            root = Path(bootstrap.tenant_root)
            if not (root / "master_cv.json").is_file():
                raise ValueError("master_cv.json is missing")
            if not (root / "templates/cv_template.tex").is_file():
                raise ValueError("CV template is missing")
            if not (root / "templates/cover_letter_template.tex").is_file():
                raise ValueError("Cover-letter template is missing")
            if live:
                Container(settings).tenant(key)
            typer.echo(f"OK {key}")
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    if live:
        try:
            _validate_litellm_groups(settings)
            typer.echo("OK configured LiteLLM capability groups")
        except Exception as exc:
            failures.append(f"shared LLM routes: {exc}")

    if failures:
        typer.echo("\n".join(failures), err=True)
        raise typer.Exit(1)


@config_app.command("seed")
def validate_seed(path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    typer.echo(read_seed(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
