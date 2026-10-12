from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANNER = ROOT / "scripts" / "deploy-plan.sh"


def plan(*paths: str) -> set[str]:
    result = subprocess.run(
        ["bash", str(PLANNER), *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    return {output} if output == "pull-only" else set(output.split(","))


def test_repository_only_changes_do_not_touch_runtime() -> None:
    for path in (
        "README.md",
        "docs/operations.md",
        "tests/unit/test_worker.py",
        ".github/workflows/ci.yml",
        ".env.example",
        "config/seeds/prompts.json",
        "scripts/deploy-vps.sh",
        "scripts/deploy-plan.sh",
    ):
        assert plan(path) == {"pull-only"}


def test_llm_routing_change_only_quiesces_llm_consumers_and_proxy() -> None:
    actions = plan("config/llm-routing.json")
    assert actions == {
        "reload-litellm",
        "premerge-drain-fast",
        "premerge-drain-documents",
        "litellm-drain-fast",
        "litellm-drain-documents",
        "validate-live",
    }
    assert "recreate-api" not in actions
    assert "recreate-beat" not in actions
    assert "recreate-notifications" not in actions


def test_tenant_template_change_preserves_active_tasks_without_image_build() -> None:
    actions = plan("tenants/nova/templates/cv_template.tex")
    assert actions == {"premerge-drain-fast", "premerge-drain-documents", "latex-check"}


def test_api_only_change_rebuilds_and_recreates_only_api_runtime() -> None:
    actions = plan("src/job_hunt/api/app.py")
    assert actions == {"build-api", "recreate-api"}


def test_cli_only_change_builds_image_without_restarting_services() -> None:
    actions = plan("src/job_hunt/cli.py")
    assert actions == {"build-api"}


def test_litellm_generator_change_builds_tool_image_and_refreshes_proxy_only() -> None:
    actions = plan("scripts/generate-litellm-config.py")
    assert actions == {
        "build-api",
        "reload-litellm",
        "litellm-drain-fast",
        "litellm-drain-documents",
        "validate-live",
    }


def test_worker_imported_application_change_keeps_both_images_compatible() -> None:
    actions = plan("src/job_hunt/application/discovery.py")
    assert {"build-api", "build-documents", "recreate-fast", "recreate-documents"} <= actions
    assert "recreate-beat" in actions
    assert "recreate-notifications" in actions


def test_rendering_change_rebuilds_only_document_service() -> None:
    actions = plan("src/job_hunt/rendering/base.py")
    assert actions == {"build-documents", "recreate-documents", "latex-check"}


def test_shared_runtime_change_keeps_both_pipeline_stages_on_same_version() -> None:
    actions = plan("src/job_hunt/worker.py")
    assert {"build-api", "build-documents", "recreate-fast", "recreate-documents"} <= actions


def test_dependency_change_reloads_proxy_and_both_images() -> None:
    actions = plan("pyproject.toml")
    assert {"build-api", "build-documents", "reload-litellm"} <= actions
    assert {"litellm-drain-fast", "litellm-drain-documents"} <= actions


def test_mixed_change_uses_union_of_required_actions() -> None:
    actions = plan("src/job_hunt/rendering/base.py", "src/job_hunt/api/app.py")
    assert "build-api" in actions
    assert "build-documents" in actions
    assert "recreate-api" in actions
    assert "recreate-documents" in actions
    assert "recreate-fast" not in actions


def test_unknown_path_defaults_to_safe_shared_runtime_plan() -> None:
    actions = plan("config/new-runtime-file.json")
    assert {"build-api", "build-documents", "recreate-fast", "recreate-documents"} <= actions
