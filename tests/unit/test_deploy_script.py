from pathlib import Path

SCRIPT = Path("scripts/deploy-vps.sh").read_text()


def test_deploy_readiness_defaults_allow_slow_vps_startup() -> None:
    assert "DEPLOY_HTTP_READY_TIMEOUT_SECONDS:-240" in SCRIPT
    assert "DEPLOY_LITELLM_READY_TIMEOUT_SECONDS:-240" in SCRIPT
    assert "DEPLOY_WORKER_DRAIN_TIMEOUT_SECONDS:-1800" in SCRIPT


def test_deploys_detach_by_default_to_survive_ssh_disconnects() -> None:
    assert "DEPLOY_VPS_DETACHED_CHILD" in SCRIPT
    assert "nohup env" in SCRIPT
    assert "runs/deployments" in SCRIPT
    assert "latest.log" in SCRIPT
    assert "latest.status" in SCRIPT
    assert "Deployment started in the background and will survive SSH disconnects." in SCRIPT


def test_foreground_override_is_available() -> None:
    foreground_case = SCRIPT.split("--foreground)", 1)[1].split(";;", 1)[0]
    assert "foreground=true" in foreground_case
    assert "shift" in foreground_case
    assert "Use --foreground" in SCRIPT


def test_detached_deploy_records_final_status() -> None:
    assert "DEPLOY_STATUS_FILE" in SCRIPT
    assert "success\\n" in SCRIPT
    assert "failed exit_code=%s\\n" in SCRIPT
    assert "Check final result" in SCRIPT
    assert "Latest shortcuts" in SCRIPT


def test_deployment_is_service_aware_in_auto_mode() -> None:
    assert 'source "$repo_root/scripts/deploy-plan.sh"' in SCRIPT
    assert 'if [[ "$requested_level" == "auto" ]]' in SCRIPT
    assert 'deploy_plan_add_path "$path"' in SCRIPT
    assert 'plan_actions="$(deploy_plan_actions)"' in SCRIPT


def test_worker_shutdown_is_graceful_and_redis_backed() -> None:
    drain = SCRIPT.split("drain_service() {", 1)[1].split("reload_litellm() {", 1)[0]
    assert 'docker compose stop -t "$worker_drain_timeout_seconds" "$service"' in drain
    assert "Queued work remains in Redis" in drain


def test_bind_mounted_consumers_are_drained_before_fast_forward() -> None:
    merge = SCRIPT.index('git merge --ff-only "$upstream"')
    premerge_fast = SCRIPT.index('"$PLAN_PREMERGE_DRAIN_FAST" && drain_service worker-fast')
    premerge_documents = SCRIPT.index('"$PLAN_PREMERGE_DRAIN_DOCUMENTS" && drain_service worker-documents')
    assert premerge_fast < merge
    assert premerge_documents < merge


def test_builds_finish_before_image_consumers_are_drained() -> None:
    build = SCRIPT.index('docker compose build "${build_targets[@]}"')
    drain_fast = SCRIPT.index('"$PLAN_RECREATE_FAST" && drain_service worker-fast')
    drain_documents = SCRIPT.index('"$PLAN_RECREATE_DOCUMENTS" && drain_service worker-documents')
    assert build < drain_fast
    assert build < drain_documents


def test_preflight_happens_before_image_consumer_cutover() -> None:
    preflight = SCRIPT.index("preflight_runtime\nfi")
    drain_fast = SCRIPT.index('"$PLAN_RECREATE_FAST" && drain_service worker-fast')
    assert preflight < drain_fast
    assert "from job_hunt.api.app import create_app" in SCRIPT
    assert "from job_hunt.worker import celery_app" in SCRIPT
    assert "Container" in SCRIPT


def test_litellm_consumers_are_drained_before_proxy_restart() -> None:
    drain_fast = SCRIPT.index('"$PLAN_LITELLM_DRAIN_FAST" && drain_service worker-fast')
    drain_documents = SCRIPT.index('"$PLAN_LITELLM_DRAIN_DOCUMENTS" && drain_service worker-documents')
    reload_call = SCRIPT.index('if "$PLAN_RELOAD_LITELLM"; then\n    reload_litellm')
    assert drain_fast < reload_call
    assert drain_documents < reload_call


def test_beat_is_not_globally_stopped_during_worker_drain() -> None:
    assert "Stopping beat before worker drain" not in SCRIPT
    beat_section = SCRIPT.split('if "$PLAN_RECREATE_BEAT"; then', 1)[1]
    assert "docker compose stop beat" in beat_section
    assert "docker compose up -d --force-recreate beat" in beat_section


def test_missed_discovery_detection_runs_before_fast_worker_resumes() -> None:
    recovery = SCRIPT.index("run_recovery_detection\nfi")
    resume_fast = SCRIPT.index("if service_was_stopped worker-fast; then", recovery + 1)
    assert recovery < resume_fast
    assert "python -m job_hunt.recovery" in SCRIPT


def test_successful_applied_commit_is_tracked_separately_from_checkout_head() -> None:
    assert 'applied_head_file="$repo_root/runs/deployments/applied-head"' in SCRIPT
    assert 'git diff --name-only "$baseline_head" "$new_head"' in SCRIPT
    assert "record_applied_head" in SCRIPT
    assert "Last successfully applied commit" in SCRIPT


def test_failed_deployment_rolls_checkout_and_image_tags_back() -> None:
    rollback = SCRIPT.split("restore_transaction_state() {", 1)[1].split("recover_cutover() {", 1)[0]
    assert 'git reset --hard "$baseline_head"' in rollback
    assert "job-hunt-automation-api:latest" in rollback
    assert "job-hunt-automation-documents:latest" in rollback
    assert "litellm.runtime.yaml" in rollback


def test_failed_runtime_transition_recreates_previous_runtime() -> None:
    recovery = SCRIPT.split("recover_cutover() {", 1)[1].split("finish_deploy() {", 1)[0]
    assert "restore_transaction_state" in recovery
    assert "--force-recreate" in recovery
    for service in ("litellm", "api", "worker-fast", "worker-documents", "worker-notifications", "flower", "beat"):
        assert service in recovery


def test_failed_pre_cutover_build_remains_pending_and_preserves_runtime() -> None:
    finish = SCRIPT.split("finish_deploy() {", 1)[1].split("trap finish_deploy", 1)[0]
    assert "restore_transaction_state" in finish
    assert "recover_cutover" in finish
    assert "previous runtime remains active" in finish


def test_tracked_vps_drift_is_refused_before_deployment() -> None:
    assert "git diff --quiet" in SCRIPT
    assert "git diff --cached --quiet" in SCRIPT
    assert "Tracked VPS files have local modifications" in SCRIPT


def test_overlapping_deployments_are_rejected() -> None:
    assert 'exec 9>"$repo_root/runs/deployments/deploy.lock"' in SCRIPT
    assert "flock -n 9" in SCRIPT
    assert "Another deployment is already running" in SCRIPT


def test_deploy_verifies_rebuilt_running_images() -> None:
    assert "verify_service_image api job-hunt-automation-api:latest" in SCRIPT
    assert "verify_service_image worker-fast job-hunt-automation-api:latest" in SCRIPT
    assert "verify_service_image worker-documents job-hunt-automation-documents:latest" in SCRIPT


def test_dry_run_argument_advances_parser() -> None:
    dry_run_case = SCRIPT.split("--dry-run)", 1)[1].split(";;", 1)[0]
    assert "dry_run=true" in dry_run_case
    assert "shift" in dry_run_case


def test_legacy_level_overrides_remain_for_operator_recovery() -> None:
    assert "apply_legacy_level_override" in SCRIPT
    assert "auto|0|1|2|3" in SCRIPT
