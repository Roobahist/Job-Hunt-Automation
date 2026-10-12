#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
source "$repo_root/scripts/deploy-plan.sh"

original_args=("$@")
requested_level="auto"
dry_run=false
foreground=false
detached_child="${DEPLOY_VPS_DETACHED_CHILD:-false}"
deploy_status_file="${DEPLOY_STATUS_FILE:-}"
http_ready_timeout_seconds="${DEPLOY_HTTP_READY_TIMEOUT_SECONDS:-240}"
litellm_ready_timeout_seconds="${DEPLOY_LITELLM_READY_TIMEOUT_SECONDS:-240}"
worker_drain_timeout_seconds="${DEPLOY_WORKER_DRAIN_TIMEOUT_SECONDS:-1800}"
applied_head_file="$repo_root/runs/deployments/applied-head"
cutover_started=false
cutover_completed=false
checkout_advanced=false
stopped_services=()
rollback_api_tag="job-hunt-automation-api:deploy-rollback"
rollback_documents_tag="job-hunt-automation-documents:deploy-rollback"
rollback_litellm_config=""
baseline_head=""
new_head=""

usage() {
    cat <<'USAGE'
Usage: bash scripts/deploy-vps.sh [--level auto|0|1|2|3] [--dry-run] [--foreground]

Auto mode is service-aware. It derives the minimum safe action set from changed paths instead of
forcing every change through one coarse cutover. Builds happen while the old runtime is still live.
Only workers that consume changed code/config are gracefully drained, and Beat remains live unless
its own image must be replaced. Redis holds queued work throughout worker pauses.

The script records the last successfully applied commit separately from Git HEAD. A failed build or
cutover therefore remains pending on the next deployment even if the checkout had already advanced.
Tracked local modifications are refused so failure rollback can safely restore the previous commit.

Legacy explicit levels remain available for operator recovery:
  0  Pull only.
  1  Refresh all runtime containers without rebuilding images; reload LiteLLM.
  2  Rebuild/recreate lightweight application services; keep the TeX image.
  3  Rebuild/recreate both application and TeX document services.

Deployments run detached by default so they survive SSH disconnects. Output is written under
runs/deployments/. Use --foreground only when intentionally tying deployment to the terminal.
USAGE
}

while (($#)); do
    case "$1" in
        --level)
            requested_level="${2:-}"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --foreground)
            foreground=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$requested_level" in
    auto|0|1|2|3) ;;
    *)
        printf 'Invalid deployment level: %s\n' "$requested_level" >&2
        exit 2
        ;;
esac

for timeout_name in http_ready_timeout_seconds litellm_ready_timeout_seconds worker_drain_timeout_seconds; do
    timeout_value="${!timeout_name}"
    if [[ ! "$timeout_value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got %q\n' "$timeout_name" "$timeout_value" >&2
        exit 2
    fi
done

if ! "$dry_run" && ! "$foreground" && [[ "$detached_child" != "true" ]]; then
    deploy_dir="$repo_root/runs/deployments"
    mkdir -p "$deploy_dir"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    log_file="$deploy_dir/deploy-$timestamp.log"
    status_file="$deploy_dir/deploy-$timestamp.status"
    pid_file="$deploy_dir/deploy-$timestamp.pid"

    ln -sfn "$(basename "$log_file")" "$deploy_dir/latest.log"
    ln -sfn "$(basename "$status_file")" "$deploy_dir/latest.status"
    ln -sfn "$(basename "$pid_file")" "$deploy_dir/latest.pid"

    nohup env \
        DEPLOY_VPS_DETACHED_CHILD=true \
        DEPLOY_STATUS_FILE="$status_file" \
        bash "$repo_root/scripts/deploy-vps.sh" "${original_args[@]}" \
        >"$log_file" 2>&1 </dev/null &
    deploy_pid=$!
    printf '%s\n' "$deploy_pid" >"$pid_file"

    printf 'Deployment started in the background and will survive SSH disconnects.\n'
    printf 'PID: %s\n' "$deploy_pid"
    printf 'Log: %s\n' "$log_file"
    printf '\nFollow live output:\n  tail -f %q\n' "$log_file"
    printf '\nCheck final result:\n  cat %q\n' "$status_file"
    printf '\nLatest shortcuts:\n  tail -f runs/deployments/latest.log\n  cat runs/deployments/latest.status\n'
    exit 0
fi

mkdir -p "$repo_root/runs/deployments"
exec 9>"$repo_root/runs/deployments/deploy.lock"
if ! flock -n 9; then
    printf 'Another deployment is already running. Refusing overlapping deployments.\n' >&2
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    printf 'Tracked VPS files have local modifications. Commit the intended change upstream or restore it before deploying.\n' >&2
    git status --short >&2
    exit 1
fi

service_was_stopped() {
    local target="$1"
    local item
    for item in "${stopped_services[@]:-}"; do
        [[ "$item" == "$target" ]] && return 0
    done
    return 1
}

snapshot_image() {
    local source_tag="$1"
    local rollback_tag="$2"
    if docker image inspect "$source_tag" >/dev/null 2>&1; then
        docker tag "$source_tag" "$rollback_tag"
    fi
}

restore_image() {
    local rollback_tag="$1"
    local destination_tag="$2"
    if docker image inspect "$rollback_tag" >/dev/null 2>&1; then
        docker tag "$rollback_tag" "$destination_tag"
    fi
}

snapshot_runtime_state() {
    if "$PLAN_BUILD_API"; then
        snapshot_image job-hunt-automation-api:latest "$rollback_api_tag"
    fi
    if "$PLAN_BUILD_DOCUMENTS"; then
        snapshot_image job-hunt-automation-documents:latest "$rollback_documents_tag"
    fi
    if "$PLAN_RELOAD_LITELLM" && [[ -f "$repo_root/config/litellm.runtime.yaml" ]]; then
        rollback_litellm_config="$(mktemp "$repo_root/runs/deployments/litellm-runtime.rollback.XXXXXX")"
        cp "$repo_root/config/litellm.runtime.yaml" "$rollback_litellm_config"
    fi
}

restore_transaction_state() {
    if [[ -n "$baseline_head" ]] && git cat-file -e "$baseline_head^{commit}" 2>/dev/null; then
        git reset --hard "$baseline_head" >&2 || true
        checkout_advanced=false
    fi

    restore_image "$rollback_api_tag" job-hunt-automation-api:latest
    restore_image "$rollback_documents_tag" job-hunt-automation-documents:latest

    if [[ -n "$rollback_litellm_config" && -f "$rollback_litellm_config" ]]; then
        cp "$rollback_litellm_config" "$repo_root/config/litellm.runtime.yaml" || true
    fi
}

recover_cutover() {
    if ! "$cutover_started" || "$cutover_completed"; then
        return
    fi
    printf 'Deployment failed during runtime transition; restoring previous checkout, images, proxy config, and services.\n' >&2
    restore_transaction_state
    docker compose up -d --force-recreate redis litellm api worker-fast worker-documents worker-notifications flower beat >&2 || true
}

finish_deploy() {
    local status=$?
    if ((status != 0)); then
        if "$cutover_started"; then
            recover_cutover
        else
            restore_transaction_state
            printf 'Deployment failed before any runtime service was stopped; previous runtime remains active.\n' >&2
        fi
    fi
    if [[ -n "$rollback_litellm_config" ]]; then
        rm -f "$rollback_litellm_config" || true
    fi
    docker image rm "$rollback_api_tag" "$rollback_documents_tag" >/dev/null 2>&1 || true
    if [[ -n "$deploy_status_file" ]]; then
        if ((status == 0)); then
            printf 'success\n' >"$deploy_status_file"
        else
            printf 'failed exit_code=%s\n' "$status" >"$deploy_status_file"
        fi
    fi
    exit "$status"
}
trap finish_deploy EXIT

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout_seconds="${3:-$http_ready_timeout_seconds}"
    local response
    for ((attempt = 1; attempt <= timeout_seconds; attempt++)); do
        if response=$(curl --fail --silent --show-error "$url" 2>/dev/null); then
            printf '%s: %s\n' "$name" "$response"
            return 0
        fi
        sleep 1
    done
    printf '%s did not become ready after %s seconds\n' "$name" "$timeout_seconds" >&2
    docker compose logs --tail=100 api litellm >&2 || true
    return 1
}

wait_for_litellm() {
    local timeout_seconds="${1:-$litellm_ready_timeout_seconds}"
    for ((attempt = 1; attempt <= timeout_seconds; attempt++)); do
        if docker compose exec -T litellm python -c \
            "import urllib.request; urllib.request.urlopen('http://localhost:4000/health/liveliness', timeout=2)" \
            >/dev/null 2>&1; then
            printf 'LiteLLM live\n'
            return 0
        fi
        sleep 1
    done
    printf 'LiteLLM did not become ready after %s seconds\n' "$timeout_seconds" >&2
    docker compose logs --tail=150 litellm >&2 || true
    return 1
}

drain_service() {
    local service="$1"
    local container_id
    service_was_stopped "$service" && return 0
    container_id="$(docker compose ps -q "$service")"
    if [[ -z "$container_id" ]]; then
        return 0
    fi
    cutover_started=true
    printf 'Draining %s (graceful timeout %ss)...\n' "$service" "$worker_drain_timeout_seconds"
    docker compose stop -t "$worker_drain_timeout_seconds" "$service"
    stopped_services+=("$service")
    printf 'Drained %s. Queued work remains in Redis.\n' "$service"
}

reload_litellm() {
    local runtime_tmp
    runtime_tmp="$(mktemp ./config/litellm.runtime.yaml.XXXXXX)"
    trap 'rm -f "$runtime_tmp"' RETURN
    docker compose run --rm -T --no-deps \
        -e LITELLM_CONFIG_STDOUT=true \
        api python scripts/generate-litellm-config.py >"$runtime_tmp"
    mv "$runtime_tmp" ./config/litellm.runtime.yaml
    trap - RETURN

    cutover_started=true
    docker compose up -d redis
    docker compose up -d --force-recreate litellm
    wait_for_litellm "$litellm_ready_timeout_seconds"
}

run_recovery_detection() {
    printf 'Checking for discovery windows missed while processing was paused...\n'
    docker compose run --rm -T --no-deps api python -m job_hunt.recovery
}

preflight_runtime() {
    printf 'Running pre-cutover validation against newly built images...\n'
    if "$PLAN_BUILD_API"; then
        docker compose run --rm -T --no-deps api python -c \
            "from job_hunt.api.app import create_app; from job_hunt.worker import celery_app; from job_hunt.domain.models import EnqueueResponse, RetryResponse, RunState, RunStatus; from job_hunt.integrations.baserow import BaserowClient; assert callable(create_app); assert celery_app is not None; assert RunState.QUEUED.value == 'queued'; assert all(x is not None for x in (RunStatus, EnqueueResponse, RetryResponse)); assert callable(getattr(BaserowClient, 'list_fields', None))"
        docker compose run --rm -T --no-deps api python -c \
            "from job_hunt.config import Settings, load_registry; from job_hunt.container import Container; s=Settings(); registry=load_registry(s.registry_path); c=Container(s); [c.tenant(key, checkpoint_namespace=f'preflight:{key}') for key in registry]"
    fi
    if "$PLAN_LATEX_CHECK"; then
        docker compose run --rm -T --no-deps worker-documents python scripts/check-atlas-latex.py
    fi
    printf 'Pre-cutover validation passed.\n'
}

verify_service_image() {
    local service="$1"
    local image_name="$2"
    local container_id expected_image_id running_image_id
    container_id="$(docker compose ps -q "$service")"
    [[ -n "$container_id" ]] || return 1
    expected_image_id="$(docker image inspect "$image_name" --format '{{.Id}}')"
    running_image_id="$(docker inspect "$container_id" --format '{{.Image}}')"
    if [[ "$expected_image_id" != "$running_image_id" ]]; then
        printf 'Image verification failed for %s: running=%s expected=%s\n' \
            "$service" "$running_image_id" "$expected_image_id" >&2
        return 1
    fi
    printf 'Image verified: %s -> %s\n' "$service" "$running_image_id"
}

record_applied_head() {
    local tmp
    tmp="$(mktemp "$repo_root/runs/deployments/applied-head.XXXXXX")"
    printf '%s\n' "$new_head" >"$tmp"
    mv "$tmp" "$applied_head_file"
}

apply_legacy_level_override() {
    local level="$1"
    deploy_plan_reset
    case "$level" in
        0)
            ;;
        1)
            PLAN_RECREATE_API=true
            PLAN_RECREATE_FAST=true
            PLAN_RECREATE_DOCUMENTS=true
            PLAN_RECREATE_NOTIFICATIONS=true
            PLAN_RECREATE_BEAT=true
            PLAN_RECREATE_FLOWER=true
            PLAN_RELOAD_LITELLM=true
            PLAN_LITELLM_DRAIN_FAST=true
            PLAN_LITELLM_DRAIN_DOCUMENTS=true
            PLAN_VALIDATE_LIVE=true
            ;;
        2)
            PLAN_BUILD_API=true
            PLAN_PREFLIGHT=true
            PLAN_RECREATE_API=true
            PLAN_RECREATE_FAST=true
            PLAN_RECREATE_NOTIFICATIONS=true
            PLAN_RECREATE_BEAT=true
            PLAN_RECREATE_FLOWER=true
            PLAN_RELOAD_LITELLM=true
            PLAN_LITELLM_DRAIN_FAST=true
            PLAN_LITELLM_DRAIN_DOCUMENTS=true
            PLAN_VALIDATE_LIVE=true
            ;;
        3)
            PLAN_BUILD_API=true
            PLAN_BUILD_DOCUMENTS=true
            PLAN_PREFLIGHT=true
            PLAN_LATEX_CHECK=true
            PLAN_RECREATE_API=true
            PLAN_RECREATE_FAST=true
            PLAN_RECREATE_DOCUMENTS=true
            PLAN_RECREATE_NOTIFICATIONS=true
            PLAN_RECREATE_BEAT=true
            PLAN_RECREATE_FLOWER=true
            PLAN_RELOAD_LITELLM=true
            PLAN_LITELLM_DRAIN_FAST=true
            PLAN_LITELLM_DRAIN_DOCUMENTS=true
            PLAN_VALIDATE_LIVE=true
            ;;
    esac
}

checkout_head="$(git rev-parse HEAD)"
baseline_head="$checkout_head"
if [[ -s "$applied_head_file" ]]; then
    candidate="$(tr -d '[:space:]' <"$applied_head_file")"
    if git cat-file -e "$candidate^{commit}" 2>/dev/null; then
        baseline_head="$candidate"
    else
        printf 'Ignoring invalid applied-head state: %s\n' "$candidate" >&2
    fi
fi

git fetch --prune origin
upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || printf 'origin/main')"
new_head="$(git rev-parse "$upstream")"

if ! git merge-base --is-ancestor "$baseline_head" "$new_head"; then
    printf 'Applied commit %s is not an ancestor of upstream %s. Refusing ambiguous auto deployment.\n' \
        "$baseline_head" "$new_head" >&2
    exit 1
fi

mapfile -t changed_files < <(git diff --name-only "$baseline_head" "$new_head")

deploy_plan_reset
if [[ "$requested_level" == "auto" ]]; then
    for path in "${changed_files[@]}"; do
        deploy_plan_add_path "$path"
    done
else
    apply_legacy_level_override "$requested_level"
fi
plan_actions="$(deploy_plan_actions)"

printf 'Checkout commit: %s\n' "$checkout_head"
printf 'Last successfully applied commit: %s\n' "$baseline_head"
printf 'Upstream commit: %s\n' "$new_head"
printf 'Deployment plan: %s\n' "$plan_actions"
printf 'Worker graceful-drain timeout: %ss\n' "$worker_drain_timeout_seconds"
if ((${#changed_files[@]})); then
    printf 'Changed files since last successful deployment:\n'
    printf '  %s\n' "${changed_files[@]}"
else
    printf 'Changed files since last successful deployment: none\n'
fi

if "$dry_run"; then
    printf 'Dry run only; repository and containers were not changed.\n'
    exit 0
fi

snapshot_runtime_state

# Bind-mounted routing and tenant assets can change underneath a running task. Their consumers are
# quiesced before the checkout changes. Source code is baked into images, so code-only preparation
# remains fully online until its narrow service cutover.
"$PLAN_PREMERGE_DRAIN_FAST" && drain_service worker-fast
"$PLAN_PREMERGE_DRAIN_DOCUMENTS" && drain_service worker-documents
"$PLAN_PREMERGE_DRAIN_NOTIFICATIONS" && drain_service worker-notifications

git merge --ff-only "$upstream"
checkout_advanced=true

if [[ "$plan_actions" == "pull-only" ]]; then
    record_applied_head
    cutover_completed=true
    printf 'Deployment complete: repository updated; runtime unchanged.\n'
    exit 0
fi

build_targets=()
"$PLAN_BUILD_API" && build_targets+=(api)
"$PLAN_BUILD_DOCUMENTS" && build_targets+=(worker-documents)
if ((${#build_targets[@]})); then
    docker compose build "${build_targets[@]}"
fi

if "$PLAN_PREFLIGHT" || "$PLAN_LATEX_CHECK"; then
    preflight_runtime
fi

# A proxy restart must never interrupt an active model call. Drain only LLM-consuming queues and do
# it after build/preflight so expensive preparation does not pause processing.
"$PLAN_LITELLM_DRAIN_FAST" && drain_service worker-fast
"$PLAN_LITELLM_DRAIN_DOCUMENTS" && drain_service worker-documents

# Drain only workers whose code image is being replaced. Beat can continue enqueueing scheduled
# dispatch tasks into Redis while consumers are paused.
"$PLAN_RECREATE_FAST" && drain_service worker-fast
"$PLAN_RECREATE_DOCUMENTS" && drain_service worker-documents
"$PLAN_RECREATE_NOTIFICATIONS" && drain_service worker-notifications

if "$PLAN_RECREATE_API"; then
    cutover_started=true
    docker compose up -d --force-recreate --remove-orphans api
fi
if "$PLAN_RECREATE_FLOWER"; then
    cutover_started=true
    docker compose up -d --force-recreate flower
fi

if "$PLAN_RELOAD_LITELLM"; then
    reload_litellm
fi

# When fast processing was paused, run the same missed-slot detector used at Beat startup before any
# queued dispatch can be consumed. The next normal discovery then expands f_TPR for missed blocks.
if service_was_stopped worker-fast; then
    run_recovery_detection
fi

if service_was_stopped worker-documents; then
    if "$PLAN_RECREATE_DOCUMENTS"; then
        docker compose up -d --force-recreate worker-documents
    else
        docker compose up -d worker-documents
    fi
fi
if service_was_stopped worker-fast; then
    if "$PLAN_RECREATE_FAST"; then
        docker compose up -d --force-recreate worker-fast
    else
        docker compose up -d worker-fast
    fi
fi
if service_was_stopped worker-notifications; then
    if "$PLAN_RECREATE_NOTIFICATIONS"; then
        docker compose up -d --force-recreate worker-notifications
    else
        docker compose up -d worker-notifications
    fi
fi

# Beat is replaced last and only when its own image/runtime changed. Its startup recovery hook is an
# additional safety net if this short recreation crosses a discovery boundary.
if "$PLAN_RECREATE_BEAT"; then
    cutover_started=true
    docker compose stop beat || true
    docker compose up -d --force-recreate beat
fi

if "$PLAN_RECREATE_API"; then
    wait_for_http "API live" "http://127.0.0.1:8000/health/live" "$http_ready_timeout_seconds"
    wait_for_http "API ready" "http://127.0.0.1:8000/health/ready" "$http_ready_timeout_seconds"
fi

docker compose exec -T redis redis-cli ping

if "$PLAN_VALIDATE_LIVE"; then
    docker compose run --rm --no-deps api job-hunt config validate --live
fi

if "$PLAN_BUILD_API"; then
    "$PLAN_RECREATE_API" && verify_service_image api job-hunt-automation-api:latest
    "$PLAN_RECREATE_FAST" && verify_service_image worker-fast job-hunt-automation-api:latest
    "$PLAN_RECREATE_NOTIFICATIONS" && verify_service_image worker-notifications job-hunt-automation-api:latest
    "$PLAN_RECREATE_BEAT" && verify_service_image beat job-hunt-automation-api:latest
    "$PLAN_RECREATE_FLOWER" && verify_service_image flower job-hunt-automation-api:latest
fi
if "$PLAN_BUILD_DOCUMENTS" && "$PLAN_RECREATE_DOCUMENTS"; then
    verify_service_image worker-documents job-hunt-automation-documents:latest
fi

record_applied_head
cutover_completed=true
checkout_advanced=false
printf 'Deployment complete. Applied actions: %s\n' "$plan_actions"
