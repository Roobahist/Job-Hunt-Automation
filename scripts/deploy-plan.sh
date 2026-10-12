#!/usr/bin/env bash

# Deployment planning is capability-based rather than level-based. Call deploy_plan_reset once,
# then deploy_plan_add_path for every changed repository path. The exported PLAN_* booleans tell
# deploy-vps.sh exactly which images and services need action.

deploy_plan_reset() {
    PLAN_BUILD_API=false
    PLAN_BUILD_DOCUMENTS=false
    PLAN_RECREATE_API=false
    PLAN_RECREATE_FAST=false
    PLAN_RECREATE_DOCUMENTS=false
    PLAN_RECREATE_NOTIFICATIONS=false
    PLAN_RECREATE_BEAT=false
    PLAN_RECREATE_FLOWER=false
    PLAN_RELOAD_LITELLM=false
    PLAN_PREFLIGHT=false
    PLAN_LATEX_CHECK=false
    PLAN_VALIDATE_LIVE=false
    PLAN_PREMERGE_DRAIN_FAST=false
    PLAN_PREMERGE_DRAIN_DOCUMENTS=false
    PLAN_PREMERGE_DRAIN_NOTIFICATIONS=false
    PLAN_LITELLM_DRAIN_FAST=false
    PLAN_LITELLM_DRAIN_DOCUMENTS=false
}

_deploy_plan_build_api() {
    PLAN_BUILD_API=true
    PLAN_PREFLIGHT=true
}

_deploy_plan_api_runtime() {
    _deploy_plan_build_api
    PLAN_RECREATE_API=true
}

_deploy_plan_shared_runtime() {
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
    PLAN_VALIDATE_LIVE=true
}

_deploy_plan_document_runtime() {
    PLAN_BUILD_DOCUMENTS=true
    PLAN_PREFLIGHT=true
    PLAN_LATEX_CHECK=true
    PLAN_RECREATE_DOCUMENTS=true
}

_deploy_plan_litellm_reload() {
    PLAN_RELOAD_LITELLM=true
    PLAN_VALIDATE_LIVE=true
    # Fast and document queues can both make model calls. Drain immediately before proxy restart,
    # after builds/preflight, so image preparation does not create avoidable processing downtime.
    PLAN_LITELLM_DRAIN_FAST=true
    PLAN_LITELLM_DRAIN_DOCUMENTS=true
}

deploy_plan_add_path() {
    local path="$1"

    # Repository-only changes. Deployment mechanics themselves never require rebuilding runtime
    # images; the newly pulled script is used on the next invocation.
    case "$path" in
        README.md|docs/*|tests/*|.github/*|.env.example|config/seeds/*|scripts/deploy-vps.sh|scripts/deploy-plan.sh|scripts/deploy-path-level.sh)
            return
            ;;
    esac

    # LLM routing is bind-mounted and LiteLLM consumes a generated derivative. Quiesce both LLM
    # queues before Git changes the source, then regenerate/restart only the proxy and resume them.
    case "$path" in
        config/llm-routing.json|config/llm-model-limits.json)
            _deploy_plan_litellm_reload
            PLAN_PREMERGE_DRAIN_FAST=true
            PLAN_PREMERGE_DRAIN_DOCUMENTS=true
            return
            ;;
    esac

    # Tenant master data/templates are bind-mounted. Template text is read at render time, while the
    # master CV is loaded when task services are constructed, so active fast/document tasks finish
    # before these files are replaced. Beat can continue queuing into Redis during the pause.
    case "$path" in
        tenants/*)
            PLAN_PREMERGE_DRAIN_FAST=true
            PLAN_PREMERGE_DRAIN_DOCUMENTS=true
            case "$path" in
                tenants/*/master_cv.json|tenants/*/templates/*)
                    PLAN_LATEX_CHECK=true
                    ;;
            esac
            return
            ;;
    esac

    # Registry values are reloaded per operation. Nginx files are repository-managed references,
    # not Compose bind mounts. Neither requires a container restart by itself.
    case "$path" in
        config/users.toml|deploy/nginx/*)
            return
            ;;
    esac

    # API-only modules do not justify touching workers that happen to share the same image tag.
    case "$path" in
        src/job_hunt/api/*|src/job_hunt/queueing.py|src/job_hunt/application/runs.py)
            _deploy_plan_api_runtime
            return
            ;;
    esac

    # CLI/operator-only code needs a fresh lightweight image for future `docker compose run api ...`
    # invocations, but no already-running service imports it as part of its active workload.
    case "$path" in
        src/job_hunt/cli.py|src/job_hunt/application/triage.py)
            _deploy_plan_build_api
            return
            ;;
    esac

    # LiteLLM config generation runs from the lightweight image but is not part of the API/worker
    # process itself. Build it, then refresh only the proxy while preserving active LLM calls.
    case "$path" in
        src/job_hunt/integrations/litellm_config.py|scripts/generate-litellm-config.py)
            _deploy_plan_build_api
            _deploy_plan_litellm_reload
            return
            ;;
    esac

    # Renderer code is isolated to the TeX worker.
    case "$path" in
        src/job_hunt/rendering/*)
            _deploy_plan_document_runtime
            return
            ;;
    esac

    # Discovery/normalization/compatibility and other shared application code are imported through
    # worker.py/Container by both Celery images. Keep those images on one compatible Python version.
    case "$path" in
        src/job_hunt/application/compatibility.py|\
        src/job_hunt/application/discovery.py|\
        src/job_hunt/application/normalization.py|\
        src/job_hunt/application/page_content.py)
            _deploy_plan_shared_runtime
            return
            ;;
    esac

    # Dependency/build/Compose changes can alter both images and service definitions.
    case "$path" in
        Dockerfile|docker-compose.yml|pyproject.toml|uv.lock)
            _deploy_plan_shared_runtime
            _deploy_plan_litellm_reload
            return
            ;;
    esac

    # Unknown application/runtime code is conservatively treated as shared. This preserves task and
    # serialized-data contracts between fast and document stages.
    case "$path" in
        src/job_hunt/*|scripts/*.py)
            _deploy_plan_shared_runtime
            return
            ;;
    esac

    # Unclassified repository files also take the conservative shared-runtime path. New non-runtime
    # categories should be explicitly added above rather than silently under-deployed.
    _deploy_plan_shared_runtime
}

deploy_plan_actions() {
    local actions=()
    "$PLAN_BUILD_API" && actions+=(build-api)
    "$PLAN_BUILD_DOCUMENTS" && actions+=(build-documents)
    "$PLAN_RELOAD_LITELLM" && actions+=(reload-litellm)
    "$PLAN_RECREATE_API" && actions+=(recreate-api)
    "$PLAN_RECREATE_FAST" && actions+=(recreate-fast)
    "$PLAN_RECREATE_DOCUMENTS" && actions+=(recreate-documents)
    "$PLAN_RECREATE_NOTIFICATIONS" && actions+=(recreate-notifications)
    "$PLAN_RECREATE_BEAT" && actions+=(recreate-beat)
    "$PLAN_RECREATE_FLOWER" && actions+=(recreate-flower)
    "$PLAN_PREMERGE_DRAIN_FAST" && actions+=(premerge-drain-fast)
    "$PLAN_PREMERGE_DRAIN_DOCUMENTS" && actions+=(premerge-drain-documents)
    "$PLAN_PREMERGE_DRAIN_NOTIFICATIONS" && actions+=(premerge-drain-notifications)
    "$PLAN_LITELLM_DRAIN_FAST" && actions+=(litellm-drain-fast)
    "$PLAN_LITELLM_DRAIN_DOCUMENTS" && actions+=(litellm-drain-documents)
    "$PLAN_LATEX_CHECK" && actions+=(latex-check)
    "$PLAN_VALIDATE_LIVE" && actions+=(validate-live)
    if ((${#actions[@]} == 0)); then
        printf 'pull-only'
    else
        local IFS=,
        printf '%s' "${actions[*]}"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    deploy_plan_reset
    for path in "$@"; do
        deploy_plan_add_path "$path"
    done
    deploy_plan_actions
    printf '\n'
fi
