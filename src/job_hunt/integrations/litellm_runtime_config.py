from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


def application_owned_fallback_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove proxy-level cross-group fallbacks while preserving in-group deployment failover."""
    router_settings = config.get("router_settings")
    if isinstance(router_settings, Mapping):
        normalized = dict(router_settings)
        normalized.pop("fallbacks", None)
        config["router_settings"] = normalized
    return config


def _group_model_priorities(registry: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Return provider/model priority tiers in the exact order listed for each routing group."""
    raw_groups = registry.get("groups")
    if not isinstance(raw_groups, Mapping):
        return {}

    priorities: dict[str, dict[str, int]] = {}
    for raw_group, raw_entries in raw_groups.items():
        if not isinstance(raw_entries, list):
            continue
        group = str(raw_group)
        group_priorities: dict[str, int] = {}
        for order, raw_entry in enumerate(raw_entries, start=1):
            if not isinstance(raw_entry, Mapping):
                continue
            provider = str(raw_entry.get("provider", "")).strip().strip("/")
            model = str(raw_entry.get("model", "")).strip().strip("/")
            if provider and model:
                group_priorities[f"{provider}/{model}"] = order
        priorities[group] = group_priorities
    return priorities


def _in_group_retry_budget(model_list: object) -> int:
    """Allow one attempt per deployment in the largest routing group before the caller downgrades groups."""
    if not isinstance(model_list, list):
        return 0
    counts = Counter(
        str(entry.get("model_name", ""))
        for entry in model_list
        if isinstance(entry, Mapping) and str(entry.get("model_name", ""))
    )
    return max((count - 1 for count in counts.values()), default=0)


def prioritize_group_deployments(config: dict[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    """Let LiteLLM exhaust stronger deployment tiers before weaker models in the same group."""
    priorities = _group_model_priorities(registry)
    model_list = config.get("model_list")
    if isinstance(model_list, list):
        for entry in model_list:
            if not isinstance(entry, dict):
                continue
            group = str(entry.get("model_name", ""))
            group_priorities = priorities.get(group)
            if not group_priorities:
                continue
            params = entry.get("litellm_params")
            if not isinstance(params, Mapping):
                continue
            provider_model = str(params.get("model", ""))
            order = group_priorities.get(provider_model)
            if order is None:
                continue
            normalized_params = dict(params)
            normalized_params["order"] = order
            entry["litellm_params"] = normalized_params

    router_settings = config.get("router_settings")
    normalized_router = dict(router_settings) if isinstance(router_settings, Mapping) else {}
    normalized_router["enable_pre_call_checks"] = True
    normalized_router["enable_weighted_failover"] = True
    normalized_router["num_retries"] = _in_group_retry_budget(model_list)

    # A failed deployment should leave the eligible pool quickly enough for retries
    # to make progress through the remaining keys and models in the same group.
    raw_allowed_fails = normalized_router.get("allowed_fails", 1)
    if not isinstance(raw_allowed_fails, int) or isinstance(raw_allowed_fails, bool):
        raw_allowed_fails = 1
    normalized_router["allowed_fails"] = max(1, raw_allowed_fails)

    # Keep exhausted deployments out of adjacent calls long enough to avoid repeatedly
    # spending the retry budget on a key that is still provider-rate-limited.
    raw_cooldown = normalized_router.get("cooldown_time", 65)
    if not isinstance(raw_cooldown, (int, float)) or isinstance(raw_cooldown, bool):
        raw_cooldown = 65
    normalized_router["cooldown_time"] = max(65, raw_cooldown)

    config["router_settings"] = normalized_router
    return config
