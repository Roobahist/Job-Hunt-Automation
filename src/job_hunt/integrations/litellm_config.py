from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

CHAT_MODEL_BLOCKLIST = (
    "whisper",
    "speech",
    "tts",
    "orpheus",
    "guard",
    "safeguard",
    "moderation",
    "embedding",
    "embed",
    "rerank",
)
CAPABILITY_GROUPS = ("job-fast", "job-balanced", "job-powerful")
REPAIR_GROUP_MAP = {"job-fast": "repair-fast", "job-balanced": "repair-balanced"}
REPAIR_GROUPS = tuple(REPAIR_GROUP_MAP.values())
ALL_GROUPS = CAPABILITY_GROUPS + REPAIR_GROUPS
DEFAULT_PROVIDER_REGISTRY_PATH = Path("config/llm-routing.json")
DEFAULT_MODEL_LIMITS_PATH = Path("config/llm-model-limits.json")


class ConfigGenerationError(RuntimeError):
    pass


def indexed_env_values(env: Mapping[str, str], prefix: str) -> list[tuple[str, str]]:
    values: list[tuple[int, str, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for name, value in env.items():
        match = pattern.match(name)
        if match and value.strip():
            values.append((int(match.group(1)), name, value.strip()))
    values.sort(key=lambda item: item[0])
    return [(name, value) for _, name, value in values]


def load_provider_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigGenerationError(f"LiteLLM routing config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigGenerationError(f"LiteLLM routing config is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("providers"), list):
        raise ConfigGenerationError("LiteLLM routing config must contain a providers array")
    groups = payload.get("groups")
    if groups is not None and not isinstance(groups, Mapping):
        raise ConfigGenerationError("LiteLLM routing config groups must be an object")
    fallbacks = payload.get("fallbacks")
    if fallbacks is not None and not isinstance(fallbacks, Mapping):
        raise ConfigGenerationError("LiteLLM routing config fallbacks must be an object")
    return payload


def load_model_limits(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigGenerationError(f"LiteLLM model limits config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigGenerationError(f"LiteLLM model limits config is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ConfigGenerationError("LiteLLM model limits config must contain a models array")

    metadata: dict[str, dict[str, Any]] = {}
    for entry in payload["models"]:
        if not isinstance(entry, Mapping):
            raise ConfigGenerationError("Every LiteLLM model limits entry must be an object")
        provider = str(entry.get("provider", "")).strip().strip("/")
        model = str(entry.get("model", "")).strip().strip("/")
        if not provider or not model:
            raise ConfigGenerationError("LiteLLM model limits entries require provider and model")
        values: dict[str, Any] = {}
        for field in ("rpm", "tpm"):
            raw = entry.get(field)
            if raw is None:
                continue
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                raise ConfigGenerationError(f"LiteLLM model limit {provider}/{model} {field} must be nonnegative")
            values[field] = raw
        supports_schema = entry.get("supports_native_json_schema")
        if supports_schema is not None:
            if not isinstance(supports_schema, bool):
                raise ConfigGenerationError(
                    f"LiteLLM model metadata {provider}/{model} supports_native_json_schema must be boolean"
                )
            values["supports_native_json_schema"] = supports_schema
        metadata[f"{provider}/{model}"] = values
    return metadata


def discover_models(
    api_key: str,
    *,
    url: str,
    client_factory: Callable[[], httpx.Client] = httpx.Client,
) -> list[str]:
    with client_factory() as client:
        response = client.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    models = payload.get("data", []) if isinstance(payload, dict) else []
    return sorted(
        {
            str(item.get("id", "")).strip()
            for item in models
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
    )


def discover_with_key_fallback(
    keys: Sequence[tuple[str, str]],
    discover: Callable[[str], Sequence[str]],
) -> list[str]:
    failures: list[str] = []
    for key_name, key in keys:
        try:
            models = list(discover(key))
            if models:
                return models
            failures.append(f"{key_name}: empty model catalog")
        except Exception as exc:
            failures.append(f"{key_name}: {exc}")
    raise ConfigGenerationError("Unable to discover provider models with configured keys: " + "; ".join(failures))


def is_chat_candidate(model: str, blocklist: Sequence[str] = CHAT_MODEL_BLOCKLIST) -> bool:
    lowered = model.lower()
    return not any(marker.lower() in lowered for marker in blocklist)


def classify_model(model: str) -> str:
    lowered = model.lower()
    powerful_markers = ("120b", "90b", "70b", "72b", "65b", "405b")
    balanced_markers = ("34b", "32b", "31b", "27b", "24b", "20b", "17b", "14b")
    if any(marker in lowered for marker in powerful_markers):
        return "job-powerful"
    if any(marker in lowered for marker in balanced_markers):
        return "job-balanced"
    return "job-fast"


def _fill_missing_groups(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    if not any(groups.values()):
        return groups
    preferred_sources = {
        "job-fast": ("job-balanced", "job-powerful"),
        "job-balanced": ("job-powerful", "job-fast"),
        "job-powerful": ("job-balanced", "job-fast"),
    }
    for group in CAPABILITY_GROUPS:
        if groups[group]:
            continue
        for source in preferred_sources[group]:
            if groups[source]:
                groups[group] = list(groups[source])
                break
    return groups


def group_models(
    discovered: Sequence[str],
    *,
    explicit: Mapping[str, Sequence[str]] | None = None,
    excluded: Sequence[str] = (),
    blocklist: Sequence[str] = CHAT_MODEL_BLOCKLIST,
    allowlist: Sequence[str] = (),
    fill_missing: bool = True,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {group: [] for group in CAPABILITY_GROUPS}
    explicit_groups = explicit or {}
    excluded_set = set(excluded)
    allowlist_set = set(allowlist)
    explicitly_classified = {model for models in explicit_groups.values() for model in models}
    for group in CAPABILITY_GROUPS:
        groups[group].extend(model for model in explicit_groups.get(group, ()) if model not in excluded_set)
    for model in discovered:
        if model in excluded_set or model in explicitly_classified:
            continue
        if allowlist_set and model not in allowlist_set:
            continue
        if not is_chat_candidate(model, blocklist):
            continue
        groups[classify_model(model)].append(model)
    deduplicated = {group: sorted(set(models)) for group, models in groups.items()}
    return _fill_missing_groups(deduplicated) if fill_missing else deduplicated


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return normalized or "unknown"


def _deployment_id(model_name: str, provider_model: str, key_env_name: str) -> str:
    key_match = re.search(r"(\d+)$", key_env_name)
    key_alias = f"key-{key_match.group(1)}" if key_match else _slug(key_env_name)
    return f"{_slug(model_name)}--{_slug(provider_model)}--{key_alias}"


def deployment(
    model_name: str,
    provider_model: str,
    key_env_name: str,
    *,
    extra_params: Mapping[str, Any] | None = None,
    model_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params = dict(extra_params or {})
    params.update({"model": provider_model, "api_key": f"os.environ/{key_env_name}"})
    info: dict[str, Any] = {"id": _deployment_id(model_name, provider_model, key_env_name)}
    if model_info:
        info.update(model_info)
    return {"model_name": model_name, "litellm_params": params, "model_info": info}


def _provider_groups(provider: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = provider.get("models", {})
    if not isinstance(raw, Mapping):
        raise ConfigGenerationError(f"Provider {provider.get('name', '<unknown>')} models must be an object")
    return {
        "job-fast": [str(item) for item in raw.get("fast", [])],
        "job-balanced": [str(item) for item in raw.get("balanced", [])],
        "job-powerful": [str(item) for item in raw.get("powerful", [])],
    }


def _provider_repair_groups(provider: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = provider.get("repair_models", {})
    if not isinstance(raw, Mapping):
        raise ConfigGenerationError(f"Provider {provider.get('name', '<unknown>')} repair_models must be an object")
    return {
        "repair-fast": [str(item) for item in raw.get("fast", [])],
        "repair-balanced": [str(item) for item in raw.get("balanced", [])],
    }


def _explicit_group_assignments(registry: Mapping[str, Any]) -> dict[str, dict[str, list[str]]] | None:
    raw_groups = registry.get("groups")
    if raw_groups is None:
        return None
    if not isinstance(raw_groups, Mapping):
        raise ConfigGenerationError("LiteLLM routing config groups must be an object")

    assignments: dict[str, dict[str, list[str]]] = {}
    for group, entries in raw_groups.items():
        group_name = str(group)
        if group_name not in ALL_GROUPS:
            raise ConfigGenerationError(f"Unknown LiteLLM group in routing config: {group_name}")
        if not isinstance(entries, list):
            raise ConfigGenerationError(f"LiteLLM group {group_name} must contain a list of model entries")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ConfigGenerationError(f"LiteLLM group {group_name} entries must be objects")
            provider_name = str(entry.get("provider", "")).strip().strip("/")
            model = str(entry.get("model", "")).strip().strip("/")
            if not provider_name or not model:
                raise ConfigGenerationError(
                    f"LiteLLM group {group_name} entries require non-empty provider and model fields"
                )
            if model.startswith(f"{provider_name}/"):
                raise ConfigGenerationError(f"LiteLLM group {group_name} model must omit its provider prefix: {model}")
            provider_groups = assignments.setdefault(provider_name, {name: [] for name in ALL_GROUPS})
            provider_groups[group_name].append(model)
    return assignments


def _provider_enabled(provider: Mapping[str, Any], env: Mapping[str, str]) -> bool:
    flag = str(provider.get("enabled_env", "")).strip()
    if not flag:
        return bool(provider.get("enabled", True))
    raw = env.get(flag, str(provider.get("enabled", True))).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _provider_name(provider: Mapping[str, Any]) -> str:
    name = str(provider.get("name", "")).strip().strip("/")
    if not name:
        raise ConfigGenerationError("Each provider must define a non-empty name")
    if "/" in name:
        raise ConfigGenerationError(f"Provider name must not contain '/': {name}")
    return name


def _provider_keys(provider: Mapping[str, Any], env: Mapping[str, str]) -> list[tuple[str, str]]:
    prefix = str(provider.get("api_key_prefix", "")).strip()
    if not prefix:
        raise ConfigGenerationError(f"Provider {provider.get('name', '<unknown>')} must define api_key_prefix")
    return indexed_env_values(env, prefix)


def _discover_provider_models(
    provider: Mapping[str, Any],
    keys: Sequence[tuple[str, str]],
    *,
    discoverer: Callable[[str, str], Sequence[str]],
) -> list[str]:
    discovery = provider.get("discovery")
    if not isinstance(discovery, Mapping) or not discovery.get("enabled", False):
        return []
    url = str(discovery.get("url", "")).strip()
    if not url:
        raise ConfigGenerationError(f"Provider {provider.get('name', '<unknown>')} discovery requires a url")
    return discover_with_key_fallback(keys, lambda key: discoverer(key, url))


def _provider_extra_params(provider: Mapping[str, Any]) -> dict[str, Any]:
    raw = provider.get("litellm_params", {})
    if not isinstance(raw, Mapping):
        raise ConfigGenerationError(f"Provider {provider.get('name', '<unknown>')} litellm_params must be an object")
    return {str(key): value for key, value in raw.items()}


def _model_metadata(
    model_metadata: Mapping[str, Mapping[str, Any]],
    provider: str,
    model: str,
) -> dict[str, Any]:
    return dict(model_metadata.get(f"{provider}/{model}", {}))


def _deployment_params_for_model(extra_params: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(extra_params)
    for field in ("rpm", "tpm"):
        if field in metadata:
            params[field] = metadata[field]
    if metadata.get("supports_native_json_schema") is False:
        drop_params = list(params.get("additional_drop_params", []))
        if "response_format" not in drop_params:
            drop_params.append("response_format")
        params["additional_drop_params"] = drop_params
    return params


def _append_provider_models(
    model_list: list[dict[str, Any]],
    groups: Mapping[str, Sequence[str]],
    *,
    provider_name: str,
    keys: Sequence[tuple[str, str]],
    extra_params: Mapping[str, Any],
    model_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    for group, models in groups.items():
        for model in models:
            if model.startswith(f"{provider_name}/"):
                raise ConfigGenerationError(f"Model must omit its provider prefix: {model}")
            provider_model = f"{provider_name}/{model}"
            metadata = _model_metadata(model_metadata, provider_name, model)
            params = _deployment_params_for_model(extra_params, metadata)
            supports_schema = metadata.get("supports_native_json_schema")
            model_info = {"supports_response_schema": supports_schema} if isinstance(supports_schema, bool) else None
            for key_name, _ in keys:
                model_list.append(
                    deployment(group, provider_model, key_name, extra_params=params, model_info=model_info)
                )


def add_provider_deployments(
    model_list: list[dict[str, Any]],
    provider: Mapping[str, Any],
    *,
    env: Mapping[str, str],
    discoverer: Callable[[str, str], Sequence[str]],
    explicit_groups: Mapping[str, Sequence[str]] | None = None,
    model_limits: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if not _provider_enabled(provider, env):
        return
    keys = _provider_keys(provider, env)
    if not keys:
        return

    provider_name = _provider_name(provider)
    extra_params = _provider_extra_params(provider)
    metadata = model_limits or {}

    if explicit_groups is not None:
        _append_provider_models(
            model_list,
            explicit_groups,
            provider_name=provider_name,
            keys=keys,
            extra_params=extra_params,
            model_metadata=metadata,
        )
        return

    discovered = _discover_provider_models(provider, keys, discoverer=discoverer)
    explicit = _provider_groups(provider)
    excluded = [str(item) for item in provider.get("exclude_models", [])]
    blocklist = [str(item) for item in provider.get("blocklist", CHAT_MODEL_BLOCKLIST)]
    allowlist = [str(item) for item in provider.get("discovery_allowlist", [])]
    generation_groups = group_models(
        discovered,
        explicit=explicit,
        excluded=excluded,
        blocklist=blocklist,
        allowlist=allowlist,
        fill_missing=bool(provider.get("fill_missing_groups", True)),
    )
    repair_groups = _provider_repair_groups(provider)

    _append_provider_models(
        model_list,
        generation_groups,
        provider_name=provider_name,
        keys=keys,
        extra_params=extra_params,
        model_metadata=metadata,
    )
    _append_provider_models(
        model_list,
        repair_groups,
        provider_name=provider_name,
        keys=keys,
        extra_params=extra_params,
        model_metadata=metadata,
    )


def add_repair_deployments(model_list: list[dict[str, Any]]) -> None:
    source_entries = list(model_list)
    for entry in source_entries:
        group = str(entry["model_name"])
        repair_group = REPAIR_GROUP_MAP.get(group)
        if repair_group is None:
            continue
        copied: dict[str, Any] = {
            "model_name": repair_group,
            "litellm_params": dict(entry["litellm_params"]),
        }
        info = dict(entry.get("model_info", {}))
        provider_model = str(copied["litellm_params"].get("model", ""))
        api_key_ref = str(copied["litellm_params"].get("api_key", ""))
        if provider_model and api_key_ref.startswith("os.environ/"):
            info["id"] = _deployment_id(repair_group, provider_model, api_key_ref.removeprefix("os.environ/"))
        if info:
            copied["model_info"] = info
        model_list.append(copied)


def _configured_fallbacks(
    registry: Mapping[str, Any],
    present_groups: set[str],
) -> list[dict[str, list[str]]]:
    raw = registry.get("fallbacks")
    if raw is None:
        raw = {
            "job-powerful": ["job-balanced", "job-fast"],
            "job-balanced": ["job-fast"],
            "repair-fast": ["repair-balanced"],
        }
    if not isinstance(raw, Mapping):
        raise ConfigGenerationError("LiteLLM routing config fallbacks must be an object")

    fallbacks: list[dict[str, list[str]]] = []
    for source, targets in raw.items():
        source_name = str(source)
        if source_name not in ALL_GROUPS:
            raise ConfigGenerationError(f"Unknown LiteLLM fallback source: {source_name}")
        if not isinstance(targets, list):
            raise ConfigGenerationError(f"LiteLLM fallback {source_name} must contain a list")
        normalized_targets = [str(target) for target in targets]
        unknown = [target for target in normalized_targets if target not in ALL_GROUPS]
        if unknown:
            raise ConfigGenerationError(f"Unknown LiteLLM fallback target(s) for {source_name}: {unknown}")
        available_targets = [target for target in normalized_targets if target in present_groups]
        if source_name in present_groups and available_targets:
            fallbacks.append({source_name: available_targets})
    return fallbacks


def _nonnegative_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        value = int(env.get(name, str(default)))
    except ValueError as exc:
        raise ConfigGenerationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigGenerationError(f"{name} must be zero or greater")
    return value


def _env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _litellm_settings(env: Mapping[str, str]) -> dict[str, Any]:
    settings: dict[str, Any] = {"drop_params": True}
    if not _env_flag(env, "JOB_HUNT_LANGFUSE_ENABLED"):
        return settings
    missing = [name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not env.get(name, "").strip()]
    if missing:
        raise ConfigGenerationError("JOB_HUNT_LANGFUSE_ENABLED requires " + ", ".join(missing))
    settings["callbacks"] = ["langfuse_otel"]
    return settings


def _deployment_counts(model_list: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(entry["model_name"]) for entry in model_list)


def _max_failover_hops(model_list: Sequence[Mapping[str, Any]]) -> int:
    """Return a safe upper bound that grows automatically with providers and keys."""
    return max(1, len(model_list) - 1)


def build_litellm_config(
    *,
    env: Mapping[str, str],
    registry: Mapping[str, Any],
    model_limits: Mapping[str, Mapping[str, Any]] | None = None,
    discoverer: Callable[[str, str], Sequence[str]] = lambda key, url: discover_models(key, url=url),
) -> dict[str, Any]:
    providers = registry.get("providers")
    if not isinstance(providers, list):
        raise ConfigGenerationError("LiteLLM routing config must contain a providers array")

    explicit_assignments = _explicit_group_assignments(registry)
    known_provider_names = {_provider_name(provider) for provider in providers if isinstance(provider, Mapping)}
    if explicit_assignments is not None:
        unknown_providers = sorted(set(explicit_assignments) - known_provider_names)
        if unknown_providers:
            raise ConfigGenerationError(
                f"LiteLLM groups reference provider(s) not defined in providers: {unknown_providers}"
            )

    model_list: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, Mapping):
            raise ConfigGenerationError("Every provider registry entry must be an object")
        provider_name = _provider_name(provider)
        provider_groups = explicit_assignments.get(provider_name) if explicit_assignments is not None else None
        add_provider_deployments(
            model_list,
            provider,
            env=env,
            discoverer=discoverer,
            explicit_groups=provider_groups,
            model_limits=model_limits,
        )

    generation_entries = [item for item in model_list if str(item["model_name"]) in CAPABILITY_GROUPS]
    if not generation_entries:
        raise ConfigGenerationError("No LiteLLM chat-model deployments were generated from configured provider keys")
    if explicit_assignments is None:
        add_repair_deployments(model_list)

    present_groups = {str(item["model_name"]) for item in model_list}
    fallbacks = _configured_fallbacks(registry, present_groups)

    router_settings: dict[str, Any] = {
        "routing_strategy": "simple-shuffle",
        "enable_weighted_failover": True,
        "max_fallbacks": _max_failover_hops(model_list),
        "num_retries": 0,
        "allowed_fails": _nonnegative_int(env, "LITELLM_ALLOWED_FAILS", 0),
        "cooldown_time": _nonnegative_int(env, "LITELLM_COOLDOWN_SECONDS", 65),
        "fallbacks": fallbacks,
    }

    return {
        "model_list": model_list,
        "router_settings": router_settings,
        "litellm_settings": _litellm_settings(env),
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


def write_config(config: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def generate_litellm_config(
    *,
    env: Mapping[str, str],
    registry: Mapping[str, Any],
    destination: Path,
    model_limits: Mapping[str, Mapping[str, Any]] | None = None,
    discoverer: Callable[[str, str], Sequence[str]] = lambda key, url: discover_models(key, url=url),
) -> dict[str, Any]:
    config = build_litellm_config(
        env=env,
        registry=registry,
        model_limits=model_limits,
        discoverer=discoverer,
    )
    write_config(config, destination)
    return config
