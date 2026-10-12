from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

from job_hunt.integrations.litellm_config import (
    DEFAULT_MODEL_LIMITS_PATH,
    DEFAULT_PROVIDER_REGISTRY_PATH,
    ConfigGenerationError,
    build_litellm_config,
    discover_models,
    load_model_limits,
    load_provider_registry,
    write_config,
)
from job_hunt.integrations.litellm_runtime_config import (
    application_owned_fallback_config,
    prioritize_group_deployments,
)


def main() -> int:
    destination = Path(os.getenv("LITELLM_CONFIG_PATH", "config/litellm.runtime.yaml"))
    routing_path = Path(os.getenv("LITELLM_ROUTING_CONFIG_PATH", str(DEFAULT_PROVIDER_REGISTRY_PATH)))
    limits_path = Path(os.getenv("LITELLM_MODEL_LIMITS_PATH", str(DEFAULT_MODEL_LIMITS_PATH)))
    stdout_mode = os.getenv("LITELLM_CONFIG_STDOUT", "false").strip().lower() in {"1", "true", "yes"}
    try:
        registry = load_provider_registry(routing_path)
        model_limits = load_model_limits(limits_path)
        config = build_litellm_config(
            env=os.environ,
            registry=registry,
            model_limits=model_limits,
            discoverer=lambda key, url: discover_models(key, url=url),
        )
        config = application_owned_fallback_config(config)
        config = prioritize_group_deployments(config, registry)
        if stdout_mode:
            sys.stdout.write(json.dumps(config, indent=2) + "\n")
            return 0
        write_config(config, destination)
    except (ConfigGenerationError, httpx.HTTPError, OSError, ValueError) as exc:
        print(f"LiteLLM config generation failed: {exc}", file=sys.stderr)
        return 1

    groups: dict[str, int] = {}
    for entry in config["model_list"]:
        group = str(entry["model_name"])
        groups[group] = groups.get(group, 0) + 1
    print(f"Generated {destination}: {groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
