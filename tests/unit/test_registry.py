from pathlib import Path

import pytest

from job_hunt.config import load_registry
from job_hunt.errors import ConfigurationError
from job_hunt.rendering.profiles import AtlasCvRenderer, NovaCvRenderer
from job_hunt.tenants.registry import TenantRegistry


def test_registry_builds_each_tenant_renderer() -> None:
    registry = TenantRegistry(load_registry(Path("config/users.toml")))
    assert isinstance(registry.get("atlas").renderer.cv_renderer, AtlasCvRenderer)
    assert isinstance(registry.get("nova").renderer.cv_renderer, NovaCvRenderer)
    with pytest.raises(ConfigurationError, match="Unknown tenant"):
        registry.get("nobody")
