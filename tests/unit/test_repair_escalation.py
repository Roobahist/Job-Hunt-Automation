from __future__ import annotations

from job_hunt.config import Settings
from job_hunt.integrations.llm_routing import LiteLLMGatewayClient, _build_repair_routes


def test_repair_routes_chain_schema_validation_fallbacks_in_order() -> None:
    settings = Settings(
        litellm_api_key="secret",
        llm_repair_routes="repair-fast,repair-balanced",
    )

    routes = dict((route.model, client) for route, client in _build_repair_routes(settings, state=None))

    fast = routes["repair-fast"]
    balanced = routes["repair-balanced"]
    assert isinstance(fast, LiteLLMGatewayClient)
    assert isinstance(balanced, LiteLLMGatewayClient)
    assert fast.repair_client is balanced
    assert balanced.repair_client is None
