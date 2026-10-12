from __future__ import annotations

import pytest
from redis import Redis
from redis.retry import Retry

from job_hunt.redis_client import configure_redis_retry, create_redis_client, redis_retry_policy


def test_configure_redis_retry_sets_four_connection_retries() -> None:
    client = Redis.from_url("redis://localhost:6379/0")

    configured = configure_redis_retry(client)

    assert configured is client
    retry = configured.connection_pool.connection_kwargs["retry"]
    assert isinstance(retry, Retry)
    assert retry.get_retries() == 4


def test_create_redis_client_applies_retry_policy() -> None:
    client = create_redis_client("redis://localhost:6379/0")

    retry = client.connection_pool.connection_kwargs["retry"]
    assert isinstance(retry, Retry)
    assert retry.get_retries() == 4


def test_redis_retry_policy_rejects_invalid_delays() -> None:
    with pytest.raises(ValueError, match="cap"):
        redis_retry_policy(base_seconds=2.0, cap_seconds=1.0)
