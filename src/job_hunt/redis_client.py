from __future__ import annotations

from redis import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

_REDIS_RETRY_COUNT = 4
_REDIS_RETRY_BASE_SECONDS = 0.25
_REDIS_RETRY_CAP_SECONDS = 4.0


def redis_retry_policy(
    *,
    retries: int = _REDIS_RETRY_COUNT,
    base_seconds: float = _REDIS_RETRY_BASE_SECONDS,
    cap_seconds: float = _REDIS_RETRY_CAP_SECONDS,
) -> Retry:
    """Build the shared Redis connection retry policy."""
    if retries < 0:
        raise ValueError("Redis retry count must be non-negative")
    if base_seconds < 0 or cap_seconds < 0:
        raise ValueError("Redis retry delays must be non-negative")
    if cap_seconds < base_seconds:
        raise ValueError("Redis retry cap must be at least the base delay")
    return Retry(ExponentialBackoff(base=base_seconds, cap=cap_seconds), retries)


def configure_redis_retry(
    redis: Redis,
    *,
    retries: int = _REDIS_RETRY_COUNT,
    base_seconds: float = _REDIS_RETRY_BASE_SECONDS,
    cap_seconds: float = _REDIS_RETRY_CAP_SECONDS,
) -> Redis:
    """Apply connection retries while preserving injected Redis-compatible test doubles."""
    connection_pool = getattr(redis, "connection_pool", None)
    set_retry = getattr(connection_pool, "set_retry", None)
    if callable(set_retry):
        set_retry(
            redis_retry_policy(
                retries=retries,
                base_seconds=base_seconds,
                cap_seconds=cap_seconds,
            )
        )
    return redis


def create_redis_client(url: str, *, decode_responses: bool = True) -> Redis:
    """Create a Redis client with the shared transient connection retry policy."""
    return configure_redis_retry(Redis.from_url(url, decode_responses=decode_responses))
