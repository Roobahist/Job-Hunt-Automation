from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, cast

from redis import Redis

from job_hunt.redis_client import configure_redis_retry


class RedisState:
    """Small shared-state abstraction used by workers without introducing another datastore."""

    def __init__(self, redis: Redis, *, namespace: str = "job-hunt") -> None:
        self.redis = configure_redis_retry(redis)
        self.namespace = namespace.rstrip(":")

    def _key(self, *parts: str) -> str:
        return ":".join((self.namespace, *parts))

    def set_json(self, key: str, value: object, *, ttl_seconds: int) -> None:
        self.redis.setex(self._key("json", key), ttl_seconds, json.dumps(value, default=str))

    def get_json(self, key: str) -> Any | None:
        raw = self.redis.get(self._key("json", key))
        if raw is None:
            return None
        return json.loads(cast(str, raw))

    def set_checkpoint(self, digest: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        self.set_json(f"checkpoint:{digest}", dict(value), ttl_seconds=ttl_seconds)

    def get_checkpoint(self, digest: str) -> dict[str, Any] | None:
        value = self.get_json(f"checkpoint:{digest}")
        return dict(value) if isinstance(value, dict) else None

    def set_snapshot(self, snapshot_id: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        self.set_json(f"snapshot:{snapshot_id}", dict(value), ttl_seconds=ttl_seconds)

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        value = self.get_json(f"snapshot:{snapshot_id}")
        return dict(value) if isinstance(value, dict) else None

    def cooldown(self, scope: str, resource: str, *, seconds: float) -> None:
        ttl = max(1, int(seconds))
        self.redis.setex(self._key("cooldown", scope, resource), ttl, str(time.time() + ttl))

    def available(self, scope: str, resource: str) -> bool:
        return not bool(self.redis.exists(self._key("cooldown", scope, resource)))

    def remaining_cooldown(self, scope: str, resource: str) -> int:
        ttl = cast(int, self.redis.ttl(self._key("cooldown", scope, resource)))
        return max(0, ttl)

    def increment_window(
        self,
        scope: str,
        resource: str,
        window: str,
        *,
        ttl_seconds: int,
        amount: int = 1,
    ) -> int:
        key = self._key("counter", scope, resource, window)
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.incrby(key, amount)
        pipeline.expire(key, ttl_seconds, nx=True)
        result = pipeline.execute()
        return int(result[0])

    def checkpoints(self, checkpoint_namespace: str | None) -> RedisState:
        if not checkpoint_namespace:
            return self
        return CheckpointScopedRedisState(
            self.redis,
            namespace=self.namespace,
            checkpoint_namespace=checkpoint_namespace,
        )


class CheckpointScopedRedisState(RedisState):
    """Share provider state globally while isolating LLM checkpoints by run lineage."""

    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str,
        checkpoint_namespace: str,
    ) -> None:
        super().__init__(redis, namespace=namespace)
        self.checkpoint_namespace = checkpoint_namespace

    def set_checkpoint(self, digest: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        super().set_checkpoint(
            f"{self.checkpoint_namespace}:{digest}",
            value,
            ttl_seconds=ttl_seconds,
        )

    def get_checkpoint(self, digest: str) -> dict[str, Any] | None:
        return super().get_checkpoint(f"{self.checkpoint_namespace}:{digest}")
