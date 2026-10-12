from __future__ import annotations

from job_hunt.state import RedisState


class Pipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple[object, ...]]] = []

    def incrby(self, key: str, amount: int) -> None:
        self.operations.append(("incrby", (key, amount)))

    def expire(self, key: str, ttl: int, nx: bool = False) -> None:
        self.operations.append(("expire", (key, ttl, nx)))

    def execute(self) -> list[object]:
        result: list[object] = []
        for name, args in self.operations:
            if name == "incrby":
                key, amount = str(args[0]), int(args[1])
                value = int(self.redis.data.get(key, "0")) + amount
                self.redis.data[key] = str(value)
                result.append(value)
            else:
                result.append(True)
        return result


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value
        self.ttls[key] = ttl

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def exists(self, key: str) -> int:
        return int(key in self.data)

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -2)

    def pipeline(self, transaction: bool = True) -> Pipeline:
        del transaction
        return Pipeline(self)


def test_json_snapshots_checkpoints_and_cooldowns() -> None:
    redis = FakeRedis()
    state = RedisState(redis)  # type: ignore[arg-type]
    state.set_snapshot("batch", {"config": {"x": 1}}, ttl_seconds=20)
    assert state.get_snapshot("batch") == {"config": {"x": 1}}
    assert state.get_snapshot("missing") is None

    state.set_checkpoint("digest", {"value": "done"}, ttl_seconds=30)
    assert state.get_checkpoint("digest") == {"value": "done"}

    assert state.available("gemini", "candidate")
    state.cooldown("gemini", "candidate", seconds=17)
    assert not state.available("gemini", "candidate")
    assert state.remaining_cooldown("gemini", "candidate") == 17


def test_scoped_checkpoints_are_isolated_by_run_lineage() -> None:
    redis = FakeRedis()
    state = RedisState(redis)  # type: ignore[arg-type]
    first = state.checkpoints("run-a")
    second = state.checkpoints("run-b")

    first.set_checkpoint("same-digest", {"value": "first"}, ttl_seconds=30)
    assert first.get_checkpoint("same-digest") == {"value": "first"}
    assert second.get_checkpoint("same-digest") is None

    second.set_checkpoint("same-digest", {"value": "second"}, ttl_seconds=30)
    assert first.get_checkpoint("same-digest") == {"value": "first"}
    assert second.get_checkpoint("same-digest") == {"value": "second"}


def test_increment_window_is_shared_and_incremental() -> None:
    redis = FakeRedis()
    state = RedisState(redis)  # type: ignore[arg-type]
    assert state.increment_window("rpm", "a", "minute", ttl_seconds=90) == 1
    assert state.increment_window("rpm", "a", "minute", ttl_seconds=90, amount=4) == 5
