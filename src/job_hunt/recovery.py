from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from redis import Redis

from job_hunt.config import Settings, load_registry
from job_hunt.container import Container
from job_hunt.integrations.telegram import TelegramNotifier
from job_hunt.logging import configure_logging, logger
from job_hunt.redis_client import create_redis_client

_SLOT_FORMAT = "%Y-%m-%dT%H%z"
_PENDING_KEY_PREFIX = "job-hunt:pending-auto-discovery-recovery"


@dataclass(frozen=True, slots=True)
class MissedBeatRecoveryPlan:
    missed_slots: tuple[datetime, ...]
    window_seconds: int
    latest_slot: str
    clipped_to_lookback: bool


class RecoveryNotifier(TelegramNotifier):
    def send_text(self, chat_id: str, text: str) -> None:
        self._post("/sendMessage", json={"chat_id": chat_id, "text": text[:4096]})


def _slot_string(value: datetime) -> str:
    return value.strftime(_SLOT_FORMAT)


def _parse_slot(value: str, timezone: ZoneInfo) -> datetime:
    parsed = datetime.strptime(value, _SLOT_FORMAT)
    return parsed.astimezone(timezone)


def pending_recovery_key(tenant: str) -> str:
    return f"{_PENDING_KEY_PREFIX}:{tenant}"


def plan_missed_discovery_recovery(
    *,
    last_slot: str | None,
    now: datetime,
    interval_hours: int,
    lookback_hours: int,
    timezone: ZoneInfo,
) -> MissedBeatRecoveryPlan | None:
    if not last_slot:
        return None
    if interval_hours <= 0 or lookback_hours <= 0:
        raise ValueError("Discovery interval and recovery lookback must both be positive")

    previous = _parse_slot(last_slot, timezone)
    local_now = now.astimezone(timezone)
    current_hour = local_now.replace(minute=0, second=0, microsecond=0)
    cutoff = local_now - timedelta(hours=lookback_hours)

    missed: list[datetime] = []
    cursor = previous + timedelta(hours=1)
    while cursor <= current_hour:
        if cursor.hour % interval_hours == 0 and cursor >= cutoff:
            missed.append(cursor)
        cursor += timedelta(hours=1)

    if not missed:
        return None

    recovery_start = max(previous, cutoff)
    window_seconds = max(1, math.ceil((local_now - recovery_start).total_seconds()))
    return MissedBeatRecoveryPlan(
        missed_slots=tuple(missed),
        window_seconds=window_seconds,
        latest_slot=_slot_string(missed[-1]),
        clipped_to_lookback=previous < cutoff,
    )


def recovery_window_seconds(
    marker: dict[str, Any],
    *,
    now: datetime,
    timezone: ZoneInfo,
) -> int:
    last_slot = str(marker["last_slot"])
    lookback_hours = int(marker["lookback_hours"])
    previous = _parse_slot(last_slot, timezone)
    local_now = now.astimezone(timezone)
    cutoff = local_now - timedelta(hours=lookback_hours)
    recovery_start = max(previous, cutoff)
    return max(1, math.ceil((local_now - recovery_start).total_seconds()))


def load_pending_recovery(redis: Redis, tenant: str) -> dict[str, Any] | None:
    raw = redis.get(pending_recovery_key(tenant))
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        logger().warning("missed_discovery_recovery_marker_invalid", tenant=tenant)
        return None
    return parsed if isinstance(parsed, dict) else None


def clear_pending_recovery(redis: Redis, tenant: str, marker: dict[str, Any]) -> None:
    key = pending_recovery_key(tenant)
    raw = redis.get(key)
    if not raw:
        return
    try:
        current = json.loads(str(raw))
    except (TypeError, ValueError):
        return
    if isinstance(current, dict) and current.get("detected_at") == marker.get("detected_at"):
        redis.delete(key)


def notify_recovery_complete(
    *,
    settings: Settings,
    chat_id: str,
    tenant: str,
    marker: dict[str, Any],
    window_seconds: int,
    counts: dict[str, int],
) -> None:
    missed_count = int(marker.get("missed_count", 0))
    clipped = bool(marker.get("clipped_to_lookback", False))
    clipped_text = "\nRecovery was capped by the configured lookback window." if clipped else ""
    text = (
        "⚠️ LinkedIn discovery recovery\n"
        f"Tenant: {tenant}\n"
        f"Missed scheduled runs: {missed_count}\n"
        f"Catch-up applied on the next scheduled discovery.\n"
        f"Search window: {window_seconds} seconds\n"
        f"Jobs discovered: {counts['discovered']}\n"
        f"Existing jobs skipped: {counts['duplicates']}\n"
        f"New jobs queued: {counts['queued']}"
        f"{clipped_text}"
    )
    RecoveryNotifier(
        settings.shared_telegram_token(),
        timeout_seconds=settings.telegram_request_timeout_seconds,
    ).send_text(chat_id, text)


def recover_missed_discoveries(settings: Settings | None = None, *, now: datetime | None = None) -> dict[str, int]:
    """Detect missed beats at startup and defer catch-up to the next normal discovery run."""
    active_settings = settings or Settings()
    redis = create_redis_client(active_settings.redis_url)
    timezone = ZoneInfo(active_settings.scheduler_timezone)
    current_time = now or datetime.now(UTC)
    pending = 0
    missed_slots = 0
    failed = 0

    for tenant, bootstrap in load_registry(active_settings.registry_path).items():
        if not bootstrap.enabled:
            continue
        lock = redis.lock(
            f"job-hunt:lock:missed-discovery-recovery:{tenant}",
            timeout=300,
            blocking_timeout=0,
        )
        if not lock.acquire(blocking=False):
            logger().info("missed_discovery_recovery_detection_already_running", tenant=tenant)
            continue
        try:
            services = Container(active_settings).tenant(tenant)
            due_key = f"job-hunt:last-auto-discovery-slot:{tenant}"
            last_slot = redis.get(due_key)
            try:
                plan = plan_missed_discovery_recovery(
                    last_slot=str(last_slot) if last_slot else None,
                    now=current_time,
                    interval_hours=services.config.linkedin_schedule_interval_hours,
                    lookback_hours=services.config.linkedin_missed_beat_recovery_hours,
                    timezone=timezone,
                )
            except ValueError as exc:
                logger().warning(
                    "missed_discovery_recovery_invalid_slot",
                    tenant=tenant,
                    last_slot=last_slot,
                    error=str(exc),
                )
                failed += 1
                continue

            if plan is None:
                logger().info(
                    "missed_discovery_recovery_not_needed",
                    tenant=tenant,
                    last_slot=last_slot,
                    lookback_hours=services.config.linkedin_missed_beat_recovery_hours,
                )
                continue

            marker = {
                "last_slot": str(last_slot),
                "detected_at": current_time.isoformat(),
                "missed_count": len(plan.missed_slots),
                "missed_slots": [_slot_string(slot) for slot in plan.missed_slots],
                "interval_hours": services.config.linkedin_schedule_interval_hours,
                "lookback_hours": services.config.linkedin_missed_beat_recovery_hours,
                "clipped_to_lookback": plan.clipped_to_lookback,
            }
            redis.set(pending_recovery_key(tenant), json.dumps(marker, separators=(",", ":")))
            pending += 1
            missed_slots += len(plan.missed_slots)
            logger().warning(
                "missed_discovery_recovery_pending",
                tenant=tenant,
                missed_count=len(plan.missed_slots),
                missed_slots=marker["missed_slots"],
                normal_interval_hours=services.config.linkedin_schedule_interval_hours,
                lookback_hours=services.config.linkedin_missed_beat_recovery_hours,
                current_window_seconds=plan.window_seconds,
                clipped_to_lookback=plan.clipped_to_lookback,
                recovery_timing="next_scheduled_discovery",
            )
        except Exception as exc:
            failed += 1
            logger().exception("missed_discovery_recovery_detection_failed", tenant=tenant, error=str(exc))
        finally:
            if lock.owned():
                lock.release()

    return {"pending": pending, "missed_slots": missed_slots, "failed": failed}


def main() -> None:
    settings = Settings()
    configure_logging(json_logs=settings.environment != "development")
    result = recover_missed_discoveries(settings)
    logger().info("missed_discovery_recovery_detection_complete", **result)


if __name__ == "__main__":
    main()
