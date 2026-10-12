from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from job_hunt.worker import (
    _automatic_discovery_slot,
    _completed_day_slot,
    _end_of_day_window_seconds,
    celery_app,
    settings,
)


def test_scheduler_defaults_to_calgary_timezone() -> None:
    assert settings.scheduler_timezone == "America/Edmonton"
    assert celery_app.conf.timezone == "America/Edmonton"
    assert celery_app.conf.enable_utc is True


def test_beat_dispatches_on_top_of_every_hour() -> None:
    schedule = celery_app.conf.beat_schedule["dispatch-due-tenants"]["schedule"]
    assert schedule.minute == {0}


def test_beat_has_no_independent_end_of_day_schedule() -> None:
    assert "dispatch-end-of-day-tenants" not in celery_app.conf.beat_schedule


def test_midnight_slot_targets_the_previous_completed_calgary_day() -> None:
    midnight_calgary = datetime(2026, 9, 20, 6, 0, tzinfo=UTC)
    one_am_calgary = datetime(2026, 9, 20, 7, 0, tzinfo=UTC)

    assert _completed_day_slot(midnight_calgary) == "2026-09-19"
    assert _completed_day_slot(one_am_calgary) is None


def test_hourly_slot_uses_calgary_clock_and_timezone_rules() -> None:
    calgary = ZoneInfo("America/Edmonton")
    summer = datetime(2026, 8, 20, 19, 0, tzinfo=UTC)
    winter = datetime(2026, 12, 20, 19, 0, tzinfo=UTC)

    assert _automatic_discovery_slot(summer, 1) == summer.astimezone(calgary).strftime("%Y-%m-%dT%H%z")
    assert _automatic_discovery_slot(winter, 1) == winter.astimezone(calgary).strftime("%Y-%m-%dT%H%z")


def test_multi_hour_intervals_align_to_local_clock_hours() -> None:
    noon_calgary = datetime(2026, 8, 20, 18, 0, tzinfo=UTC)
    one_pm_calgary = datetime(2026, 8, 20, 19, 0, tzinfo=UTC)

    assert _automatic_discovery_slot(noon_calgary, 2) == "2026-08-20T12-0600"
    assert _automatic_discovery_slot(one_pm_calgary, 2) is None


def test_end_of_day_window_includes_delay_after_midnight() -> None:
    seven_minutes_thirty_seconds_after_midnight = datetime(2026, 9, 20, 6, 7, 30, tzinfo=UTC)

    assert (
        _end_of_day_window_seconds(
            "2026-09-19",
            seven_minutes_thirty_seconds_after_midnight,
        )
        == 86_850
    )
