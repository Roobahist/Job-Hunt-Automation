from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import job_hunt.recovery as recovery
from job_hunt.application.discovery import build_search_url
from job_hunt.config import parse_configuration_rows
from job_hunt.recovery import notify_recovery_complete, plan_missed_discovery_recovery


def test_recovery_plan_detects_multiple_hourly_beats_and_covers_elapsed_gap() -> None:
    timezone = ZoneInfo("America/Edmonton")
    plan = plan_missed_discovery_recovery(
        last_slot="2026-08-25T14-0600",
        now=datetime.fromisoformat("2026-08-25T16:12:30-06:00"),
        interval_hours=1,
        lookback_hours=6,
        timezone=timezone,
    )

    assert plan is not None
    assert [slot.hour for slot in plan.missed_slots] == [15, 16]
    assert plan.latest_slot == "2026-08-25T16-0600"
    assert plan.window_seconds == 7950
    assert plan.clipped_to_lookback is False


def test_recovery_plan_respects_group_interval() -> None:
    timezone = ZoneInfo("America/Edmonton")
    plan = plan_missed_discovery_recovery(
        last_slot="2026-08-25T10-0600",
        now=datetime.fromisoformat("2026-08-25T16:05:00-06:00"),
        interval_hours=2,
        lookback_hours=6,
        timezone=timezone,
    )

    assert plan is not None
    assert [slot.hour for slot in plan.missed_slots] == [12, 14, 16]
    assert plan.latest_slot == "2026-08-25T16-0600"


def test_recovery_plan_caps_old_gap_to_configured_lookback() -> None:
    timezone = ZoneInfo("America/Edmonton")
    plan = plan_missed_discovery_recovery(
        last_slot="2026-08-25T06-0600",
        now=datetime.fromisoformat("2026-08-25T16:30:00-06:00"),
        interval_hours=1,
        lookback_hours=6,
        timezone=timezone,
    )

    assert plan is not None
    assert [slot.hour for slot in plan.missed_slots] == [11, 12, 13, 14, 15, 16]
    assert plan.window_seconds == 21600
    assert plan.clipped_to_lookback is True


def test_recovery_plan_does_nothing_without_prior_scheduler_state() -> None:
    plan = plan_missed_discovery_recovery(
        last_slot=None,
        now=datetime.fromisoformat("2026-08-25T16:30:00-06:00"),
        interval_hours=1,
        lookback_hours=6,
        timezone=ZoneInfo("America/Edmonton"),
    )

    assert plan is None


def test_single_missed_beat_sends_recovery_notice(monkeypatch: object) -> None:
    sent: list[tuple[str, str]] = []

    class FakeNotifier:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def send_text(self, chat_id: str, text: str) -> None:
            sent.append((chat_id, text))

    monkeypatch.setattr(recovery, "RecoveryNotifier", FakeNotifier)  # type: ignore[attr-defined]
    settings = SimpleNamespace(
        shared_telegram_token=lambda: "token",
        telegram_request_timeout_seconds=1,
    )

    notify_recovery_complete(
        settings=settings,  # type: ignore[arg-type]
        chat_id="42",
        tenant="nova",
        marker={"missed_count": 1, "clipped_to_lookback": False},
        window_seconds=4019,
        counts={"discovered": 7, "duplicates": 0, "queued": 7},
    )

    assert len(sent) == 1
    assert sent[0][0] == "42"
    assert "Missed scheduled runs: 1" in sent[0][1]
    assert "New jobs queued: 7" in sent[0][1]


def test_search_url_accepts_exact_recovery_window_seconds() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {"Keywords": "data scientist", "Location": "Alberta", "Sort By": "Relevance"},
        1,
        window_seconds=7950,
    )
    query = parse_qs(urlparse(url).query)

    assert query["f_TPR"] == ["r7950"]
    assert query["keywords"] == ["data scientist"]


def test_recovery_lookback_defaults_to_six_hours_when_baserow_key_is_absent() -> None:
    rows = [
        {"configKey": "tenant_key", "valueType": "text", "value": "test"},
        {"configKey": "applicant_filename", "valueType": "text", "value": "Test"},
        {
            "configKey": "baserow_table_ids",
            "valueType": "json",
            "value": '{"jobs":1,"searchCriteria":2,"prompts":3}',
        },
        {"configKey": "status_option_ids", "valueType": "json", "value": '{"new":1,"dropped":2}'},
        {"configKey": "linkedin_base_search_url", "valueType": "text", "value": "https://example.com"},
        {
            "configKey": "linkedin_job_url_template",
            "valueType": "text",
            "value": "https://example.com/{jobId}",
        },
        {"configKey": "telegram_chat_id", "valueType": "text", "value": "1"},
    ]

    config = parse_configuration_rows(rows)

    assert config.linkedin_missed_beat_recovery_hours == 6


def test_beat_runs_recovery_hook_before_celery_scheduler() -> None:
    compose = Path("docker-compose.yml").read_text()
    beat_section = compose.split("  beat:\n", 1)[1].split("\n  flower:\n", 1)[0]

    recovery_index = beat_section.index("python -m job_hunt.recovery")
    scheduler_index = beat_section.index("celery -A job_hunt.worker.celery_app beat")
    assert recovery_index < scheduler_index
