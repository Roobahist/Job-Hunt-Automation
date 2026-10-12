from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from job_hunt.queueing import CeleryQueue


def test_celery_queue_dispatches_submission_and_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    submission_calls: list[tuple[object, ...]] = []
    discovery_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "job_hunt.queueing.process_submission",
        SimpleNamespace(delay=lambda *args: submission_calls.append(args) or SimpleNamespace(id="s1")),
    )
    monkeypatch.setattr(
        "job_hunt.queueing.discover_tenant",
        SimpleNamespace(delay=lambda *args: discovery_calls.append(args) or SimpleNamespace(id="d1")),
    )
    queue = CeleryQueue()
    run_id = uuid4()
    assert (
        queue.submission(
            "atlas",
            {"x": 1},
            run_id,
            True,
            "snapshot",
            "checkpoint-lineage",
            4758,
        )
        == "s1"
    )
    assert submission_calls == [
        (
            "atlas",
            {"x": 1},
            str(run_id),
            True,
            "snapshot",
            "checkpoint-lineage",
            4758,
        )
    ]
    assert queue.discovery("nova", run_id) == "d1"
    assert discovery_calls == [("nova", str(run_id))]
