from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar
from uuid import UUID

from fastapi.testclient import TestClient

from job_hunt.api.app import create_app
from job_hunt.config import Settings
from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.run_store import RunStore


class Pipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.pending: tuple[str, str] | None = None

    def watch(self, _: str) -> None:
        return None

    def get(self, key: str) -> str | None:
        return self.redis.get(key)

    def multi(self) -> None:
        return None

    def setex(self, key: str, _: int, value: str) -> None:
        self.pending = (key, value)

    def execute(self) -> list[bool]:
        if self.pending:
            self.redis.data[self.pending[0]] = self.pending[1]
        return [True]

    def reset(self) -> None:
        self.pending = None


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def setex(self, key: str, _: int, value: str) -> None:
        self.data[key] = value

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def ping(self) -> bool:
        return True

    def pipeline(self) -> Pipeline:
        return Pipeline(self)


class FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def submission(
        self,
        tenant: str,
        payload: dict[str, Any],
        run_id: UUID,
        force: bool,
        snapshot_id: str | None = None,
        checkpoint_namespace: str | None = None,
        resume_row_id: int | None = None,
    ) -> str:
        self.calls.append(
            (
                "submission",
                tenant,
                payload,
                run_id,
                force,
                snapshot_id,
                checkpoint_namespace,
                resume_row_id,
            )
        )
        return "task-1"

    def discovery(self, tenant: str, run_id: UUID) -> str:
        self.calls.append(("discovery", tenant, run_id))
        return "task-2"


class FakeBootstrap:
    enabled = True

    def secret(self, name: str, *, required: bool = True) -> str:
        values = {"fillout": "webhook-secret"}
        value = values.get(name, "")
        if required and not value:
            raise KeyError(name)
        return value


class FakeNotifier:
    def __init__(self) -> None:
        self.answers: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str, str, str, int | None]] = []
        self.processing_messages: list[tuple[str, str, str | None, int | None]] = []
        self.fail_answers = False
        self.fail_processing = False

    def answer_callback(self, callback_id: str, text: str) -> None:
        if self.fail_answers:
            raise ProviderError(
                "callback acknowledgement failed",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="telegram",
            )
        self.answers.append((callback_id, text))

    def send_processing_message(
        self,
        chat_id: str,
        *,
        caption: str,
        job_url: str | None,
        row_id: int | None = None,
        document_content: str | bytes | None = None,
    ) -> str:
        del document_content
        if self.fail_processing:
            raise ProviderError(
                "processing acknowledgement failed",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="telegram",
            )
        self.processing_messages.append((chat_id, caption, job_url, row_id))
        return "901"

    def edit_processing_message(
        self,
        chat_id: str,
        message_id: str,
        *,
        caption: str,
        job_url: str,
        row_id: int | None = None,
    ) -> str:
        self.edits.append((chat_id, message_id, caption, job_url, row_id))
        return message_id


class FakeRepository:
    def __init__(self) -> None:
        self.statuses: list[tuple[int, str]] = []
        self.rows: list[dict[str, object]] = []

    def set_status(self, row_id: int, status: str) -> None:
        self.statuses.append((row_id, status))

    def find_reference(self, *, external_id: str | None = None, url: str | None = None) -> dict[str, object] | None:
        for row in self.rows:
            if external_id is not None and str(row.get("Job ID") or "") == external_id:
                return row
            if url is not None and row.get("Link") == url:
                return row
        return None


class FakeContainer:
    notifier: ClassVar[FakeNotifier] = FakeNotifier()
    repositories: ClassVar[dict[str, FakeRepository]] = {
        "atlas": FakeRepository(),
        "nova": FakeRepository(),
    }
    bootstraps: ClassVar[dict[str, FakeBootstrap]] = {
        "atlas": FakeBootstrap(),
        "nova": FakeBootstrap(),
    }

    def __init__(self) -> None:
        self.registry = SimpleNamespace(get=lambda _: object(), bootstraps=self.bootstraps)

    def telegram_route(self, chat_id: str) -> object | None:
        tenants = {"100": "atlas", "200": "nova"}
        tenant = tenants.get(chat_id)
        if tenant is None:
            return None
        return SimpleNamespace(tenant=tenant, repository=self.repositories[tenant], notifier=self.notifier)

    def tenant(self, tenant: str) -> object:
        chat_ids = {"atlas": "100", "nova": "200"}
        return SimpleNamespace(
            context=SimpleNamespace(bootstrap=self.bootstraps[tenant]),
            config=SimpleNamespace(
                fillout_field_ids={},
                telegram_chat_id=chat_ids[tenant],
            ),
            notifier=self.notifier,
            repository=self.repositories[tenant],
            normalizer=SimpleNamespace(linkedin_template="https://www.linkedin.com/jobs/view/{jobId}"),
        )


def client() -> tuple[TestClient, FakeQueue]:
    queue = FakeQueue()
    store = RunStore(FakeRedis())  # type: ignore[arg-type]
    settings = Settings(
        operator_token="operator",
        redis_url="redis://unused",
        telegram_bot_token="bot-token",
        telegram_webhook_secret="telegram-secret",
    )
    app = create_app(settings=settings, queue=queue, store=store, container_factory=FakeContainer)
    return TestClient(app), queue


def auth() -> dict[str, str]:
    return {"Authorization": "Bearer operator"}


def reset_fillout_rows() -> None:
    for repository in FakeContainer.repositories.values():
        repository.rows.clear()


def test_health_and_operator_auth() -> None:
    api, _ = client()
    assert api.get("/health/live").json() == {"status": "ok"}
    assert api.get("/health/ready").status_code == 200
    assert api.post("/v1/tenants/atlas/jobs", json={}).status_code == 401


def test_submit_status_and_retry() -> None:
    api, queue = client()
    response = api.post(
        "/v1/tenants/atlas/jobs?force=true",
        headers=auth(),
        json={
            "entry_type": "external",
            "company_name": "C",
            "job_title": "T",
            "job_description": "D",
            "job_url": "https://example.com/j",
        },
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert api.get(f"/v1/runs/{run_id}", headers=auth()).json()["task_id"] == "task-1"
    assert api.post(f"/v1/runs/{run_id}/retry", headers=auth()).status_code == 202
    assert queue.calls[0][4] is True
    assert queue.calls[0][-1] is None


def test_discovery_and_fillout_authentication() -> None:
    reset_fillout_rows()
    FakeContainer.notifier.processing_messages.clear()
    api, queue = client()
    assert api.post("/v1/tenants/nova/discoveries", headers=auth()).status_code == 202
    assert queue.calls[-1][0] == "discovery"

    # A stale formId from an older imported form is deliberately ignored. The tenant
    # endpoint plus that tenant's bearer secret are the Fillout identity boundary.
    payload = {"formId": "stale-legacy-value", "entryType": "linkedin", "linkedinJobId": 1}
    accepted = api.post(
        "/webhooks/fillout/atlas",
        json=payload,
        headers={"Authorization": "Bearer webhook-secret"},
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run_id"]
    assert queue.calls[-1][4] is True
    assert queue.calls[-1][-1] is None
    assert len(FakeContainer.notifier.processing_messages) == 1
    chat_id, caption, job_url, row_id = FakeContainer.notifier.processing_messages[0]
    assert chat_id == "100"
    assert "LinkedIn job 1 received" in caption
    assert "Status: ⚙️ Received, starting normalization" in caption
    assert job_url == "https://www.linkedin.com/jobs/view/1"
    assert row_id is None
    status_payload = api.get(f"/v1/runs/{run_id}", headers=auth()).json()
    assert status_payload["notification"]["message_id"] == "901"
    assert status_payload["notification"]["init_queued"] is True


def test_fillout_existing_job_queues_stored_data_without_rescraping() -> None:
    reset_fillout_rows()
    FakeContainer.repositories["nova"].rows.append(
        {
            "id": 77,
            "Job ID": "4452378707",
            "Company Name": "Existing Co",
            "Title": "Existing Role",
            "Job Description": "Stored scraped description",
            "Location": "Calgary, AB",
            "Contract Type": {"value": "Full-time"},
            "Link": "https://www.linkedin.com/jobs/view/4452378707",
            "Score": 82,
        }
    )
    api, queue = client()

    response = api.post(
        "/webhooks/fillout/nova",
        json={"entryType": "linkedin", "linkedinJobId": 4452378707},
        headers={"Authorization": "Bearer webhook-secret"},
    )

    assert response.status_code == 202
    payload = queue.calls[-1][2]
    assert isinstance(payload, dict)
    assert payload["entry_type"] == "external"
    assert payload["company_name"] == "Existing Co"
    assert payload["job_description"] == "Stored scraped description"
    assert payload["reuse_existing_data"] is True
    assert payload["reuse_existing_score"] is True


def test_fillout_existing_row_with_missing_data_still_scrapes_but_reuses_score() -> None:
    reset_fillout_rows()
    FakeContainer.repositories["nova"].rows.append(
        {
            "id": 78,
            "Job ID": "4452378708",
            "Company Name": "Existing Co",
            "Title": "Existing Role",
            "Job Description": "",
            "Link": "https://www.linkedin.com/jobs/view/4452378708",
            "Score": 74,
        }
    )
    api, queue = client()

    response = api.post(
        "/webhooks/fillout/nova",
        json={"entryType": "linkedin", "linkedinJobId": 4452378708},
        headers={"Authorization": "Bearer webhook-secret"},
    )

    assert response.status_code == 202
    payload = queue.calls[-1][2]
    assert isinstance(payload, dict)
    assert payload["entry_type"] == "linkedin"
    assert payload["linkedin_job_id"] == 4452378708
    assert payload["reuse_existing_data"] is False
    assert payload["reuse_existing_score"] is True


def test_fillout_does_not_require_form_id() -> None:
    reset_fillout_rows()
    api, queue = client()
    response = api.post(
        "/webhooks/fillout/nova",
        json={"entryType": "linkedin", "linkedinJobId": 3},
        headers={"Authorization": "Bearer webhook-secret"},
    )
    assert response.status_code == 202
    assert queue.calls[-1][0] == "submission"


def test_fillout_rejects_wrong_tenant_secret() -> None:
    reset_fillout_rows()
    api, _ = client()
    response = api.post(
        "/webhooks/fillout/atlas",
        json={"entryType": "linkedin", "linkedinJobId": 4},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401


def test_fillout_telegram_failure_does_not_block_processing() -> None:
    reset_fillout_rows()
    FakeContainer.notifier.processing_messages.clear()
    FakeContainer.notifier.fail_processing = True
    api, queue = client()
    response = api.post(
        "/webhooks/fillout/atlas",
        json={"entryType": "linkedin", "linkedinJobId": 2},
        headers={"Authorization": "Bearer webhook-secret"},
    )
    FakeContainer.notifier.fail_processing = False
    assert response.status_code == 202
    assert queue.calls[-1][0] == "submission"
    run_id = response.json()["run_id"]
    status_payload = api.get(f"/v1/runs/{run_id}", headers=auth()).json()
    assert status_payload["notification"]["init_queued"] is False
    assert "intake_notification_error" in status_payload["notification"]


def test_fillout_invalid_values_return_structured_validation_error() -> None:
    reset_fillout_rows()
    api, _ = client()
    response = api.post(
        "/webhooks/fillout/nova",
        json={"entryType": "url", "jobUrl": ""},
        headers={"Authorization": "Bearer webhook-secret"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["message"] == "Submitted values are incomplete or invalid"


def test_shared_telegram_callback_routes_by_chat_id() -> None:
    FakeContainer.repositories["atlas"].statuses.clear()
    FakeContainer.repositories["nova"].statuses.clear()
    FakeContainer.notifier.answers.clear()
    FakeContainer.notifier.edits.clear()
    FakeContainer.notifier.fail_answers = False
    api, _ = client()
    response = api.post(
        "/webhooks/telegram",
        headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        json={
            "callback_query": {
                "id": "cb",
                "data": "status:applied:42",
                "message": {"chat": {"id": 100}},
            }
        },
    )
    assert response.status_code == 200
    assert FakeContainer.repositories["atlas"].statuses == [(42, "applied")]
    assert FakeContainer.repositories["nova"].statuses == []
    assert FakeContainer.notifier.answers == [("cb", "Status updated")]


def test_skip_processing_updates_same_telegram_message_to_dropped() -> None:
    FakeContainer.repositories["nova"].statuses.clear()
    FakeContainer.notifier.answers.clear()
    FakeContainer.notifier.edits.clear()
    api, _ = client()
    caption = "Designer\n\nStatus: ⏳ Waiting to retry\n▶ Qualification: started 2:41:37 PM\nRetry in: 300s"
    response = api.post(
        "/webhooks/telegram",
        headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        json={
            "callback_query": {
                "id": "cb-drop",
                "data": "status:dropped:77",
                "message": {
                    "message_id": 555,
                    "chat": {"id": 200},
                    "caption": caption,
                    "reply_markup": {"inline_keyboard": [[{"text": "Open Job", "url": "https://example.com/job"}]]},
                },
            }
        },
    )
    assert response.status_code == 200
    assert FakeContainer.repositories["nova"].statuses == [(77, "dropped")]
    assert len(FakeContainer.notifier.edits) == 1
    chat_id, message_id, updated_caption, job_url, row_id = FakeContainer.notifier.edits[0]
    assert chat_id == "200"
    assert message_id == "555"
    assert "Status: ⛔ Dropped manually" in updated_caption
    assert "■ Qualification: stopped manually" in updated_caption
    assert "Retry in:" not in updated_caption
    assert job_url == "https://example.com/job"
    assert row_id is None
    assert FakeContainer.notifier.answers == [("cb-drop", "Status updated")]


def test_shared_telegram_callback_ignores_unknown_chat() -> None:
    FakeContainer.repositories["atlas"].statuses.clear()
    api, _ = client()
    response = api.post(
        "/webhooks/telegram",
        headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        json={
            "callback_query": {
                "id": "cb",
                "data": "status:applied:42",
                "message": {"chat": {"id": 999}},
            }
        },
    )
    assert response.status_code == 200
    assert FakeContainer.repositories["atlas"].statuses == []


def test_telegram_acknowledgement_failure_does_not_retry_action() -> None:
    FakeContainer.repositories["nova"].statuses.clear()
    FakeContainer.notifier.fail_answers = True
    api, _ = client()
    response = api.post(
        "/webhooks/telegram",
        headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        json={
            "callback_query": {
                "id": "cb-fail",
                "data": "status:applied:43",
                "message": {"chat": {"id": 200}},
            }
        },
    )
    FakeContainer.notifier.fail_answers = False
    assert response.status_code == 200
    assert FakeContainer.repositories["nova"].statuses == [(43, "applied")]


def test_validation_error_is_structured() -> None:
    api, _ = client()
    response = api.post(
        "/v1/tenants/atlas/jobs",
        headers=auth(),
        json={"entry_type": "linkedin", "linkedin_job_id": -1},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
