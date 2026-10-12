from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import httpx

from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.http import raise_provider_error
from job_hunt.url_utils import valid_http_url

_DROPPED_STATUS_MARKER = "Status: ⛔ Dropped"
_FAILED_STATUS_MARKER = "Status: ❌ Failed"


def _should_cleanup(caption: str) -> bool:
    return _DROPPED_STATUS_MARKER in caption


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: int = 120,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self._client = client or httpx.Client(
            base_url=f"https://api.telegram.org/bot{token}",
            timeout=timeout_seconds,
        )

    def _visible_caption(self, caption: str) -> str:
        if self.debug:
            return caption
        lines = caption.splitlines()
        status_index = next((index for index, line in enumerate(lines) if line.startswith("Status:")), None)
        if status_index is None:
            return caption
        visible = lines[: status_index + 1]
        error_line = next((line for line in lines[status_index + 1 :] if line.startswith("Error:")), None)
        if error_line:
            visible.append(error_line)
        return "\n".join(visible)

    def _post(self, path: str, **kwargs: Any) -> dict[str, object]:
        try:
            response = self._client.post(path, **kwargs)
        except httpx.TransportError as exc:
            raise ProviderError(
                f"Telegram network request failed: {exc}",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="telegram",
            ) from exc
        raise_provider_error("telegram", response)
        return cast(dict[str, object], response.json())

    @staticmethod
    def _actions(job_url: str | None, row_id: int, run_id: str) -> dict[str, object]:
        rows: list[list[dict[str, object]]] = []
        normalized_url = valid_http_url(job_url)
        if normalized_url:
            rows.append([{"text": "Open Job", "url": normalized_url}])
        rows.append(
            [
                {"text": "Mark Applied", "callback_data": f"status:applied:{row_id}"},
                {"text": "Skip", "callback_data": f"status:dropped:{row_id}"},
            ]
        )
        rows.append([{"text": "Regenerate", "callback_data": f"retry:{run_id}"}])
        return {"inline_keyboard": rows}

    @staticmethod
    def _processing_actions(
        job_url: str | None,
        row_id: int | None,
        *,
        failed_run_id: str | None = None,
    ) -> dict[str, object]:
        rows: list[list[dict[str, object]]] = []
        normalized_url = valid_http_url(job_url)
        if normalized_url:
            rows.append([{"text": "Open Job", "url": normalized_url}])
        if row_id is not None:
            rows.append(
                [
                    {
                        "text": "Skip Processing",
                        "callback_data": f"status:dropped:{row_id}",
                    }
                ]
            )
        if failed_run_id:
            rows.append([{"text": "Regenerate", "callback_data": f"retry:{failed_run_id}"}])
        return {"inline_keyboard": rows}

    @staticmethod
    def _message_id(payload: dict[str, object], error_message: str) -> str:
        result = payload.get("result")
        if not isinstance(result, dict) or "message_id" not in result:
            raise ProviderError(
                error_message,
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="telegram",
            )
        return str(result["message_id"])

    def delete_message(self, chat_id: str, message_id: str) -> None:
        payload = self._post(
            "/deleteMessage",
            json={"chat_id": chat_id, "message_id": int(message_id)},
        )
        if payload.get("result") is not True:
            raise ProviderError(
                "Telegram did not confirm message deletion",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="telegram",
            )

    def send_processing_message(
        self,
        chat_id: str,
        *,
        caption: str,
        job_url: str | None,
        row_id: int | None = None,
        document_content: str | bytes | None = None,
    ) -> str:
        caption = self._visible_caption(caption)
        if _should_cleanup(caption):
            return ""
        placeholder = b" ".join(
            (
                b"Job processing is in progress.",
                b"The application ZIP will replace this placeholder when complete.\n",
            )
        )
        if isinstance(document_content, str):
            attachment = document_content.encode("utf-8")
        elif isinstance(document_content, bytes):
            attachment = document_content
        else:
            attachment = placeholder
        data: dict[str, object] = {
            "chat_id": chat_id,
            "caption": caption[:1024],
        }
        actions = self._processing_actions(job_url, row_id)
        if actions["inline_keyboard"]:
            data["reply_markup"] = json.dumps(actions)
        payload = self._post(
            "/sendDocument",
            data=data,
            files={"document": ("processing.txt", attachment, "text/plain")},
        )
        return self._message_id(payload, "Telegram returned no processing message")

    def edit_processing_message(
        self,
        chat_id: str,
        message_id: str,
        *,
        caption: str,
        job_url: str | None,
        row_id: int | None = None,
        run_id: str | None = None,
    ) -> str:
        caption = self._visible_caption(caption)
        if _should_cleanup(caption):
            self.delete_message(chat_id, message_id)
            return message_id
        data: dict[str, object] = {
            "chat_id": chat_id,
            "message_id": int(message_id),
            "caption": caption[:1024],
        }
        if "Status: ✅ Complete" not in caption:
            actions = self._processing_actions(
                job_url,
                row_id,
                failed_run_id=run_id if _FAILED_STATUS_MARKER in caption else None,
            )
            if actions["inline_keyboard"]:
                data["reply_markup"] = actions
        payload = self._post("/editMessageCaption", json=data)
        return self._message_id(payload, "Telegram returned no edited processing message")

    def edit_final_caption(
        self,
        chat_id: str,
        message_id: str,
        *,
        caption: str,
        job_url: str | None,
        row_id: int,
        run_id: str,
    ) -> str:
        caption = self._visible_caption(caption)
        payload = self._post(
            "/editMessageCaption",
            json={
                "chat_id": chat_id,
                "message_id": int(message_id),
                "caption": caption[:1024],
                "reply_markup": self._actions(job_url, row_id, run_id),
            },
        )
        return self._message_id(payload, "Telegram returned no edited final caption")

    def finalize_processing_message(
        self,
        chat_id: str,
        message_id: str,
        archive: Path,
        *,
        caption: str,
        job_url: str | None,
        row_id: int,
        run_id: str,
    ) -> str:
        caption = self._visible_caption(caption)
        with archive.open("rb") as handle:
            payload = self._post(
                "/editMessageMedia",
                data={
                    "chat_id": chat_id,
                    "message_id": int(message_id),
                    "media": json.dumps(
                        {
                            "type": "document",
                            "media": "attach://document",
                            "caption": caption[:1024],
                        }
                    ),
                    "reply_markup": json.dumps(self._actions(job_url, row_id, run_id)),
                },
                files={"document": (archive.name, handle, "application/zip")},
            )
        return self._message_id(payload, "Telegram returned no finalized processing message")

    def answer_callback(self, callback_id: str, text: str) -> None:
        self._post("/answerCallbackQuery", json={"callback_query_id": callback_id, "text": text})

    def send_documents(self, chat_id: str, artifacts: Iterable[Path], caption: str) -> str:
        caption = self._visible_caption(caption)
        paths = list(artifacts)
        if not 2 <= len(paths) <= 10:
            raise ValueError("Telegram media groups support 2 to 10 documents")
        media = []
        with ExitStack() as stack:
            files: dict[str, tuple[str, Any, str]] = {}
            for index, path in enumerate(paths):
                handle = stack.enter_context(path.open("rb"))
                name = f"file_{index}"
                files[name] = (path.name, handle, "application/octet-stream")
                media.append(
                    {
                        "type": "document",
                        "media": f"attach://{name}",
                        "caption": caption[:1024] if index == 0 else "",
                    }
                )
            payload = self._post(
                "/sendMediaGroup",
                data={"chat_id": chat_id, "media": json.dumps(media)},
                files=files,
            )
        result = payload.get("result")
        if not isinstance(result, list) or not result:
            raise ProviderError(
                "Telegram returned no media group messages",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="telegram",
            )
        first = result[0]
        if not isinstance(first, dict) or "message_id" not in first:
            raise ProviderError(
                "Telegram returned malformed media group result",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="telegram",
            )
        return str(first["message_id"])

    def send_document_with_actions(
        self,
        chat_id: str,
        archive: Path,
        *,
        caption: str,
        job_url: str | None,
        row_id: int,
        run_id: str,
    ) -> str:
        caption = self._visible_caption(caption)
        with archive.open("rb") as handle:
            payload = self._post(
                "/sendDocument",
                data={
                    "chat_id": chat_id,
                    "caption": caption[:1024],
                    "reply_markup": json.dumps(self._actions(job_url, row_id, run_id)),
                },
                files={"document": (archive.name, handle, "application/zip")},
            )
        return self._message_id(payload, "Telegram returned no application archive message")

    def send_application_bundle(
        self,
        chat_id: str,
        artifacts: Iterable[Path],
        *,
        caption: str,
        job_url: str | None,
        row_id: int,
        run_id: str,
    ) -> str:
        paths = list(artifacts)
        archives = [path for path in paths if path.suffix.casefold() == ".zip"]
        if len(archives) != 1:
            raise ValueError("Telegram application bundle requires exactly one ZIP archive")
        return self.send_document_with_actions(
            chat_id,
            archives[0],
            caption=caption,
            job_url=job_url,
            row_id=row_id,
            run_id=run_id,
        )

    def send_application_actions(
        self,
        chat_id: str,
        *,
        caption: str,
        job_url: str | None,
        row_id: int,
        run_id: str,
    ) -> str:
        caption = self._visible_caption(caption)
        payload = self._post(
            "/sendMessage",
            json={
                "chat_id": chat_id,
                "text": caption[:4096],
                "reply_markup": self._actions(job_url, row_id, run_id),
            },
        )
        return self._message_id(payload, "Telegram returned no application action message")
