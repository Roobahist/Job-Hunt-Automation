from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from job_hunt.application.job_equivalence import (
    best_processed_equivalent,
    equivalent_rows,
    job_from_baserow_row,
    jobs_equivalent,
    select_option_id,
    stored_score,
)
from job_hunt.domain.models import Job, Qualification
from job_hunt.errors import ErrorKind, ProviderError
from job_hunt.integrations.discovery_skip import recent_baserow_rows
from job_hunt.integrations.http import raise_provider_error
from job_hunt.retry import retry_transient
from job_hunt.url_utils import valid_http_url

LOG_FIELD = "Log"
_DERIVED_FILE_FIELDS = ("CV", "Cover Letter", "Application ZIP")


def _row_missing(exc: Exception) -> bool:
    return isinstance(exc, ProviderError) and exc.provider == "baserow" and exc.status_code == 404


class BaserowClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.baserow.io",
        client: httpx.Client | None = None,
        *,
        timeout_seconds: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._base_origin = urlsplit(self._base_url)
        self._client = client or httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Token {token}"},
            timeout=timeout_seconds,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise ProviderError(
                f"Baserow network request failed: {exc}",
                ErrorKind.TRANSIENT_PROVIDER,
                retryable=True,
                provider="baserow",
            ) from exc
        raise_provider_error("baserow", response)
        return response

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            content_type = response.headers.get("content-type", "unknown")
            preview = response.text[:200].replace("\n", " ").strip() or "<empty body>"
            raise ProviderError(
                f"Baserow returned a non-JSON response (status={response.status_code}, "
                f"content_type={content_type}, body={preview!r})",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                retryable=True,
                provider="baserow",
                status_code=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError(
                "Baserow returned JSON that was not an object",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                retryable=True,
                provider="baserow",
                status_code=response.status_code,
            )
        return cast(dict[str, Any], payload)

    def _row_page(self, url: str, query: Mapping[str, Any]) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {"params": dict(query)} if query else {}
        response = self._request("GET", url, **request_kwargs)
        payload = self._json_object(response)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ProviderError(
                "Baserow row page is missing a results array",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                retryable=True,
                provider="baserow",
                status_code=response.status_code,
            )
        return payload

    def _pagination_url(self, value: object) -> str | None:
        if not value:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        parsed = urlsplit(raw)
        if not parsed.scheme and not parsed.netloc:
            return raw
        if parsed.hostname != self._base_origin.hostname:
            raise ProviderError(
                f"Baserow pagination URL changed host from {self._base_origin.hostname!r} to {parsed.hostname!r}",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                provider="baserow",
            )
        path = parsed.path or "/"
        return f"{path}?{parsed.query}" if parsed.query else path

    def iter_rows(self, table_id: int, **params: object) -> Iterator[dict[str, Any]]:
        url: str | None = f"/api/database/rows/table/{table_id}/"
        query: dict[str, Any] = {"user_field_names": "true", "size": 200, **params}
        while url:
            payload = retry_transient(self._row_page, url, query)
            yield from payload["results"]
            url = self._pagination_url(payload.get("next"))
            query = {}

    def get_row(self, table_id: int, row_id: int) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/api/database/rows/table/{table_id}/{row_id}/?user_field_names=true",
        )
        return self._json_object(response)

    def find_equal(self, table_id: int, field: str, value: object) -> dict[str, Any] | None:
        rows = list(self.iter_rows(table_id, **{f"filter__{field}__equal": value, "size": 2}))
        return rows[0] if rows else None

    def create_row(self, table_id: int, values: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST", f"/api/database/rows/table/{table_id}/?user_field_names=true", json=dict(values)
        )
        return self._json_object(response)

    def update_row(self, table_id: int, row_id: int, values: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request(
            "PATCH",
            f"/api/database/rows/table/{table_id}/{row_id}/?user_field_names=true",
            json=dict(values),
        )
        return self._json_object(response)

    def delete_row(self, table_id: int, row_id: int) -> None:
        try:
            self._request("DELETE", f"/api/database/rows/table/{table_id}/{row_id}/")
        except ProviderError as exc:
            if _row_missing(exc):
                return
            raise

    def upload_via_url(self, url: str) -> dict[str, Any]:
        response = self._request("POST", "/api/user-files/upload-via-url/", json={"url": url})
        return self._json_object(response)

    def upload_file(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("rb") as handle:
                response = self._request(
                    "POST",
                    "/api/user-files/upload-file/",
                    files={"file": (path.name, handle, "application/octet-stream")},
                )
        except OSError as exc:
            raise ProviderError(
                f"Could not read artifact for Baserow upload: {path}",
                ErrorKind.VALIDATION,
                provider="baserow",
            ) from exc
        return self._json_object(response)

    def list_fields(self, table_id: int) -> list[dict[str, Any]]:
        response = self._request("GET", f"/api/database/fields/table/{table_id}/")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                "Baserow returned a non-JSON field list",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                retryable=True,
                provider="baserow",
                status_code=response.status_code,
            ) from exc
        if not isinstance(payload, list):
            raise ProviderError(
                "Baserow field list response was not an array",
                ErrorKind.MALFORMED_PROVIDER_RESPONSE,
                retryable=True,
                provider="baserow",
                status_code=response.status_code,
            )
        return cast(list[dict[str, Any]], payload)


class BaserowJobRepository:
    def __init__(
        self,
        client: BaserowClient,
        table_id: int,
        status_options: Mapping[str, int],
        contract_type_options: Mapping[str, int] | None = None,
        to_apply_score_threshold: int | None = None,
        equivalence_lookback_seconds: int = 0,
    ) -> None:
        self.client = client
        self.table_id = table_id
        self.status_options = status_options
        self.contract_type_options = contract_type_options or {}
        self.to_apply_score_threshold = to_apply_score_threshold
        self.equivalence_lookback_seconds = equivalence_lookback_seconds
        self._discovery_representatives: list[Job] = []
        self._discovery_recent_rows: list[Mapping[str, Any]] | None = None

    @staticmethod
    def _display_job_id(job: Job) -> str:
        if job.external_id:
            return job.external_id
        return str(job.internal_id)

    @staticmethod
    def _file_references(value: object) -> object:
        if not isinstance(value, list):
            return value
        references: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                references.append({"name": name.strip()})
        return references

    @staticmethod
    def _reuse_log_text(target: Mapping[str, Any], source: Mapping[str, Any]) -> str:
        timestamp = datetime.now(UTC).isoformat()
        source_row_id = source.get("id")
        source_job_id = str(source.get("Job ID") or "unknown").strip() or "unknown"
        target_job_id = str(target.get("Job ID") or "unknown").strip() or "unknown"
        return (
            f"{timestamp} INFO equivalent_job_result_reused source_row_id={source_row_id} "
            f"source_job_id={source_job_id} target_job_id={target_job_id}\n"
            f"{timestamp} INFO reuse_notice This job was not independently processed. "
            "It was identified as equivalent to another job in the short reuse window; "
            "qualification score/status and generated application files are reused from "
            "the source job as they become available.\n"
        )

    def _reuse_log_file(self, target: Mapping[str, Any], source: Mapping[str, Any]) -> list[dict[str, Any]]:
        target_id = target.get("id")
        target_job_id = str(target.get("Job ID") or target_id or "unknown").strip() or "unknown"
        with TemporaryDirectory(prefix="job-hunt-reuse-") as directory:
            path = Path(directory) / f"job-{target_job_id}-reuse.log"
            path.write_text(self._reuse_log_text(target, source), encoding="utf-8")
            uploaded = self.client.upload_file(path)
        return [uploaded]

    def _recent_rows(self) -> list[Mapping[str, Any]]:
        if self.equivalence_lookback_seconds <= 0:
            return []
        return recent_baserow_rows(
            self.client,
            self.table_id,
            lookback_seconds=self.equivalence_lookback_seconds,
        )

    def _recent_discovery_rows(self) -> list[Mapping[str, Any]]:
        if self._discovery_recent_rows is None:
            self._discovery_recent_rows = self._recent_rows()
        return self._discovery_recent_rows

    def _result_fields(self, row: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        status_id = select_option_id(row.get("Status"))
        if status_id is not None:
            values["Status"] = status_id
        score = stored_score(row)
        if score is not None:
            values["Score"] = score
        for field in _DERIVED_FILE_FIELDS:
            raw = row.get(field)
            if raw:
                values[field] = self._file_references(raw)
        return values

    def _safe_target_values(self, target: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        new_status_id = self.status_options.get("new")
        target_status_id = select_option_id(target.get("Status"))
        for field, value in values.items():
            if field == "Status":
                if new_status_id is not None and target_status_id == new_status_id:
                    safe[field] = value
            elif field == "Score":
                if stored_score(target) is None or (new_status_id is not None and target_status_id == new_status_id):
                    safe[field] = value
            elif field in _DERIVED_FILE_FIELDS and not target.get(field):
                safe[field] = self._file_references(value)
        return safe

    def _propagate_result(self, row_id: int, values: Mapping[str, Any]) -> None:
        if self.equivalence_lookback_seconds <= 0 or not values:
            return
        source = self.client.get_row(self.table_id, row_id)
        source_job = job_from_baserow_row(source)
        if source_job is None:
            return
        for target in equivalent_rows(source_job, self._recent_rows()):
            target_id = target.get("id")
            if not isinstance(target_id, int) or isinstance(target_id, bool) or target_id == row_id:
                continue
            safe = self._safe_target_values(target, values)
            if safe and not target.get(LOG_FIELD):
                safe[LOG_FIELD] = self._reuse_log_file(target, source)
            if safe:
                self.client.update_row(self.table_id, target_id, safe)

    def _create_discovery_duplicate(self, job: Job, source: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        values = self._job_fields(job, include_status=True)
        if source is not None:
            values.update(self._result_fields(source))
            log_target = {**values, "Job ID": self._display_job_id(job)}
            values[LOG_FIELD] = self._reuse_log_file(log_target, source)
        created = self.client.create_row(self.table_id, values)
        if self._discovery_recent_rows is not None:
            self._discovery_recent_rows.append(created)
        return created

    def find_reference(self, *, external_id: str | None = None, url: str | None = None) -> Mapping[str, Any] | None:
        if external_id:
            found = self.client.find_equal(self.table_id, "Job ID", str(external_id))
            if found:
                return found
        job_url = valid_http_url(url or "")
        if job_url is None:
            return None
        return self.client.find_equal(self.table_id, "Link", job_url)

    def find(self, job: Job) -> Mapping[str, Any] | None:
        exact = self.find_reference(external_id=job.external_id, url=job.url)
        if exact is not None:
            return exact
        if self.equivalence_lookback_seconds <= 0 or job.provider_data.get("_scheduled_discovery") is not True:
            return None

        recent = self._recent_discovery_rows()
        processed = best_processed_equivalent(
            job,
            recent,
            new_status_id=self.status_options["new"],
        )
        if processed is not None:
            return self._create_discovery_duplicate(job, processed)

        if any(jobs_equivalent(job, representative) for representative in self._discovery_representatives):
            return self._create_discovery_duplicate(job)

        self._discovery_representatives.append(job)
        return None

    def create(self, job: Job) -> Mapping[str, Any]:
        return self.client.create_row(self.table_id, self._job_fields(job, include_status=True))

    def reset(self, row_id: int, job: Job) -> Mapping[str, Any]:
        return self.client.update_row(self.table_id, row_id, self._job_fields(job))

    def clear_qualification(self, row_id: int) -> None:
        self.client.update_row(self.table_id, row_id, {"Score": None})

    def save_qualification(self, row_id: int, result: Qualification) -> None:
        values: dict[str, Any] = {"Score": result.score}
        if self.to_apply_score_threshold is not None and result.score >= self.to_apply_score_threshold:
            values["Status"] = self.status_options["toApply"]
        self.client.update_row(self.table_id, row_id, values)
        self._propagate_result(row_id, values)

    def save_artifacts(self, row_id: int, uploaded_files: Mapping[str, Any]) -> None:
        values = dict(uploaded_files)
        self.client.update_row(self.table_id, row_id, values)
        self._propagate_result(row_id, values)

    def save_log(self, row_id: int, path: Path) -> None:
        # The Jobs schema is managed explicitly. Runtime log attachment should only
        # require ordinary row/file permissions, not database schema permissions.
        uploaded = self.client.upload_file(path)
        values = {LOG_FIELD: [uploaded]}
        self.client.update_row(self.table_id, row_id, values)

    def set_status(self, row_id: int, status: str) -> None:
        if status not in self.status_options:
            raise KeyError(f"Unknown status key: {status}")
        values = {"Status": self.status_options[status]}
        self.client.update_row(self.table_id, row_id, values)
        self._propagate_result(row_id, values)

    def has_status(self, row_id: int, status: str) -> bool:
        if status not in self.status_options:
            raise KeyError(f"Unknown status key: {status}")
        row = self.client.get_row(self.table_id, row_id)
        current = row.get("Status")
        expected = self.status_options[status]
        if isinstance(current, Mapping):
            return current.get("id") == expected
        if isinstance(current, int):
            return current == expected
        return False

    def _has_status_if_present(self, row_id: int, status: str) -> bool:
        try:
            return bool(retry_transient(self.has_status, row_id, status))
        except ProviderError as exc:
            if _row_missing(exc):
                return False
            raise

    def _delete_row_idempotently(self, row_id: int) -> bool:
        had_ambiguous_failure = False

        def delete_once() -> bool:
            nonlocal had_ambiguous_failure
            try:
                self.client.delete_row(self.table_id, row_id)
                return True
            except ProviderError as exc:
                if _row_missing(exc):
                    return had_ambiguous_failure
                if exc.retryable:
                    had_ambiguous_failure = True
                raise

        return bool(retry_transient(delete_once))

    def delete_status(self, status: str) -> int:
        if status not in self.status_options:
            raise KeyError(f"Unknown status key: {status}")
        expected = self.status_options[status]
        row_ids: list[int] = []
        seen_row_ids: set[int] = set()
        for row in self.client.iter_rows(self.table_id):
            row_id = row.get("id")
            if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id in seen_row_ids:
                continue
            if select_option_id(row.get("Status")) == expected:
                row_ids.append(row_id)
                seen_row_ids.add(row_id)

        deleted = 0
        for row_id in row_ids:
            if not self._has_status_if_present(row_id, status):
                continue
            if self._delete_row_idempotently(row_id):
                deleted += 1
        return deleted

    def _job_fields(self, job: Job, *, include_status: bool = False) -> dict[str, Any]:
        normalized_contract = "".join(
            character for character in (job.contract_type or "").casefold() if character.isalnum()
        )
        aliases = {
            "fulltime": "fullTime",
            "parttime": "partTime",
            "contract": "contract",
            "temporary": "temporary",
            "internship": "internship",
            "volunteer": "volunteer",
        }
        contract_key = aliases.get(normalized_contract)
        contract_value: object = self.contract_type_options.get(contract_key) if contract_key else None
        values: dict[str, Any] = {
            "Job ID": self._display_job_id(job),
            "Company Name": job.company_name,
            "Title": job.title,
            "Location": job.location or "",
            "Job Description": job.description,
            "Contract Type": contract_value,
            "Link": valid_http_url(job.url),
            "Date": job.published_at.date().isoformat() if job.published_at else None,
        }
        if include_status:
            values["Status"] = self.status_options["new"]
        return values
