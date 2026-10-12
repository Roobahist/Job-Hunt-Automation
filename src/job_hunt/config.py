from __future__ import annotations

import csv
import json
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from job_hunt.errors import ConfigurationError

_SUPPORTED_DISCOVERY_STRATEGIES = {"roobahist", "brightdata", "coregent", "legacy"}
_SUPPORTED_APIFY_DISCOVERY_STRATEGIES = {"coregent", "legacy"}
_SUPPORTED_LINKEDIN_JOB_FETCH_STRATEGIES = {"roobahist", "apify", "brightdata"}
LINKEDIN_BASE_SEARCH_URL = "https://www.linkedin.com/jobs/search/"
LINKEDIN_JOB_URL_TEMPLATE = "https://www.linkedin.com/jobs/view/{jobId}"
_CODE_OWNED_CONFIG_KEYS = {"linkedin_base_search_url", "linkedin_job_url_template"}

# These IDs belong to the one application-owned Fillout template in config/fillout.
# They are code-owned contract data, not tenant configuration.
_FILLOUT_FIELD_IDS = {
    "entryType": "entryType",
    "linkedinJobId": "demoFielda903bb44",
    "indeedJobId": "demoField2d374ace",
    "companyName": "demoFieldbfe4c4e1",
    "jobTitle": "demoFielde14b9014",
    "jobDescription": "demoField11f633b8",
    "jobUrl": "demoField36472c85",
    "location": "demoField0d32379e",
    "contractType": "demoField8cf83086",
    "publishedAt": "demoFieldad51d672",
    "externalJobId": "demoFieldd1f77d14",
    "pageContent": "demoFielddfea9138",
}


class LlmRoute(BaseModel):
    provider: str
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JOB_HUNT_", env_file=".env", extra="ignore")

    environment: str = "development"
    debug: bool = False
    registry_path: Path = Path("config/users.toml")
    artifact_root: Path = Path("runs")
    redis_url: str = "redis://localhost:6379/0"
    operator_token: str = ""
    run_ttl_seconds: int = 604800
    discovery_snapshot_ttl_seconds: int = Field(default=7200, ge=300)
    llm_checkpoint_ttl_seconds: int = Field(default=604800, ge=300)
    llm_parallelism: int = Field(default=3, ge=1, le=8)
    scheduler_timezone: str = "America/Edmonton"
    end_of_day_catchup_enabled: bool = True
    end_of_day_dropped_cleanup_enabled: bool = True

    # Operational timing is environment-configurable. A task time limit of zero disables
    # Celery's global deadline so long document jobs are not killed mid-generation.
    litellm_request_timeout_seconds: int = Field(default=360, ge=1)
    telegram_request_timeout_seconds: int = Field(default=120, ge=1)
    baserow_request_timeout_seconds: int = Field(default=60, ge=1)
    latex_compile_timeout_seconds: int = Field(default=180, ge=1)
    job_lock_timeout_seconds: int = Field(default=1800, ge=60)
    task_soft_time_limit_seconds: int = Field(default=0, ge=0)
    task_time_limit_seconds: int = Field(default=0, ge=0)
    task_max_retries: int = Field(default=8, ge=0)
    rate_limit_fallback_seconds: int = Field(default=65, ge=1)
    transient_base_delay_seconds: int = Field(default=5, ge=1)
    transient_max_delay_seconds: int = Field(default=300, ge=1)

    discovery_order: str = "roobahist,brightdata,coregent,legacy"
    linkedin_job_fetch_order: str = "roobahist,apify,brightdata"
    apify_tokens: str = ""
    apify_discovery_order: str = "coregent,legacy"
    brightdata_api_keys: str = ""
    brightdata_linkedin_jobs_dataset_id: str = ""
    brightdata_indeed_jobs_dataset_id: str = ""
    brightdata_request_timeout_seconds: int = Field(default=75, ge=1)
    brightdata_snapshot_timeout_seconds: int = Field(default=300, ge=1)
    brightdata_snapshot_poll_seconds: int = Field(default=5, ge=1)
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = ""

    # Job-hunt workers call a centralized LiteLLM Proxy using logical model aliases.
    litellm_base_url: str = "http://litellm:4000"
    litellm_api_key: str = ""
    langfuse_enabled: bool = False
    llm_routes: str = ""
    llm_repair_routes: str = ""

    provider_quota_cooldown_seconds: int = Field(default=3600, ge=1)

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @staticmethod
    def _validate_strategy_order(strategies: list[str], *, variable_name: str, supported: set[str]) -> list[str]:
        if not strategies:
            raise ConfigurationError(f"{variable_name} must contain at least one strategy")
        if len(set(strategies)) != len(strategies):
            raise ConfigurationError(f"{variable_name} must not contain duplicate strategies")
        unknown = set(strategies) - supported
        if unknown:
            raise ConfigurationError(f"{variable_name} contains unknown strategies: {sorted(unknown)}")
        return strategies

    def discovery_strategies(self) -> list[str]:
        return self._validate_strategy_order(
            self._split_csv(self.discovery_order),
            variable_name="JOB_HUNT_DISCOVERY_ORDER",
            supported=_SUPPORTED_DISCOVERY_STRATEGIES,
        )

    def linkedin_job_fetch_strategies(self) -> list[str]:
        return self._validate_strategy_order(
            self._split_csv(self.linkedin_job_fetch_order),
            variable_name="JOB_HUNT_LINKEDIN_JOB_FETCH_ORDER",
            supported=_SUPPORTED_LINKEDIN_JOB_FETCH_STRATEGIES,
        )

    def shared_apify_tokens(self) -> list[str]:
        tokens = self._split_csv(self.apify_tokens)
        if not tokens:
            raise ConfigurationError("JOB_HUNT_APIFY_TOKENS must contain at least one token")
        return tokens

    def apify_discovery_strategies(self) -> list[str]:
        return self._validate_strategy_order(
            self._split_csv(self.apify_discovery_order),
            variable_name="JOB_HUNT_APIFY_DISCOVERY_ORDER",
            supported=_SUPPORTED_APIFY_DISCOVERY_STRATEGIES,
        )

    def shared_brightdata_api_keys(self) -> list[str]:
        return self._split_csv(self.brightdata_api_keys)

    def shared_brightdata_linkedin_jobs_dataset_id(self) -> str:
        dataset_id = self.brightdata_linkedin_jobs_dataset_id.strip()
        if not dataset_id:
            raise ConfigurationError(
                "JOB_HUNT_BRIGHTDATA_LINKEDIN_JOBS_DATASET_ID must be set when Bright Data is configured"
            )
        return dataset_id

    def shared_brightdata_indeed_jobs_dataset_id(self) -> str:
        dataset_id = self.brightdata_indeed_jobs_dataset_id.strip()
        if not dataset_id:
            raise ConfigurationError(
                "JOB_HUNT_BRIGHTDATA_INDEED_JOBS_DATASET_ID must be set when Bright Data is configured"
            )
        return dataset_id

    def shared_litellm_key(self) -> str:
        if not self.litellm_api_key:
            raise ConfigurationError("JOB_HUNT_LITELLM_API_KEY must be set")
        return self.litellm_api_key

    def litellm_key_for_tenant(self, tenant: str | None) -> str:
        if tenant:
            normalized = re.sub(r"[^A-Za-z0-9]+", "_", tenant).strip("_").upper()
            if normalized:
                tenant_key = os.getenv(f"JOB_HUNT_LITELLM_TENANT_KEY_{normalized}", "").strip()
                if tenant_key:
                    return tenant_key
        return self.shared_litellm_key()

    def shared_telegram_token(self) -> str:
        if not self.telegram_bot_token:
            raise ConfigurationError("JOB_HUNT_TELEGRAM_BOT_TOKEN must be set")
        return self.telegram_bot_token

    def shared_telegram_webhook_secret(self) -> str:
        if not self.telegram_webhook_secret:
            raise ConfigurationError("JOB_HUNT_TELEGRAM_WEBHOOK_SECRET must be set")
        return self.telegram_webhook_secret

    @classmethod
    def _gateway_routes(cls, raw: str, variable_name: str) -> list[LlmRoute]:
        routes: list[LlmRoute] = []
        for alias in cls._split_csv(raw):
            if ":" in alias:
                raise ConfigurationError(
                    f"{variable_name} must contain LiteLLM model aliases, not provider:model entries: {alias}"
                )
            routes.append(LlmRoute(provider="litellm", model=alias))
        return routes

    def llm_route_specs(self) -> list[LlmRoute]:
        routes = self._gateway_routes(self.llm_routes, "JOB_HUNT_LLM_ROUTES")
        if not routes:
            raise ConfigurationError("JOB_HUNT_LLM_ROUTES must contain at least one LiteLLM model alias")
        return routes

    def llm_repair_route_specs(self) -> list[LlmRoute]:
        routes = self._gateway_routes(self.llm_repair_routes, "JOB_HUNT_LLM_REPAIR_ROUTES")
        if not routes:
            raise ConfigurationError("JOB_HUNT_LLM_REPAIR_ROUTES must contain at least one LiteLLM model alias")
        return routes


class SecretAliases(BaseModel):
    baserow: str
    fillout: str


class TenantBootstrap(BaseModel):
    key: str
    enabled: bool = True
    renderer: str
    config_table_id: int = Field(ge=0)
    baserow_base_url: str = "https://api.baserow.io"
    tenant_root: Path
    secrets: SecretAliases

    def secret(self, name: str, *, required: bool = True) -> str:
        alias = getattr(self.secrets, name)
        if not alias:
            if required:
                raise ConfigurationError(f"Missing secret alias for {name} on tenant {self.key}")
            return ""
        value = os.getenv(alias, "")
        if required and not value:
            raise ConfigurationError(f"Missing {alias} for tenant {self.key}")
        return value


class TenantRuntimeConfig(BaseModel):
    tenant_key: str
    tenant_enabled: bool = True
    applicant_filename: str
    baserow_table_ids: dict[str, int]
    contract_type_option_ids: dict[str, int] = Field(default_factory=dict)
    status_option_ids: dict[str, int]
    apify_proxy_country: str = "ca"
    apify_max_concurrency: int = Field(default=1, ge=1)
    linkedin_max_items: int = Field(default=150, ge=1)
    linkedin_schedule_interval_hours: int = Field(default=1, ge=1)
    linkedin_missed_beat_recovery_hours: int = Field(default=6, ge=1)
    linkedin_end_of_day_catchup_enabled: bool = True
    end_of_day_dropped_cleanup_enabled: bool = True
    company_inclusions: list[str] = Field(default_factory=list)
    company_exclusions: list[str] = Field(default_factory=list)
    title_inclusions: list[str] = Field(default_factory=list)
    title_exclusions: list[str] = Field(default_factory=list)
    description_inclusions: list[str] = Field(default_factory=list)
    description_exclusions: list[str] = Field(default_factory=list)
    qualification_threshold: int = Field(default=33, ge=0, le=100)
    to_apply_score_threshold: int = Field(default=70, ge=0, le=100)
    telegram_chat_id: str
    project_selection_count: int | None = None
    work_experience_selection_count: int = Field(default=3, ge=1)

    @property
    def linkedin_base_search_url(self) -> str:
        return LINKEDIN_BASE_SEARCH_URL

    @property
    def linkedin_job_url_template(self) -> str:
        return LINKEDIN_JOB_URL_TEMPLATE

    @property
    def fillout_field_ids(self) -> dict[str, str]:
        return dict(_FILLOUT_FIELD_IDS)

    @model_validator(mode="after")
    def table_contract(self) -> TenantRuntimeConfig:
        missing = {"jobs", "searchCriteria", "prompts"} - self.baserow_table_ids.keys()
        if missing:
            raise ValueError(f"baserow_table_ids missing: {sorted(missing)}")
        return self


def load_registry(path: Path) -> dict[str, TenantBootstrap]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle).get("users", {})
    result: dict[str, TenantBootstrap] = {}
    for key, value in raw.items():
        aliases = SecretAliases(
            baserow=value["baserow_token_env"],
            fillout=value["fillout_secret_env"],
        )
        result[key] = TenantBootstrap(
            key=key,
            secrets=aliases,
            **{k: v for k, v in value.items() if not k.endswith("_env")},
        )
    return result


def decode_config_value(value_type: str, value: str) -> Any:
    match value_type:
        case "boolean":
            return value.strip().lower() in {"true", "1", "yes"}
        case "number":
            number = float(value)
            return int(number) if number.is_integer() else number
        case "json":
            stripped = value.strip()
            return json.loads(stripped) if stripped else []
        case _:
            return value


def parse_configuration_rows(rows: list[Mapping[str, Any]]) -> TenantRuntimeConfig:
    values: dict[str, Any] = {}
    for row in rows:
        if not bool(row.get("enabled", True)):
            continue
        key = str(row.get("configKey", "")).strip()
        if key in _CODE_OWNED_CONFIG_KEYS:
            continue
        if key:
            values[key] = decode_config_value(str(row.get("valueType", "text")), str(row["value"]))
    try:
        return TenantRuntimeConfig.model_validate(values)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid tenant configuration: {exc}") from exc


def read_seed(path: Path) -> TenantRuntimeConfig:
    with path.open(newline="", encoding="utf-8") as handle:
        return parse_configuration_rows(list(csv.DictReader(handle)))
