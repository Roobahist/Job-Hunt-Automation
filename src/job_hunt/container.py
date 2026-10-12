from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redis import Redis

from job_hunt.application.compatibility import GeminiCompatibilityFilter
from job_hunt.application.normalization import SubmissionNormalizer
from job_hunt.application.workflow import ApplicationWorkflow
from job_hunt.config import Settings, TenantRuntimeConfig, load_registry
from job_hunt.domain.models import PromptDefinition
from job_hunt.integrations.apify import ApifyProvider
from job_hunt.integrations.apify_actors import (
    INDEED_SINGLE_JOB_ACTOR,
    LINKEDIN_COREGENT_SEARCH_ACTOR,
    LINKEDIN_LEGACY_SEARCH_ACTOR,
    LINKEDIN_ROOBAHIST_ACTOR,
    LINKEDIN_SINGLE_JOB_ACTOR,
)
from job_hunt.integrations.apify_indeed import ApifyIndeedProvider
from job_hunt.integrations.apify_linkedin_roobahist import RoobahistLinkedInProvider
from job_hunt.integrations.artifacts import BaserowArtifactPublisher
from job_hunt.integrations.baserow import BaserowClient, BaserowJobRepository
from job_hunt.integrations.brightdata_resilient import ResilientBrightDataProvider
from job_hunt.integrations.configuration import (
    ATLAS_PROMPT_KEYS,
    NOVA_PROMPT_KEYS,
    BaserowConfigurationRepository,
    validate_prompt_contract,
)
from job_hunt.integrations.constrained_workflow_ai import (
    ConstrainedParallelGeminiWorkflowAI,
    ConstrainedParallelAtlasGeminiWorkflowAI,
)
from job_hunt.integrations.discovery_filters import FilteredDiscoveryProvider
from job_hunt.integrations.discovery_router import DiscoveryRouter
from job_hunt.integrations.discovery_skip import recent_baserow_linkedin_job_ids
from job_hunt.integrations.gemini import GeminiWorkflowAI
from job_hunt.integrations.indeed_router import IndeedFetchRouter
from job_hunt.integrations.job_extractor import StructuredJobExtractor
from job_hunt.integrations.llm_routing_runtime import build_routed_structured_client
from job_hunt.integrations.telegram import TelegramNotifier
from job_hunt.ports import DiscoveryProvider
from job_hunt.state import RedisState
from job_hunt.tenants.registry import TenantContext, TenantRegistry

_END_OF_DAY_ROOBAHIST_TIMEOUT_SECONDS = 0
_END_OF_DAY_ROOBAHIST_MAX_TOTAL_CHARGE_USD = 0.1


@dataclass(slots=True)
class TenantServices:
    context: TenantContext
    config: TenantRuntimeConfig
    baserow: BaserowClient
    config_repository: BaserowConfigurationRepository
    prompts: dict[str, PromptDefinition]
    repository: BaserowJobRepository
    normalizer: SubmissionNormalizer
    workflow: ApplicationWorkflow
    discovery: DiscoveryProvider
    end_of_day_discovery: RoobahistLinkedInProvider
    notifier: TelegramNotifier

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.model_dump(mode="json"),
            "prompts": {key: definition.model_dump(mode="json") for key, definition in self.prompts.items()},
        }


@dataclass(slots=True)
class TelegramRoute:
    tenant: str
    config: TenantRuntimeConfig
    repository: BaserowJobRepository
    notifier: TelegramNotifier


class Container:
    def __init__(self, settings: Settings | None = None, project_root: Path = Path(".")) -> None:
        self.settings = settings or Settings()
        self.project_root = project_root
        self.registry = TenantRegistry(
            load_registry(self.settings.registry_path),
            project_root,
            latex_compile_timeout_seconds=self.settings.latex_compile_timeout_seconds,
        )
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        self.state = RedisState(self.redis)

    def _baserow(self, token: str, base_url: str) -> BaserowClient:
        return BaserowClient(
            token,
            base_url,
            timeout_seconds=self.settings.baserow_request_timeout_seconds,
        )

    def _telegram(self) -> TelegramNotifier:
        return TelegramNotifier(
            self.settings.shared_telegram_token(),
            timeout_seconds=self.settings.telegram_request_timeout_seconds,
            debug=self.settings.debug,
        )

    def telegram_route(self, chat_id: str) -> TelegramRoute | None:
        notifier = self._telegram()
        for tenant, bootstrap in self.registry.bootstraps.items():
            if not bootstrap.enabled:
                continue
            baserow = self._baserow(bootstrap.secret("baserow"), bootstrap.baserow_base_url)
            config_repository = BaserowConfigurationRepository(baserow, bootstrap.config_table_id)
            config = config_repository.load()
            if str(config.telegram_chat_id) != chat_id:
                continue
            repository = BaserowJobRepository(
                baserow,
                config.baserow_table_ids["jobs"],
                config.status_option_ids,
                config.contract_type_option_ids,
                config.to_apply_score_threshold,
            )
            return TelegramRoute(tenant, config, repository, notifier)
        return None

    def tenant(
        self,
        key: str,
        snapshot: dict[str, Any] | None = None,
        checkpoint_namespace: str | None = None,
    ) -> TenantServices:
        context = self.registry.get(key)
        bootstrap = context.bootstrap
        baserow = self._baserow(bootstrap.secret("baserow"), bootstrap.baserow_base_url)
        config_repository = BaserowConfigurationRepository(baserow, bootstrap.config_table_id)

        if snapshot is None:
            config = config_repository.load()
            config_repository.validate_job_table(config.baserow_table_ids["jobs"])
            prompts = config_repository.prompts(config.baserow_table_ids["prompts"])
        else:
            config = TenantRuntimeConfig.model_validate(snapshot.get("config"))
            raw_prompts = snapshot.get("prompts")
            if not isinstance(raw_prompts, dict):
                raise ValueError("Discovery snapshot has no prompt definitions")
            prompts = {
                str(prompt_key): PromptDefinition.model_validate(value) for prompt_key, value in raw_prompts.items()
            }

        if config.tenant_key not in {key, key.replace("_", "-")} and not config.tenant_key.startswith(key):
            raise ValueError(f"Registry tenant {key} does not match Baserow tenant_key {config.tenant_key}")

        jobs_table_id = config.baserow_table_ids["jobs"]
        repository = BaserowJobRepository(
            baserow,
            jobs_table_id,
            config.status_option_ids,
            config.contract_type_option_ids,
            config.to_apply_score_threshold,
            equivalence_lookback_seconds=config.linkedin_schedule_interval_hours * 2 * 3600,
        )
        apify_tokens = self.settings.shared_apify_tokens()
        apify = ApifyProvider(
            apify_tokens,
            LINKEDIN_LEGACY_SEARCH_ACTOR,
            LINKEDIN_SINGLE_JOB_ACTOR,
            search_actor_ids={
                "coregent": LINKEDIN_COREGENT_SEARCH_ACTOR,
                "legacy": LINKEDIN_LEGACY_SEARCH_ACTOR,
            },
            discovery_order=self.settings.apify_discovery_strategies(),
            skip_job_ids_provider=lambda: recent_baserow_linkedin_job_ids(
                baserow,
                jobs_table_id,
                lookback_hours=config.linkedin_schedule_interval_hours * 2,
            ),
            company_exclusions=config.company_exclusions,
            title_exclusions=config.title_exclusions,
            quota_cooldown_seconds=self.settings.provider_quota_cooldown_seconds,
            state=self.state,
        )
        roobahist_linkedin = RoobahistLinkedInProvider(
            apify,
            LINKEDIN_ROOBAHIST_ACTOR,
            company_inclusions=config.company_inclusions,
            company_exclusions=config.company_exclusions,
            title_inclusions=config.title_inclusions,
            title_exclusions=config.title_exclusions,
            description_inclusions=config.description_inclusions,
            description_exclusions=config.description_exclusions,
        )
        end_of_day_linkedin = RoobahistLinkedInProvider(
            apify,
            LINKEDIN_ROOBAHIST_ACTOR,
            company_inclusions=config.company_inclusions,
            company_exclusions=config.company_exclusions,
            title_inclusions=config.title_inclusions,
            title_exclusions=config.title_exclusions,
            description_inclusions=config.description_inclusions,
            description_exclusions=config.description_exclusions,
            run_timeout_seconds=_END_OF_DAY_ROOBAHIST_TIMEOUT_SECONDS,
            run_max_total_charge_usd=_END_OF_DAY_ROOBAHIST_MAX_TOTAL_CHARGE_USD,
        )
        indeed_apify = ApifyIndeedProvider(
            apify_tokens,
            actor_id=INDEED_SINGLE_JOB_ACTOR,
            quota_cooldown_seconds=self.settings.provider_quota_cooldown_seconds,
            state=self.state,
        )
        brightdata_keys = self.settings.shared_brightdata_api_keys()
        brightdata = (
            ResilientBrightDataProvider(
                brightdata_keys,
                dataset_id=self.settings.shared_brightdata_linkedin_jobs_dataset_id(),
                company_exclusions=config.company_exclusions,
                title_exclusions=config.title_exclusions,
                request_timeout_seconds=self.settings.brightdata_request_timeout_seconds,
                snapshot_timeout_seconds=self.settings.brightdata_snapshot_timeout_seconds,
                snapshot_poll_seconds=self.settings.brightdata_snapshot_poll_seconds,
                quota_cooldown_seconds=self.settings.provider_quota_cooldown_seconds,
                state=self.state,
            )
            if brightdata_keys
            else None
        )
        indeed: Any = IndeedFetchRouter(
            indeed_apify,
            brightdata=None,
            country=config.apify_proxy_country,
            order=("apify",),
        )
        discovery_router = DiscoveryRouter(
            apify,
            roobahist=roobahist_linkedin,
            brightdata=brightdata,
            indeed=indeed,
            discovery_order=self.settings.discovery_strategies(),
            recent_job_ids_provider=lambda lookback_seconds: recent_baserow_linkedin_job_ids(
                baserow,
                jobs_table_id,
                lookback_seconds=lookback_seconds,
            ),
            default_window_seconds=config.linkedin_schedule_interval_hours * 3600,
            linkedin_fetch_order=self.settings.linkedin_job_fetch_strategies(),
        )
        discovery: DiscoveryProvider = FilteredDiscoveryProvider(
            discovery_router,
            company_inclusions=config.company_inclusions,
            company_exclusions=config.company_exclusions,
            title_inclusions=config.title_inclusions,
            title_exclusions=config.title_exclusions,
            description_inclusions=config.description_inclusions,
            description_exclusions=config.description_exclusions,
        )
        structured_client = build_routed_structured_client(
            self.settings,
            state=self.state.checkpoints(checkpoint_namespace),
            tenant=key,
            session_id=checkpoint_namespace,
        )
        compatibility_filter = GeminiCompatibilityFilter(structured_client)

        ai: GeminiWorkflowAI
        if bootstrap.renderer == "atlas":
            validate_prompt_contract(prompts, ATLAS_PROMPT_KEYS, "atlas")
            ai = ConstrainedParallelAtlasGeminiWorkflowAI(
                structured_client,
                prompts,
                project_selection_count=config.project_selection_count,
                work_experience_selection_count=config.work_experience_selection_count,
                parallelism=self.settings.llm_parallelism,
            )
        else:
            validate_prompt_contract(prompts, NOVA_PROMPT_KEYS, "nova")
            ai = ConstrainedParallelGeminiWorkflowAI(
                structured_client,
                prompts,
                project_selection_count=config.project_selection_count,
                work_experience_selection_count=config.work_experience_selection_count,
                parallelism=self.settings.llm_parallelism,
            )

        workflow = ApplicationWorkflow(
            repository,
            ai,
            ai,
            context.renderer,
            BaserowArtifactPublisher(baserow),
            self.settings.artifact_root,
            compatibility_filter=compatibility_filter,
            tenant_key=key,
            langfuse_enabled=self.settings.langfuse_enabled,
        )
        extractor = StructuredJobExtractor(
            structured_client,
            prompts["job_page_content_extraction"],
            tenant=key,
            session_id=checkpoint_namespace,
            langfuse_enabled=self.settings.langfuse_enabled,
        )
        normalizer = SubmissionNormalizer(discovery, extractor, config.linkedin_job_url_template)
        return TenantServices(
            context,
            config,
            baserow,
            config_repository,
            prompts,
            repository,
            normalizer,
            workflow,
            discovery,
            end_of_day_linkedin,
            self._telegram(),
        )
