from __future__ import annotations

import json
from collections.abc import Mapping

from job_hunt.domain.models import PromptDefinition
from job_hunt.errors import ConfigurationError
from job_hunt.integrations.gemini import GeminiWorkflowAI
from job_hunt.integrations.gemini_parallel import (
    ParallelGeminiWorkflowAI,
    ParallelAtlasGeminiWorkflowAI,
)


def _allowed_skills(values: Mapping[str, object]) -> list[str]:
    groups = values.get("skills_json")
    if not isinstance(groups, list):
        raise ConfigurationError("cv_skills_tailoring requires skills_json as an array")

    allowed: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ConfigurationError("cv_skills_tailoring skills_json entries must be objects")
        skills = group.get("skills")
        if not isinstance(skills, list):
            raise ConfigurationError("cv_skills_tailoring skills_json entries must contain a skills array")
        for skill in skills:
            if not isinstance(skill, str) or not skill.strip():
                raise ConfigurationError("cv_skills_tailoring allowed skills must be non-empty strings")
            exact = skill.strip()
            if exact not in seen:
                allowed.append(exact)
                seen.add(exact)

    if not allowed:
        raise ConfigurationError("cv_skills_tailoring requires at least one allowed skill")
    return allowed


def _skill_constrained_definition(
    definition: PromptDefinition,
    values: Mapping[str, object],
) -> PromptDefinition:
    runtime_definition = GeminiWorkflowAI._runtime_definition(definition, values)
    if definition.key != "cv_skills_tailoring":
        return runtime_definition

    allowed = _allowed_skills(values)
    schema = json.loads(json.dumps(runtime_definition.output_structure))
    properties = schema.get("properties")
    groups = properties.get("groups") if isinstance(properties, dict) else None
    group_items = groups.get("items") if isinstance(groups, dict) else None
    group_properties = group_items.get("properties") if isinstance(group_items, dict) else None
    skills = group_properties.get("skills") if isinstance(group_properties, dict) else None
    skill_items = skills.get("items") if isinstance(skills, dict) else None

    if not isinstance(groups, dict) or groups.get("type") != "array":
        raise ConfigurationError("cv_skills_tailoring output schema must define groups as an array")
    if not isinstance(group_items, dict) or group_items.get("type") != "object":
        raise ConfigurationError("cv_skills_tailoring output schema groups must contain objects")
    if not isinstance(skills, dict) or skills.get("type") != "array":
        raise ConfigurationError("cv_skills_tailoring output schema groups must define skills as an array")
    if not isinstance(skill_items, dict) or skill_items.get("type") != "string":
        raise ConfigurationError("cv_skills_tailoring output schema skill items must be strings")

    # The runtime master CV is the source of truth. Adding the whitelist to the
    # schema lets the existing structured-output repair path correct model drift
    # instead of publishing unsupported skills or surfacing a normal model error.
    skill_items["enum"] = allowed
    skills["uniqueItems"] = True
    return runtime_definition.model_copy(update={"output_structure": schema})


class SkillConstrainedWorkflowMixin:
    @staticmethod
    def _runtime_definition(
        definition: PromptDefinition,
        values: Mapping[str, object],
    ) -> PromptDefinition:
        return _skill_constrained_definition(definition, values)


class ConstrainedParallelGeminiWorkflowAI(SkillConstrainedWorkflowMixin, ParallelGeminiWorkflowAI):
    pass


class ConstrainedParallelAtlasGeminiWorkflowAI(SkillConstrainedWorkflowMixin, ParallelAtlasGeminiWorkflowAI):
    pass
