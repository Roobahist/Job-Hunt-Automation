from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from langchain_google_genai import ChatGoogleGenerativeAI

from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import Job, PromptDefinition, Qualification, TailoredContent
from job_hunt.errors import ConfigurationError, ErrorKind, ProviderError
from job_hunt.prompt_rendering import render_prompt


def _compact(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _render(template: str, values: Mapping[str, object], prompt_key: str) -> str:
    return render_prompt(template, values, prompt_key)


def _message_text(raw: object) -> str:
    value = getattr(raw, "text", "")
    if callable(value):
        value = value()
    if isinstance(value, str) and value:
        return value
    content = getattr(raw, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        return "\n".join(chunks)
    return ""


def _validate_json_schema(payload: object, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        raise ConfigurationError(f"Invalid Baserow Output Structure JSON Schema: {exc.message}") from exc
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise JsonSchemaValidationError(f"{location}: {error.message}")
    if not isinstance(payload, dict):
        raise JsonSchemaValidationError("$: response must be a JSON object")
    return payload


class GeminiStructuredClient:
    def __init__(self, primary_key: str, backup_key: str, model: str) -> None:
        self.keys = [key for key in (primary_key, backup_key) if key]
        self.model = model.removeprefix("models/")
        if not self.keys:
            raise ValueError("At least one Gemini API key is required")

    def _model(self, key: str, temperature: float) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model,
            api_key=key,
            temperature=temperature,
            max_retries=2,
        )

    def _structured_call(
        self,
        key: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        model = self._model(key, temperature)
        runnable = model.with_structured_output(
            schema=schema,
            method="json_schema",
            include_raw=True,
        )
        result = runnable.invoke(prompt)
        if not isinstance(result, dict):
            return None, "", "LangChain returned an unexpected structured-output wrapper"
        parsed = result.get("parsed")
        raw_text = _message_text(result.get("raw"))
        parsing_error = result.get("parsing_error")
        if isinstance(parsed, dict):
            return parsed, raw_text or _compact(parsed), str(parsing_error) if parsing_error else None
        return None, raw_text, str(parsing_error) if parsing_error else "No parsed JSON object"

    def _repair(
        self,
        key: str,
        raw_output: str,
        schema: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any]:
        repair_prompt = (
            "Repair the following model output so it conforms exactly to the supplied JSON Schema. "
            "Preserve the original meaning and facts. Do not add facts that were not present. "
            "Return only the repaired structured result.\n\n"
            f"VALIDATION ERROR:\n{validation_error}\n\n"
            f"JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"ORIGINAL OUTPUT:\n{raw_output}"
        )
        parsed, _, parsing_error = self._structured_call(key, repair_prompt, schema, 0.0)
        if parsed is None:
            raise ValueError(f"auto-fix model did not return parsed JSON: {parsing_error}")
        return _validate_json_schema(parsed, schema)

    def generate(self, prompt: str, definition: PromptDefinition) -> dict[str, Any]:
        failures: list[str] = []
        for index, key in enumerate(self.keys):
            try:
                parsed, raw_output, parsing_error = self._structured_call(
                    key,
                    prompt,
                    definition.output_structure,
                    definition.temperature,
                )
                if parsed is not None:
                    try:
                        return _validate_json_schema(parsed, definition.output_structure)
                    except JsonSchemaValidationError as exc:
                        parsing_error = str(exc)
                        raw_output = raw_output or _compact(parsed)
                if not raw_output:
                    raise ValueError(parsing_error or "Gemini returned an empty response")
                repair_key = self.keys[1] if len(self.keys) > 1 and index == 0 else key
                return self._repair(
                    repair_key,
                    raw_output,
                    definition.output_structure,
                    parsing_error or "structured output did not validate",
                )
            except ConfigurationError:
                raise
            except Exception as exc:
                failures.append(f"{definition.key}: {exc}")
        raise ProviderError(
            "Gemini failed for all configured keys: " + "; ".join(failures),
            ErrorKind.MALFORMED_PROVIDER_RESPONSE,
            provider="gemini",
        )


class GeminiWorkflowAI:
    def __init__(
        self,
        client: GeminiStructuredClient,
        prompts: Mapping[str, PromptDefinition] | None = None,
        *,
        project_selection_count: int | None = None,
        work_experience_selection_count: int = 3,
    ) -> None:
        self.client = client
        self.prompts = dict(prompts or {})
        self.project_selection_count = project_selection_count
        self.work_experience_selection_count = work_experience_selection_count

    @staticmethod
    def _definition(prompts: Mapping[str, PromptDefinition], key: str) -> PromptDefinition:
        try:
            return prompts[key]
        except KeyError as exc:
            raise ConfigurationError(f"Missing active prompt: {key}") from exc

    @staticmethod
    def _job_values(job: Job) -> dict[str, object]:
        return {
            "job_description": job.description,
            "job_title": job.title,
            "company_name": job.company_name,
            "job_url": job.url,
            "job_json": job.model_dump(mode="json"),
        }

    @staticmethod
    def _selection_indices(
        raw_indices: object,
        *,
        prompt_key: str,
        item_label: str,
        available_count: int,
        max_count: int,
        exact_count: int | None = None,
    ) -> list[int]:
        if not isinstance(raw_indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) for index in raw_indices
        ):
            raise ValueError(f"{prompt_key} must return selected_indices as integers")
        if exact_count is not None and len(raw_indices) != exact_count:
            raise ValueError(f"{item_label} selection must contain exactly {exact_count} indices")
        if len(raw_indices) != len(set(raw_indices)):
            raise ValueError(f"Selected {item_label} indices must be unique")
        if len(raw_indices) > max_count:
            raise ValueError(f"Selected {item_label} count exceeds configured count")
        for index in raw_indices:
            if index < 0 or index >= available_count:
                raise ValueError(f"Invalid {item_label} index: {index}")
        return sorted(raw_indices)

    @staticmethod
    def _runtime_definition(
        definition: PromptDefinition,
        values: Mapping[str, object],
    ) -> PromptDefinition:
        if definition.key in {"cv_project_selection", "cv_work_experience_selection"}:
            selection_count = values.get("selection_count")
            if (
                not isinstance(selection_count, int)
                or isinstance(selection_count, bool)
                or selection_count < 0
            ):
                raise ConfigurationError(f"{definition.key} requires selection_count as a non-negative integer")

            schema = json.loads(json.dumps(definition.output_structure))
            properties = schema.get("properties")
            selected_indices = properties.get("selected_indices") if isinstance(properties, dict) else None
            if not isinstance(selected_indices, dict) or selected_indices.get("type") != "array":
                raise ConfigurationError(f"{definition.key} output schema must define selected_indices as an array")

            selected_indices["minItems"] = selection_count
            selected_indices["maxItems"] = selection_count
            selected_indices["uniqueItems"] = True
            return definition.model_copy(update={"output_structure": schema})

        if definition.key not in {"cv_project_rewrite", "cv_work_experience_rewrite"}:
            return definition
        rewrite_inputs = values.get("rewrite_inputs_json")
        if not isinstance(rewrite_inputs, list):
            raise ConfigurationError(f"{definition.key} requires rewrite_inputs_json as an array")

        schema = json.loads(json.dumps(definition.output_structure))
        properties = schema.get("properties")
        contents = properties.get("contents") if isinstance(properties, dict) else None
        if not isinstance(contents, dict) or contents.get("type") != "array":
            raise ConfigurationError(f"{definition.key} output schema must define contents as an array")

        expected_count = len(rewrite_inputs)
        contents["minItems"] = expected_count
        contents["maxItems"] = expected_count
        return definition.model_copy(update={"output_structure": schema})

    def _run(
        self,
        definition: PromptDefinition,
        values: Mapping[str, object],
    ) -> dict[str, Any]:
        runtime_definition = self._runtime_definition(definition, values)
        return self.client.generate(
            _render(definition.template, values, definition.key),
            runtime_definition,
        )

    def qualify(
        self,
        job: Job,
        master_cv: Mapping[str, object],
        prompts: Mapping[str, PromptDefinition],
    ) -> Qualification:
        definition = self._definition(prompts, "qualification_scoring")
        values = {
            **self._job_values(job),
            "master_cv": master_cv,
            "master_cv_json": master_cv,
        }
        generated = self._run(definition, values)
        score = generated.get("score", generated.get("qualification_score"))
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError("qualification_scoring must return an integer score")
        should_apply = generated.get("should_apply", generated.get("apply", False))
        if not isinstance(should_apply, bool):
            raise ValueError("qualification_scoring must return should_apply as a boolean")
        reasoning = generated.get(
            "reasoning",
            generated.get("qualification_reasoning", generated.get("reason", "Qualification scored")),
        )
        return Qualification(
            score=score,
            should_apply=should_apply,
            reasoning=str(reasoning),
        )

    def _project_pipeline(
        self,
        job: Job,
        master_cv: dict[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> tuple[list[int], list[dict[str, Any]]]:
        projects = master_cv.get("projects", [])
        if not isinstance(projects, list):
            raise ValueError("master_cv.projects must be an array")
        count = min(self.project_selection_count or len(projects), len(projects))
        inputs = []
        for index, project in enumerate(projects):
            if not isinstance(project, dict):
                raise ValueError("Each project must be an object")
            content = project.get("content")
            if not isinstance(content, list) or any(not isinstance(x, str) for x in content):
                raise ValueError(f"Project at index {index} must have string content bullets")
            inputs.append({"index": index, **project})

        selection = self._run(
            self._definition(prompts, "cv_project_selection"),
            {
                **self._job_values(job),
                "selection_count": count,
                "project_summaries_json": inputs,
                "projects_json": inputs,
                "project_inputs_json": inputs,
            },
        )
        indices = self._selection_indices(
            selection.get("selected_indices"),
            prompt_key="cv_project_selection",
            item_label="project",
            available_count=len(projects),
            max_count=count,
            exact_count=count,
        )
        selected = [projects[index] for index in indices]
        rewrite_inputs = [dict(project) for project in selected]
        rewritten = self._run(
            self._definition(prompts, "cv_project_rewrite"),
            {
                **self._job_values(job),
                "selected_projects_json": rewrite_inputs,
                "rewrite_inputs_json": rewrite_inputs,
            },
        )
        contents = rewritten.get("contents")
        if not isinstance(contents, list) or len(contents) != len(indices):
            raise ValueError("Project rewrite contents count must match selected projects")
        tailored: list[dict[str, Any]] = []
        for position, index in enumerate(indices):
            bullets = contents[position]
            if not isinstance(bullets, list) or any(not isinstance(x, str) for x in bullets):
                raise ValueError("Each rewritten project content item must be an array of strings")
            rebuilt = dict(projects[index])
            rebuilt["content"] = bullets
            tailored.append(rebuilt)
        return indices, tailored

    def _work_pipeline(
        self,
        job: Job,
        master_cv: dict[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> tuple[list[int], list[dict[str, Any]]]:
        experiences = master_cv.get("work_experience", [])
        if not isinstance(experiences, list):
            raise ValueError("master_cv.work_experience must be an array")
        count = min(self.work_experience_selection_count, len(experiences))
        inputs = []
        for index, experience in enumerate(experiences):
            if not isinstance(experience, dict):
                raise ValueError("Each work experience must be an object")
            content = experience.get("content")
            if not isinstance(content, list) or any(not isinstance(x, str) for x in content):
                raise ValueError(f"Work experience at index {index} must have string content bullets")
            inputs.append({"index": index, **experience})

        selection = self._run(
            self._definition(prompts, "cv_work_experience_selection"),
            {
                **self._job_values(job),
                "selection_count": count,
                "work_experience_summaries_json": inputs,
                "experience_summaries_json": inputs,
                "work_experiences_json": inputs,
                "experience_inputs_json": inputs,
            },
        )
        indices = self._selection_indices(
            selection.get("selected_indices"),
            prompt_key="cv_work_experience_selection",
            item_label="work-experience",
            available_count=len(experiences),
            max_count=count,
            exact_count=count,
        )
        selected = [experiences[index] for index in indices]
        rewrite_inputs = [dict(experience) for experience in selected]
        rewritten = self._run(
            self._definition(prompts, "cv_work_experience_rewrite"),
            {
                **self._job_values(job),
                "selected_work_experiences_json": rewrite_inputs,
                "rewrite_inputs_json": rewrite_inputs,
            },
        )
        contents = rewritten.get("contents")
        if not isinstance(contents, list) or len(contents) != len(indices):
            raise ValueError("Work rewrite contents count must match selected experiences")
        tailored: list[dict[str, Any]] = []
        for position, index in enumerate(indices):
            bullets = contents[position]
            if not isinstance(bullets, list) or any(not isinstance(x, str) for x in bullets):
                raise ValueError("Each rewritten work content item must be an array of strings")
            rebuilt = dict(experiences[index])
            rebuilt["content"] = bullets
            tailored.append(rebuilt)
        return indices, tailored

    def _skills_pipeline(
        self,
        job: Job,
        master_cv: dict[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> list[dict[str, str]]:
        skills = master_cv.get("skills", [])
        if not isinstance(skills, list):
            raise ValueError("master_cv.skills must be an array")
        normalized = []
        for group in skills:
            if not isinstance(group, dict):
                raise ValueError("Each skill group must be an object")
            label = str(group.get("label", ""))
            raw = group.get("value", group.get("content", []))
            if isinstance(raw, str):
                values = [part.strip() for part in re.split(r"[,;|]", raw) if part.strip()]
            elif isinstance(raw, list):
                values = [part.strip() for part in raw if isinstance(part, str) and part.strip()]
            else:
                values = []
            normalized.append({"label": label, "skills": values})
        generated = self._run(
            self._definition(prompts, "cv_skills_tailoring"),
            {
                **self._job_values(job),
                "original_skills_json": normalized,
                "skills_json": normalized,
            },
        )
        groups = generated.get("groups")
        if not isinstance(groups, list):
            raise ValueError("cv_skills_tailoring must return a groups array")
        tailored: list[dict[str, str]] = []
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Every returned skill group must be an object")
            returned_label = group.get("label")
            returned = group.get("skills")
            if not isinstance(returned_label, str) or not returned_label.strip() or not isinstance(returned, list):
                raise ValueError("Every skill group requires label and skills")
            deduped: list[str] = []
            seen: set[str] = set()
            for skill in returned:
                if not isinstance(skill, str) or not skill.strip():
                    raise ValueError("Every returned skill must be a non-empty string")
                cleaned = skill.strip()
                folded = cleaned.casefold()
                if folded not in seen:
                    deduped.append(cleaned)
                    seen.add(folded)
            tailored.append({"label": returned_label.strip(), "value": ", ".join(deduped)})
        return tailored

    def _summary_pipeline(
        self,
        job: Job,
        master_cv: dict[str, Any],
        tailored_skills: list[dict[str, str]],
        tailored_projects: list[dict[str, Any]],
        tailored_experiences: list[dict[str, Any]],
        prompts: Mapping[str, PromptDefinition],
    ) -> list[str]:
        original_summary = master_cv.get("summary", [])
        if not isinstance(original_summary, list):
            raise ValueError("master_cv.summary must be an array")
        context = {
            "master_cv": master_cv,
            "original_summary": original_summary,
            "tailored_skills": tailored_skills,
            "project_context": tailored_projects,
            "experience_context": tailored_experiences,
        }
        generated = self._run(
            self._definition(prompts, "cv_summary_rewrite"),
            {
                **self._job_values(job),
                "master_cv": master_cv,
                "master_cv_json": master_cv,
                "summary_context_json": context,
                "original_summary_json": original_summary,
                "tailored_skills_json": tailored_skills,
                "project_context_json": tailored_projects,
                "tailored_projects_json": tailored_projects,
                "experience_context_json": tailored_experiences,
                "tailored_work_experience_json": tailored_experiences,
            },
        )
        summary = generated.get("summary")
        if not isinstance(summary, list) or any(not isinstance(line, str) for line in summary):
            raise ValueError("cv_summary_rewrite must return summary as an array of strings")
        return summary

    def tailor(
        self,
        job: Job,
        master_cv: Mapping[str, object],
        prompts: Mapping[str, PromptDefinition],
    ) -> TailoredContent:
        source = json.loads(json.dumps(master_cv, ensure_ascii=False, default=str))
        if not isinstance(source, dict):
            raise ValueError("master_cv must be an object")
        _, projects = self._project_pipeline(job, source, prompts)
        _, experiences = self._work_pipeline(job, source, prompts)
        skills = self._skills_pipeline(job, source, prompts)
        summary = self._summary_pipeline(job, source, skills, projects, experiences, prompts)
        cv = {
            **source,
            "summary": summary,
            "skills": skills,
            "projects": projects,
            "work_experience": experiences,
        }
        cover_definition = self._definition(prompts, "cover_letter_generation")
        cover = self._run(
            cover_definition,
            {
                **self._job_values(job),
                "master_cv": source,
                "master_cv_json": source,
                "cv_json": cv,
                "tailored_cv_json": cv,
            },
        )
        cover.setdefault("company_name", job.company_name)
        cover.setdefault("date", datetime.now().strftime("%B %d, %Y").replace(" 0", " "))
        return TailoredContent(cv=cv, cover_letter=cover)

    def extract_job(self, content: str, source_url: str) -> Job:
        definition = self._definition(self.prompts, "job_page_content_extraction")
        generated = self._run(
            definition,
            {
                "page_content": content,
                "content": content,
                "source_url": source_url,
                "job_url": source_url,
            },
        )
        company_name = generated.get("company_name", generated.get("company"))
        title = generated.get("title", generated.get("job_title"))
        description = generated.get("description", generated.get("job_description"))
        if not all(isinstance(value, str) and value.strip() for value in (company_name, title, description)):
            raise ValueError("job_page_content_extraction did not return company, title, and description")
        job = Job(
            source=str(generated.get("source", "web")),
            external_id=(str(generated["external_id"]) if generated.get("external_id") is not None else None),
            url=str(generated.get("url", generated.get("job_url", source_url))),
            company_name=str(company_name),
            title=str(title),
            description=str(description),
            location=(str(generated["location"]) if generated.get("location") is not None else None),
            contract_type=(str(generated["contract_type"]) if generated.get("contract_type") is not None else None),
        )
        return assign_identity(job)