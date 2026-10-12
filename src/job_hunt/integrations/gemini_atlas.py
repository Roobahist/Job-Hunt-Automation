from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from job_hunt.domain.models import Job, PromptDefinition, TailoredContent
from job_hunt.integrations.gemini import GeminiWorkflowAI


class AtlasGeminiWorkflowAI(GeminiWorkflowAI):
    """Atlas-specific CV assembly on top of the shared Gemini workflow primitives."""

    @staticmethod
    def _section(source: dict[str, Any], section_type: str) -> dict[str, Any]:
        sections = source.get("sections")
        if not isinstance(sections, list):
            raise ValueError("master_cv.sections must be an array")
        matches = [section for section in sections if isinstance(section, dict) and section.get("type") == section_type]
        if len(matches) != 1:
            raise ValueError(f"master_cv must contain exactly one {section_type} section")
        return matches[0]

    def _work_pipeline_sections(
        self,
        job: Job,
        source: dict[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> list[dict[str, Any]]:
        experiences = self._section(source, "entries").get("entries")
        if not isinstance(experiences, list):
            raise ValueError("The entries section must contain an entries array")
        count = min(self.work_experience_selection_count, len(experiences))
        inputs: list[dict[str, Any]] = []
        for index, experience in enumerate(experiences):
            if not isinstance(experience, dict):
                raise ValueError("Each experience must be an object")
            content = experience.get("content")
            if not isinstance(content, list):
                raise ValueError(f"Experience at index {index} must have a content array")
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

        title_to_main_index = {
            experience.get("title"): index
            for index, experience in enumerate(experiences)
            if isinstance(experience, dict) and not experience.get("parent")
        }
        expanded = set(indices)
        for index in indices:
            experience = experiences[index]
            if not isinstance(experience, dict):
                continue
            parent = experience.get("parent")
            if parent:
                parent_index = title_to_main_index.get(parent)
                if parent_index is None:
                    raise ValueError(f"Selected nested experience references unknown parent: {parent}")
                expanded.add(parent_index)
        selected_indices = sorted(expanded)
        selected = [experiences[index] for index in selected_indices]

        rewrite_inputs = []
        for experience in selected:
            if not isinstance(experience, dict):
                raise ValueError("Each selected experience must be an object")
            rewrite_inputs.append(dict(experience))
        rewritten = self._run(
            self._definition(prompts, "cv_work_experience_rewrite"),
            {
                **self._job_values(job),
                "selected_work_experiences_json": rewrite_inputs,
                "selected_experiences_json": rewrite_inputs,
                "rewrite_inputs_json": rewrite_inputs,
            },
        )
        contents = rewritten.get("contents")
        if not isinstance(contents, list) or len(contents) != len(selected_indices):
            raise ValueError("Work rewrite contents count must match selected experiences")
        tailored: list[dict[str, Any]] = []
        for position, index in enumerate(selected_indices):
            bullets = contents[position]
            if not isinstance(bullets, list) or any(not isinstance(item, str) for item in bullets):
                raise ValueError("Each rewritten work content item must be an array of strings")
            rebuilt = dict(experiences[index])
            rebuilt["content"] = bullets
            tailored.append(rebuilt)
        return tailored

    def _skills_pipeline_sections(
        self,
        job: Job,
        source: dict[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> list[dict[str, str]]:
        rows = self._section(source, "label_rows").get("rows")
        if not isinstance(rows, list):
            raise ValueError("The label_rows section must contain a rows array")
        normalized = []
        for group in rows:
            if not isinstance(group, dict):
                raise ValueError("Every skill row must be an object")
            raw = group.get("value", "")
            skills = [part.strip() for part in re.split(r"[,;|]", raw) if part.strip()] if isinstance(raw, str) else []
            normalized.append({"label": str(group.get("label", "")), "skills": skills})
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
                raise ValueError("Every skill group must be an object")
            label = group.get("label")
            returned = group.get("skills")
            if not isinstance(label, str) or not label.strip() or not isinstance(returned, list):
                raise ValueError("Every skill group requires label and skills")
            unique: list[str] = []
            seen: set[str] = set()
            for skill in returned:
                if not isinstance(skill, str) or not skill.strip():
                    raise ValueError("Every returned skill must be a non-empty string")
                cleaned = skill.strip()
                folded = cleaned.casefold()
                if folded not in seen:
                    unique.append(cleaned)
                    seen.add(folded)
            tailored.append({"label": label.strip(), "value": " | ".join(unique)})
        return tailored

    def _references_decision(
        self,
        job: Job,
        source: dict[str, Any],
        prompts: Mapping[str, PromptDefinition],
    ) -> bool:
        references = self._section(source, "references").get("items")
        if not isinstance(references, list):
            raise ValueError("The references section must contain an items array")
        generated = self._run(
            self._definition(prompts, "cv_references_inclusion"),
            {
                **self._job_values(job),
                "references_json": references,
                "master_cv_json": source,
                "master_cv": source,
            },
        )
        include = generated.get("include_references")
        if not isinstance(include, bool):
            raise ValueError("cv_references_inclusion must return include_references as boolean")
        return include

    def _summary_pipeline_sections(
        self,
        job: Job,
        source: dict[str, Any],
        skills: list[dict[str, str]],
        experiences: list[dict[str, Any]],
        prompts: Mapping[str, PromptDefinition],
    ) -> list[str]:
        summary = self._section(source, "text").get("content")
        if not isinstance(summary, list):
            raise ValueError("The text section must contain a content array")
        context = {
            "master_cv": source,
            "original_summary": summary,
            "tailored_skills": skills,
            "experience_context": experiences,
        }
        generated = self._run(
            self._definition(prompts, "cv_summary_rewrite"),
            {
                **self._job_values(job),
                "master_cv": source,
                "master_cv_json": source,
                "summary_context_json": context,
                "original_summary_json": summary,
                "tailored_skills_json": skills,
                "project_context_json": [],
                "tailored_projects_json": [],
                "experience_context_json": experiences,
                "tailored_work_experience_json": experiences,
            },
        )
        result = generated.get("summary")
        if not isinstance(result, list) or any(not isinstance(line, str) for line in result):
            raise ValueError("cv_summary_rewrite must return summary as an array of strings")
        return result

    def tailor(
        self,
        job: Job,
        master_cv: Mapping[str, object],
        prompts: Mapping[str, PromptDefinition],
    ) -> TailoredContent:
        source = json.loads(json.dumps(master_cv, ensure_ascii=False, default=str))
        if not isinstance(source, dict):
            raise ValueError("master_cv must be an object")

        experiences = self._work_pipeline_sections(job, source, prompts)
        skills = self._skills_pipeline_sections(job, source, prompts)
        include_references = self._references_decision(job, source, prompts)
        summary = self._summary_pipeline_sections(job, source, skills, experiences, prompts)

        sections = source.get("sections")
        if not isinstance(sections, list):
            raise ValueError("master_cv.sections must be an array")
        final_sections: list[dict[str, Any]] = []
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("Every master_cv section must be an object")
            rebuilt = dict(section)
            section_type = rebuilt.get("type")
            if section_type == "text":
                rebuilt["content"] = summary
            elif section_type == "entries":
                rebuilt["entries"] = experiences
            elif section_type == "label_rows":
                rebuilt["rows"] = skills
            elif section_type == "references" and not include_references:
                rebuilt["items"] = []
            final_sections.append(rebuilt)
        cv = {"sections": final_sections}

        cover = self._run(
            self._definition(prompts, "cover_letter_generation"),
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
