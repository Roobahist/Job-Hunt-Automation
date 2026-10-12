from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime
from typing import Any

from job_hunt.domain.models import Job, PromptDefinition, TailoredContent
from job_hunt.integrations.gemini import GeminiWorkflowAI
from job_hunt.integrations.gemini_atlas import AtlasGeminiWorkflowAI


class ParallelGeminiWorkflowAI(GeminiWorkflowAI):
    def __init__(self, *args: object, parallelism: int = 3, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.parallelism = parallelism

    def tailor(
        self,
        job: Job,
        master_cv: Mapping[str, object],
        prompts: Mapping[str, PromptDefinition],
    ) -> TailoredContent:
        source = json.loads(json.dumps(master_cv, ensure_ascii=False, default=str))
        if not isinstance(source, dict):
            raise ValueError("master_cv must be an object")

        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            project_future = executor.submit(copy_context().run, self._project_pipeline, job, source, prompts)
            work_future = executor.submit(copy_context().run, self._work_pipeline, job, source, prompts)
            skills_future = executor.submit(copy_context().run, self._skills_pipeline, job, source, prompts)
            _, projects = project_future.result()
            _, experiences = work_future.result()
            skills = skills_future.result()

        summary = self._summary_pipeline(job, source, skills, projects, experiences, prompts)
        cv = {
            **source,
            "summary": summary,
            "skills": skills,
            "projects": projects,
            "work_experience": experiences,
        }
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


class ParallelAtlasGeminiWorkflowAI(AtlasGeminiWorkflowAI):
    def __init__(self, *args: object, parallelism: int = 3, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.parallelism = parallelism

    def tailor(
        self,
        job: Job,
        master_cv: Mapping[str, object],
        prompts: Mapping[str, PromptDefinition],
    ) -> TailoredContent:
        source = json.loads(json.dumps(master_cv, ensure_ascii=False, default=str))
        if not isinstance(source, dict):
            raise ValueError("master_cv must be an object")

        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            work_future = executor.submit(copy_context().run, self._work_pipeline_sections, job, source, prompts)
            skills_future = executor.submit(copy_context().run, self._skills_pipeline_sections, job, source, prompts)
            references_future = executor.submit(copy_context().run, self._references_decision, job, source, prompts)
            experiences = work_future.result()
            skills = skills_future.result()
            include_references = references_future.result()

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
