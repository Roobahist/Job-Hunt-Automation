"""Offline portfolio example using the real job domain and identity code.

All applicants and postings are fictional. Scoring is an illustrative deterministic
heuristic, NOT the production LLM-based qualification pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from job_hunt.domain.identity import assign_identity
from job_hunt.domain.models import EntryType, ExternalSubmission, Job, Qualification

DEFAULT_INPUT = Path(__file__).with_name("sample_jobs.json")
DEFAULT_CV = Path(__file__).resolve().parents[1] / "tenants/nova/master_cv.json"


def _skills(cv: dict[str, Any]) -> str:
    values = cv.get("skills", [])
    return " ".join(str(row.get("value", "")) for row in values if isinstance(row, dict)).lower()


def _qualification(job: Job, required: list[str], known_skills: str) -> tuple[Qualification, list[str]]:
    matched = [name for name in required if name.lower() in known_skills]
    coverage = len(matched) / max(1, len(required))
    # Demo-only rule, deliberately distinct from production LLM qualification.
    score = round(
        40 * coverage
        + (20 if any(word in job.title.lower() for word in ("engineer", "developer")) else 0)
        + (20 if "remote" in (job.location or "").lower() else 0)
        + (20 if job.contract_type == "full-time" else 0)
    )
    return (
        Qualification(
            score=min(score, 100),
            reasoning=f"Matched {len(matched)} of {len(required)} requested technologies",
        ),
        matched,
    )


def run_demo(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = Path("runs/demo"),
    cv_path: Path = DEFAULT_CV,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    known_skills = _skills(cv)
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0

    for data in payload["jobs"]:
        submission = ExternalSubmission(
            entry_type=EntryType.EXTERNAL,
            company_name=data["company_name"],
            job_title=data["title"],
            job_description=data["description"],
            job_url=data["url"],
            location=data.get("location"),
            contract_type=data.get("contract_type"),
        )
        job = assign_identity(
            Job(
                source="synthetic",
                url=str(submission.job_url),
                company_name=submission.company_name,
                title=submission.job_title,
                description=submission.job_description,
                location=submission.location,
                contract_type=submission.contract_type,
            )
        )
        if job.identity in seen:
            duplicates += 1
            continue
        seen.add(job.identity)
        required = data.get("required_skills", [])
        qualification, matched = _qualification(job, required, known_skills)
        decisions.append(
            {
                "job_id": job.internal_id,
                "company": job.company_name,
                "title": job.title,
                "canonical_url": job.url,
                "required_skills": required,
                "matched_skills": matched,
                "score": qualification.score,
                "reason": qualification.reasoning,
                "decision": "generate_documents" if qualification.passes(65) else "review_manually",
            }
        )

    report = {
        "notice": "Synthetic offline illustration; not production LLM qualification.",
        "applicant": "Nova Example",
        "input_count": len(payload["jobs"]),
        "unique_count": len(decisions),
        "duplicates": duplicates,
        "decisions": decisions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decisions.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a credential-free synthetic pipeline demo")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    report = run_demo(args.input, args.output)
    print(f"Processed {report['unique_count']} unique jobs; skipped {report['duplicates']} duplicates.")
    print(f"Decisions: {args.output / 'decisions.json'}")


if __name__ == "__main__":
    main()
