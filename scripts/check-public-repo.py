"""Conservative guard for a public repository with synthetic applicant data."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_TENANTS = {"atlas", "nova"}
PERSONAL_EMAIL = re.compile(
    r"[A-Z0-9._%+-]+@(?!example\.org\b|example\.com\b|test\.com\b)"
    r"[A-Z0-9.-]+\.[A-Z]{2,}",
    re.I,
)
CREDENTIAL = re.compile(
    r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AIza[A-Za-z0-9_-]{30,}|sk-[A-Za-z0-9_-]{24,})"
)
FORBIDDEN_PATH = re.compile(r"(?:^|/)(?:\.env$|id_rsa|[^/]+\.pem$|[^/]+\.key$)", re.I)


def inspect_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    names = result.stdout.decode().split("\0")

    failures: list[str] = []
    for name in filter(None, names):
        path = ROOT / name
        parts = Path(name).parts
        if FORBIDDEN_PATH.search(name):
            failures.append(f"{name}: sensitive file path")
            continue
        if len(parts) > 1 and parts[0] == "tenants" and parts[1] not in DEMO_TENANTS:
            failures.append(f"{name}: applicant must be a fictional demo tenant")
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # Avoid matching the scanner's literal example patterns.
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PERSONAL_EMAIL.search(body):
            failures.append(f"{name}: non-example email address")
        if CREDENTIAL.search(body):
            failures.append(f"{name}: credential-like value")
    return failures


if __name__ == "__main__":
    issues = inspect_paths()
    if issues:
        raise SystemExit("Public safety check failed:\n" + "\n".join(issues))
    print("Public repository safety checks passed")
