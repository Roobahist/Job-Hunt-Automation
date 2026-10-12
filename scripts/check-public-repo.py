"""Safety check for the public portfolio repository's tracked text files."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_NAME = re.compile(r"\b(?:mahsa|mojtaba|kanani|golmaghani|sarcheshmeh)\b", re.I)
PERSONAL_EMAIL = re.compile(r"[A-Z0-9._%+-]+@(?!example\.org\b|example\.com\b|test\.com\b)[A-Z0-9.-]+\.[A-Z]{2,}", re.I)


def inspect_paths() -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode().split("\0")
    failures: list[str] = []
    for name in filter(None, tracked):
        path = ROOT / name
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_NAME.search(name) or PRIVATE_NAME.search(body):
            failures.append(f"{name}: private identity reference")
        if PERSONAL_EMAIL.search(body):
            failures.append(f"{name}: non-example email address")
    return failures


if __name__ == "__main__":
    issues = inspect_paths()
    if issues:
        raise SystemExit("Public safety check failed:\n" + "\n".join(issues))
    print("Public identity safety check passed")
