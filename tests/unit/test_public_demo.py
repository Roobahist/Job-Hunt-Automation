"""Credential-free portfolio demonstration contract tests."""
import json
from pathlib import Path

from examples.demo import DEFAULT_INPUT, run_demo


def test_demo_is_deterministic_and_deduplicates_urls(tmp_path: Path) -> None:
    result = run_demo(DEFAULT_INPUT, tmp_path)
    assert result["input_count"] == 4
    assert result["unique_count"] == 3
    assert result["duplicates"] == 1
    assert {row["decision"] for row in result["decisions"]} == {
        "generate_documents",
        "review_manually",
    }
    assert json.loads((tmp_path / "decisions.json").read_text()) == result


def test_demo_does_not_require_external_credentials(monkeypatch, tmp_path: Path) -> None:
    for key in ("GEMINI_API_KEY_1", "JOB_HUNT_LITELLM_API_KEY", "ATLAS_BASEROW_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    result = run_demo(DEFAULT_INPUT, tmp_path)
    assert result["unique_count"] == 3
