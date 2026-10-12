from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from job_hunt.errors import DocumentRenderingError, WorkflowError
from job_hunt.rendering.latex import PdfLatexCompiler
from job_hunt.security import fetch_public_text


def test_public_fetch_redirect_text_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("job_hunt.security.validate_public_url", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/final"})
        return httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"posting")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        assert fetch_public_text("https://x/start", client=client) == "posting"
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={"Content-Type": "image/png"}, content=b"x")
            )
        ) as client,
        pytest.raises(WorkflowError, match="supported text"),
    ):
        fetch_public_text("https://x/image", client=client)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={"Content-Type": "text/plain"}, content=b"too big")
            )
        ) as client,
        pytest.raises(WorkflowError, match="size limit"),
    ):
        fetch_public_text("https://x/big", client=client, max_bytes=2)


def test_pdflatex_two_passes_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tex = tmp_path / "cv.tex"
    tex.write_text("document")
    calls = 0

    def success(command: list[str], *, cwd: Path, **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        (Path(cwd) / "cv.pdf").write_bytes(b"pdf")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("job_hunt.rendering.latex.subprocess.run", success)
    assert PdfLatexCompiler().compile(tex).read_bytes() == b"pdf"
    assert calls == 2
    monkeypatch.setattr(
        "job_hunt.rendering.latex.subprocess.run",
        lambda command, **_: subprocess.CompletedProcess(command, 1, "bad latex", ""),
    )
    with pytest.raises(DocumentRenderingError, match="bad latex"):
        PdfLatexCompiler().compile(tex)
