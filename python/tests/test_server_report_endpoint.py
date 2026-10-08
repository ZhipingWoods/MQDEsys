"""GET /api/v1/jobs/<id>/report.<lang>.html on the real mqde server: content, headers, and that no part of the request
can become a path. Skips when the server binary has not been built."""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from mqde.report.html import render_html

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "python" / "tests" / "fixtures" / "reports" / "llm_vllm.json"
JOB = "20261008-000000-aaaaaa"


def _binary() -> Path | None:
    for p in ("build/dev/bin/mqde", "build/mac/bin/mqde"):
        if (ROOT / p).is_file():
            return ROOT / p
    return None


pytestmark = pytest.mark.skipif(_binary() is None, reason="mqde server binary not built")


def get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.putrequest("GET", path, skip_host=False)  # path sent as is, no normalisation
        c.endheaders()
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        c.close()


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[int, Path]]:
    runs = tmp_path_factory.mktemp("runs")
    report = json.loads(FIXTURE.read_text())
    d = runs / JOB
    d.mkdir()
    (d / "report.json").write_text(json.dumps(report))
    (d / "report.en.html").write_text(render_html(report, "en"), encoding="utf-8")  # no zh page on purpose
    (runs / "secret.txt").write_text("TOP-SECRET")
    (runs / "other").mkdir()
    (runs / "other" / "report.en.html").write_text("OTHER-DIR")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([str(_binary()), "serve", "--repo", str(ROOT), "--runs", str(runs), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                if get(port, "/api/v1/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("server did not start")
        yield port, runs
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_serves_the_report_with_restrictive_headers(server: tuple[int, Path]) -> None:
    port, runs = server
    status, h, body = get(port, f"/api/v1/jobs/{JOB}/report.en.html")
    assert status == 200 and body == (runs / JOB / "report.en.html").read_bytes()
    assert h["content-type"].startswith("text/html") and "charset=utf-8" in h["content-type"]
    assert h["content-security-policy"] == "default-src 'none'; style-src 'unsafe-inline'"
    assert h["x-content-type-options"] == "nosniff"


def test_missing_page_or_job_is_404(server: tuple[int, Path]) -> None:
    port, _ = server
    assert get(port, f"/api/v1/jobs/{JOB}/report.zh.html")[0] == 404  # job exists, page was not written
    assert get(port, "/api/v1/jobs/20990101-000000-bbbbbb/report.en.html")[0] == 404


@pytest.mark.parametrize("path", [
    f"/api/v1/jobs/{JOB}/report.fr.html",
    f"/api/v1/jobs/{JOB}/report.en.htmlx",
    f"/api/v1/jobs/{JOB}/report.en.html/extra",
    "/api/v1/jobs/..%2Fother/report.en.html",
    "/api/v1/jobs/../other/report.en.html",
    f"/api/v1/jobs/{JOB}/..%2F..%2Fsecret.txt",
    f"/api/v1/jobs/{JOB}/report.en.html/../../../secret.txt",
    f"/api/v1/jobs/{JOB}%2F..%2Fother/report.en.html",
])
def test_no_part_of_the_request_becomes_a_path(server: tuple[int, Path], path: str) -> None:
    port, _ = server
    status, _, body = get(port, path)
    assert status != 200 and b"TOP-SECRET" not in body and b"OTHER-DIR" not in body
