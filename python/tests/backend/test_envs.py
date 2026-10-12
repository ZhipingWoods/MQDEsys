"""Managed backend environments (mqde.backend.envs / scripts/setup_backend.py): versions, resolution, update checks and
the install flows, against a local fake PyPI / GitHub server and a fake ``uv`` -- no network, no real packages."""

from __future__ import annotations

import io
import json
import stat
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mqde.backend import envs


def sh(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# ---------------------------------------------------------------- versions
@pytest.mark.parametrize(("lo", "hi"), [
    ("0.5.21", "0.5.22.dev586+g86624f14e"),  # a dev build of the next release is ahead of the last release...
    ("0.5.22.dev586", "0.5.22"),             # ...and behind that release itself
    ("0.5.22", "0.5.22.post1"),
    ("0.5.9", "0.5.10"),                      # numeric, not lexicographic
    ("0.30.0", "0.31.0"),
    ("1.0rc1", "1.0"),
    ("b11564", "b11565"),
    ("b999", "b11565"),
])
def test_version_order(lo: str, hi: str) -> None:
    assert envs.version_key(lo) < envs.version_key(hi)
    assert envs.is_newer(hi, lo) and not envs.is_newer(lo, hi)


def test_is_newer_edge_cases() -> None:
    assert not envs.is_newer(None, "1.0") and not envs.is_newer("", "1.0")
    assert envs.is_newer("1.0", None), "nothing installed: any version is an update"
    assert not envs.is_newer("1.0", "1.0.0") and not envs.is_newer("0.5.21", "0.5.21")


def test_prerelease_detection() -> None:
    assert all(envs.is_prerelease(v) for v in ("1.0rc1", "0.5.22.dev3", "2.0a1", "2.0.b2"))
    assert not any(envs.is_prerelease(v) for v in ("1.0", "0.5.15.post1", "b11565"))


@pytest.mark.parametrize("v", ["0.31.0", "b11565", "0.5.22.dev586+g86624f14e", "1!2.0"])
def test_valid_versions(v: str) -> None:
    assert envs.valid_version(v)


@pytest.mark.parametrize("v", ["", "1.0; rm -rf /", "$(id)", "1 2", "../x", "-1", "a" * 41, "1.0\n", "==1.0"])
def test_hostile_versions_are_rejected(v: str) -> None:
    assert not envs.valid_version(v)
    with pytest.raises(ValueError, match="invalid version"):
        envs.install("vllm", v, lambda _l: None, dry_run=True)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        envs.install("pip", None, lambda _l: None, dry_run=True)
    with pytest.raises(ValueError, match="unknown backend"):
        envs.resolve("pip")


# ---------------------------------------------------------------- PyPI simple index
INDEX_HTML = """<html><body>
<a href="https://x/vllm-0.29.0.tar.gz#sha256=1">vllm-0.29.0.tar.gz</a>
<a href="https://x/vllm-0.30.0-cp38-abi3-manylinux1_x86_64.whl#sha256=2">vllm-0.30.0-cp38-abi3-manylinux1_x86_64.whl</a>
<a href="https://x/vllm-0.31.0-cp38-abi3-manylinux1_x86_64.whl#sha256=3">vllm-0.31.0-cp38-abi3-manylinux1_x86_64.whl</a>
<a href="https://x/vllm-0.32.0rc1-cp38-abi3-manylinux1_x86_64.whl#sha256=4">vllm-0.32.0rc1-cp38-abi3-manylinux1_x86_64.whl</a>
<a href="https://x/vllm-0.99.0-cp38-abi3-manylinux1_x86_64.whl#sha256=5" data-yanked="broken">vllm-0.99.0-cp38-abi3-manylinux1_x86_64.whl</a>
<a href="https://x/vllm_flash_attn-9.0.0.tar.gz">vllm_flash_attn-9.0.0.tar.gz</a>
</body></html>"""


def test_simple_index_parsing_skips_yanked_and_other_projects() -> None:
    vs = envs.simple_index_versions(INDEX_HTML, "vllm")
    assert vs == ["0.29.0", "0.30.0", "0.31.0", "0.32.0rc1"]


class Fake:
    """A throwaway HTTP server: ``routes`` maps a path to bytes (or a callable returning bytes)."""

    def __init__(self, routes: dict[str, object]) -> None:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def do_GET(self) -> None:
                outer.hits.append(self.path)
                body = routes.get(self.path.split("?")[0])
                if callable(body):
                    body = body()
                if body is None:
                    self.send_error(404)
                    return
                assert isinstance(body, bytes)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.hits: list[str] = []
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()


@pytest.fixture
def pypi(monkeypatch: pytest.MonkeyPatch) -> Iterator[Fake]:
    f = Fake({"/simple/vllm/": INDEX_HTML.encode(), "/simple/sglang/": b'<a href="x/sglang-0.5.21-py3-none-any.whl">sglang-0.5.21-py3-none-any.whl</a>'})
    monkeypatch.setenv("MQDE_PIP_INDEX", f"{f.url}/simple")
    yield f
    f.close()


def test_latest_is_the_newest_stable_release(pypi: Fake) -> None:
    assert envs.latest("vllm") == "0.31.0", "0.32.0rc1 is a pre-release and 0.99.0 is yanked"
    assert envs.latest("sglang") == "0.5.21"


@pytest.fixture
def no_index_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for k in ("MQDE_PIP_INDEX", "PIP_INDEX_URL", "UV_INDEX_URL", "PIP_CONFIG_FILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def no_subprocess(*a: object, **k: object) -> None:
        raise AssertionError("the index must be read from the config file, not by running pip")

    monkeypatch.setattr(envs, "_run_out", no_subprocess)
    return tmp_path / "home"


def test_index_comes_from_pip_conf_without_needing_pip_installed(no_index_env: Path) -> None:
    """The system python on the dev machine has no pip module, but the user's pip.conf still names the mirror."""
    assert envs.pip_index_url() == "https://pypi.org/simple"
    (no_index_env / ".config/pip").mkdir(parents=True)
    (no_index_env / ".config/pip/pip.conf").write_text("[global]\nindex-url = https://mirror.example/pypi/simple/\ntrusted-host = mirror.example\n")
    assert envs.pip_index_url() == "https://mirror.example/pypi/simple"


def test_explicit_index_variable_beats_pip_conf(no_index_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (no_index_env / ".pip").mkdir(parents=True)
    (no_index_env / ".pip/pip.conf").write_text("[global]\nindex-url = https://mirror.example/simple\n")
    monkeypatch.setenv("MQDE_PIP_INDEX", "https://mine.example/simple/")
    assert envs.pip_index_url() == "https://mine.example/simple"


def test_latest_reports_an_unreachable_index_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_PIP_INDEX", "http://127.0.0.1:9/simple")
    with pytest.raises(OSError):
        envs.latest("vllm")


# ---------------------------------------------------------------- llama.cpp releases and assets
def asset(tag: str, kind: str) -> str:
    return f"llama-{tag}-bin-{kind}.tar.gz"


TAG = "b11565"
NAMES = [asset(TAG, k) for k in ("ubuntu-x64", "ubuntu-arm64", "ubuntu-cuda-12.8-x64", "ubuntu-cuda-13.4-x64", "macos-arm64", "macos-x64", "ubuntu-vulkan-x64")] + [
    f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz", f"llama-{TAG}-bin-win-cpu-x64.zip"]


@pytest.mark.parametrize(("system", "machine", "cuda", "want"), [
    ("Linux", "x86_64", None, [asset(TAG, "ubuntu-x64")]),
    ("Linux", "x86_64", (13, 0), [asset(TAG, "ubuntu-cuda-12.8-x64"), f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"]),  # 13.4 > driver's 13.0
    ("Linux", "x86_64", (13, 4), [asset(TAG, "ubuntu-cuda-13.4-x64"), f"cudart-llama-{TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz"]),
    ("Linux", "x86_64", (11, 8), [asset(TAG, "ubuntu-x64")]),      # driver older than every CUDA build: CPU build, not a broken one
    ("Linux", "aarch64", None, [asset(TAG, "ubuntu-arm64")]),
    ("Darwin", "arm64", None, [asset(TAG, "macos-arm64")]),
    ("Darwin", "x86_64", None, [asset(TAG, "macos-x64")]),
])
def test_asset_choice(system: str, machine: str, cuda: tuple[int, int] | None, want: list[str]) -> None:
    assert envs.pick_llamacpp_assets(NAMES, TAG, system, machine, cuda) == want


def test_asset_choice_errors_clearly_when_nothing_fits() -> None:
    with pytest.raises(RuntimeError, match="no llama.cpp b11565 release asset for Linux/riscv64"):
        envs.pick_llamacpp_assets(NAMES, TAG, "Linux", "riscv64", None)
    with pytest.raises(RuntimeError, match="Windows"):
        envs.pick_llamacpp_assets(NAMES, TAG, "Windows", "AMD64", None)


def release(tag: str, names: list[str], base: str, sizes: dict[str, int] | None = None) -> dict[str, object]:
    return {"tag_name": tag, "draft": False,
            "assets": [{"name": n, "size": (sizes or {}).get(n, 100), "browser_download_url": f"{base}/dl/{n}"} for n in names]}


def test_only_numbered_builds_count_as_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    f = Fake({})
    f.srv.RequestHandlerClass  # noqa: B018 - keep the server alive for the duration of the test
    body = json.dumps([{"tag_name": "v0.6.0", "assets": []}, {"tag_name": "b11564", "assets": []}, {"tag_name": "b11565", "assets": []},
                       {"tag_name": "b99999", "draft": True, "assets": []}]).encode()
    g = Fake({"/repos/ggml-org/llama.cpp/releases": body})
    monkeypatch.setenv("MQDE_GITHUB_API", g.url)
    try:
        assert [r["tag_name"] for r in envs.llamacpp_releases()] == ["b11565", "b11564"]
        assert envs.latest("llamacpp") == "b11565"
    finally:
        f.close()
        g.close()


# ---------------------------------------------------------------- resolution and version probing
@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for k in ("MQDE_VLLM_BIN", "MQDE_SGLANG_PYTHON", "MQDE_LLAMA_SERVER_BIN", "MQDE_ENVS_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    monkeypatch.setenv("MQDE_ENVS_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(envs, "repo_root", lambda: tmp_path / "repo")
    return tmp_path


def test_nothing_installed_resolves_to_none(clean: Path) -> None:
    assert [envs.resolve(b) for b in envs.BACKENDS] == [None, None, None]
    assert [r["installed"] for r in envs.status()] == [False, False, False]


def test_vllm_precedence_is_env_var_then_managed_then_path_then_legacy(clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = sh(clean / "repo/build/vllm/bin/vllm", "true")
    assert envs.resolve("vllm") == envs.Resolved(legacy, "legacy")
    onpath = sh(clean / "emptybin/vllm", "true")
    assert envs.resolve("vllm") == envs.Resolved(onpath, "path")
    managed = sh(clean / "envs/vllm/bin/vllm", "true")
    assert envs.resolve("vllm") == envs.Resolved(managed, "managed"), "an install through the script must win over older copies"
    explicit = sh(clean / "mine/vllm", "true")
    monkeypatch.setenv("MQDE_VLLM_BIN", str(explicit))
    assert envs.resolve("vllm") == envs.Resolved(explicit, "env"), "an explicit variable always wins"


def test_sglang_and_llamacpp_precedence(clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_py = sh(clean / "repo/build/sglang/bin/python", "true")
    assert envs.resolve("sglang") == envs.Resolved(legacy_py, "legacy")
    managed_py = sh(clean / "envs/sglang/bin/python", "true")
    assert envs.resolve("sglang") == envs.Resolved(managed_py, "managed")
    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(legacy_py))
    assert envs.resolve("sglang") == envs.Resolved(legacy_py, "env")

    legacy_srv = sh(clean / "repo/build/llamacpp-server/bin/llama-server", "true")
    assert envs.resolve("llamacpp") == envs.Resolved(legacy_srv, "legacy")
    rel = clean / "envs/llamacpp/b11565"
    srv = sh(rel / "llama-b11565/llama-server", "true")
    (clean / "envs/llamacpp/current").symlink_to("b11565")
    got = envs.resolve("llamacpp")
    assert got is not None and got.source == "managed" and got.path.resolve() == srv.resolve()


def test_installed_versions_are_read_from_the_binaries(clean: Path) -> None:
    sh(clean / "envs/llamacpp/b11565/llama-server", 'echo "version: 11565 (abc1234)"; echo "built with gcc" >&2')
    (clean / "envs/llamacpp/current").symlink_to("b11565")
    sh(clean / "envs/vllm/bin/vllm", "true")
    sh(clean / "envs/vllm/bin/python", 'echo "some warning"; echo 0.30.0')
    sh(clean / "envs/sglang/bin/python", "echo 0.5.22.dev586+g86624f14e")
    rows = {r["id"]: r for r in envs.status()}
    assert rows["llamacpp"]["version"] == "b11565" and rows["llamacpp"]["managed"] is True
    assert rows["vllm"]["version"] == "0.30.0", "the last output line, not warnings printed before it"
    assert rows["sglang"]["version"] == "0.5.22.dev586+g86624f14e"
    assert all(r["installed"] for r in rows.values())


@pytest.mark.parametrize(("out", "want"), [
    # verbatim from a real b11567 release binary
    ("version: 0.6.0-dev (build 11567, commit 972b5fcc1)\nbuilt with GNU 11.4.0 for Linux x86_64", "b11567"),
    ("version: 11565 (abc1234)\nbuilt with gcc", "b11565"),
    # a local build without a git history reports build 1: not a release number, keep what it says
    ("version: 0.5.0-dev (build 1, commit 81ff93e)", "0.5.0-dev (build 1)"),
    ("version: 0.5.0-dev", "0.5.0-dev"),
    ("no version here", None),
])
def test_llama_version_parsing(out: str, want: str | None) -> None:
    assert envs.parse_llama_version(out) == want


def test_a_release_is_newer_than_a_local_build_and_equal_to_itself() -> None:
    assert envs.is_newer("b11567", "0.5.0-dev (build 1)")
    assert not envs.is_newer("b11567", "b11567") and not envs.is_newer("b11567", "b11568")
    assert envs.is_newer("b11568", "b11567")


# ---------------------------------------------------------------- resumable download
class RangeServer:
    """Serves `data`; the first `drops` responses are cut off half way (a stalled CDN connection). `honor_range=False`
    models a server that ignores Range and always answers 200 with the whole body."""

    def __init__(self, data: bytes, drops: int = 0, honor_range: bool = True) -> None:
        outer = self
        self.requests: list[str | None] = []
        self.left = drops

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def do_GET(self) -> None:
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                start = int(rng.split("=")[1].rstrip("-")) if rng and honor_range else 0
                if start >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
                body = data[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body)))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                if outer.left > 0:
                    outer.left -= 1
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.connection.close()  # cut mid-body: the client sees an incomplete read
                    return
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/f.bin"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()


DATA = bytes(range(256)) * 4096  # 1 MiB


def test_a_dropped_connection_is_resumed_from_where_it_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs.time, "sleep", lambda _s: None)
    srv = RangeServer(DATA, drops=2)
    logs: list[str] = []
    try:
        envs.download(srv.url, tmp_path / "f.bin", len(DATA), logs.append)
    finally:
        srv.close()
    assert (tmp_path / "f.bin").read_bytes() == DATA, "the resumed file is byte-identical, not truncated or duplicated"
    assert srv.requests[0] is None and srv.requests[1].startswith("bytes=") and srv.requests[2].startswith("bytes="), srv.requests
    assert int(srv.requests[2].split("=")[1].rstrip("-")) > int(srv.requests[1].split("=")[1].rstrip("-")), "each retry continues further on"
    assert sum("resuming" in line for line in logs) == 2


def test_a_server_that_ignores_range_restarts_the_download_instead_of_corrupting_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs.time, "sleep", lambda _s: None)
    srv = RangeServer(DATA, drops=1, honor_range=False)
    try:
        envs.download(srv.url, tmp_path / "f.bin", len(DATA), lambda _l: None)
    finally:
        srv.close()
    assert (tmp_path / "f.bin").read_bytes() == DATA


def test_a_download_that_never_completes_gives_up_with_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs.time, "sleep", lambda _s: None)
    srv = RangeServer(DATA, drops=99)
    try:
        with pytest.raises(RuntimeError, match=r"download of f\.bin failed after \d+ attempts"):
            envs.download(srv.url, tmp_path / "f.bin", len(DATA), lambda _l: None, retries=3)
    finally:
        srv.close()


def test_a_failed_download_of_unknown_size_is_an_error_not_an_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs.time, "sleep", lambda _s: None)
    srv = RangeServer(DATA, drops=99)  # always cut off half way, and the caller does not know the size
    try:
        with pytest.raises(RuntimeError, match=r"download of f\.bin failed"):
            envs.download(srv.url, tmp_path / "f.bin", None, lambda _l: None, retries=2)
    finally:
        srv.close()


def test_unknown_size_download_still_resumes_and_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs.time, "sleep", lambda _s: None)
    srv = RangeServer(DATA, drops=1)
    try:
        envs.download(srv.url, tmp_path / "f.bin", None, lambda _l: None)
    finally:
        srv.close()
    assert (tmp_path / "f.bin").read_bytes() == DATA


def test_a_file_of_the_wrong_size_is_not_accepted(tmp_path: Path) -> None:
    srv = RangeServer(DATA)
    try:
        with pytest.raises(RuntimeError, match="expected 1000 bytes"):
            envs.download(srv.url, tmp_path / "f.bin", 1000, lambda _l: None, retries=0)
    finally:
        srv.close()


# ---------------------------------------------------------------- llama.cpp install against a fake GitHub
def tarball(files: dict[str, tuple[str, int]], extra: Iterator[tarfile.TarInfo] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, (content, mode) in files.items():
            data = content.encode()
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), mode
            tf.addfile(ti, io.BytesIO(data))
        for ti in extra or []:
            tf.addfile(ti)
    return buf.getvalue()


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch, clean: Path) -> Iterator[dict[str, object]]:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr(envs, "driver_cuda_version", lambda: None)
    state: dict[str, object] = {"assets": {}, "tags": []}
    holder: dict[str, Fake] = {}

    def releases() -> bytes:
        base = holder["f"].url
        sizes = {n: len(b) for n, b in state["assets"].items()}  # type: ignore[attr-defined]
        return json.dumps([release(t, [asset(t, "ubuntu-x64")], base, sizes) for t in sorted(state["tags"], reverse=True)]).encode()  # type: ignore[arg-type]

    routes: dict[str, object] = {"/repos/ggml-org/llama.cpp/releases": releases}

    class Routes(dict):  # type: ignore[type-arg]
        def get(self, k: str, d: object = None) -> object:
            if k.startswith("/dl/"):
                return state["assets"].get(k[4:], d)  # type: ignore[attr-defined]
            return super().get(k, d)

    f = Fake(Routes(routes))
    holder["f"] = f
    monkeypatch.setenv("MQDE_GITHUB_API", f.url)
    state["fake"] = f
    yield state
    f.close()


def put_release(state: dict[str, object], tag: str, server_script: str = '#!/bin/sh\necho "version: {n} (abc)"') -> None:
    n = tag[1:]
    state["tags"].append(tag)  # type: ignore[attr-defined]
    state["assets"][asset(tag, "ubuntu-x64")] = tarball({  # type: ignore[index]
        f"llama-{tag}/llama-server": (server_script.replace("{n}", n), 0o644),  # archived without the exec bit: install must set it
        f"llama-{tag}/llama-quantize": ("#!/bin/sh\ntrue", 0o644), f"llama-{tag}/libggml.so": ("lib", 0o644)})


def test_llamacpp_install_downloads_verifies_and_switches_current(github: dict[str, object], clean: Path) -> None:
    put_release(github, "b11565")
    logs: list[str] = []
    assert envs.install("llamacpp", None, logs.append) == "b11565"
    r = envs.resolve("llamacpp")
    assert r is not None and r.source == "managed"
    assert envs.installed_version("llamacpp", r) == "b11565"
    cur = clean / "envs/llamacpp/current"
    assert cur.is_symlink() and cur.readlink() == Path("b11565")
    assert (r.path.parent / "llama-quantize").stat().st_mode & 0o111, "executables get their exec bit"
    assert not (r.path.parent / "libggml.so").stat().st_mode & 0o111, "libraries do not"
    assert not [p for p in (clean / "envs/llamacpp").iterdir() if p.name.startswith(".b")], "no temp directory left behind"
    assert any("downloading" in line for line in logs)


def test_update_keeps_the_previous_release_for_rollback_and_drops_older_ones(github: dict[str, object], clean: Path) -> None:
    for t in ("b11565", "b11566", "b11567"):
        put_release(github, t)
        envs.install("llamacpp", t, lambda _l: None)
    root = clean / "envs/llamacpp"
    assert sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()) == ["b11566", "b11567"]
    assert (root / "current").readlink() == Path("b11567")


def test_a_release_whose_llama_server_does_not_start_is_not_installed(github: dict[str, object], clean: Path) -> None:
    put_release(github, "b11565")
    envs.install("llamacpp", "b11565", lambda _l: None)
    put_release(github, "b11566", server_script="#!/bin/sh\nexit 3")
    with pytest.raises(RuntimeError, match="does not start"):
        envs.install("llamacpp", "b11566", lambda _l: None)
    root = clean / "envs/llamacpp"
    assert (root / "current").readlink() == Path("b11565"), "the working install is untouched"
    assert not (root / "b11566").exists()


def test_a_server_that_does_not_start_reports_why_in_the_error(github: dict[str, object], clean: Path) -> None:
    put_release(github, "b11566", server_script="#!/bin/sh\necho 'error while loading shared libraries: libcudart.so.12: cannot open shared object file' >&2\nexit 127")
    with pytest.raises(RuntimeError, match="libcudart.so.12: cannot open shared object file"):
        envs.install("llamacpp", "b11566", lambda _l: None)


def test_downloaded_archives_are_kept_for_the_next_attempt_and_dropped_after_success(github: dict[str, object], clean: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """A 700 MB download must not be thrown away because the install step after it failed (here: a transient start failure)."""
    put_release(github, "b11566")
    real = envs._run_out
    monkeypatch.setattr(envs, "_run_out", lambda *a, **k: None)  # the post-download `llama-server --version` check fails once
    with pytest.raises(RuntimeError, match="does not start"):
        envs.install("llamacpp", "b11566", lambda _l: None)
    fake: Fake = github["fake"]  # type: ignore[assignment]
    assert len([h for h in fake.hits if h.startswith("/dl/")]) == 1
    assert (clean / "envs/llamacpp/.cache" / asset("b11566", "ubuntu-x64")).is_file(), "the archive is still there"
    monkeypatch.setattr(envs, "_run_out", real)
    logs: list[str] = []
    envs.install("llamacpp", "b11566", logs.append)
    assert len([h for h in fake.hits if h.startswith("/dl/")]) == 1, "the retry used the cached archive: no second download"
    assert any("using the downloaded" in line for line in logs)
    assert (clean / "envs/llamacpp/current").readlink() == Path("b11566")
    assert not list((clean / "envs/llamacpp/.cache").glob("*")) if (clean / "envs/llamacpp/.cache").exists() else True, "emptied after success"


def test_a_cached_archive_of_the_wrong_size_is_downloaded_again(github: dict[str, object], clean: Path) -> None:
    put_release(github, "b11566")
    cache = clean / "envs/llamacpp/.cache"
    cache.mkdir(parents=True)
    (cache / asset("b11566", "ubuntu-x64")).write_bytes(b"truncated garbage")
    envs.install("llamacpp", "b11566", lambda _l: None)
    assert envs.resolve("llamacpp") is not None


def test_archive_members_cannot_escape_the_install_directory(github: dict[str, object], clean: Path) -> None:
    github["tags"].append("b11565")  # type: ignore[attr-defined]
    evil = tarfile.TarInfo("../../escaped.txt")
    github["assets"][asset("b11565", "ubuntu-x64")] = tarball({"llama-b11565/llama-server": ("#!/bin/sh\ntrue", 0o755)}, iter([evil]))  # type: ignore[index]
    with pytest.raises(RuntimeError, match="outside"):
        envs.install("llamacpp", "b11565", lambda _l: None)
    assert not (clean / "escaped.txt").exists() and not (clean.parent / "escaped.txt").exists()
    assert not (clean / "envs/llamacpp/current").exists()


def test_dry_run_touches_nothing(github: dict[str, object], clean: Path) -> None:
    put_release(github, "b11565")
    logs: list[str] = []
    envs.install("llamacpp", None, logs.append, dry_run=True)
    assert any("download" in line and "b11565" in line for line in logs)
    assert not (clean / "envs").exists()


# ---------------------------------------------------------------- PyPI install (vllm / sglang) with a fake uv
@pytest.fixture
def fake_uv(clean: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `uv` that records its arguments and, for `venv`, creates an env whose python reports $FAKE_VERSION."""
    rec = clean / "uv.calls"
    uv = sh(clean / "toolbin/uv", f'''echo "$@" >> {rec}
case "$1" in
  python) echo /usr/bin/python3.12 ;;
  venv) mkdir -p "$2/bin"; printf '#!/bin/sh\\necho "${{FAKE_VERSION}}"\\n' > "$2/bin/python"; chmod +x "$2/bin/python"; touch "$2/bin/vllm"; chmod +x "$2/bin/vllm" ;;
  pip) [ -n "$FAKE_PIP_FAIL" ] && {{ echo "boom: no matching distribution" >&2; exit 1; }}; exit 0 ;;
esac''')
    monkeypatch.setenv("PATH", f"{uv.parent}:{clean / 'emptybin'}:/bin:/usr/bin")
    monkeypatch.setenv("MQDE_PIP_INDEX", "http://index.invalid/simple")
    return rec


def test_fresh_install_creates_the_env_then_installs_the_pinned_version(fake_uv: Path, clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VERSION", "0.31.0")
    logs: list[str] = []
    assert envs.install("vllm", "0.31.0", logs.append) == "0.31.0"
    calls = fake_uv.read_text().splitlines()
    env = clean / "envs/vllm"
    assert calls[0] == "python find 3.12"
    assert calls[1] == f"venv {env} --python /usr/bin/python3.12 --seed"
    assert calls[2] == (f"pip install --python {env}/bin/python --index-url http://index.invalid/simple "
                        "--index-strategy unsafe-best-match vllm==0.31.0")
    r = envs.resolve("vllm")
    assert r is not None and r.source == "managed" and envs.installed_version("vllm", r) == "0.31.0"


def test_update_reuses_the_existing_env(fake_uv: Path, clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VERSION", "0.30.0")
    envs.install("vllm", "0.30.0", lambda _l: None)
    fake_uv.write_text("")
    monkeypatch.setenv("FAKE_VERSION", "0.31.0")
    envs.install("vllm", "0.31.0", lambda _l: None)
    calls = fake_uv.read_text().splitlines()
    assert len(calls) == 1 and calls[0].startswith("pip install") and calls[0].endswith("vllm==0.31.0"), "no new venv for an update"


def test_a_failed_first_install_leaves_no_env_that_resolve_would_pick_up(fake_uv: Path, clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VERSION", "0.31.0")
    monkeypatch.setenv("FAKE_PIP_FAIL", "1")
    logs: list[str] = []
    with pytest.raises(RuntimeError, match="command failed"):
        envs.install("sglang", "0.5.21", logs.append)
    assert not (clean / "envs/sglang").exists() and envs.resolve("sglang") is None
    assert any("boom: no matching distribution" in line for line in logs), "uv's own error text reaches the log"


def test_a_failed_update_keeps_the_working_env(fake_uv: Path, clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VERSION", "0.30.0")
    envs.install("vllm", "0.30.0", lambda _l: None)
    monkeypatch.setenv("FAKE_PIP_FAIL", "1")
    with pytest.raises(RuntimeError):
        envs.install("vllm", "0.31.0", lambda _l: None)
    r = envs.resolve("vllm")
    assert r is not None and envs.installed_version("vllm", r) == "0.30.0"


def test_install_is_refused_when_the_installed_version_is_not_what_was_asked(fake_uv: Path, clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VERSION", "0.30.0")  # uv "succeeds" but the env still reports the old version
    with pytest.raises(RuntimeError, match=r"reports '0\.30\.0', expected '0\.31\.0'"):
        envs.install("vllm", "0.31.0", lambda _l: None)


def test_default_version_is_the_newest_release(fake_uv: Path, clean: Path, pypi: Fake, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VERSION", "0.31.0")
    assert envs.install("vllm", None, lambda _l: None) == "0.31.0"
    assert fake_uv.read_text().splitlines()[-1].endswith("vllm==0.31.0")


def test_uv_is_bootstrapped_into_a_private_venv_when_missing(clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", f"{clean / 'emptybin'}:/bin:/usr/bin")
    logs: list[str] = []
    envs.ensure_uv(envs.Runner(logs.append, dry_run=True))
    assert any("venv" in line and ".tools" in line for line in logs)
    assert any("pip install" in line and line.rstrip().endswith("uv") for line in logs)


def test_a_second_install_while_one_is_running_is_refused_not_interleaved(clean: Path) -> None:
    import fcntl

    (clean / "envs").mkdir()
    with (clean / "envs/.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="another install is already running"):
            envs.install("llamacpp", "b11565", lambda _l: None)
    envs.install("vllm", "0.31.0", lambda _l: None, dry_run=True)  # a dry run never needs the lock


# ---------------------------------------------------------------- CLI
def test_cli_status_json_and_check(clean: Path, pypi: Fake, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sh(clean / "envs/vllm/bin/vllm", "true")
    sh(clean / "envs/vllm/bin/python", "echo 0.30.0")
    monkeypatch.setenv("MQDE_GITHUB_API", "http://127.0.0.1:9")  # unreachable: reported per backend, the command still succeeds
    assert envs.main(["status", "--check", "--json"]) == 0
    rows = {r["id"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["vllm"]["latest"] == "0.31.0" and rows["vllm"]["update_available"] is True and rows["vllm"]["version"] == "0.30.0"
    assert rows["sglang"]["installed"] is False and rows["sglang"]["update_available"] is True, "not installed: installing is the 'update'"
    assert rows["llamacpp"]["latest"] is None and rows["llamacpp"]["check_error"]


def test_cli_up_to_date_is_not_an_update(clean: Path, pypi: Fake, capsys: pytest.CaptureFixture[str]) -> None:
    sh(clean / "envs/vllm/bin/vllm", "true")
    sh(clean / "envs/vllm/bin/python", "echo 0.31.0")
    sh(clean / "envs/sglang/bin/python", "echo 0.5.22.dev586+g86624f14e")  # a source build ahead of PyPI's 0.5.21
    envs.main(["status", "--check", "--json"])
    rows = {r["id"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["vllm"]["update_available"] is False and rows["sglang"]["update_available"] is False


def test_cli_rejects_bad_input_with_exit_code_1(clean: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert envs.main(["install", "vllm", "--version", "1; rm -rf /", "--dry-run"]) == 1
    assert "invalid version" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        envs.main(["install", "tensorrt"])


def test_ensure_only_installs_what_is_missing_and_only_what_was_asked_for(clean: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sh(clean / "envs/sglang/bin/python", "echo 0.5.21")
    assert envs.main(["ensure", "vllm", "sglang", "--dry-run"]) == 0  # dry run does not look up the newest version
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "already installed: sglang", "llamacpp was not asked for, so it is not mentioned"
    assert "uv venv" in out and "vllm==<latest>" in out and "sglang==" not in out
