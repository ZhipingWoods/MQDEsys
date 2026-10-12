"""Managed inference-backend environments: vLLM, SGLang and llama.cpp (resolve, probe, check for updates, install).

Standard library only on purpose: this runs before any backend (or torch) exists, from ``scripts/setup_backend.py``
and from the server's "update" button, and the adapters import it just to *find* a binary.

Layout, under ``MQDE_ENVS_DIR`` (default ``<repo>/build/envs``)::

    vllm/          a uv-created venv holding ``vllm`` from PyPI
    sglang/        a uv-created venv holding ``sglang`` from PyPI
    llamacpp/<tag>/  an official llama.cpp release (llama-server, llama-quantize, ... and its shared libraries)
    llamacpp/current -> <tag>
    .tools/        a venv holding ``uv`` itself, only when ``uv`` is not already on PATH

Resolution order for every backend is: the explicit ``MQDE_*`` variable the adapters have always honoured, then the
managed environment above (so an install or update through this module takes effect without exporting anything),
then whatever was found before (PATH, repo ``build/`` trees). Updates are pinned (``vllm==X``): what is installed is
always a version someone can name, and the "update" button can say which one.
"""

from __future__ import annotations

import argparse
import configparser
import contextlib
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKENDS = ("vllm", "sglang", "llamacpp")
PYPI_PACKAGE = {"vllm": "vllm", "sglang": "sglang"}
LLAMACPP_REPO = "ggml-org/llama.cpp"
# What a version string may look like when it comes from a client: never anything a shell or pip would treat specially.
VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+!_-]{0,39}$")
PYTHON_CANDIDATES = ("3.12", "3.11", "3.13", "3.10")

Log = Callable[[str], None]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def envs_dir() -> Path:
    return Path(os.environ.get("MQDE_ENVS_DIR") or repo_root() / "build" / "envs")


# ------------------------------------------------------------------------------------------------ versions
def version_key(v: str) -> tuple[tuple[int, ...], int, int]:
    """Order PEP 440-like versions: ``0.5.22.dev5`` < ``0.5.22`` < ``0.5.22.post1``; build tags ``b11565`` compare by number."""
    s = v.strip().lower().lstrip("v")
    if re.fullmatch(r"b\d+", s):
        return ((int(s[1:]),), 0, 0)
    m = re.match(r"(\d+(?:\.\d+)*)(.*)$", s)
    if not m:
        return ((0,), -3, 0)
    rel = [int(x) for x in m.group(1).split(".")]
    while len(rel) > 1 and rel[-1] == 0:
        rel.pop()
    tail = m.group(2).split("+")[0]
    t = re.match(r"[.\-_]?(dev|a|alpha|b|beta|rc|c|pre|preview|post|rev|r)[.\-_]?(\d*)", tail)
    if not t:
        return (tuple(rel), 0, 0)
    kind, n = t.group(1), int(t.group(2) or 0)
    rank = {"dev": -3, "a": -2, "alpha": -2, "b": -2, "beta": -2, "rc": -1, "c": -1, "pre": -1, "preview": -1}.get(kind, 1)
    return (tuple(rel), rank, n)


def is_prerelease(v: str) -> bool:
    return version_key(v)[1] < 0


def is_newer(latest: str | None, installed: str | None) -> bool:
    """True when `latest` should be offered as an update over `installed` (unknown installed version: offer it)."""
    if not latest:
        return False
    if not installed:
        return True
    try:
        return version_key(latest) > version_key(installed)
    except (ValueError, TypeError):
        return latest != installed


def valid_version(v: str) -> bool:
    return bool(VERSION_RE.fullmatch(v))


# ------------------------------------------------------------------------------------------------ resolution
@dataclass(frozen=True)
class Resolved:
    path: Path  # vllm: the `vllm` binary; sglang: the env's python; llamacpp: `llama-server`
    source: str  # "env" | "managed" | "path" | "legacy"


def _first(cands: Iterable[tuple[str, str]], ok: Callable[[Path], bool]) -> Resolved | None:
    for c, source in cands:
        if c and ok(Path(c)):
            return Resolved(Path(c), source)
    return None


def managed_llamacpp_server() -> Path | None:
    cur = envs_dir() / "llamacpp" / "current"
    if not cur.exists():
        return None
    for p in sorted(cur.resolve().rglob("llama-server")):
        if p.is_file():
            return p
    return None


def resolve(backend: str) -> Resolved | None:
    """Where this backend's executable lives right now, or None. Same order the adapters use."""
    root, e = repo_root(), envs_dir()
    if backend == "vllm":
        return _first([(os.environ.get("MQDE_VLLM_BIN", ""), "env"), (str(e / "vllm/bin/vllm"), "managed"),
                       (shutil.which("vllm") or "", "path"), (str(root / "build/vllm/bin/vllm"), "legacy")], Path.is_file)
    if backend == "sglang":
        return _first([(os.environ.get("MQDE_SGLANG_PYTHON", ""), "env"), (str(e / "sglang/bin/python"), "managed"),
                       (str(root / "build/sglang/bin/python"), "legacy")], Path.is_file)
    if backend == "llamacpp":
        m = managed_llamacpp_server()
        return _first([(os.environ.get("MQDE_LLAMA_SERVER_BIN", ""), "env"), (str(m or ""), "managed"),
                       (str(root / "build/llamacpp-server/bin/llama-server"), "legacy"),
                       (str(root / "build/llamacpp-metal/bin/llama-server"), "legacy")], Path.is_file)
    raise ValueError(f"unknown backend {backend!r}")


def _run_out(cmd: list[str], timeout: float = 90) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout + r.stderr) if r.returncode == 0 else None


def parse_llama_version(text: str) -> str | None:
    """``version: 11565 (abc123)`` (releases) -> ``b11565``; ``version: 0.5.0-dev (build 1, commit 81ff93e)`` (local builds) as is."""
    m = re.search(r"version:\s*(\S+)", text)
    if not m:
        return None
    v = m.group(1)
    return f"b{v}" if v.isdigit() else v


def installed_version(backend: str, r: Resolved) -> str | None:
    if backend == "llamacpp":
        out = _run_out([str(r.path), "--version"], 30)
        return parse_llama_version(out) if out else None
    py = r.path if backend == "sglang" else r.path.parent / "python"
    pkg = PYPI_PACKAGE[backend]
    if py.is_file():
        out = _run_out([str(py), "-c", f"import importlib.metadata as m; print(m.version('{pkg}'))"], 60)
        if out:
            return out.strip().splitlines()[-1]
    if backend == "vllm":  # a `vllm` on PATH that is not in a venv we can ask
        out = _run_out([str(r.path), "--version"], 120)
        m = re.search(r"\d+\.\d+[\w.+-]*", out or "")
        return m.group(0) if m else None
    return None


def status() -> list[dict[str, Any]]:
    out = []
    for b in BACKENDS:
        r = resolve(b)
        managed = bool(r and r.source == "managed")
        out.append({"id": b, "installed": r is not None, "version": installed_version(b, r) if r else None,
                    "path": str(r.path) if r else None, "source": r.source if r else None, "managed": managed,
                    "envs_dir": str(envs_dir())})
    return out


# ------------------------------------------------------------------------------------------------ latest versions
def _urlopen(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mqde-setup", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - https URLs we construct, or a test server
        data: bytes = r.read()
    return data


def _pip_conf_index() -> str | None:
    """`index-url` from the first pip config file that sets one -- read directly, because the system python that runs
    this script often has no pip module at all while the user's pip.conf still names the mirror to use."""
    home = Path.home()
    files = [os.environ.get("PIP_CONFIG_FILE", ""), str(home / ".config/pip/pip.conf"), str(home / ".pip/pip.conf"), "/etc/pip.conf"]
    for f in files:
        if not f or not Path(f).is_file():
            continue
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read(f)
        except configparser.Error:
            continue
        for sect in ("global", "install"):
            if cp.has_option(sect, "index-url"):
                return cp.get(sect, "index-url").strip().rstrip("/")
    return None


def pip_index_url() -> str:
    for k in ("MQDE_PIP_INDEX", "PIP_INDEX_URL", "UV_INDEX_URL"):
        if os.environ.get(k):
            return os.environ[k].rstrip("/")
    return _pip_conf_index() or "https://pypi.org/simple"


def simple_index_versions(html: str, package: str) -> list[str]:
    """Versions listed on a PEP 503 project page (wheel and sdist file names), skipping yanked files."""
    norm = re.sub(r"[-_.]+", "-", package).lower()
    found: set[str] = set()
    for m in re.finditer(r"<a\b([^>]*)>([^<]*)</a>", html):
        attrs, name = m.group(1), m.group(2).strip()
        if "data-yanked" in attrs:
            continue
        stem = re.sub(r"\.(whl|tar\.gz|zip)$", "", name)
        if name.endswith(".whl"):
            parts = stem.split("-")
            dist, ver = parts[0], parts[1] if len(parts) > 1 else ""
        else:
            dist, _, ver = stem.rpartition("-")
        if re.sub(r"[-_.]+", "-", dist).lower() == norm and ver:
            found.add(ver)
    return sorted(found, key=version_key)


def pypi_latest(package: str, include_pre: bool = False, timeout: float = 20) -> str:
    html = _urlopen(f"{pip_index_url()}/{package}/", timeout).decode("utf-8", "replace")
    vs = simple_index_versions(html, package)
    if not include_pre:
        vs = [v for v in vs if not is_prerelease(v)]
    if not vs:
        raise RuntimeError(f"no releases of {package} found on {pip_index_url()}")
    return vs[-1]


def github_api() -> str:
    return os.environ.get("MQDE_GITHUB_API", "https://api.github.com").rstrip("/")


def llamacpp_releases(timeout: float = 20) -> list[dict[str, Any]]:
    """Releases tagged ``b<number>`` (the repo also carries unrelated tags: its ``latest`` marker is not a build)."""
    data = json.loads(_urlopen(f"{github_api()}/repos/{LLAMACPP_REPO}/releases?per_page=30", timeout))
    rel = [r for r in data if re.fullmatch(r"b\d+", str(r.get("tag_name", ""))) and not r.get("draft")]
    return sorted(rel, key=lambda r: version_key(r["tag_name"]), reverse=True)


def latest(backend: str) -> str:
    if backend == "llamacpp":
        rel = llamacpp_releases()
        if not rel:
            raise RuntimeError("no llama.cpp release builds found")
        return str(rel[0]["tag_name"])
    return pypi_latest(PYPI_PACKAGE[backend])


# ------------------------------------------------------------------------------------------------ llama.cpp assets
def driver_cuda_version() -> tuple[int, int] | None:
    out = _run_out(["nvidia-smi"], 15)
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def pick_llamacpp_assets(names: list[str], tag: str, system: str, machine: str,
                         cuda: tuple[int, int] | None) -> list[str]:
    """Asset file names to download for this platform (a CUDA build also needs its ``cudart`` bundle). Raises if none fit."""
    m = machine.lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x64" if m in ("x86_64", "amd64") else m
    if system == "Linux":
        if cuda and arch in ("x64", "arm64"):
            best: tuple[int, int] | None = None
            for n in names:
                mm = re.fullmatch(rf"llama-{tag}-bin-ubuntu-cuda-(\d+)\.(\d+)-{arch}\.tar\.gz", n)
                if mm and (int(mm.group(1)), int(mm.group(2))) <= cuda and (best is None or (int(mm.group(1)), int(mm.group(2))) > best):
                    best = (int(mm.group(1)), int(mm.group(2)))
            if best:
                ver = f"{best[0]}.{best[1]}"
                want = [f"llama-{tag}-bin-ubuntu-cuda-{ver}-{arch}.tar.gz", f"cudart-llama-{tag}-bin-ubuntu-cuda-{ver}-{arch}.tar.gz"]
                return [n for n in want if n in names] if want[0] in names else []
        cpu = f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"
        if cpu in names:
            return [cpu]
    elif system == "Darwin":
        mac = f"llama-{tag}-bin-macos-{arch}.tar.gz"
        if mac in names:
            return [mac]
    raise RuntimeError(f"no llama.cpp {tag} release asset for {system}/{machine}; build it from source and set MQDE_LLAMA_SERVER_BIN")


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for m in tf.getmembers():
        target = (dest / m.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"refusing to extract {m.name!r} outside {dest}")
        if m.issym() or m.islnk():
            link = (target.parent / m.linkname).resolve() if m.issym() else (dest / m.linkname).resolve()
            if root != link and root not in link.parents:
                raise RuntimeError(f"refusing link {m.name!r} -> {m.linkname!r} outside {dest}")
    tf.extractall(dest)  # noqa: S202 - members validated above


def download(url: str, dest: Path, expected_size: int | None, log: Log, retries: int = 8, timeout: float = 60) -> None:
    """Download `url` to `dest`, resuming with HTTP Range after a stalled or dropped connection (GitHub's asset CDN does
    that on slow links) and restarting from zero when the server ignores Range. With `expected_size` the result must have
    exactly that many bytes; without it, a response that was read to its end counts as complete (a body cut short of its
    Content-Length raises and is retried like any other drop)."""
    have = 0
    dest.unlink(missing_ok=True)
    for attempt in range(retries + 1):
        headers = {"User-Agent": "mqde-setup"}
        if have:
            headers["Range"] = f"bytes={have}-"
        complete = False
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp, \
                    dest.open("ab" if have else "wb") as out:  # noqa: S310
                if have and resp.status != 206:  # Range ignored: the body is the whole file again
                    have = 0
                    out.seek(0)
                    out.truncate()
                start = have
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    have += len(chunk)
                    if have // (64 << 20) != (have - len(chunk)) // (64 << 20):
                        log(f"  {dest.name}: {have >> 20} MiB" + (f" of {expected_size >> 20}" if expected_size else ""))
                # HTTPResponse.read(amt) does not raise when the peer closes early: compare with Content-Length ourselves
                clen = resp.headers.get("Content-Length")
                if clen and clen.isdigit() and have - start < int(clen):
                    raise OSError(f"connection closed after {have - start} of {clen} bytes")
            complete = expected_size is None or have >= expected_size
        except urllib.error.HTTPError as e:
            if e.code == 416 and expected_size and have >= expected_size:  # everything was already there
                complete = True
            elif e.code < 500 and e.code != 429:
                raise RuntimeError(f"download of {dest.name} failed: HTTP {e.code}") from e
        except (OSError, http.client.HTTPException):
            pass
        if complete:
            break
        if attempt == retries:
            raise RuntimeError(f"download of {dest.name} failed after {retries + 1} attempts ({have} bytes"
                               + (f" of {expected_size})" if expected_size else ")"))
        log(f"  {dest.name}: connection interrupted at {have >> 20} MiB, resuming (attempt {attempt + 2}/{retries + 1})")
        time.sleep(min(2 ** attempt, 20))
    if expected_size is not None and dest.stat().st_size != expected_size:
        raise RuntimeError(f"download of {dest.name} has {dest.stat().st_size} bytes, expected {expected_size} bytes")


# ------------------------------------------------------------------------------------------------ install
class Runner:
    """Runs (or, with ``dry_run``, only prints) the commands an install needs; output is streamed to `log`."""

    def __init__(self, log: Log, dry_run: bool = False) -> None:
        self.log, self.dry_run = log, dry_run

    def run(self, cmd: list[str], env: dict[str, str] | None = None) -> None:
        self.log("+ " + " ".join(cmd))
        if self.dry_run:
            return
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL,
                             env={**os.environ, **(env or {})})
        assert p.stdout is not None
        for line in p.stdout:
            self.log(line.rstrip())
        if p.wait() != 0:
            raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd[:4])} ...")


def ensure_uv(r: Runner) -> str:
    u = shutil.which("uv")
    if u:
        return u
    tools = envs_dir() / ".tools"
    uv = tools / "bin" / "uv"
    if uv.is_file():
        return str(uv)
    r.log("uv is not installed: bootstrapping it into a private venv")
    r.run([sys.executable, "-m", "venv", str(tools)])
    r.run([str(tools / "bin" / "python"), "-m", "pip", "install", "--disable-pip-version-check", "-q",
           "--index-url", pip_index_url(), "uv"])
    return str(uv)


def pick_python(uv: str, r: Runner) -> str:
    """A Python for the venv: an explicit MQDE_BACKEND_PYTHON, else the first of 3.12/3.11/3.13/3.10 that is already on this
    machine (downloading interpreters is disabled: it needs network access the dev machines do not all have)."""
    explicit = os.environ.get("MQDE_BACKEND_PYTHON")
    if explicit:
        return explicit
    env = {**os.environ, "UV_PYTHON_DOWNLOADS": "never"}
    for c in PYTHON_CANDIDATES:
        if r.dry_run:
            return c
        p = subprocess.run([uv, "python", "find", c], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[-1]
    if sys.version_info >= (3, 10):
        return sys.executable
    raise RuntimeError("no Python >= 3.10 found (set MQDE_BACKEND_PYTHON)")


def install_pypi(backend: str, version: str | None, r: Runner) -> str:
    pkg = PYPI_PACKAGE[backend]
    env = envs_dir() / backend
    py = env / "bin" / "python"
    if version is None:
        version = latest(backend) if not r.dry_run else "<latest>"
        r.log(f"latest {pkg}: {version}")
    uv = ensure_uv(r) if not r.dry_run else "uv"
    fresh = not py.is_file()
    index = pip_index_url()
    try:
        if fresh:
            r.run([uv, "venv", str(env), "--python", pick_python(uv, r), "--seed"], {"UV_PYTHON_DOWNLOADS": "never"})
        r.run([uv, "pip", "install", "--python", str(py), "--index-url", index, "--index-strategy", "unsafe-best-match",
               f"{pkg}=={version}"], {"UV_PYTHON_DOWNLOADS": "never"})
        if not r.dry_run:
            got = installed_version(backend, Resolved(py if backend == "sglang" else env / "bin" / "vllm", "managed"))
            if got != version:
                raise RuntimeError(f"installed {pkg} reports {got!r}, expected {version!r}")
    except BaseException:
        if fresh and not r.dry_run:
            shutil.rmtree(env, ignore_errors=True)  # never leave a half-built env that resolve() would then pick up
        raise
    return version


def install_llamacpp(version: str | None, r: Runner) -> str:
    rels = llamacpp_releases()
    if version:
        rel = next((x for x in rels if x["tag_name"] == version), None)
        if rel is None:
            rel = json.loads(_urlopen(f"{github_api()}/repos/{LLAMACPP_REPO}/releases/tags/{version}", 20))
    else:
        if not rels:
            raise RuntimeError("no llama.cpp release builds found")
        rel = rels[0]
    tag = str(rel["tag_name"])
    by_name = {a["name"]: a for a in rel.get("assets", [])}
    names = pick_llamacpp_assets(list(by_name), tag, platform.system(), platform.machine(), driver_cuda_version())
    if not names:
        raise RuntimeError(f"llama.cpp {tag} has no asset matching this machine's CUDA driver")
    root = envs_dir() / "llamacpp"
    dest = root / tag
    r.log(f"llama.cpp {tag}: " + ", ".join(f"{n} ({by_name[n].get('size', 0) // 2**20} MiB)" for n in names))
    if r.dry_run:
        for n in names:
            r.log(f"+ download {by_name[n]['browser_download_url']} -> {dest}")
        return tag
    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{tag}.", dir=root))
    try:
        for n in names:
            r.log(f"downloading {n}")
            f = tmp / n
            download(by_name[n]["browser_download_url"], f, int(by_name[n].get("size") or 0) or None, r.log)
            with tarfile.open(f) as tf:
                _safe_extract(tf, tmp / "x")
            f.unlink()
        srv = next((p for p in sorted((tmp / "x").rglob("llama-server")) if p.is_file()), None)
        if srv is None:
            raise RuntimeError(f"llama.cpp {tag} archive has no llama-server")
        for p in srv.parent.iterdir():
            if p.is_file() and not p.suffix and not p.name.startswith("lib"):
                p.chmod(p.stat().st_mode | 0o111)
        got = parse_llama_version(_run_out([str(srv), "--version"], 30) or "")
        if got is None:
            raise RuntimeError("the downloaded llama-server does not start (--version failed)")
        shutil.rmtree(dest, ignore_errors=True)
        (tmp / "x").rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    link, new = root / "current", root / ".current.new"
    with contextlib.suppress(FileNotFoundError):
        new.unlink()
    new.symlink_to(tag)
    os.replace(new, link)
    # keep this release and the previous one (rollback), drop the rest
    older = sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and re.fullmatch(r"b\d+", p.name) and p.name != tag),
                   key=lambda p: version_key(p.name), reverse=True)
    for old in older[1:]:
        shutil.rmtree(old, ignore_errors=True)
    return tag


@contextlib.contextmanager
def install_lock() -> Iterator[None]:
    """One install at a time per envs dir (a manual run and the UI button must not interleave two uv runs on one venv)."""
    import fcntl

    envs_dir().mkdir(parents=True, exist_ok=True)
    with (envs_dir() / ".lock").open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another install is already running in this environments directory") from None
        yield


def install(backend: str, version: str | None, log: Log, dry_run: bool = False) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if version is not None and not valid_version(version):
        raise ValueError(f"invalid version {version!r}")
    if platform.system() == "Windows":
        raise RuntimeError("managed installs support Linux and macOS")
    r = Runner(log, dry_run)
    with contextlib.nullcontext() if dry_run else install_lock():
        got = install_llamacpp(version, r) if backend == "llamacpp" else install_pypi(backend, version, r)
    log(f"{backend} {got} ready" + (" (dry run)" if dry_run else ""))
    return got


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="setup_backend", description="Manage the vLLM / SGLang / llama.cpp environments.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="installed version and location of each backend")
    s.add_argument("--check", action="store_true", help="also look up the newest version (needs network)")
    s.add_argument("--json", action="store_true")
    i = sub.add_parser("install", help="install or update one backend (pinned to --version, default the newest)")
    i.add_argument("backend", choices=BACKENDS)
    i.add_argument("--version")
    i.add_argument("--dry-run", action="store_true", help="print what would run, change nothing")
    e = sub.add_parser("ensure", help="install every backend that is not installed yet")
    e.add_argument("backends", nargs="*", choices=BACKENDS, default=list(BACKENDS))
    e.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    def log(line: str) -> None:
        print(line, flush=True)

    try:
        if a.cmd == "status":
            rows = status()
            if a.check:
                for row in rows:
                    try:
                        row["latest"] = latest(row["id"])
                        row["update_available"] = is_newer(row["latest"], row["version"]) if row["installed"] else True
                    except Exception as ex:  # network trouble is a per-backend fact, not a failure of the command
                        row["latest"], row["update_available"], row["check_error"] = None, False, str(ex)[:200]
            if a.json:
                print(json.dumps(rows))
            else:
                for row in rows:
                    print(f"{row['id']:9} {'v' + str(row['version']) if row['installed'] else 'not installed':18} "
                          f"{row.get('source') or '-':8} {row.get('path') or ''}"
                          + (f"  latest {row['latest']}" if row.get("latest") else ""))
            return 0
        if a.cmd == "install":
            install(a.backend, a.version, log, a.dry_run)
            return 0
        wanted = list(dict.fromkeys(a.backends or BACKENDS))
        todo = [b for b in wanted if resolve(b) is None]
        log("already installed: " + (", ".join(b for b in wanted if b not in todo) or "none"))
        for b in todo:
            install(b, None, log, a.dry_run)
        return 0
    except (RuntimeError, ValueError, OSError) as ex:
        print(f"error: {ex}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
