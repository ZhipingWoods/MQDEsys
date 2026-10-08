"""scripts/privacy_scan.sh: what it blocks before a push, and that it does not cry wolf."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "privacy_scan.sh"
ME = "me@example.org"


def git(repo: Path, *a: str, env: dict[str, str] | None = None) -> str:
    e = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", **(env or {})}
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True, env=e).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Me")
    git(tmp_path, "config", "user.email", ME)
    (tmp_path / "a.txt").write_text("base\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "tag", "base")
    return tmp_path


def commit(repo: Path, files: dict[str, str | bytes], msg: str = "change", env: dict[str, str] | None = None) -> None:
    for name, body in files.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body if isinstance(body, bytes) else body.encode())
    git(repo, "add", ".")
    git(repo, "commit", "-qm", msg, env=env)


def scan(repo: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("MQDE_PRIVACY_PATTERNS", "MQDE_ALLOWED_EMAILS")}
    env.update({"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", **(extra_env or {})})
    return subprocess.run(["bash", str(SCRIPT), "base..HEAD"], cwd=repo, capture_output=True, text=True, env=env)


def test_clean_change_passes(repo: Path) -> None:
    commit(repo, {"doc.md": "Run `mqde estimate` on your model.\n"})
    r = scan(repo)
    assert r.returncode == 0 and "0 fail" in r.stdout


def ip(*octets: int) -> str:
    """Built at run time: a literal private address here would make the scanner fail on its own test file."""
    return ".".join(map(str, octets))


@pytest.mark.parametrize(("line", "what"), [
    ("host = " + ip(10, 20, 30, 40), "private IP"),
    ("gw = " + ip(192, 168, 1, 5), "private IP"),
    ("net = " + ip(172, 20, 0, 9), "private IP"),
    ("token = ghp_" + "a" * 30, "token"),
    ("HF_TOKEN=hf_" + "b" * 28, "token"),
    ("-----BEGIN " + "OPENSSH PRIVATE KEY-----", "token"),
])
def test_secrets_and_private_addresses_fail(repo: Path, line: str, what: str) -> None:
    commit(repo, {"cfg.txt": line + "\n"})
    r = scan(repo)
    assert r.returncode == 1 and "FAIL" in r.stdout and what in r.stdout


def test_public_addresses_and_version_numbers_are_not_flagged(repo: Path) -> None:
    commit(repo, {"v.txt": "version 10.2.3\nserver 8.8.8.8\nrange 172.32.0.1\nip 1.10.20.30.40\n"})
    assert scan(repo).returncode == 0


def test_absolute_paths_only_warn(repo: Path) -> None:
    commit(repo, {"d.md": "models live in /cache/someone/hub\n"})
    r = scan(repo)
    assert r.returncode == 0 and "WARN" in r.stdout and "1 warn" in r.stdout


def test_binary_and_large_files_fail(repo: Path) -> None:
    commit(repo, {"m/model.gguf": b"GGUF"})
    assert "binary / media file" in scan(repo).stdout
    git(repo, "tag", "-f", "mid")
    commit(repo, {"big.txt": b"x" * (1024 * 1024 + 10)})
    r = subprocess.run(["bash", str(SCRIPT), "mid..HEAD"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1 and "over 1 MiB" in r.stdout


def test_foreign_identity_fails_and_can_be_allowed(repo: Path) -> None:
    other = {"GIT_AUTHOR_EMAIL": "old@corp.example", "GIT_COMMITTER_EMAIL": "old@corp.example"}
    commit(repo, {"x.txt": "x\n"}, env=other)
    r = scan(repo)
    assert r.returncode == 1 and "old@corp.example" in r.stdout
    assert scan(repo, {"MQDE_ALLOWED_EMAILS": f"{ME},old@corp.example"}).returncode == 0


def test_secret_in_the_commit_message_fails(repo: Path) -> None:
    commit(repo, {"y.txt": "y\n"}, msg="debug with ghp_" + "c" * 30)
    r = scan(repo)
    assert r.returncode == 1 and "commit message" in r.stdout


def test_a_secret_added_then_removed_is_still_caught(repo: Path) -> None:
    commit(repo, {"s.txt": "key = ghp_" + "d" * 30 + "\n"})
    commit(repo, {"s.txt": "key = <removed>\n"})
    assert scan(repo).returncode == 1  # it is in the history that would be pushed


def test_private_patterns_file_outside_the_repo(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    pats = tmp_path_factory.mktemp("priv") / "patterns.txt"
    pats.write_text("UnreleasedModel-[0-9]+\n")
    commit(repo, {"n.md": "tested on UnreleasedModel-5\n"}, msg="note about UnreleasedModel-5")
    assert scan(repo).returncode == 0  # without the file the scan cannot know
    r = scan(repo, {"MQDE_PRIVACY_PATTERNS": str(pats)})
    assert r.returncode == 1 and r.stdout.count("private pattern") >= 2  # the added line and the message


def test_empty_range_and_bad_range(repo: Path) -> None:
    assert subprocess.run(["bash", str(SCRIPT), "HEAD..HEAD"], cwd=repo, capture_output=True, text=True).returncode == 0
    assert subprocess.run(["bash", str(SCRIPT), "nope..HEAD"], cwd=repo, capture_output=True, text=True).returncode == 2
