"""scripts/check_abi.sh (T-1101): the HAL ABI layout is compared with a golden file; any layout change must fail it."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HDR = "cpp/include/mqde/hal/mqde_hal.h"
GOLD = "cpp/include/mqde/hal/abi_v1.golden"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    for rel in (HDR, GOLD, "scripts/check_abi.sh"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "scripts/check_abi.sh", *args], cwd=repo, capture_output=True, text=True, check=False)


def edit(repo: Path, old: str, new: str) -> None:
    """Replace one declaration. Matching ignores the comments and the spacing the formatter chooses, so reformatting the
    header never breaks these tests, but the line that is replaced must still exist exactly once."""
    p = repo / HDR
    lines = p.read_text().split("\n")
    want = [x.strip() for x in old.split("\n")]
    hits = [k for k in range(len(lines) - len(want) + 1)
            if [re.sub(r"\s*/\*.*?\*/", "", ln).strip() for ln in lines[k : k + len(want)]] == want]
    assert len(hits) == 1, f"{old!r} matched {len(hits)} times"
    lines[hits[0] : hits[0] + len(want)] = new.split("\n")
    p.write_text("\n".join(lines))


def test_the_committed_header_matches_its_golden_file(repo: Path) -> None:
    r = run(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "unchanged" in r.stdout


@pytest.mark.parametrize(
    ("old", "new", "what"),
    [
        ("char name[128];\nchar arch[32];", "char name[64];\nchar arch[32];", "a string shrinks"),
        ("int32_t kind;\nint32_t index;", "int32_t index;\nint32_t kind;", "two fields swap"),
        ("uint64_t mem_total_bytes;\nuint64_t mem_free_bytes;", "uint64_t mem_free_bytes;\nuint64_t mem_total_bytes;",
         "memory fields swap"),
        ("int (*alloc)(int device, size_t bytes, void** ptr);\nint (*free)(int device, void* ptr);",
         "int (*free)(int device, void* ptr);\nint (*alloc)(int device, size_t bytes, void** ptr);", "function pointers swap"),
        ("#define MQDE_HAL_ABI_VERSION 1", "#define MQDE_HAL_ABI_VERSION 2", "the version number changes"),
        ("double peak_tflops;", "float peak_tflops;", "a field changes type"),
    ],
)
def test_every_kind_of_layout_change_fails_the_check(repo: Path, old: str, new: str, what: str) -> None:
    edit(repo, old, new)
    r = run(repo)
    assert r.returncode == 1, f"{what}: not detected\n{r.stdout}"
    assert "breaking change" in r.stderr and r.stdout.startswith("---"), "the diff is shown"


def test_a_comment_only_edit_is_not_an_abi_change(repo: Path) -> None:
    edit(repo, "double peak_tflops;", "double peak_tflops; /* reworded comment */")
    assert run(repo).returncode == 0


def test_update_rewrites_the_golden_file_and_a_missing_golden_is_an_error(repo: Path) -> None:
    edit(repo, "char name[128];\nchar arch[32];", "char name[64];\nchar arch[32];")
    assert run(repo).returncode == 1
    assert run(repo, "--update").returncode == 0
    assert run(repo).returncode == 0, "after an approved update the new layout is the reference"
    (repo / GOLD).unlink()
    r = run(repo)
    assert r.returncode == 1 and "missing" in r.stderr
