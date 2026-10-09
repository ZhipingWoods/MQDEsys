"""scripts/check_abi.sh (T-1101): the HAL ABI layout is compared with a golden file; any layout change must fail it."""

from __future__ import annotations

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
    p = repo / HDR
    s = p.read_text()
    assert s.count(old) == 1, old
    p.write_text(s.replace(old, new))


def test_the_committed_header_matches_its_golden_file(repo: Path) -> None:
    r = run(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "unchanged" in r.stdout


@pytest.mark.parametrize(
    ("old", "new", "what"),
    [
        ("  char name[128];\n  char arch[32];", "  char name[64];\n  char arch[32];", "a string shrinks"),
        ("  int32_t kind;           /* mqde_hal_dev_kind */\n  int32_t index;", "  int32_t index;\n  int32_t kind;",
         "two fields swap"),
        ("  uint64_t mem_total_bytes;\n  uint64_t mem_free_bytes;", "  uint64_t mem_free_bytes;\n  uint64_t mem_total_bytes;", "memory fields swap"),
        ("  int (*alloc)(int device, size_t bytes, void** ptr);\n  int (*free)(int device, void* ptr);",
         "  int (*free)(int device, void* ptr);\n  int (*alloc)(int device, size_t bytes, void** ptr);", "function pointers swap"),
        ("#define MQDE_HAL_ABI_VERSION 1", "#define MQDE_HAL_ABI_VERSION 2", "the version number changes"),
        ("  double peak_tflops;     /* 0 when unknown */", "  float peak_tflops;      /* 0 when unknown */", "a field changes type"),
    ],
)
def test_every_kind_of_layout_change_fails_the_check(repo: Path, old: str, new: str, what: str) -> None:
    edit(repo, old, new)
    r = run(repo)
    assert r.returncode == 1, f"{what}: not detected\n{r.stdout}"
    assert "breaking change" in r.stderr and r.stdout.startswith("---"), "the diff is shown"


def test_a_comment_only_edit_is_not_an_abi_change(repo: Path) -> None:
    edit(repo, "  double peak_tflops;     /* 0 when unknown */", "  double peak_tflops;     /* reworded comment */")
    assert run(repo).returncode == 0


def test_update_rewrites_the_golden_file_and_a_missing_golden_is_an_error(repo: Path) -> None:
    edit(repo, "  char name[128];\n  char arch[32];", "  char name[64];\n  char arch[32];")
    assert run(repo).returncode == 1
    assert run(repo, "--update").returncode == 0
    assert run(repo).returncode == 0, "after an approved update the new layout is the reference"
    (repo / GOLD).unlink()
    r = run(repo)
    assert r.returncode == 1 and "missing" in r.stderr
