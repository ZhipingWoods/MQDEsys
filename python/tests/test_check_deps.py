"""scripts/check_deps.py: the module dependency direction, including the HAL leaf module (T-1101)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "scripts/check_deps.py"), "--root", str(root)], capture_output=True, text=True, check=False)


def tree(tmp: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp


def test_the_real_repository_is_clean() -> None:
    r = run(REPO)
    assert r.returncode == 0, r.stderr


def test_everything_above_may_use_hal_and_hal_may_use_nothing(tmp_path: Path) -> None:
    ok = tree(tmp_path / "ok", {
        "cpp/include/mqde/hal/mqde_hal.h": "#include <stdint.h>\n",
        "cpp/src/server/s.cpp": '#include "mqde/hal/mqde_hal.h"\n',
        "cpp/apps/a/main.cpp": '#include "mqde/hal/mqde_hal.h"\n',
        "cpp/tests/unit/t.cpp": '#include "mqde/hal/mqde_hal.h"\n',
    })
    assert run(ok).returncode == 0
    bad = tree(tmp_path / "bad", {"cpp/include/mqde/hal/mqde_hal.h": '#include "mqde/estimator/estimator.h"\n'})
    r = run(bad)
    assert r.returncode == 1 and "'hal' must not depend on 'estimator'" in r.stderr


def test_a_lower_module_may_not_use_hal(tmp_path: Path) -> None:
    r = run(tree(tmp_path, {"cpp/src/estimator/e.cpp": '#include "mqde/hal/mqde_hal.h"\n'}))
    assert r.returncode == 1 and "'estimator' must not depend on 'hal'" in r.stderr


def test_an_unknown_module_is_reported(tmp_path: Path) -> None:
    r = run(tree(tmp_path, {"cpp/include/mqde/newthing/x.h": "int x;\n"}))
    assert r.returncode == 1 and "unknown module 'newthing'" in r.stderr
