#!/usr/bin/env python3
"""Install, update and inspect the vLLM / SGLang / llama.cpp environments (uv-managed). See mqde/backend/envs.py.

    scripts/setup_backend.py status [--check] [--json]
    scripts/setup_backend.py install vllm|sglang|llamacpp [--version X] [--dry-run]
    scripts/setup_backend.py ensure [vllm sglang llamacpp]       # install whatever is missing

Environments live in $MQDE_ENVS_DIR (default <repo>/build/envs). Needs only a Python >= 3.10; uv is bootstrapped when absent.
"""

import importlib.util
import sys
from pathlib import Path

# Load the module by file path: importing the `mqde` package would pull in the compiled core, which may not exist yet.
_spec = importlib.util.spec_from_file_location("mqde_backend_envs", Path(__file__).resolve().parents[1] / "python/mqde/backend/envs.py")
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    raise SystemExit(_mod.main())
