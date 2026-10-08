# Testing and the local gate

`scripts/check_all.sh` is the executable form of the project's rules and is meant to be equivalent to CI (no CI workflow exists in the repository yet; see the proposal in `docs/site/pages-workflow.proposal.yml` for the docs build only).

```bash
scripts/check_all.sh                     # everything available in this environment
scripts/check_all.sh --fast              # skip the C++ rebuild and the pytest run
SKIP=web,cpp scripts/check_all.sh        # skip groups: deps, format, cpp, python, web
BUILD_DIR=build/mac scripts/check_all.sh # use another C++ build directory (default build/dev)
```

| Check | Tool |
|---|---|
| module dependency direction | `scripts/check_deps.py` |
| C++ format | `clang-format --dry-run --Werror` (`.clang-format`) |
| C++ build and unit tests | `cmake --build`, `ctest` |
| C++ static analysis | `clang-tidy` |
| Python lint and types | `ruff check python`, `mypy` (strict on untyped defs) |
| Python tests | `pytest` in `python/` |
| front end | `npm run build` (`tsc -b` + `vite build`) |

The summary table at the end marks each check PASS, FAIL or SKIP. **SKIP means the tool is not installed here, not that the check passed.** The script runs `python3 -m ruff` and `python3 -m mypy`, so they must be installed in the Python that `python3` resolves to; the script does not take a Python path.

Run subsets directly:

```bash
ctest --preset dev -R estimator_                     # one C++ module
build/mac/cpp/mqde_unit_tests --gtest_filter="fit.*"   # the test binary is in <build>/cpp/
PYTHONPATH=python python -m pytest python/tests -q   # Python (some tests skip without CUDA, vLLM or a GGUF fixture)
```

Tests use offline fixtures (the project rule is that network tests must be opt-in; the variable `MQDE_ONLINE_TESTS` named in `AGENTS.md` is not referenced by any test yet). Tests that need a real llama.cpp binary or GGUF read an environment variable (for example `MQDE_TEST_GGUF`, `MQDE_TEST_NGL`) and skip otherwise. `python/tests/test_numerics_golden.py` checks the Torch numerics against the C++ core; when it fails, suspect the code first. (The `data/golden/` directory described in the design documents does not exist yet.)

Before pushing, `scripts/privacy_scan.sh` scans the commits a push would publish for private IP addresses, token formats, binary and large files, and author identities.

## Building these docs

```bash
python3 -m venv build/venv-docs && build/venv-docs/bin/pip install -r docs/site/requirements.txt
cd docs/site && make html            # both languages, warnings are errors
```
