# Testing and quality gates

`scripts/check_all.sh` aggregates the project's local checks: dependency direction, formatting, build, unit tests and frontend build.

```bash
scripts/check_all.sh                     # all checks available in the current environment
scripts/check_all.sh --fast              # skip the C++ rebuild and pytest
SKIP=web,cpp scripts/check_all.sh        # skip groups: deps, format, cpp, python, web
BUILD_DIR=build/mac scripts/check_all.sh # use another C++ build directory (default build/dev)
```

| Check | Tool |
|---|---|
| Module dependency direction | `scripts/check_deps.py` |
| C++ formatting | `clang-format --dry-run --Werror` (`.clang-format`) |
| C++ build and unit tests | `cmake --build`, `ctest` |
| C++ static analysis | `clang-tidy` |
| Python lint and types | `ruff check python`, `mypy` |
| Python tests | `pytest` in `python/` |
| Frontend build | `npm run build` (`tsc -b` + `vite build`) |

The script prints a summary table with PASS, FAIL or SKIP for each item. SKIP means the tool is not installed locally and does not indicate a passing check. Ruff and mypy are invoked as `python3 -m ruff` and `python3 -m mypy`, so they must be installed in the environment referenced by `python3`.

Individual checks can also be run directly:

```bash
ctest --preset dev -R estimator_                      # one C++ module
build/mac/cpp/mqde_unit_tests --gtest_filter="fit.*"  # test binary under <build>/cpp/
PYTHONPATH=python python -m pytest python/tests -q    # Python tests
```

Tests use offline fixtures by default. Network tests require `MQDE_ONLINE_TESTS=1`. Tests that depend on real inference engines or GGUF files are enabled through environment variables such as `MQDE_TEST_GGUF` and `MQDE_TEST_NGL`, and are skipped when those variables are unset.

`python/tests/test_numerics_golden.py` compares the PyTorch numerics against the C++ core. A failure should first be investigated as an implementation issue rather than addressed by changing golden data.

Before pushing, `scripts/privacy_scan.sh` scans the commits to be published for internal addresses, credential patterns, binary files, large files and author information.

## Building the documentation

```bash
python3 -m venv build/venv-docs && build/venv-docs/bin/pip install -r docs/site/requirements.txt
cd docs/site && make html            # build both languages, warnings treated as errors
```
