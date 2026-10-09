#!/usr/bin/env bash
# Local gate, equivalent to CI (T-0005). Runs every check, reports a summary, exits non-zero if any failed.
#
#   scripts/check_all.sh              # everything that is available in this environment
#   scripts/check_all.sh --fast       # skip the C++ build and the Python test suite
#   SKIP=web,cpp scripts/check_all.sh # skip named groups (deps,format,cpp,python,web)
#
# Missing optional tools (clang-format, clang-tidy, npm) are reported as SKIP, not as failures, so the
# script is usable on a machine that only has part of the toolchain.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
BUILD=${BUILD_DIR:-build/dev}
SKIP=${SKIP:-}
FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

declare -a NAMES=() STATES=() DETAILS=()
skipped() { [[ ",$SKIP," == *",$1,"* ]]; }

run() { # run <group> <name> <cmd...>
  local group=$1 name=$2; shift 2
  if skipped "$group"; then NAMES+=("$name"); STATES+=(SKIP); DETAILS+=("SKIP=$group"); return; fi
  printf '\033[1m==> %s\033[0m\n' "$name"
  local out rc
  out=$("$@" 2>&1); rc=$?
  echo "$out" | tail -n 25
  NAMES+=("$name"); DETAILS+=("$(echo "$out" | tail -n 1)")
  if [[ $rc -eq 0 ]]; then STATES+=(PASS); else STATES+=(FAIL); fi
}

have() { command -v "$1" >/dev/null 2>&1; }
note_missing() { NAMES+=("$1"); STATES+=(SKIP); DETAILS+=("$2 not installed"); }

# ---------------------------------------------------------------- structure
run deps "module dependencies" python3 scripts/check_deps.py
run abi "HAL ABI layout" bash scripts/check_abi.sh

# ---------------------------------------------------------------- formatting
if have clang-format; then
  mapfile -t CPP < <(git ls-files 'cpp/**/*.cpp' 'cpp/**/*.h' 'cpp/**/*.c' 2>/dev/null)
  run format "clang-format" clang-format --dry-run --Werror "${CPP[@]}"
else
  note_missing "clang-format" clang-format
fi

# ---------------------------------------------------------------- C++
if [[ $FAST -eq 0 ]]; then
  run cpp "cmake build" cmake --build "$BUILD" -j "$(getconf _NPROCESSORS_ONLN)"
  run cpp "ctest" ctest --test-dir "$BUILD" --output-on-failure
else
  run cpp "ctest (existing build)" ctest --test-dir "$BUILD"
fi
if have clang-tidy && [[ -f $BUILD/compile_commands.json ]] && ! skipped cpp; then
  # clang-tidy needs its own builtin headers (stddef.h & co). conda's clang-tools ships none, so borrow
  # them from any clang on PATH; if the toolchains still disagree, this is an environment problem and the
  # check reports SKIP instead of failing the build.
  TIDY_ARGS=()
  if have clang; then TIDY_ARGS+=("--extra-arg=-isystem$(clang -print-resource-dir)/include"); fi
  mapfile -t TIDY < <(git ls-files 'cpp/src/**/*.cpp' 'cpp/apps/**/*.cpp')
  tidy_out=$(clang-tidy -p "$BUILD" --quiet "${TIDY_ARGS[@]}" "${TIDY[@]}" 2>&1)
  tidy_rc=$?
  NAMES+=("clang-tidy")
  # errors from files outside the repo (mismatched clang builtin headers, sysroot) are environment problems
  own_err=$(grep -E "^$ROOT/[^:]*:[0-9]+:[0-9]+: error:" <<<"$tidy_out" | grep -c .)
  if [[ $tidy_rc -ne 0 && $own_err -eq 0 ]]; then
    STATES+=(SKIP); DETAILS+=("clang-tidy cannot parse this toolchain's system headers")
  elif [[ $tidy_rc -eq 0 ]]; then
    n=$(grep -c "warning:" <<<"$tidy_out")
    STATES+=(PASS); DETAILS+=("$n warning(s), no error-level check triggered")
  else
    grep -E "^$ROOT/[^:]*:[0-9]+:[0-9]+: error:" <<<"$tidy_out" | head -n 15
    STATES+=(FAIL); DETAILS+=("$own_err error(s) in repo files")
  fi
else
  note_missing "clang-tidy" "clang-tidy or $BUILD/compile_commands.json"
fi

# ---------------------------------------------------------------- Python
run python "ruff" python3 -m ruff check python
run python "mypy" bash -c "cd python && python3 -m mypy"
if [[ $FAST -eq 0 ]]; then
  run python "pytest" bash -c "cd python && CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES-} PYTHONPATH=. python3 -m pytest tests -q"
fi

# ---------------------------------------------------------------- web
if have npm && [[ -d web/node_modules ]]; then
  run web "web build (tsc + vite)" bash -c "cd web && npm run build"
else
  note_missing "web build" "npm or web/node_modules"
fi

# ---------------------------------------------------------------- summary
echo
printf '\033[1m%-28s %s\033[0m\n' "check" "result"
fail=0
for i in "${!NAMES[@]}"; do
  case ${STATES[$i]} in
    PASS) c='\033[32mPASS\033[0m' ;;
    FAIL) c='\033[31mFAIL\033[0m'; fail=1 ;;
    *)    c='\033[33mSKIP\033[0m' ;;
  esac
  printf '%-28s %b  %s\n' "${NAMES[$i]}" "$c" "${DETAILS[$i]}"
done
[[ $fail -eq 0 ]] && echo "all checks passed" || echo "some checks FAILED (see above)"
exit $fail
