---
name: mqde-dev-gate
description: >
  The checks and steps before MQDEsys code leaves your machine: sync to the GPU/dev server, run the local gate
  (scripts/check_all.sh), scan what a push would publish (scripts/privacy_scan.sh), commit with the project's message
  format, push, and update docs/TASKS.md and docs/PROGRESS.md. Use before every commit or push, when the user says
  "run the gate", "is this ready to push", "commit this", or "sync to the server". Also lists the mistakes that have
  already cost time here. Do NOT use to decide what to build next (read docs/TASKS.md and AGENTS.md).
user-invocable: true
allowed-tools: Bash(git:*) Bash(rsync:*) Bash(bash:*) Bash(python:*) Bash(python3:*) Bash(ssh:*) Bash(cmake:*) Bash(ctest:*) Read Edit Grep Glob
---

# Gate, scan, commit, push

Two machines. The **local machine** (e.g. a Mac) holds the main git checkout: edit, commit and push there. The **dev
server** (GPUs, conda env `mqde`) builds, runs the GPU tests and serves the UI: it only pulls. Never commit on the
server; two writers diverge.

## 1. Sync for testing (not for publishing)

```bash
rsync -a --exclude .git --exclude build --exclude runs --exclude node_modules --exclude dist --exclude 3rdpart \
      --exclude __pycache__ --exclude '*.so' --exclude '*.dylib' --exclude .DS_Store \
      ./ <SERVER>:<SERVER_REPO>/
```

Always exclude `*.so`: a copy of a stale compiled extension overwrote the server's freshly built one once
(`module 'mqde._core' has no attribute 'pack_int'`). Do not use `--delete`. Rebuild after C++ changes:
`cmake --build --preset dev -j 32` on the server. After the push, align the server:
`git fetch origin && git reset --hard origin/main` (it holds no work of its own; the rsync copy is replaced by what
was committed).

## 2. The gate

On the server, in the `mqde` conda env, from the repository root:

```bash
MQDE_GGML_LIB=build/llamacpp/bin/libggml-base.so bash scripts/check_all.sh
```

It runs: module-dependency check, clang-format, C++ build + ctest, clang-tidy (SKIP when the toolchain cannot parse
system headers: that is expected, say so), ruff, mypy, pytest, web build. `SKIP=web,cpp` skips groups; `--fast` skips
the build and pytest. A long run belongs in the background (`nohup ... &`) with its log read afterwards.
Tests that need a GPU, vLLM, a llama.cpp binary or a reference library skip themselves when it is absent; count them
and say how many were skipped. On a laptop, `PYTHONPATH=python python -m pytest python/tests -q` covers everything
that does not need those.

Platform notes: macOS builds the C++ core with Homebrew dependencies
(`cmake -S . -B build/mac -G Ninja -DCMAKE_PREFIX_PATH=/opt/homebrew -DMQDE_WITH_PYTHON=ON -DPython_EXECUTABLE=<venv python>`);
timing-sensitive tests have separate tolerances per platform and must not be loosened to make a run pass.

## 3. Scan what would be published

```bash
MQDE_PRIVACY_PATTERNS=<file outside the repo> bash scripts/privacy_scan.sh      # default range: @{upstream}..HEAD
```

FAIL (exit 1): private IPv4 addresses, token / key formats, binary or media files, files over 1 MiB, a commit identity
that is not allowed (`MQDE_ALLOWED_EMAILS`, default `git config user.email`), anything matching your private
patterns file. WARN: absolute paths in added lines (developer paths in docs are tolerated; do not add new ones to
user-facing docs). The private patterns (internal hostnames, unreleased model names, old e-mails) live in a file
**outside** the repository so they are never published themselves. It scans every commit in the range, so a secret
added and removed later is still caught: fix it before the first push (`git rebase` / `git filter-repo`), because
once pushed it is public history. Take a backup bundle (`git bundle create ../backup.bundle --all`) before rewriting,
and never force-push without the maintainer's approval.

## 4. Commit

```
<type>(<scope>): <imperative summary, <= 72 chars>

<why and how, short>

Refs: T-0803
<the attribution trailer your harness requires>
```

`type`: feat fix refactor perf test docs build ci chore. One logical change per commit. Files that do not belong in git:
`runs/`, `build/`, weights, `*.gguf`, videos, `web/tsconfig.tsbuildinfo`. 3rdpart/ entries are submodule pointers.

## 5. Update the books (every session)

- `docs/TASKS.md`: status of each touched task with a short, honest reason it is PARTIAL (what exists, what is missing).
  DONE only when its acceptance criterion was met and verified.
- `docs/PROGRESS.md`: a new entry at the top of the log (done, verification output, what is not done, decisions
  needed). Numbers in it must come from real output.
- Anything needing a maintainer decision goes into the decisions table there as `D-<n>` and the work stops at that
  boundary.

## Mistakes that already happened

- A command that printed "clean" unconditionally was mistaken for a scan result. A check you run must be able to fail;
  prove it on a known-bad input (the scan tests do).
- Regexes with full-width characters in a character class misbehave in byte mode (`filter-repo --replace-text`): use
  a non-greedy `.*?` instead.
- Rewriting history strips GPG signatures, so commits created on github.com get new hashes; merge the remote's commit
  with `git merge -s ours --allow-unrelated-histories origin/main` instead of force-pushing.
- Quantizing on CUDA and then exporting crashed on `tensor.numpy()` (missing `.cpu()`); `python/tests/export/test_cuda_export.py`
  guards it when a GPU is present.
- A check run through a chained shell one-liner hid a failure behind a later command's exit code: run the gate on its
  own and read its summary.
