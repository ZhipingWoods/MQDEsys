#!/usr/bin/env bash
# Pre-push scan of what a push would publish: lines added by the commits, their messages, author identities, and the
# files they add. Exits 1 if anything FAILs; WARN lines are advisory.
#
#   scripts/privacy_scan.sh [RANGE]      # default: @{upstream}..HEAD, else origin/main..HEAD
#
# Built-in rules are generic (private IPv4 addresses, token / key formats, binary and large files). Identifiers that are
# specific to you or your organisation (internal hostnames, unreleased model names, old e-mail addresses ...) do NOT
# belong in this repository: put them, one extended regex per line, in a file outside the repo and point
# MQDE_PRIVACY_PATTERNS at it. Allowed author / committer e-mails: MQDE_ALLOWED_EMAILS (comma separated),
# default `git config user.email` plus GitHub's noreply address.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

RANGE=${1:-}
if [[ -z "$RANGE" ]]; then
  if git rev-parse --verify -q '@{upstream}' >/dev/null; then RANGE='@{upstream}..HEAD'; else RANGE='origin/main..HEAD'; fi
fi
if ! git rev-list "$RANGE" >/dev/null 2>&1; then echo "privacy_scan: bad revision range '$RANGE'" >&2; exit 2; fi
COMMITS=$(git rev-list "$RANGE")
if [[ -z "$COMMITS" ]]; then echo "privacy_scan: nothing to publish in $RANGE"; exit 0; fi

FAILS=0; WARNS=0
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
warn() { echo "WARN  $*"; WARNS=$((WARNS + 1)); }

RE_IP='(^|[^0-9.])(10(\.[0-9]{1,3}){3}|172\.(1[6-9]|2[0-9]|3[01])(\.[0-9]{1,3}){2}|192\.168(\.[0-9]{1,3}){2})($|[^0-9.])'
RE_SECRET='(ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
RE_PATH='(/cache/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/)'
RE_FILES='\.(gguf|safetensors|pt|pth|bin|npz|so|dylib|mp4|mov|webm|aiff|wav)$'
EXTRA=${MQDE_PRIVACY_PATTERNS:-}
[[ -n "$EXTRA" && ! -f "$EXTRA" ]] && { echo "privacy_scan: MQDE_PRIVACY_PATTERNS=$EXTRA not found" >&2; exit 2; }
ALLOWED=${MQDE_ALLOWED_EMAILS:-$(git config user.email 2>/dev/null)}
ALLOWED="$ALLOWED,noreply@github.com"

for c in $COMMITS; do
  short=$(git rev-parse --short "$c")
  # ---- identities (author, committer)
  while read -r email; do
    [[ -z "$email" ]] && continue
    [[ ",$ALLOWED," == *",$email,"* ]] || fail "$short identity '$email' is not in the allowed list (MQDE_ALLOWED_EMAILS)"
  done < <(git show -s --format='%ae%n%ce' "$c" | sort -u)
  # ---- message
  msg=$(git show -s --format='%B' "$c")
  grep -Eq "$RE_SECRET" <<<"$msg" && fail "$short commit message contains a token / key"
  grep -Eq "$RE_IP" <<<"$msg" && fail "$short commit message contains a private IP address"
  if [[ -n "$EXTRA" ]] && grep -Eqf "$EXTRA" <<<"$msg"; then fail "$short commit message matches a private pattern"; fi
  # ---- files added or modified
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if [[ "$f" =~ $RE_FILES ]]; then fail "$short adds a binary / media file: $f (keep these out of git)"; continue; fi
    size=$(git cat-file -s "$c:$f" 2>/dev/null || echo 0)
    [[ "$size" -gt 1048576 ]] && fail "$short adds a file over 1 MiB: $f ($((size / 1024)) KiB)"
  done < <(git diff-tree --no-commit-id --name-only --diff-filter=AM -r "$c")
  # ---- added lines
  added=$(git show -U0 --format= "$c" | grep -E '^\+' | grep -vE '^\+\+\+' || true)
  [[ -z "$added" ]] && continue
  while IFS= read -r l; do fail "$short added line has a token / key: ${l:0:100}"; done < <(grep -E "$RE_SECRET" <<<"$added" | head -3)
  while IFS= read -r l; do fail "$short added line has a private IP address: ${l:0:100}"; done < <(grep -E "$RE_IP" <<<"$added" | head -3)
  while IFS= read -r l; do warn "$short added line has an absolute path: ${l:0:100}"; done < <(grep -E "$RE_PATH" <<<"$added" | head -2)
  if [[ -n "$EXTRA" ]]; then
    while IFS= read -r l; do fail "$short added line matches a private pattern: ${l:0:100}"; done < <(grep -Ef "$EXTRA" <<<"$added" | head -3)
  fi
done

echo "privacy_scan: $(wc -w <<<"$COMMITS" | tr -d ' ') commit(s) in $RANGE, $FAILS fail, $WARNS warn"
[[ $FAILS -eq 0 ]]
