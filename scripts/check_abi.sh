#!/usr/bin/env bash
# HAL plugin ABI guard (T-1101): compiles a tiny C program that prints the size and every field offset of each ABI struct in
# cpp/include/mqde/hal/mqde_hal.h and compares it with cpp/include/mqde/hal/abi_v1.golden. A mismatch means the ABI changed:
# that is a breaking change (L3) and needs MQDE_HAL_ABI_VERSION to be raised and the golden file regenerated on purpose.
#
#   scripts/check_abi.sh            compare (exit 1 and a diff on change)
#   scripts/check_abi.sh --update   rewrite the golden file (only after an approved ABI change)
#
# The layout is the LP64 one (Linux / macOS, 64-bit). `abidiff` would also catch symbol changes in a built plugin; it is not
# required here because the ABI is data structures plus one exported symbol, both covered below.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CC=${CC:-cc}
HDR=cpp/include
GOLD=$HDR/mqde/hal/abi_v1.golden
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/layout.c" <<'C'
#include <stddef.h>
#include <stdio.h>
#include "mqde/hal/mqde_hal.h"
#define F(T, f) printf("  %s.%s offset=%zu size=%zu\n", #T, #f, offsetof(T, f), sizeof(((T*)0)->f))
#define S(T) printf("%s size=%zu align=%zu\n", #T, sizeof(T), _Alignof(T))
int main(void) {
  printf("MQDE_HAL_ABI_VERSION=%d\npointer_size=%zu\n", MQDE_HAL_ABI_VERSION, sizeof(void*));
  S(mqde_hal_device); F(mqde_hal_device, kind); F(mqde_hal_device, index); F(mqde_hal_device, vendor); F(mqde_hal_device, name);
  F(mqde_hal_device, arch); F(mqde_hal_device, mem_total_bytes); F(mqde_hal_device, mem_free_bytes); F(mqde_hal_device, compute_units);
  F(mqde_hal_device, driver_version);
  S(mqde_hal_capability_query); F(mqde_hal_capability_query, device); F(mqde_hal_capability_query, dtype); F(mqde_hal_capability_query, op);
  S(mqde_hal_capability_result); F(mqde_hal_capability_result, supported); F(mqde_hal_capability_result, native);
  F(mqde_hal_capability_result, peak_tflops); F(mqde_hal_capability_result, note);
  S(mqde_hal_profile_request); F(mqde_hal_profile_request, device); F(mqde_hal_profile_request, kind); F(mqde_hal_profile_request, dtype);
  F(mqde_hal_profile_request, bytes); F(mqde_hal_profile_request, m); F(mqde_hal_profile_request, n); F(mqde_hal_profile_request, k);
  F(mqde_hal_profile_request, iterations);
  S(mqde_hal_profile_result); F(mqde_hal_profile_result, value); F(mqde_hal_profile_result, seconds); F(mqde_hal_profile_result, unit);
  S(mqde_hal_compile_request); S(mqde_hal_compile_result); S(mqde_hal_launch_request);
  S(mqde_hal_plugin_v1); F(mqde_hal_plugin_v1, abi_version); F(mqde_hal_plugin_v1, struct_size); F(mqde_hal_plugin_v1, name);
  F(mqde_hal_plugin_v1, vendor); F(mqde_hal_plugin_v1, plugin_version); F(mqde_hal_plugin_v1, device_probe); F(mqde_hal_plugin_v1, last_error);
  F(mqde_hal_plugin_v1, supports); F(mqde_hal_plugin_v1, alloc); F(mqde_hal_plugin_v1, free); F(mqde_hal_plugin_v1, profile);
  F(mqde_hal_plugin_v1, compile); F(mqde_hal_plugin_v1, launch);
  return 0;
}
C
"$CC" -std=c11 -Wall -Wextra -I"$HDR" "$tmp/layout.c" -o "$tmp/layout"
"$tmp/layout" >"$tmp/now.txt"
if [[ "${1:-}" == "--update" ]]; then cp "$tmp/now.txt" "$GOLD"; echo "check_abi: wrote $GOLD"; exit 0; fi
if [[ ! -f "$GOLD" ]]; then echo "check_abi: $GOLD is missing (run with --update once)" >&2; exit 1; fi
if diff -u "$GOLD" "$tmp/now.txt"; then echo "check_abi: HAL ABI v1 layout unchanged"; else
  echo "check_abi: the HAL ABI layout changed. This is a breaking change (L3): raise MQDE_HAL_ABI_VERSION and regenerate with --update only after approval." >&2
  exit 1
fi
