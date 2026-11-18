/* FlagTree-style target strings (T-R14 host side): vendor/arch -> backend + warp size, without touching the
 * plugin ABI. Plugins still only describe devices; this is the common spelling the CLI, server and planner use. */
#include "mqde_probe.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

/* Defined by probe.c: target errors use the same last_error() surface as the rest of the probe API. */
void mqde_probe_set_last_error(const char* msg);

static int fail(const char* msg) {
  mqde_probe_set_last_error(msg);
  return -1;
}

int mqde_target_parse(const char* text, mqde_target* out) {
  if (!text || !out) return fail("target: null argument");
  const char* c1 = strchr(text, ':');
  if (!c1) return fail("target must be 'backend:arch[:warp]'");
  const char* c2 = strchr(c1 + 1, ':');
  const char* arch_end = c2 ? c2 : text + strlen(text);
  const size_t blen = (size_t)(c1 - text), alen = (size_t)(arch_end - (c1 + 1));
  if (blen == 0 || blen >= sizeof out->backend) return fail("target backend is empty or too long");
  if (alen == 0 || alen >= sizeof out->arch) return fail("target arch is empty or too long");
  for (size_t i = 0; i < blen; ++i)
    if (!(islower((unsigned char)text[i]) || isdigit((unsigned char)text[i]) || text[i] == '_' ||
          text[i] == '-'))
      return fail("target backend must be lowercase letters, digits, '_' or '-'");
  memset(out, 0, sizeof *out);
  memcpy(out->backend, text, blen);
  memcpy(out->arch, c1 + 1, alen);
  int warp = 0;
  if (c2) {
    if (c2[1] == '\0') return fail("target warp size is empty");
    const char* p = c2 + 1;
    while (*p) {
      if (!isdigit((unsigned char)*p)) return fail("target warp size must be an integer");
      warp = warp * 10 + (*p - '0');
      if (warp > 4096) return fail("target warp size is too large");
      ++p;
    }
    if (warp <= 0) return fail("target warp size must be positive");
  } else {
    if (!strcmp(out->backend, "rocm") || !strcmp(out->backend, "maca"))
      warp = 64;
    else if (!strcmp(out->backend, "cpu"))
      warp = 1;
    else
      warp = 32;
  }
  out->warp_size = warp;
  return 0;
}

int mqde_target_for_device(const mqde_device_info* dev, mqde_target* out) {
  if (!dev || !out) return fail("target: null device");
  memset(out, 0, sizeof *out);
  const char* v = dev->vendor;
  if (dev->kind == MQDE_DEV_CPU)
    snprintf(out->backend, sizeof out->backend, "%s", "cpu");
  else if (!strcmp(v, "nvidia"))
    snprintf(out->backend, sizeof out->backend, "%s", "cuda");
  else if (!strcmp(v, "amd"))
    snprintf(out->backend, sizeof out->backend, "%s", "rocm");
  else if (!strcmp(v, "apple"))
    snprintf(out->backend, sizeof out->backend, "%s", "metal");
  else if (!strcmp(v, "huawei") || !strcmp(v, "ascend"))
    snprintf(out->backend, sizeof out->backend, "%s", "ascend");
  else if (!strcmp(v, "metax"))
    snprintf(out->backend, sizeof out->backend, "%s", "maca");
  else if (!strcmp(v, "mthreads"))
    snprintf(out->backend, sizeof out->backend, "%s", "musa");
  else if (!strcmp(v, "intel"))
    snprintf(out->backend, sizeof out->backend, "%s", "sycl");
  else
    snprintf(out->backend, sizeof out->backend, "%s", "other");
  snprintf(out->arch, sizeof out->arch, "%s", dev->arch[0] ? dev->arch : "unknown");
  out->warp_size = !strcmp(out->backend, "rocm") || !strcmp(out->backend, "maca") ? 64
                   : !strcmp(out->backend, "cpu")                          ? 1
                                                                           : 32;
  return 0;
}

int mqde_target_format(const mqde_target* t, char* buf, size_t cap) {
  if (!t || !buf || cap == 0) return fail("target: null argument");
  const int n = snprintf(buf, cap, "%s:%s:%d", t->backend, t->arch, (int)t->warp_size);
  if (n < 0 || (size_t)n >= cap) {
    buf[0] = '\0';
    return fail("target: buffer too small");
  }
  return 0;
}
