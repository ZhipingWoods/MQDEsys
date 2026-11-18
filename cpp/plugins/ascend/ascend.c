/* Ascend / CANN HAL plugin (T-1105).
 *
 * The plugin mirrors the CUDA/ROCm ones: the vendor runtime is dlopen'ed, never linked. It uses the
 * stable ACL runtime surface that llama.cpp's CANN backend also uses (aclInit, aclrtGetDeviceCount,
 * aclrtGetSocName, aclrtGetMemInfo, aclrtMalloc/Free, aclrtMemcpy). If CANN is absent the plugin is
 * INACTIVE, not broken; a broken/missing symbol is reported in last_error().
 *
 * Verified against the signatures used by llama.cpp's ggml-cann backend (3rdpart/llama.cpp):
 *   aclError aclrtGetDeviceCount(uint32_t*);
 *   const char* aclrtGetSocName(void);
 *   aclError aclrtGetMemInfo(aclrtMemType, size_t*, size_t*);
 *   aclError aclrtMalloc(void**, size_t, aclrtMemMallocPolicy);
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mqde/hal/mqde_hal.h"

typedef int (*acl_init_fn)(const char*);
typedef int (*acl_get_count_fn)(uint32_t*);
typedef int (*acl_set_device_fn)(int);
typedef int (*acl_reset_device_fn)(int);
typedef const char* (*acl_soc_name_fn)(void);
typedef int (*acl_mem_info_fn)(int, size_t*, size_t*);
typedef int (*acl_malloc_fn)(void**, size_t, int);
typedef int (*acl_free_fn)(void*);
typedef int (*acl_memcpy_fn)(void*, size_t, const void*, size_t, int);
typedef int (*acl_sync_fn)(void);

static char g_err[256];
static int set_err(const char* m, const char* detail) {
  snprintf(g_err, sizeof g_err, "%s%s%s", m, detail ? ": " : "", detail ? detail : "");
  return -1;
}
static const char* last_error(void) { return g_err; }

static struct {
  void* lib;
  acl_init_fn init;
  acl_get_count_fn count;
  acl_set_device_fn set;
  acl_reset_device_fn reset;
  acl_soc_name_fn soc;
  acl_mem_info_fn meminfo;
  acl_malloc_fn malloc;
  acl_free_fn free;
  acl_memcpy_fn memcpy;
  acl_sync_fn sync;
  int initialized;
  int failed;
} g_acl;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void* open_runtime(void) {
  const char* dir = getenv("MQDE_ASCEND_LIB_DIR");
  char path[4096];
  const char* names[] = {"libascendcl.so", "libascendcl.so.1", "libascendcl.so.2"};
  if (dir && *dir) {
    for (size_t i = 0; i < sizeof names / sizeof names[0]; ++i) {
      snprintf(path, sizeof path, "%s/%s", dir, names[i]);
      void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (h) return h;
    }
  }
  const char* home = getenv("ASCEND_HOME_PATH");
  if (home && *home) {
    snprintf(path, sizeof path, "%s/lib64/libascendcl.so", home);
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h) return h;
  }
  for (size_t i = 0; i < sizeof names / sizeof names[0]; ++i) {
    void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}

static int load_runtime(void) {
  if (g_acl.lib) return 0;
  if (g_acl.failed) return set_err("the CANN runtime was already found unusable", NULL);
  void* h = open_runtime();
  if (!h) {
    g_acl.failed = 1;
    return set_err("libascendcl.so not found (set MQDE_ASCEND_LIB_DIR or source the CANN environment)", NULL);
  }
  g_acl.lib = h;
#define SYM(field, name)                                                    \
  do {                                                                      \
    *(void**)(&g_acl.field) = dlsym(h, name);                               \
    if (!g_acl.field) {                                                     \
      dlclose(h);                                                           \
      g_acl.lib = NULL;                                                     \
      g_acl.failed = 1;                                                     \
      return set_err("CANN symbol not found", name);                        \
    }                                                                       \
  } while (0)
  SYM(init, "aclInit");
  SYM(count, "aclrtGetDeviceCount");
  SYM(set, "aclrtSetDevice");
  SYM(reset, "aclrtResetDevice");
  SYM(soc, "aclrtGetSocName");
  SYM(meminfo, "aclrtGetMemInfo");
  SYM(malloc, "aclrtMalloc");
  SYM(free, "aclrtFree");
  SYM(memcpy, "aclrtMemcpy");
  *(void**)(&g_acl.sync) = dlsym(h, "aclrtSynchronizeDevice"); /* optional */
#undef SYM
  return 0;
}

static int ensure_init(void) {
  if (g_acl.initialized) return 0;
  int rc = g_acl.init(NULL);
  if (rc != 0) return set_err("aclInit failed", NULL);
  g_acl.initialized = 1;
  return 0;
}

static void sanitize(char* dst, size_t cap, const char* src) {
  size_t j = 0;
  for (size_t i = 0; src[i] && j + 1 < cap; ++i) {
    unsigned char c = (unsigned char)src[i];
    if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-') dst[j++] = (char)c;
  }
  dst[j] = '\0';
  if (!dst[0]) snprintf(dst, cap, "ascend");
}

static int device_probe(mqde_hal_device* out, int max, int* n) {
  if (!out || !n || max < 0) return set_err("invalid arguments", NULL);
  *n = 0;
  if (load_runtime() != 0 || ensure_init() != 0) return 0; /* absent/unusable runtime == no devices */
  uint32_t count = 0;
  if (g_acl.count(&count) != 0) return set_err("aclrtGetDeviceCount failed", NULL);
  if (count > (uint32_t)max) count = (uint32_t)max;
  for (uint32_t i = 0; i < count; ++i) {
    if (g_acl.set((int)i) != 0) continue;
    mqde_hal_device* d = &out[*n];
    memset(d, 0, sizeof *d);
    d->kind = MQDE_HAL_DEV_NPU;
    d->index = (int32_t)i;
    snprintf(d->vendor, sizeof d->vendor, "%s", "huawei");
    const char* soc = g_acl.soc ? g_acl.soc() : NULL;
    sanitize(d->arch, sizeof d->arch, soc ? soc : "ascend");
    snprintf(d->name, sizeof d->name, "Ascend %s", soc && *soc ? soc : "NPU");
    size_t free_b = 0, total_b = 0;
    if (g_acl.meminfo && g_acl.meminfo(0 /* ACL_HBM_MEM */, &free_b, &total_b) == 0) {
      d->mem_free_bytes = (uint64_t)free_b;
      d->mem_total_bytes = (uint64_t)total_b;
    }
    const char* drv = getenv("MQDE_ASCEND_DRIVER");
    snprintf(d->driver_version, sizeof d->driver_version, "%s", drv && *drv ? drv : "cann");
    g_acl.reset((int)i);
    ++*n;
  }
  return 0;
}

static int supports(const mqde_hal_capability_query* q, mqde_hal_capability_result* r) {
  if (!q || !r) return set_err("null argument", NULL);
  if (q->device < 0) return set_err("no such device", NULL);
  memset(r, 0, sizeof *r);
  if (q->op[0] && strcmp(q->op, "gemm") != 0) return set_err("only the gemm op is described", q->op);
  const char* t = q->dtype;
  int native = 0, sup = 1;
  const char* note = "";
  if (!strcmp(t, "fp32"))
    native = 1, note = "native";
  else if (!strcmp(t, "fp16") || !strcmp(t, "bf16"))
    native = 1, note = "native (AI core)";
  else if (!strcmp(t, "int8"))
    native = 1, note = "native (AI core)";
  else if (!strcmp(t, "int4") || !strcmp(t, "fp8_e4m3") || !strcmp(t, "fp8_e5m2"))
    note = "emulated through int8/fp16 kernels on this CANN generation";
  else
    sup = 0, note = "dtype not known to the Ascend plugin";
  r->supported = sup;
  r->native = native;
  snprintf(r->note, sizeof r->note, "%s", note);
  return 0;
}

static int p_alloc(int device, size_t bytes, void** ptr) {
  if (!ptr || device < 0) return set_err("invalid arguments", NULL);
  if (load_runtime() != 0 || ensure_init() != 0) return set_err("CANN runtime unavailable", NULL);
  if (g_acl.set(device) != 0) return set_err("aclrtSetDevice failed", NULL);
  if (g_acl.malloc(ptr, bytes ? bytes : 1, 0 /* ACL_MEM_MALLOC_HUGE_FIRST */) != 0)
    return set_err("aclrtMalloc failed", NULL);
  return 0;
}

static int p_free(int device, void* ptr) {
  (void)device;
  if (load_runtime() != 0 || ensure_init() != 0) return set_err("CANN runtime unavailable", NULL);
  return g_acl.free(ptr) == 0 ? 0 : set_err("aclrtFree failed", NULL);
}

static int profile_gemm(void) { return set_err("gemm is not implemented in the Ascend plugin yet", NULL); }

static int profile(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!req || !res) return set_err("null argument", NULL);
  if (device < 0) return set_err("no such device", NULL);
  memset(res, 0, sizeof *res);
  if (!strcmp(req->kind, "gemm")) return profile_gemm();
  if (strcmp(req->kind, "bandwidth") != 0) return set_err("unknown profile kind", req->kind);
  if (load_runtime() != 0 || ensure_init() != 0) return set_err("CANN runtime unavailable", NULL);
  if (g_acl.set(device) != 0) return set_err("aclrtSetDevice failed", NULL);
  const size_t bytes = req->bytes ? (size_t)req->bytes : (64u << 20);
  const int iters = req->iterations > 0 ? req->iterations : 5;
  void *a = NULL, *b = NULL;
  void* host = malloc(bytes);
  if (!host) return set_err("host allocation failed", NULL);
  memset(host, 0x5a, bytes);
  if (g_acl.malloc(&a, bytes, 0) != 0 || g_acl.malloc(&b, bytes, 0) != 0) {
    free(host);
    if (a) g_acl.free(a);
    return set_err("aclrtMalloc failed", NULL);
  }
  if (g_acl.memcpy(a, bytes, host, bytes, 1 /* HOST_TO_DEVICE */) != 0) {
    free(host);
    g_acl.free(a);
    g_acl.free(b);
    return set_err("H2D copy failed", NULL);
  }
  const double t0 = now();
  for (int i = 0; i < iters; ++i)
    if (g_acl.memcpy(b, bytes, a, bytes, 3 /* DEVICE_TO_DEVICE */) != 0) {
      free(host);
      g_acl.free(a);
      g_acl.free(b);
      return set_err("D2D copy failed", NULL);
    }
  if (g_acl.sync) g_acl.sync();
  const double secs = now() - t0;
  free(host);
  g_acl.free(a);
  g_acl.free(b);
  res->seconds = secs;
  res->value = secs > 0 ? 2.0 * (double)bytes * (double)iters / secs / 1e9 : 0.0;
  snprintf(res->unit, sizeof res->unit, "%s", "GB/s");
  return 0;
}

static const mqde_hal_plugin_v1 kPlugin = {
    MQDE_HAL_ABI_VERSION, sizeof(mqde_hal_plugin_v1), "ascend", "huawei", "1.0.0",
    device_probe, last_error, supports, p_alloc, p_free, profile, NULL, NULL};

MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return &kPlugin; }
