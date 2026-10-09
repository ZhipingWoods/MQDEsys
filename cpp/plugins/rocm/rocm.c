/* ROCm / AMD HAL plugin (T-1104). Devices and the bandwidth micro-benchmark through the HIP runtime
 * (libamdhip64), loaded with dlopen: a machine without ROCm lists no device and no error. The HIP runtime
 * mirrors the CUDA runtime API one to one, so this follows plugins/cuda. Device properties come from
 * hipDeviceGetAttribute / hipDeviceGetName / hipDeviceTotalMem (stable, by id), never from the
 * hipDeviceProp_t struct, whose layout changes between ROCm releases.
 *
 * NOT YET RUN ON REAL HARDWARE (no AMD GPU was available). It is exercised against a stand-in HIP library
 * built from cpp/tests/fakes/fake_hip.c, which proves the loading, the symbol use, the error paths and the
 * arithmetic; it cannot prove that real HIP returns what this code expects. Attribute numbers below are from
 * hip_runtime_api.h (hipDeviceAttribute_t). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mqde/hal/mqde_hal.h"

static char g_err[256] = "";
static int set_err(const char* m) {
  snprintf(g_err, sizeof g_err, "%s", m);
  return -1;
}
static int set_err2(const char* m, const char* d) {
  snprintf(g_err, sizeof g_err, "%s: %s", m, d ? d : "");
  return -1;
}
static const char* last_error(void) { return g_err; }

typedef int (*hipInit_t)(unsigned);
typedef int (*hipGetDeviceCount_t)(int*);
typedef int (*hipDeviceGetName_t)(char*, int, int);
typedef int (*hipDeviceTotalMem_t)(size_t*, int);
typedef int (*hipDeviceGetAttribute_t)(int*, int, int);
typedef int (*hipDeviceGetPCIBusId_t)(char*, int, int);
typedef int (*hipDriverGetVersion_t)(int*);
typedef int (*hipSetDevice_t)(int);
typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipMemGetInfo_t)(size_t*, size_t*);
typedef int (*hipMemcpyAsync_t)(void*, const void*, size_t, int, void*);
typedef int (*hipMemset_t)(void*, int, size_t);
typedef int (*hipEventCreate_t)(void**);
typedef int (*hipEventDestroy_t)(void*);
typedef int (*hipEventRecord_t)(void*, void*);
typedef int (*hipEventSynchronize_t)(void*);
typedef int (*hipEventElapsedTime_t)(float*, void*, void*);
typedef int (*hipDeviceSynchronize_t)(void);
typedef const char* (*hipGetErrorString_t)(int);

static struct {
  int tried, ok_hip;
  void* hip;
  hipInit_t hipInit;
  hipGetDeviceCount_t hipGetDeviceCount;
  hipDeviceGetName_t hipDeviceGetName;
  hipDeviceTotalMem_t hipDeviceTotalMem;
  hipDeviceGetAttribute_t hipDeviceGetAttribute;
  hipDriverGetVersion_t hipDriverGetVersion;
  hipSetDevice_t hipSetDevice;
  hipMalloc_t hipMalloc;
  hipFree_t hipFree;
  hipMemGetInfo_t hipMemGetInfo;
  hipMemcpyAsync_t hipMemcpyAsync;
  hipMemset_t hipMemset;
  hipEventCreate_t hipEventCreate;
  hipEventDestroy_t hipEventDestroy;
  hipEventRecord_t hipEventRecord;
  hipEventSynchronize_t hipEventSynchronize;
  hipEventElapsedTime_t hipEventElapsedTime;
  hipDeviceSynchronize_t hipDeviceSynchronize;
  hipGetErrorString_t hipGetErrorString;
} L;

static int sym_ok(void** slot, void* h, const char* name) {
  *slot = dlsym(h, name);
  return *slot != NULL;
}
#define SYM(h, field) sym_ok((void**)&L.field, (h), #field)

static void* open_first(const char* const* names) {
  const char* override =
      getenv("MQDE_ROCM_LIB_DIR"); /* lets tests (and unusual installs) point at a directory */
  for (; *names; ++names) {
    char path[1024];
    if (override && *override) {
      snprintf(path, sizeof path, "%.900s/%s", override, *names);
      void* h = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
      if (h) return h;
    }
    void* h = dlopen(*names, RTLD_LAZY | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}

static void load_libs(void) {
  if (L.tried) return;
  L.tried = 1;
  static const char* const hip[] = {"libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5", NULL};
  L.hip = open_first(hip);
  if (L.hip)
    L.ok_hip = SYM(L.hip, hipInit) && SYM(L.hip, hipGetDeviceCount) && SYM(L.hip, hipDeviceGetName) &&
               SYM(L.hip, hipDeviceTotalMem) && SYM(L.hip, hipDeviceGetAttribute) &&
               SYM(L.hip, hipSetDevice) && SYM(L.hip, hipMalloc) && SYM(L.hip, hipFree) &&
               SYM(L.hip, hipMemGetInfo) && SYM(L.hip, hipMemcpyAsync) && SYM(L.hip, hipMemset) &&
               SYM(L.hip, hipEventCreate) && SYM(L.hip, hipEventDestroy) && SYM(L.hip, hipEventRecord) &&
               SYM(L.hip, hipEventSynchronize) && SYM(L.hip, hipEventElapsedTime) &&
               SYM(L.hip, hipDeviceSynchronize) && SYM(L.hip, hipGetErrorString);
  if (L.hip) SYM(L.hip, hipDriverGetVersion);
}

static int g_inited = 0, g_ndev = 0;
static int init_hip(void) {
  load_libs();
  if (!L.ok_hip) return 0;
  if (!g_inited) {
    if (L.hipInit(0) != 0 || L.hipGetDeviceCount(&g_ndev) != 0)
      g_ndev = 0; /* runtime present, no GPU: zero devices */
    g_inited = 1;
  }
  return 1;
}

/* hipDeviceAttribute_t values, read from the HIP header vendored in 3rdpart/FlagTree
 * (third_party/amd/backend/include/hip) by compiling a program that prints the enum: MultiprocessorCount=63,
 * ComputeCapabilityMajor=23, ComputeCapabilityMinor=61. */
enum { ATTR_CU_COUNT = 63, ATTR_CC_MAJOR = 23, ATTR_CC_MINOR = 61 };

static int device_probe(mqde_hal_device* out, int max, int* n) {
  if (!out || !n || max < 0) return set_err("invalid arguments");
  *n = 0;
  if (!init_hip()) return 0;
  int ver = 0;
  if (L.hipDriverGetVersion) L.hipDriverGetVersion(&ver);
  for (int i = 0; i < g_ndev && *n < max; ++i) {
    mqde_hal_device* d = &out[*n];
    memset(d, 0, sizeof *d);
    d->kind = MQDE_HAL_DEV_GPU;
    d->index = i;
    snprintf(d->vendor, sizeof d->vendor, "amd");
    L.hipDeviceGetName(d->name, (int)sizeof d->name, i);
    int cus = 0, major = 0, minor = 0;
    L.hipDeviceGetAttribute(&cus, ATTR_CU_COUNT, i);
    L.hipDeviceGetAttribute(&major, ATTR_CC_MAJOR, i);
    L.hipDeviceGetAttribute(&minor, ATTR_CC_MINOR, i);
    /* The real "gfx1100" name exists only in hipDeviceProp_t.gcnArchName, a struct whose layout changes
     * between ROCm releases, so it is not read here. The arch is reported as hip<major>.<minor> (e.g.
     * hip11.0); the major number alone selects the capability table below (9 = CDNA, 10 = RDNA1/2, 11 =
     * RDNA3). */
    snprintf(d->arch, sizeof d->arch, "hip%d.%d", major, minor);
    size_t total = 0;
    if (L.hipDeviceTotalMem(&total, i) == 0) d->mem_total_bytes = total;
    d->compute_units = (uint32_t)(cus > 0 ? cus : 0);
    snprintf(d->driver_version, sizeof d->driver_version, "hip-%d.%d", ver / 10000000, (ver / 100000) % 100);
    ++*n;
  }
  return 0;
}

static int arch_major(int dev) {
  int major = 0;
  L.hipDeviceGetAttribute(&major, ATTR_CC_MAJOR, dev);
  return major;
}

/* CDNA (major 9) and RDNA3 (major 11) have matrix instructions; fp8 only on CDNA3. These rows come from AMD's
 * published architecture descriptions and are NOT verified on hardware. */
static int supports(const mqde_hal_capability_query* q, mqde_hal_capability_result* r) {
  if (!q || !r) return set_err("null argument");
  if (!init_hip() || q->device < 0 || q->device >= g_ndev) return set_err("no such device");
  memset(r, 0, sizeof *r);
  const int mj = arch_major(q->device);
  const int cdna = mj == 9, rdna3 = mj == 11;
  const char* t = q->dtype;
  int sup = 0, nat = 0;
  const char* note = "dtype not known to the ROCm plugin";
  if (!strcmp(t, "fp32"))
    sup = 1, nat = 1, note = "vector units";
  else if (!strcmp(t, "fp16") || !strcmp(t, "bf16"))
    sup = 1, nat = cdna || rdna3,
    note = nat ? "matrix cores / WMMA" : "no matrix instructions on this generation";
  else if (!strcmp(t, "int8"))
    sup = 1, nat = cdna || rdna3, note = nat ? "matrix cores / WMMA" : "dot-product instructions only";
  else if (!strcmp(t, "fp8_e4m3") || !strcmp(t, "fp8_e5m2"))
    sup = cdna, nat = cdna, note = cdna ? "CDNA3 matrix cores (gfx94x and newer)" : "fp8 needs CDNA3";
  else if (!strcmp(t, "int4"))
    sup = 0, note = "no int4 matrix path in ROCm libraries";
  r->supported = sup;
  r->native = nat;
  snprintf(r->note, sizeof r->note, "%s", note);
  return 0;
}

static int p_alloc(int device, size_t bytes, void** ptr) {
  if (!ptr) return set_err("null pointer");
  load_libs();
  if (!L.ok_hip) return set_err("libamdhip64 not available");
  int rc = L.hipSetDevice(device);
  if (rc == 0) rc = L.hipMalloc(ptr, bytes ? bytes : 1);
  return rc == 0 ? 0 : set_err2("hipMalloc failed", L.hipGetErrorString(rc));
}
static int p_free(int device, void* ptr) {
  load_libs();
  if (!L.ok_hip) return set_err("libamdhip64 not available");
  L.hipSetDevice(device);
  return L.hipFree(ptr) == 0 ? 0 : set_err("hipFree failed");
}

#define CK(call, what)                          \
  do {                                          \
    int rc_ = (call);                           \
    if (rc_ != 0) {                             \
      set_err2(what, L.hipGetErrorString(rc_)); \
      goto fail;                                \
    }                                           \
  } while (0)

enum { MEMCPY_D2D = 3 };
static int bench_bandwidth(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  void *a = NULL, *b = NULL, *e0 = NULL, *e1 = NULL;
  size_t free_b = 0, total_b = 0;
  CK(L.hipSetDevice(device), "hipSetDevice");
  CK(L.hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");
  size_t bytes = req->bytes ? (size_t)req->bytes : (size_t)1 << 30;
  if (bytes > free_b / 3) bytes = free_b / 3;
  if (bytes < ((size_t)1 << 20)) return set_err("not enough free GPU memory for the bandwidth test");
  const int iters = req->iterations > 0 ? req->iterations : 20;
  CK(L.hipMalloc(&a, bytes), "hipMalloc");
  CK(L.hipMalloc(&b, bytes), "hipMalloc");
  CK(L.hipMemset(a, 1, bytes), "hipMemset");
  CK(L.hipEventCreate(&e0), "hipEventCreate");
  CK(L.hipEventCreate(&e1), "hipEventCreate");
  for (int i = 0; i < 3; ++i) CK(L.hipMemcpyAsync(b, a, bytes, MEMCPY_D2D, NULL), "warm-up copy");
  CK(L.hipDeviceSynchronize(), "hipDeviceSynchronize");
  CK(L.hipEventRecord(e0, NULL), "hipEventRecord");
  for (int i = 0; i < iters; ++i) CK(L.hipMemcpyAsync(b, a, bytes, MEMCPY_D2D, NULL), "device copy");
  CK(L.hipEventRecord(e1, NULL), "hipEventRecord");
  CK(L.hipEventSynchronize(e1), "hipEventSynchronize");
  float ms = 0;
  CK(L.hipEventElapsedTime(&ms, e0, e1), "hipEventElapsedTime");
  res->seconds = (double)ms / 1000.0;
  res->value = res->seconds > 0 ? 2.0 * (double)bytes * iters / res->seconds / 1e9 : 0;
  snprintf(res->unit, sizeof res->unit, "GB/s");
  L.hipEventDestroy(e0), L.hipEventDestroy(e1), L.hipFree(a), L.hipFree(b);
  return 0;
fail:
  if (e0) L.hipEventDestroy(e0);
  if (e1) L.hipEventDestroy(e1);
  if (a) L.hipFree(a);
  if (b) L.hipFree(b);
  return -1;
}

/* GEMM is deliberately not implemented yet. rocblas_gemm_ex has a long signature whose argument order and the
 * rocblas_datatype constants I could not check against a header (none is vendored in 3rdpart/FlagTree), and a
 * wrong guess crashes the host process instead of returning an error. It is added when it can be compared
 * with the real rocblas headers on a ROCm machine. */
static int bench_gemm(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  (void)device, (void)req, (void)res;
  return set_err(
      "gemm is not implemented in the ROCm plugin yet (needs verification against the rocBLAS headers on a "
      "ROCm machine)");
}

static int profile(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!req || !res) return set_err("null argument");
  if (!init_hip() || device < 0 || device >= g_ndev) return set_err("no such device");
  memset(res, 0, sizeof *res);
  if (!strcmp(req->kind, "bandwidth")) return bench_bandwidth(device, req, res);
  if (!strcmp(req->kind, "gemm")) return bench_gemm(device, req, res);
  return set_err("unknown profile kind (expected 'bandwidth' or 'gemm')");
}

static const mqde_hal_plugin_v1 kPlugin = {
    .abi_version = MQDE_HAL_ABI_VERSION,
    .struct_size = sizeof(mqde_hal_plugin_v1),
    .name = "rocm",
    .vendor = "amd",
    .plugin_version = "0.1.0-untested",
    .device_probe = device_probe,
    .last_error = last_error,
    .supports = supports,
    .alloc = p_alloc,
    .free = p_free,
    .profile = profile,
};

MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return &kPlugin; }
