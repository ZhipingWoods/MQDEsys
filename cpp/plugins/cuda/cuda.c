/* CUDA HAL plugin (T-1102). Devices come from the driver API (libcuda), micro-benchmarks from the runtime
 * (libcudart) and cuBLAS. Everything is loaded with dlopen: on a machine without NVIDIA software the plugin
 * reports zero devices and no error.
 *
 * Only stable ABI is used: driver attributes by numeric id, runtime calls by name, no struct that changes
 * between CUDA versions (no cudaDeviceProp). Device index = CUDA ordinal, so with CUDA_VISIBLE_DEVICES set it
 * differs from the NVML index the built-in probe prints, and a masked GPU can then be listed twice
 * (documented in docs/guides/add-hardware-plugin.md). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mqde/hal/mqde_hal.h"

static char g_err[256] = "";
static void set_err(const char* m) { snprintf(g_err, sizeof g_err, "%s", m); }
static void set_err2(const char* m, const char* d) { snprintf(g_err, sizeof g_err, "%s: %s", m, d); }
static const char* last_error(void) { return g_err; }

/* ---- driver API ---- */
typedef int (*cuInit_t)(unsigned);
typedef int (*cuDeviceGetCount_t)(int*);
typedef int (*cuDeviceGet_t)(int*, int);
typedef int (*cuDeviceGetName_t)(char*, int, int);
typedef int (*cuDeviceTotalMem_t)(size_t*, int);
typedef int (*cuDeviceGetAttribute_t)(int*, int, int);
typedef int (*cuDriverGetVersion_t)(int*);
/* ---- runtime API ---- */
typedef int (*cudaSetDevice_t)(int);
typedef int (*cudaMalloc_t)(void**, size_t);
typedef int (*cudaFree_t)(void*);
typedef int (*cudaMemGetInfo_t)(size_t*, size_t*);
typedef int (*cudaMemcpyAsync_t)(void*, const void*, size_t, int, void*);
typedef int (*cudaMemset_t)(void*, int, size_t);
typedef int (*cudaEventCreate_t)(void**);
typedef int (*cudaEventDestroy_t)(void*);
typedef int (*cudaEventRecord_t)(void*, void*);
typedef int (*cudaEventSynchronize_t)(void*);
typedef int (*cudaEventElapsedTime_t)(float*, void*, void*);
typedef int (*cudaDeviceSynchronize_t)(void);
typedef const char* (*cudaGetErrorString_t)(int);
/* ---- cuBLAS ---- */
typedef int (*cublasCreate_t)(void**);
typedef int (*cublasDestroy_t)(void*);
typedef int (*cublasGemmEx_t)(void*, int, int, int, int, int, const void*, const void*, int, int, const void*,
                              int, int, const void*, void*, int, int, int, int);

static struct {
  int tried, ok_driver, ok_runtime, ok_blas;
  void *drv, *rt, *blas;
  cuInit_t cuInit;
  cuDeviceGetCount_t cuDeviceGetCount;
  cuDeviceGet_t cuDeviceGet;
  cuDeviceGetName_t cuDeviceGetName;
  cuDeviceTotalMem_t cuDeviceTotalMem;
  cuDeviceGetAttribute_t cuDeviceGetAttribute;
  cuDriverGetVersion_t cuDriverGetVersion;
  cudaSetDevice_t cudaSetDevice;
  cudaMalloc_t cudaMalloc;
  cudaFree_t cudaFree;
  cudaMemGetInfo_t cudaMemGetInfo;
  cudaMemcpyAsync_t cudaMemcpyAsync;
  cudaMemset_t cudaMemset;
  cudaEventCreate_t cudaEventCreate;
  cudaEventDestroy_t cudaEventDestroy;
  cudaEventRecord_t cudaEventRecord;
  cudaEventSynchronize_t cudaEventSynchronize;
  cudaEventElapsedTime_t cudaEventElapsedTime;
  cudaDeviceSynchronize_t cudaDeviceSynchronize;
  cudaGetErrorString_t cudaGetErrorString;
  cublasCreate_t cublasCreate;
  cublasDestroy_t cublasDestroy;
  cublasGemmEx_t cublasGemmEx;
} L;

static void* open_first(const char* const* names) {
  for (; *names; ++names) {
    void* h = dlopen(*names, RTLD_LAZY | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}

/* resolves one symbol into L.<field>; true when found */
static int sym_ok(void** slot, void* h, const char* name) {
  *slot = dlsym(h, name);
  return *slot != NULL;
}
#define SYM(h, field, name) sym_ok((void**)&L.field, (h), (name))

static void load_libs(void) {
  if (L.tried) return;
  L.tried = 1;
  static const char* const drv[] = {"libcuda.so.1", "libcuda.so", NULL};
  static const char* const rt[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so",
                                   NULL};
  static const char* const bl[] = {"libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so",
                                   NULL};
  L.drv = open_first(drv);
  if (L.drv)
    L.ok_driver = SYM(L.drv, cuInit, "cuInit") && SYM(L.drv, cuDeviceGetCount, "cuDeviceGetCount") &&
                  SYM(L.drv, cuDeviceGet, "cuDeviceGet") && SYM(L.drv, cuDeviceGetName, "cuDeviceGetName") &&
                  SYM(L.drv, cuDeviceTotalMem, "cuDeviceTotalMem_v2") &&
                  SYM(L.drv, cuDeviceGetAttribute, "cuDeviceGetAttribute");
  if (L.drv) SYM(L.drv, cuDriverGetVersion, "cuDriverGetVersion");
  L.rt = open_first(rt);
  if (L.rt)
    L.ok_runtime = SYM(L.rt, cudaSetDevice, "cudaSetDevice") && SYM(L.rt, cudaMalloc, "cudaMalloc") &&
                   SYM(L.rt, cudaFree, "cudaFree") && SYM(L.rt, cudaMemGetInfo, "cudaMemGetInfo") &&
                   SYM(L.rt, cudaMemcpyAsync, "cudaMemcpyAsync") && SYM(L.rt, cudaMemset, "cudaMemset") &&
                   SYM(L.rt, cudaEventCreate, "cudaEventCreate") &&
                   SYM(L.rt, cudaEventDestroy, "cudaEventDestroy") &&
                   SYM(L.rt, cudaEventRecord, "cudaEventRecord") &&
                   SYM(L.rt, cudaEventSynchronize, "cudaEventSynchronize") &&
                   SYM(L.rt, cudaEventElapsedTime, "cudaEventElapsedTime") &&
                   SYM(L.rt, cudaDeviceSynchronize, "cudaDeviceSynchronize") &&
                   SYM(L.rt, cudaGetErrorString, "cudaGetErrorString");
  L.blas = open_first(bl);
  if (L.blas)
    L.ok_blas = SYM(L.blas, cublasCreate, "cublasCreate_v2") &&
                SYM(L.blas, cublasDestroy, "cublasDestroy_v2") && SYM(L.blas, cublasGemmEx, "cublasGemmEx");
}

static int g_inited = 0;
static int g_ndev = 0;

static int init_driver(void) {
  load_libs();
  if (!L.ok_driver) return 0;
  if (!g_inited) {
    if (L.cuInit(0) != 0) return 0; /* driver present but no usable GPU: zero devices */
    if (L.cuDeviceGetCount(&g_ndev) != 0) g_ndev = 0;
    g_inited = 1;
  }
  return 1;
}

enum { ATTR_SM_COUNT = 16, ATTR_CC_MAJOR = 75, ATTR_CC_MINOR = 76 };

static int cc_of(int dev) {
  int dh = 0, major = 0, minor = 0;
  if (L.cuDeviceGet(&dh, dev) != 0) return 0;
  L.cuDeviceGetAttribute(&major, ATTR_CC_MAJOR, dh);
  L.cuDeviceGetAttribute(&minor, ATTR_CC_MINOR, dh);
  return major * 10 + minor;
}

static int device_probe(mqde_hal_device* out, int max, int* n) {
  if (!out || !n || max < 0) return set_err("invalid arguments"), -1;
  *n = 0;
  if (!init_driver()) return 0;
  int drv_ver = 0;
  if (L.cuDriverGetVersion) L.cuDriverGetVersion(&drv_ver);
  for (int i = 0; i < g_ndev && *n < max; ++i) {
    int dh = 0;
    if (L.cuDeviceGet(&dh, i) != 0) continue;
    mqde_hal_device* d = &out[*n];
    memset(d, 0, sizeof *d);
    d->kind = MQDE_HAL_DEV_GPU;
    d->index = i;
    snprintf(d->vendor, sizeof d->vendor, "nvidia");
    L.cuDeviceGetName(d->name, (int)sizeof d->name, dh);
    int major = 0, minor = 0, sms = 0;
    L.cuDeviceGetAttribute(&major, ATTR_CC_MAJOR, dh);
    L.cuDeviceGetAttribute(&minor, ATTR_CC_MINOR, dh);
    L.cuDeviceGetAttribute(&sms, ATTR_SM_COUNT, dh);
    snprintf(d->arch, sizeof d->arch, "sm_%d%d", major, minor);
    size_t total = 0;
    if (L.cuDeviceTotalMem(&total, dh) == 0) d->mem_total_bytes = total;
    d->mem_free_bytes = 0; /* free memory needs a context; the built-in NVML probe reports it */
    d->compute_units = (uint32_t)(sms > 0 ? sms : 0);
    snprintf(d->driver_version, sizeof d->driver_version, "cuda-driver-api-%d.%d", drv_ver / 1000,
             (drv_ver % 1000) / 10);
    ++*n;
  }
  return 0;
}

/* Capability table by compute capability (cc = major*10+minor). "native" means a tensor-core / hardware path
 * exists. */
static int supports(const mqde_hal_capability_query* q, mqde_hal_capability_result* r) {
  if (!q || !r) return set_err("null argument"), -1;
  if (!init_driver() || q->device < 0 || q->device >= g_ndev) return set_err("no such device"), -1;
  const int cc = cc_of(q->device);
  memset(r, 0, sizeof *r);
  const char* t = q->dtype;
  int sup = 0, nat = 0;
  const char* note = "";
  if (!strcmp(t, "fp32"))
    sup = 1, nat = 1, note = "CUDA cores (TF32 tensor cores from sm_80)";
  else if (!strcmp(t, "fp16"))
    sup = 1, nat = cc >= 70, note = cc >= 70 ? "tensor cores" : "no tensor cores before sm_70";
  else if (!strcmp(t, "bf16"))
    sup = cc >= 80, nat = cc >= 80, note = cc >= 80 ? "tensor cores" : "bf16 needs sm_80 or newer";
  else if (!strcmp(t, "int8"))
    sup = cc >= 61, nat = cc >= 75, note = cc >= 75 ? "tensor cores" : "dp4a only before sm_75";
  else if (!strcmp(t, "fp8_e4m3") || !strcmp(t, "fp8_e5m2"))
    sup = cc >= 89, nat = cc >= 89,
    note = cc >= 89 ? "fp8 tensor cores (sm_89 and newer)" : "fp8 needs sm_89 or newer";
  else if (!strcmp(t, "int4"))
    sup = cc >= 75 && cc < 90, nat = sup,
    note = sup ? "tensor core mma only, no cuBLAS GEMM path" : "int4 tensor cores exist on sm_75 to sm_89";
  else
    note = "dtype not known to the CUDA plugin";
  r->supported = sup;
  r->native = nat;
  r->peak_tflops =
      0; /* not derivable from the device; the hardware profile database and profile() carry numbers */
  snprintf(r->note, sizeof r->note, "%s", note);
  return 0;
}

static int p_alloc(int device, size_t bytes, void** ptr) {
  if (!ptr) return set_err("null pointer"), -1;
  load_libs();
  if (!L.ok_runtime) return set_err("libcudart not available"), -1;
  int rc = L.cudaSetDevice(device);
  if (rc == 0) rc = L.cudaMalloc(ptr, bytes ? bytes : 1);
  if (rc != 0) return set_err2("cudaMalloc failed", L.cudaGetErrorString(rc)), -1;
  return 0;
}
static int p_free(int device, void* ptr) {
  load_libs();
  if (!L.ok_runtime) return set_err("libcudart not available"), -1;
  L.cudaSetDevice(device);
  return L.cudaFree(ptr) == 0 ? 0 : (set_err("cudaFree failed"), -1);
}

#define CK(call, what)                           \
  do {                                           \
    int rc_ = (call);                            \
    if (rc_ != 0) {                              \
      set_err2(what, L.cudaGetErrorString(rc_)); \
      goto fail;                                 \
    }                                            \
  } while (0)

enum { MEMCPY_D2D = 3 };
enum { R_32F = 0, R_16F = 2, R_8I = 3, R_32I = 10, R_16BF = 14 };
enum { COMPUTE_32F = 68, COMPUTE_32I = 72 };

static int bench_bandwidth(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  void *a = NULL, *b = NULL, *e0 = NULL, *e1 = NULL;
  size_t free_b = 0, total_b = 0;
  CK(L.cudaSetDevice(device), "cudaSetDevice");
  CK(L.cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
  size_t bytes = req->bytes ? (size_t)req->bytes : (size_t)1 << 30;
  if (bytes > free_b / 3) bytes = free_b / 3; /* two buffers and headroom */
  if (bytes < ((size_t)1 << 20)) return set_err("not enough free GPU memory for the bandwidth test"), -1;
  const int iters = req->iterations > 0 ? req->iterations : 20;
  CK(L.cudaMalloc(&a, bytes), "cudaMalloc");
  CK(L.cudaMalloc(&b, bytes), "cudaMalloc");
  CK(L.cudaMemset(a, 1, bytes), "cudaMemset");
  CK(L.cudaEventCreate(&e0), "cudaEventCreate");
  CK(L.cudaEventCreate(&e1), "cudaEventCreate");
  for (int i = 0; i < 3; ++i)
    CK(L.cudaMemcpyAsync(b, a, bytes, MEMCPY_D2D, NULL), "warm-up copy"); /* not timed */
  CK(L.cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  CK(L.cudaEventRecord(e0, NULL), "cudaEventRecord");
  for (int i = 0; i < iters; ++i) CK(L.cudaMemcpyAsync(b, a, bytes, MEMCPY_D2D, NULL), "device copy");
  CK(L.cudaEventRecord(e1, NULL), "cudaEventRecord");
  CK(L.cudaEventSynchronize(e1), "cudaEventSynchronize");
  float ms = 0;
  CK(L.cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");
  res->seconds = (double)ms / 1000.0;
  /* a copy reads and writes `bytes`: count both, the way vendors quote memory bandwidth */
  res->value = res->seconds > 0 ? 2.0 * (double)bytes * iters / res->seconds / 1e9 : 0;
  snprintf(res->unit, sizeof res->unit, "GB/s");
  L.cudaEventDestroy(e0), L.cudaEventDestroy(e1), L.cudaFree(a), L.cudaFree(b);
  return 0;
fail:
  if (e0) L.cudaEventDestroy(e0);
  if (e1) L.cudaEventDestroy(e1);
  if (a) L.cudaFree(a);
  if (b) L.cudaFree(b);
  return -1;
}

static int bench_gemm(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!L.ok_blas) return set_err("libcublas not available"), -1;
  int ta, tc, comp;
  size_t esz_in, esz_out;
  const char* d = req->dtype;
  if (!strcmp(d, "fp16"))
    ta = R_16F, tc = R_16F, comp = COMPUTE_32F, esz_in = esz_out = 2;
  else if (!strcmp(d, "bf16"))
    ta = R_16BF, tc = R_16BF, comp = COMPUTE_32F, esz_in = esz_out = 2;
  else if (!strcmp(d, "fp32"))
    ta = R_32F, tc = R_32F, comp = COMPUTE_32F, esz_in = esz_out = 4;
  else if (!strcmp(d, "int8"))
    ta = R_8I, tc = R_32I, comp = COMPUTE_32I, esz_in = 1, esz_out = 4;
  else
    return set_err(
               "gemm dtype not supported by this plugin (fp32, fp16, bf16, int8; fp8 needs cublasLt, not "
               "implemented)"),
           -1;
  int m = req->m > 0 ? req->m : 4096, n = req->n > 0 ? req->n : 4096, k = req->k > 0 ? req->k : 4096;
  if (ta == R_8I && (m % 4 || n % 4 || k % 4))
    return set_err("int8 gemm needs m, n and k to be multiples of 4"), -1;
  const int iters = req->iterations > 0 ? req->iterations : 20;
  void *A = NULL, *B = NULL, *C = NULL, *e0 = NULL, *e1 = NULL, *h = NULL;
  CK(L.cudaSetDevice(device), "cudaSetDevice");
  CK(L.cudaMalloc(&A, (size_t)m * k * esz_in), "cudaMalloc A");
  CK(L.cudaMalloc(&B, (size_t)k * n * esz_in), "cudaMalloc B");
  CK(L.cudaMalloc(&C, (size_t)m * n * esz_out), "cudaMalloc C");
  CK(L.cudaMemset(A, 0, (size_t)m * k * esz_in), "cudaMemset");
  CK(L.cudaMemset(B, 0, (size_t)k * n * esz_in), "cudaMemset");
  if (L.cublasCreate(&h) != 0)
    return set_err("cublasCreate failed"), L.cudaFree(A), L.cudaFree(B), L.cudaFree(C), -1;
  CK(L.cudaEventCreate(&e0), "cudaEventCreate");
  CK(L.cudaEventCreate(&e1), "cudaEventCreate");
  float af = 1.0f, bf = 0.0f;
  int ai = 1, bi = 0;
  const void* alpha = comp == COMPUTE_32I ? (const void*)&ai : (const void*)&af;
  const void* beta = comp == COMPUTE_32I ? (const void*)&bi : (const void*)&bf;
  for (int i = 0; i < iters + 3; ++i) {
    if (i == 3) {
      CK(L.cudaDeviceSynchronize(), "cudaDeviceSynchronize"); /* the first three runs are warm-up */
      CK(L.cudaEventRecord(e0, NULL), "cudaEventRecord");
    }
    int rc = L.cublasGemmEx(h, 0, 0, m, n, k, alpha, A, ta, m, B, ta, k, beta, C, tc, m, comp, -1);
    if (rc != 0) {
      char msg[64];
      snprintf(msg, sizeof msg, "cublasGemmEx returned status %d", rc);
      set_err(msg);
      goto fail;
    }
  }
  CK(L.cudaEventRecord(e1, NULL), "cudaEventRecord");
  CK(L.cudaEventSynchronize(e1), "cudaEventSynchronize");
  float ms = 0;
  CK(L.cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");
  res->seconds = (double)ms / 1000.0;
  res->value = res->seconds > 0 ? 2.0 * (double)m * n * k * iters / res->seconds / 1e12 : 0;
  /* INT8 through cublasGemmEx reached about 170 TFLOPS on an RTX 4090 (datasheet 660): this call does not use
   * the IMMA layout that cublasLt does, so for int8 the number is a lower bound of the hardware, not its
   * peak. */
  snprintf(res->unit, sizeof res->unit, "TFLOPS");
  L.cublasDestroy(h), L.cudaEventDestroy(e0), L.cudaEventDestroy(e1), L.cudaFree(A), L.cudaFree(B),
      L.cudaFree(C);
  return 0;
fail:
  if (h) L.cublasDestroy(h);
  if (e0) L.cudaEventDestroy(e0);
  if (e1) L.cudaEventDestroy(e1);
  if (A) L.cudaFree(A);
  if (B) L.cudaFree(B);
  if (C) L.cudaFree(C);
  return -1;
}

static int profile(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!req || !res) return set_err("null argument"), -1;
  if (!init_driver() || device < 0 || device >= g_ndev) return set_err("no such device"), -1;
  if (!L.ok_runtime) return set_err("libcudart not available"), -1;
  memset(res, 0, sizeof *res);
  if (!strcmp(req->kind, "bandwidth")) return bench_bandwidth(device, req, res);
  if (!strcmp(req->kind, "gemm")) return bench_gemm(device, req, res);
  return set_err("unknown profile kind (expected 'bandwidth' or 'gemm')"), -1;
}

static const mqde_hal_plugin_v1 kPlugin = {
    .abi_version = MQDE_HAL_ABI_VERSION,
    .struct_size = sizeof(mqde_hal_plugin_v1),
    .name = "cuda",
    .vendor = "nvidia",
    .plugin_version = "0.1.0",
    .device_probe = device_probe,
    .last_error = last_error,
    .supports = supports,
    .alloc = p_alloc,
    .free = p_free,
    .profile = profile,
};

MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return &kPlugin; }
