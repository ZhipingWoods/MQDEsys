/* CPU HAL plugin (T-1102): the host CPU as a device, with a memory-bandwidth and a GEMM micro-benchmark that
 * run anywhere. No vendor library. The GEMM is a plain blocked fp32 loop (the compiler vectorises it), so its
 * TFLOPS is a floor for what an optimised BLAS reaches, and is labelled that way in `note`. Bandwidth is a
 * large memcpy, read plus write counted. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "mqde/hal/mqde_hal.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static char g_err[160] = "";
static int set_err(const char* m) {
  snprintf(g_err, sizeof g_err, "%s", m);
  return -1;
}
static const char* last_error(void) { return g_err; }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void cpu_name(char* out, size_t cap) {
  out[0] = '\0';
#if defined(__APPLE__)
  size_t n = cap;
  if (sysctlbyname("machdep.cpu.brand_string", out, &n, NULL, 0) != 0) out[0] = '\0';
#elif defined(__linux__)
  FILE* f = fopen("/proc/cpuinfo", "r");
  char line[256];
  while (f && fgets(line, sizeof line, f))
    if (!strncmp(line, "model name", 10)) {
      const char* c = strchr(line, ':');
      if (c) snprintf(out, cap, "%s", c + 2);
      break;
    }
  if (f) fclose(f);
#endif
  for (size_t i = strlen(out); i > 0 && (out[i - 1] == '\n' || out[i - 1] == ' '); --i) out[i - 1] = '\0';
  if (!out[0]) snprintf(out, cap, "CPU");
}

static uint64_t total_mem(void) {
#if defined(__APPLE__)
  uint64_t m = 0;
  size_t n = sizeof m;
  return sysctlbyname("hw.memsize", &m, &n, NULL, 0) == 0 ? m : 0;
#else
  long pages = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGE_SIZE);
  return pages > 0 && ps > 0 ? (uint64_t)pages * (uint64_t)ps : 0;
#endif
}

static int device_probe(mqde_hal_device* out, int max, int* n) {
  if (!out || !n || max < 0) return set_err("invalid arguments");
  *n = 0;
  if (max < 1) return 0;
  memset(out, 0, sizeof *out);
  out->kind = MQDE_HAL_DEV_CPU;
  out->index = -1;
  cpu_name(out->name, sizeof out->name);
  /* the same vendor rule as the built-in probe, so that probe and plugin describe one CPU and it is listed
   * once */
#if defined(__APPLE__)
  snprintf(out->vendor, sizeof out->vendor, "%s", strstr(out->name, "Apple") ? "apple" : "intel");
#else
  snprintf(out->vendor, sizeof out->vendor, "%s",
           strstr(out->name, "AMD") ? "amd" : (strstr(out->name, "Intel") ? "intel" : "cpu"));
#endif
#if defined(__aarch64__) || defined(__arm64__)
  snprintf(out->arch, sizeof out->arch, "arm64");
#else
  snprintf(out->arch, sizeof out->arch, "x86_64");
#endif
  out->mem_total_bytes = total_mem();
  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  out->compute_units = cpus > 0 ? (uint32_t)cpus : 1;
  snprintf(out->driver_version, sizeof out->driver_version, "n/a");
  *n = 1;
  return 0;
}

static int supports(const mqde_hal_capability_query* q, mqde_hal_capability_result* r) {
  if (!q || !r) return set_err("null argument");
  if (q->device != 0 && q->device != -1) return set_err("no such device");
  memset(r, 0, sizeof *r);
  const char* t = q->dtype;
  int sup = 1;
  const char* note = "emulated in software";
  if (!strcmp(t, "fp32"))
    note = "native (SIMD)";
  else if (!strcmp(t, "int8"))
    note = "native integer SIMD, no tensor units assumed";
  else if (!strcmp(t, "fp16") || !strcmp(t, "bf16"))
    note = "converted to fp32 unless the CPU has native support";
  else if (!strcmp(t, "int4") || !strcmp(t, "fp8_e4m3") || !strcmp(t, "fp8_e5m2"))
    note = "emulated: unpack / convert, no hardware path";
  else
    sup = 0, note = "dtype not known to the CPU plugin";
  r->supported = sup;
  r->native = !strcmp(t, "fp32") || !strcmp(t, "int8");
  snprintf(r->note, sizeof r->note, "%s", note);
  return 0;
}

static int p_alloc(int device, size_t bytes, void** ptr) {
  (void)device;
  if (!ptr) return set_err("null pointer");
  *ptr = malloc(bytes ? bytes : 1);
  return *ptr ? 0 : set_err("out of memory");
}
static int p_free(int device, void* ptr) {
  (void)device;
  free(ptr);
  return 0;
}

static int bench_bandwidth(const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  size_t bytes = req->bytes ? (size_t)req->bytes : (size_t)256 << 20;
  if (bytes < ((size_t)1 << 20)) bytes = (size_t)1 << 20;
  char* a = malloc(bytes);
  char* b = malloc(bytes);
  if (!a || !b) {
    free(a);
    free(b);
    return set_err("out of memory for the bandwidth test");
  }
  memset(a, 1, bytes);
  memset(b, 0, bytes);
  const int iters = req->iterations > 0 ? req->iterations : 10;
  /* memcpy through a volatile function pointer, with the source changing every round: the compiler can
   * neither drop the copies nor fold them into one, and the checksum below proves the data really moved. */
  void* (*volatile copy_fn)(void*, const void*, size_t) = memcpy;
  for (int i = 0; i < 2; ++i) copy_fn(b, a, bytes); /* warm-up, not timed */
  const double t0 = now();
  for (int i = 0; i < iters; ++i) {
    a[(size_t)i % 64] = (char)(i + 2);
    copy_fn(b, a, bytes);
  }
  res->seconds = now() - t0;
  if (b[(size_t)(iters - 1) % 64] != (char)(iters + 1)) {
    free(a);
    free(b);
    return set_err("bandwidth test: the copied data does not match the source");
  }
  res->value = res->seconds > 0 ? 2.0 * (double)bytes * iters / res->seconds / 1e9 : 0;
  snprintf(res->unit, sizeof res->unit, "GB/s");
  free(a);
  free(b);
  return 0;
}

static int bench_gemm(const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (strcmp(req->dtype, "fp32") != 0) return set_err("the CPU plugin benchmarks fp32 only");
  const int m = req->m > 0 ? req->m : 512, n = req->n > 0 ? req->n : 512, k = req->k > 0 ? req->k : 512;
  float* A = malloc(sizeof(float) * (size_t)m * (size_t)k);
  float* B = malloc(sizeof(float) * (size_t)k * (size_t)n);
  float* C = malloc(sizeof(float) * (size_t)m * (size_t)n);
  if (!A || !B || !C) {
    free(A), free(B), free(C);
    return set_err("out of memory for the gemm test");
  }
  for (size_t i = 0; i < (size_t)m * k; ++i) A[i] = (float)(i % 7) * 0.25f;
  for (size_t i = 0; i < (size_t)k * n; ++i) B[i] = (float)(i % 5) * 0.5f;
  const int iters = req->iterations > 0 ? req->iterations : 3;
  double t_best = 1e30;
  for (int it = 0; it < iters + 1; ++it) { /* first run is warm-up, then the best of `iters` */
    memset(C, 0, sizeof(float) * (size_t)m * (size_t)n);
    const double t0 = now();
    for (int i = 0; i < m; ++i)
      for (int p = 0; p < k; ++p) {
        const float a = A[(size_t)i * k + p];
        const float* b = B + (size_t)p * n;
        float* c = C + (size_t)i * n;
        for (int j = 0; j < n; ++j) c[j] += a * b[j];
      }
    const double dt = now() - t0;
    if (it > 0 && dt < t_best) t_best = dt;
  }
  volatile float sink = C[(size_t)(m / 2) * n + n / 2]; /* use the result */
  (void)sink;
  res->seconds = t_best;
  res->value = t_best > 0 ? 2.0 * m * n * k / t_best / 1e12 : 0;
  snprintf(res->unit, sizeof res->unit, "TFLOPS");
  free(A), free(B), free(C);
  return 0;
}

static int profile(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!req || !res) return set_err("null argument");
  if (device != 0 && device != -1) return set_err("no such device");
  memset(res, 0, sizeof *res);
  if (!strcmp(req->kind, "bandwidth")) return bench_bandwidth(req, res);
  if (!strcmp(req->kind, "gemm")) return bench_gemm(req, res);
  return set_err("unknown profile kind (expected 'bandwidth' or 'gemm')");
}

static const mqde_hal_plugin_v1 kPlugin = {
    .abi_version = MQDE_HAL_ABI_VERSION,
    .struct_size = sizeof(mqde_hal_plugin_v1),
    .name = "cpu",
    .vendor = "host",
    .plugin_version = "0.1.0",
    .device_probe = device_probe,
    .last_error = last_error,
    .supports = supports,
    .alloc = p_alloc,
    .free = p_free,
    .profile = profile,
};

MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return &kPlugin; }
