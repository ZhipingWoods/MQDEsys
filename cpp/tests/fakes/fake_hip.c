/* Stand-in for libamdhip64 (test only). Exports the HIP functions the ROCm plugin uses, with the semantics of
 * the real ones for what the plugin relies on: attribute ids are the real ones (63 CU count, 23 / 61 compute
 * capability), copies really move bytes, events measure real wall time. Behaviour is steered with environment
 * variables so the tests can drive every branch: FAKE_HIP_DEVICES=n          number of GPUs (default 2)
 *   FAKE_HIP_MAJOR=m            compute capability major (default 11); FAKE_HIP_MINOR=m the minor (default 4)
 *   FAKE_HIP_FAIL=<function>    that function returns an error code
 *   FAKE_HIP_MEM_MB=n           total and free memory per device (default 8192) */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int env_int(const char* k, int d) {
  const char* v = getenv(k);
  return v && *v ? atoi(v) : d;
}
static int fails(const char* fn) {
  const char* f = getenv("FAKE_HIP_FAIL");
  return f && !strcmp(f, fn);
}
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
#define FAIL_IF(fn) \
  if (fails(fn)) return 719 /* hipErrorLaunchFailure */

int hipInit(unsigned flags) {
  (void)flags;
  FAIL_IF("hipInit");
  return 0;
}
int hipGetDeviceCount(int* n) {
  FAIL_IF("hipGetDeviceCount");
  *n = env_int("FAKE_HIP_DEVICES", 2);
  return 0;
}
int hipDriverGetVersion(int* v) {
  *v = 60300000; /* ROCm 6.3 style */
  return 0;
}
int hipDeviceGetName(char* name, int len, int dev) {
  snprintf(name, (size_t)len, "Fake AMD GPU %d", dev);
  return 0;
}
int hipDeviceTotalMem(size_t* bytes, int dev) {
  (void)dev;
  *bytes = (size_t)env_int("FAKE_HIP_MEM_MB", 8192) << 20;
  return 0;
}
int hipDeviceGetAttribute(int* out, int attr, int dev) {
  (void)dev;
  switch (attr) {
    case 63:
      *out = 96;
      return 0; /* hipDeviceAttributeMultiprocessorCount */
    case 23:
      *out = env_int("FAKE_HIP_MAJOR", 11);
      return 0; /* ComputeCapabilityMajor */
    case 61:
      *out = env_int("FAKE_HIP_MINOR", 4);
      return 0; /* ComputeCapabilityMinor, nonzero so that reading the wrong id changes the result */
    default:
      *out = 0;
      return 1; /* any other id: an error, so a wrong id in the plugin shows up */
  }
}
int hipSetDevice(int dev) {
  FAIL_IF("hipSetDevice");
  return dev >= 0 && dev < env_int("FAKE_HIP_DEVICES", 2) ? 0 : 101;
}
int hipMalloc(void** p, size_t n) {
  FAIL_IF("hipMalloc");
  *p = malloc(n ? n : 1);
  return *p ? 0 : 2;
}
int hipFree(void* p) {
  free(p);
  return 0;
}
int hipMemGetInfo(size_t* free_b, size_t* total) {
  *total = (size_t)env_int("FAKE_HIP_MEM_MB", 8192) << 20;
  *free_b = *total;
  return 0;
}
int hipMemcpyAsync(void* d, const void* s, size_t n, int kind, void* stream) {
  (void)stream;
  FAIL_IF("hipMemcpyAsync");
  if (kind != 3)
    return 21; /* only hipMemcpyDeviceToDevice (3) is used by the plugin; any other value is a wrong enum */
  memcpy(d, s, n);
  return 0;
}
int hipMemset(void* p, int v, size_t n) {
  memset(p, v, n);
  return 0;
}
typedef struct {
  double t;
} ev_t;
int hipEventCreate(void** e) {
  *e = calloc(1, sizeof(ev_t));
  return 0;
}
int hipEventDestroy(void* e) {
  free(e);
  return 0;
}
int hipEventRecord(void* e, void* stream) {
  (void)stream;
  ((ev_t*)e)->t = now();
  return 0;
}
int hipEventSynchronize(void* e) {
  (void)e;
  return 0;
}
int hipEventElapsedTime(float* ms, void* a, void* b) {
  *ms = (float)((((ev_t*)b)->t - ((ev_t*)a)->t) * 1000.0);
  return 0;
}
int hipDeviceSynchronize(void) { return 0; }
const char* hipGetErrorString(int rc) {
  return rc == 719 ? "hipErrorLaunchFailure (fake)" : "fake hip error";
}
