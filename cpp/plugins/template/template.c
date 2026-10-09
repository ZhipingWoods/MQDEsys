/* Template HAL plugin: a simulated device. Copy this directory to cpp/plugins/<vendor>/ and replace the
 * marked parts.
 *
 * It reports devices only when MQDE_TEMPLATE_DEVICES is set to a positive number, which is how a real plugin
 * behaves when its vendor runtime is missing: zero devices, no error. */
#define _GNU_SOURCE
#include "mqde/hal/mqde_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_err[160] = "";

static void set_err(const char* m) { snprintf(g_err, sizeof g_err, "%s", m); }
static const char* last_error(void) { return g_err; }

/* 1. DEVICE DISCOVERY. A real plugin dlopen()s its vendor runtime here and returns 0 devices when it is not
 * installed. */
static int device_count(void) {
  const char* v = getenv("MQDE_TEMPLATE_DEVICES");
  int n = v ? atoi(v) : 0;
  return n < 0 ? 0 : n > 8 ? 8 : n;
}

static int device_probe(mqde_hal_device* out, int max, int* n) {
  if (!out || !n || max < 0) {
    set_err("invalid arguments");
    return -1;
  }
  const int have = device_count();
  int w = 0;
  for (int i = 0; i < have && w < max; ++i, ++w) {
    mqde_hal_device* d = &out[w];
    memset(d, 0, sizeof *d);
    d->kind = MQDE_HAL_DEV_NPU;
    d->index = i;
    snprintf(d->vendor, sizeof d->vendor, "template");
    snprintf(d->name, sizeof d->name, "MQDE Template Accelerator %d", i);
    snprintf(d->arch, sizeof d->arch, "tmpl1");
    d->mem_total_bytes = 16ull << 30;
    d->mem_free_bytes = 12ull << 30;
    d->compute_units = 32;
    snprintf(d->driver_version, sizeof d->driver_version, "0.0-simulated");
  }
  *n = w;
  return 0;
}

/* 2. CAPABILITIES. Answer from the hardware's real support table. */
static int supports(const mqde_hal_capability_query* q, mqde_hal_capability_result* r) {
  if (!q || !r || q->device < 0 || q->device >= device_count()) {
    set_err("no such device");
    return -1;
  }
  memset(r, 0, sizeof *r);
  struct {
    const char* dtype;
    int native;
    double tflops;
  } table[] = {
      {"fp16", 1, 100.0}, {"bf16", 1, 100.0}, {"int8", 1, 200.0}, {"int4", 0, 0.0}, {"fp8_e4m3", 0, 0.0}};
  for (size_t i = 0; i < sizeof table / sizeof table[0]; ++i)
    if (strcmp(q->dtype, table[i].dtype) == 0) {
      r->supported = 1;
      r->native = table[i].native;
      r->peak_tflops = table[i].tflops;
      snprintf(r->note, sizeof r->note, "%s",
               table[i].native ? "simulated native path" : "simulated emulation");
      return 0;
    }
  r->supported = 0;
  snprintf(r->note, sizeof r->note, "dtype not known to the template");
  return 0;
}

/* 3. MEMORY. */
static int p_alloc(int device, size_t bytes, void** ptr) {
  (void)device;
  if (!ptr) return set_err("null pointer"), -1;
  *ptr = malloc(bytes ? bytes : 1);
  return *ptr ? 0 : (set_err("out of memory"), -1);
}
static int p_free(int device, void* ptr) {
  (void)device;
  free(ptr);
  return 0;
}

/* 4. MICRO-BENCHMARKS. A real plugin times a large copy and a GEMM on the device. These numbers are made up
 * and marked so. */
static int profile(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!req || !res || device < 0 || device >= device_count()) return set_err("no such device"), -1;
  memset(res, 0, sizeof *res);
  if (strcmp(req->kind, "bandwidth") == 0) {
    res->value = 500.0;
    snprintf(res->unit, sizeof res->unit, "GB/s");
  } else if (strcmp(req->kind, "gemm") == 0) {
    res->value = 50.0;
    snprintf(res->unit, sizeof res->unit, "TFLOPS");
  } else {
    set_err("unknown profile kind (expected 'bandwidth' or 'gemm')");
    return -1;
  }
  res->seconds = 0.0;
  return 0;
}

static const mqde_hal_plugin_v1 kPlugin = {
    .abi_version = MQDE_HAL_ABI_VERSION,
    .struct_size = sizeof(mqde_hal_plugin_v1),
    .name = "template",
    .vendor = "template",
    .plugin_version = "0.1.0",
    .device_probe = device_probe,
    .last_error = last_error,
    .supports = supports,
    .alloc = p_alloc,
    .free = p_free,
    .profile = profile,
    .compile = NULL, /* optional in v1 */
    .launch = NULL,
};

MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return &kPlugin; }
