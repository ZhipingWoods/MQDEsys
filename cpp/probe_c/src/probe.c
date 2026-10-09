/* Hardware probe: CPU (Linux / macOS) + NVIDIA GPUs via a dynamically loaded NVML. */
#define _GNU_SOURCE
#include "mqde_probe.h"
#include "mqde/hal/mqde_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

static char g_last_error[256] = "";

const char* mqde_probe_last_error(void) { return g_last_error; }

const char* mqde_dev_kind_name(mqde_dev_kind kind) {
  switch (kind) {
    case MQDE_DEV_CPU: return "cpu";
    case MQDE_DEV_GPU: return "gpu";
    case MQDE_DEV_NPU: return "npu";
    default: return "other";
  }
}

static void copy_str(char* dst, size_t cap, const char* src) {
  if (!src) src = "";
  snprintf(dst, cap, "%s", src);
}

#if defined(__linux__)
static void trim(char* s) {
  size_t n = strlen(s);
  while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = '\0';
}
#endif

static void probe_cpu(mqde_device_info* d) {
  memset(d, 0, sizeof(*d));
  d->kind = MQDE_DEV_CPU;
  d->index = -1;
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  d->compute_units = n > 0 ? (uint32_t)n : 0;
#if defined(__x86_64__) || defined(_M_X64)
  copy_str(d->arch, sizeof d->arch, "x86_64");
#elif defined(__aarch64__) || defined(__arm64__)
  copy_str(d->arch, sizeof d->arch, "arm64");
#else
  copy_str(d->arch, sizeof d->arch, "unknown");
#endif
#if defined(__linux__)
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[512];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "model name", 10) == 0 || strncmp(line, "Model", 5) == 0) {
        char* colon = strchr(line, ':');
        if (colon) {
          colon++;
          while (*colon == ' ' || *colon == '\t') colon++;
          copy_str(d->name, sizeof d->name, colon);
          trim(d->name);
          break;
        }
      }
    }
    fclose(f);
  }
  f = fopen("/proc/meminfo", "r");
  if (f) {
    char key[64];
    unsigned long long kb;
    char unit[16];
    while (fscanf(f, "%63s %llu %15s", key, &kb, unit) == 3) {
      if (strcmp(key, "MemTotal:") == 0) d->mem_total_bytes = (uint64_t)kb * 1024ull;
      if (strcmp(key, "MemAvailable:") == 0) d->mem_free_bytes = (uint64_t)kb * 1024ull;
    }
    fclose(f);
  }
  copy_str(d->vendor, sizeof d->vendor,
           strstr(d->name, "AMD") ? "amd" : (strstr(d->name, "Intel") ? "intel" : "cpu"));
#elif defined(__APPLE__)
  size_t len = sizeof d->name;
  sysctlbyname("machdep.cpu.brand_string", d->name, &len, NULL, 0);
  uint64_t mem = 0;
  len = sizeof mem;
  if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0) d->mem_total_bytes = mem;
  copy_str(d->vendor, sizeof d->vendor, strstr(d->name, "Apple") ? "apple" : "intel");
#endif
  if (d->name[0] == '\0') copy_str(d->name, sizeof d->name, "Unknown CPU");
}

#if !defined(_WIN32)
/* Minimal NVML surface (types match nvml.h; we avoid a build dependency on the header). */
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct {
  unsigned long long total, free, used;
} nvmlMemory_t;
typedef int (*fn_init)(void);
typedef int (*fn_shutdown)(void);
typedef int (*fn_count)(unsigned int*);
typedef int (*fn_handle)(unsigned int, nvmlDevice_t*);
typedef int (*fn_name)(nvmlDevice_t, char*, unsigned int);
typedef int (*fn_mem)(nvmlDevice_t, nvmlMemory_t*);
typedef int (*fn_cc)(nvmlDevice_t, int*, int*);
typedef int (*fn_drv)(char*, unsigned int);
typedef int (*fn_mpcount)(nvmlDevice_t, unsigned int*);

static int probe_nvidia(mqde_device_info* out, int max) {
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
  if (!lib) return 0;
  fn_init init = (fn_init)dlsym(lib, "nvmlInit_v2");
  fn_shutdown shutdown = (fn_shutdown)dlsym(lib, "nvmlShutdown");
  fn_count count = (fn_count)dlsym(lib, "nvmlDeviceGetCount_v2");
  fn_handle handle = (fn_handle)dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
  fn_name name = (fn_name)dlsym(lib, "nvmlDeviceGetName");
  fn_mem mem = (fn_mem)dlsym(lib, "nvmlDeviceGetMemoryInfo");
  fn_cc cc = (fn_cc)dlsym(lib, "nvmlDeviceGetCudaComputeCapability");
  fn_drv drv = (fn_drv)dlsym(lib, "nvmlSystemGetDriverVersion");
  fn_mpcount mpc = (fn_mpcount)dlsym(lib, "nvmlDeviceGetMultiprocessorCount"); /* optional */
  if (!init || !count || !handle || !name || !mem) {
    snprintf(g_last_error, sizeof g_last_error, "NVML found but required symbols are missing");
    dlclose(lib);
    return 0;
  }
  if (init() != 0) {
    snprintf(g_last_error, sizeof g_last_error, "nvmlInit failed");
    dlclose(lib);
    return 0;
  }
  unsigned int n = 0;
  int found = 0;
  char driver[64] = "";
  if (drv) drv(driver, sizeof driver);
  if (count(&n) == 0) {
    for (unsigned int i = 0; i < n && found < max; ++i) {
      nvmlDevice_t h;
      if (handle(i, &h) != 0) continue;
      mqde_device_info* d = &out[found];
      memset(d, 0, sizeof(*d));
      d->kind = MQDE_DEV_GPU;
      d->index = (int)i;
      copy_str(d->vendor, sizeof d->vendor, "nvidia");
      name(h, d->name, sizeof d->name);
      nvmlMemory_t m;
      if (mem(h, &m) == 0) {
        d->mem_total_bytes = m.total;
        d->mem_free_bytes = m.free;
      }
      int major = 0, minor = 0;
      if (cc && cc(h, &major, &minor) == 0) snprintf(d->arch, sizeof d->arch, "sm_%d%d", major, minor);
      unsigned int sms = 0;
      if (mpc && mpc(h, &sms) == 0) d->compute_units = sms;
      copy_str(d->driver_version, sizeof d->driver_version, driver);
      ++found;
    }
  }
  if (shutdown) shutdown();
  dlclose(lib);
  return found;
}
#endif

/* Plugin devices after the built-in ones. A plugin device that the built-in probe already found (same vendor, name and
 * index, e.g. a CUDA plugin next to NVML) is not listed twice. */
static int append_plugin_devices(mqde_device_info* out, int have, int max) {
#if defined(_WIN32)
  (void)out; (void)have; (void)max;
  return 0;
#else
  mqde_hal_ensure_loaded();
  mqde_hal_device devs[64];
  int n = mqde_hal_devices(devs, 64, NULL);
  int w = have;
  for (int i = 0; i < n && w < max; ++i) {
    int dup = 0;
    for (int j = 0; j < have && !dup; ++j)
      dup = strcmp(out[j].vendor, devs[i].vendor) == 0 && strcmp(out[j].name, devs[i].name) == 0 && out[j].index == devs[i].index;
    if (dup) continue;
    mqde_device_info* d = &out[w++];
    memset(d, 0, sizeof *d);
    d->kind = (mqde_dev_kind)devs[i].kind;
    d->index = devs[i].index;
    copy_str(d->vendor, sizeof d->vendor, devs[i].vendor);
    copy_str(d->name, sizeof d->name, devs[i].name);
    copy_str(d->arch, sizeof d->arch, devs[i].arch);
    d->mem_total_bytes = devs[i].mem_total_bytes;
    d->mem_free_bytes = devs[i].mem_free_bytes;
    d->compute_units = devs[i].compute_units;
    copy_str(d->driver_version, sizeof d->driver_version, devs[i].driver_version);
  }
  return w - have;
#endif
}

int mqde_probe_all(mqde_device_info* out, int max_devices, int* n_found) {
  g_last_error[0] = '\0';
  if (!out || max_devices < 1 || !n_found) {
    snprintf(g_last_error, sizeof g_last_error, "invalid arguments");
    return -1;
  }
  int n = 0;
  probe_cpu(&out[n++]);
#if !defined(_WIN32)
  n += probe_nvidia(out + n, max_devices - n);
#endif
  n += append_plugin_devices(out, n, max_devices);
  *n_found = n;
  return 0;
}
